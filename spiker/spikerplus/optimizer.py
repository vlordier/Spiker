"""Optimizer and quantizer for spiking neural networks.

This module provides quantization-aware optimization for spiking neural networks,
including fixed-point quantization and bit-width optimization.
"""

import json
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tabulate import tabulate

from .device import get_device
from .net_builder import SNN, NetBuilder
from .trainer import Trainer
from .types import ReadoutType

_LAYER_FC = "fc"
_LAYER_IF = "if"
_LAYER_LIF = "lif"
_LAYER_SYN = "syn"
_LAYER_RIF = "rif"
_LAYER_RLIF = "rlif"
_LAYER_RSYN = "rsyn"


class Quantizer:
    """Fixed-point quantizer for neural network parameters.

    Provides quantization functions for converting floating-point values
    to fixed-point representations with specified bit-widths.
    """

    def fixed_point(
        self,
        value: npt.NDArray[np.float64] | torch.Tensor | float,
        fp_dec: int,
        bitwidth: int,
    ) -> npt.NDArray[np.float64] | torch.Tensor | float:
        """Apply fixed-point quantization.

        Args:
            value: Value to quantize.
            fp_dec: Number of fractional bits.
            bitwidth: Total bit-width for quantization.

        Returns:
            Quantized value.

        """
        quant = value * 2**fp_dec

        return self.saturated_int(quant, bitwidth)

    def saturated_int(
        self, value: npt.NDArray[np.float64] | torch.Tensor | float, bitwidth: int
    ) -> npt.NDArray[np.float64] | torch.Tensor | float:
        """Apply saturated integer conversion.

        Args:
            value: Value to convert.
            bitwidth: Total bit-width for quantization.

        Returns:
            Converted value.

        """
        return self.saturate(self.to_int(value), bitwidth)

    def saturate(
        self, value: npt.NDArray[np.float64] | torch.Tensor | float, bitwidth: int
    ) -> npt.NDArray[np.float64] | torch.Tensor | float:
        """Saturate values to fit within bit-width limits.

        Args:
            value: Value to saturate.
            bitwidth: Total bit-width for quantization.

        Returns:
            Saturated value.

        """
        if (
            type(value).__module__ == np.__name__
            or type(value).__module__ == torch.__name__
        ):
            value[value > 2 ** (bitwidth - 1) - 1] = 2 ** (bitwidth - 1) - 1
            value[value < -(2 ** (bitwidth - 1))] = -(2 ** (bitwidth - 1))

            return value.float()

        if value > 2 ** (bitwidth - 1) - 1:
            value = 2 ** (bitwidth - 1) - 1

        elif value < -(2 ** (bitwidth - 1)):
            value = -(2 ** (bitwidth - 1))

        return float(value)

    def to_int(
        self, value: npt.NDArray[np.float64] | torch.Tensor | float
    ) -> npt.NDArray[np.float64] | torch.Tensor | float:
        """Convert value to integer representation.

        Args:
            value: Value to convert.

        Returns:
            Integer representation of the value.

        """
        if type(value).__module__ == np.__name__:
            quant = value.astype(int).astype(float)

        elif type(value).__module__ == torch.__name__:
            quant = value.type(torch.int64).float()

        else:
            quant = float(int(value))

        return quant


class QuantSNN(SNN):
    """Quantized Spiking Neural Network.

    Extends the base SNN with quantized neuron states for
    hardware-efficient inference.
    """

    def __init__(self, net_dict: dict[str, Any], neurons_bw: int) -> None:
        """Initialize quantized SNN.

        Args:
            net_dict: Network configuration dictionary.
            neurons_bw: Bit-width for neuron state quantization.

        """
        super().__init__(net_dict)

        self.neurons_bw = neurons_bw

        self.quantizer = Quantizer()

    def forward(self, input_spikes: torch.Tensor) -> None:
        """Forward pass with quantization.

        Args:
            input_spikes: Input spike tensor of shape (n_cycles, n_inputs).

        """
        self.reset()

        cur: dict[str, torch.Tensor] = {}

        if input_spikes.shape[0] != self.n_cycles:
            logging.warning(
                "Input data have a time dimension different from "
                "the network's number of steps. It's ok at this level, "
                "but remember to use a suitable number of steps in the "
                "vhdl generator"
            )

        for step in range(input_spikes.shape[0]):
            first = True
            prev_layer: str | None = None

            for layer in self.layers:
                idx = str(self.extract_index(layer))

                if _LAYER_FC in layer:
                    if first:
                        cur[layer] = self.layers[layer](input_spikes[step])
                        first = False

                    else:
                        cur[layer] = self.layers[layer](self.spk[prev_layer])

                elif layer == f"{_LAYER_IF}{idx}" or layer == f"{_LAYER_LIF}{idx}":
                    self.spk[layer], self.mem[layer] = self.layers[layer](
                        cur[prev_layer], self.mem[layer]
                    )

                elif layer == f"{_LAYER_SYN}{idx}":
                    self.spk[layer], self.syn[layer], self.mem[layer] = self.layers[
                        layer
                    ](cur[prev_layer], self.syn[layer], self.mem[layer])

                elif layer == f"{_LAYER_RIF}{idx}" or layer == f"{_LAYER_RLIF}{idx}":
                    self.spk[layer], self.mem[layer] = self.layers[layer](
                        cur[prev_layer], self.spk[layer], self.mem[layer]
                    )

                elif layer == f"{_LAYER_RSYN}{idx}":
                    self.spk[layer], self.syn[layer], self.mem[layer] = self.layers[
                        layer
                    ](
                        cur[prev_layer],
                        self.spk[layer],
                        self.syn[layer],
                        self.mem[layer],
                    )

                prev_layer = layer

                self.quantize(layer)

                self.record(layer)

        self.stack_rec()

    def quantize(self, layer: str) -> None:
        """Quantize neuron states for a layer.

        Args:
            layer: Name of the layer to quantize.

        """
        if "fc" not in layer:
            self.mem[layer] = self.quantizer.saturated_int(
                self.mem[layer], self.neurons_bw
            )

            if "syn" in layer:
                self.syn[layer] = self.quantizer.saturated_int(
                    self.syn[layer], self.neurons_bw
                )


class Optimizer:
    """Quantization-aware optimizer for spiking neural networks.

    Provides grid search over quantization parameters to find
    optimal bit-widths for weights and neuron states.
    """

    def __init__(
        self,
        net: SNN,
        net_dict: dict[str, Any],
        optim_config: dict[str, Any],
        readout_type: str | ReadoutType = ReadoutType.MEM,
    ) -> None:
        """Initialize optimizer.

        Args:
            net: Neural network to optimize.
            net_dict: Network configuration dictionary.
            optim_config: Optimizer configuration dictionary.
            readout_type: Type of readout to use.

        """
        self._trainer = Trainer(net, readout_type)
        self._net_builder = NetBuilder(net_dict)

        self.default_config: dict[str, dict[str, int]] = {
            "weights_bw": {"min": 4, "max": 8},
            "neurons_bw": {"min": 4, "max": 10},
            "fp_dec": {"min": 2, "max": 3},
        }

        self.allowed_keys = self.default_config.keys()

        self.quantizer = Quantizer()

        self.state_dict = net.state_dict()
        self._parsed_net_dict = self._net_builder.parse_config(net_dict)

        self.optim_config = self.parse_opt_config(optim_config)

        self.device = get_device(prefer_gpu=True)

    @property
    def net(self) -> SNN:
        """Get the network."""
        return self._trainer.net

    @net.setter
    def net(self, value: SNN) -> None:
        """Set the network."""
        self._trainer.net = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self._trainer.optimizer

    @property
    def loss_fn(self) -> nn.Module:
        """Get the loss function."""
        return self._trainer.loss_fn

    @property
    def net_dict(self) -> dict[str, Any]:
        """Get the parsed network configuration."""
        return self._parsed_net_dict

    def parse_config(self, net_dict: dict[str, Any]) -> dict[str, Any]:
        """Parse network configuration.

        Args:
            net_dict: Raw network configuration dictionary.

        Returns:
            Parsed network configuration dictionary.
        """
        return self._net_builder.parse_config(net_dict)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """Evaluate the network.

        Args:
            dataloader: DataLoader for evaluation data.

        Returns:
            Tuple of (loss, accuracy).
        """
        return self._trainer.evaluate(dataloader)

    def parse_opt_config(
        self, optim_config: dict[str, Any]
    ) -> dict[str, dict[str, int]]:
        """Parse optimization configuration.

        Args:
            optim_config: Optimizer configuration dictionary.

        Returns:
            Parsed optimization configuration.

        """
        optim_dict: dict[str, dict[str, int]] = {}

        for key in optim_config:
            if key in self.allowed_keys:
                if "min" in optim_config[key]:
                    if not isinstance(optim_config[key]["min"], int):
                        msg = "Range specifiers must be integers"
                        raise ValueError(msg)

                    min_value = optim_config[key]["min"]

                else:
                    min_value = self.default_config[key]["min"]

                if "max" in optim_config[key]:
                    if not isinstance(optim_config[key]["max"], int):
                        msg = "Range specifiers must be integers"
                        raise ValueError(msg)

                    max_value = optim_config[key]["max"]

                else:
                    max_value = self.default_config[key]["max"]

                optim_dict[key] = [i for i in range(min_value, max_value + 1)]

        log_message = "Optimizer configured: \n"
        log_message += json.dumps(optim_dict, indent=4)
        logging.info(log_message)

        return optim_dict

    def optimize(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Run quantization-aware optimization.

        Args:
            dataloader: DataLoader for evaluation data.

        """
        headers = [
            "Fixed-point decimals",
            "Neurons' bitwidth",
            "Weights bitwidth",
            "Loss",
            "Accuracy",
        ]
        table: list[list[str]] = []

        for fp_dec in self.optim_config["fp_dec"]:
            for w_bw in self.optim_config["weights_bw"]:
                for neuron_bw in self.optim_config["neurons_bw"]:
                    self.build_quant_snn(w_bw, neuron_bw, fp_dec)

                    loss, acc = self.evaluate(dataloader)

                    log_message = f"\nLoss: {loss:.2f}\nAcc: {acc * 100:.2f}%\n"
                    logging.info(log_message)

                    table.append(
                        [
                            str(fp_dec),
                            str(neuron_bw),
                            str(w_bw),
                            str(loss),
                            f"{acc * 100:.2f}%",
                        ]
                    )

        table_str = "\n" + tabulate(table, headers=headers, tablefmt="grid")

        logging.info(table_str)

    def build_quant_snn(self, weights_bw: int, neurons_bw: int, fp_dec: int) -> None:
        """Build quantized SNN with specified parameters.

        Args:
            weights_bw: Bit-width for weight quantization.
            neurons_bw: Bit-width for neuron state quantization.
            fp_dec: Number of fractional bits for fixed-point representation.

        """
        self.net = QuantSNN(self.net_dict, neurons_bw)

        quant_state_dict = self.state_dict.copy()

        for key in quant_state_dict:
            if "weight" in key:
                quant_state_dict[key] = self.quantizer.fixed_point(
                    quant_state_dict[key], fp_dec, weights_bw
                )

            elif "threshold" in key:
                quant_state_dict[key] = self.quantizer.fixed_point(
                    quant_state_dict[key], fp_dec, neurons_bw
                )

        self.net.load_state_dict(quant_state_dict)

        self.net.to(self.device)

        log_message = (
            f"Network ready:\n"
            f"Fixed-point decimals: {fp_dec}\n"
            f"Neurons bitwidth: {neurons_bw}\n"
            f"Weights bitwidth: {weights_bw}\n"
        )
        logging.info(log_message)
