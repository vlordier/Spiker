"""Network builder for Spiker framework.

This module provides classes for building and configuring spiking neural networks
with various neuron models and configurations.
"""

import json
import logging
import re
from typing import Any

import snntorch as snn
import torch
from torch import nn

from .exceptions import LayerConfigError, NeuronModelError
from .types import NeuronModel, ResetMechanism

_LAYER_FC = "fc"
_LAYER_IF = "if"
_LAYER_LIF = "lif"
_LAYER_SYN = "syn"
_LAYER_RIF = "rif"
_LAYER_RLIF = "rlif"
_LAYER_RSYN = "rsyn"


class SNN(nn.Module):
    """Spiking Neural Network.

    A PyTorch module implementing a spiking neural network with various
    neuron models including LIF, IF, Synaptic, and their recurrent variants.
    """

    def __init__(self, net_dict: dict[str, Any]) -> None:
        """Initialize SNN.

        Args:
            net_dict: Network configuration dictionary.

        """
        super().__init__()

        self.n_cycles: int = net_dict["n_cycles"]

        self.layers: nn.ModuleDict = nn.ModuleDict()

        self.syn: dict[str, torch.Tensor] = {}
        self.mem: dict[str, torch.Tensor] = {}
        self.spk: dict[str, torch.Tensor] = {}

        self.syn_rec: dict[str, list[torch.Tensor]] = {}
        self.mem_rec: dict[str, list[torch.Tensor]] = {}
        self.spk_rec: dict[str, list[torch.Tensor]] = {}

        self.build_snn(net_dict)

    def build_snn(self, net_dict: dict[str, Any]) -> None:  # noqa: C901
        """Build the spiking neural network layers.

        Args:
            net_dict: Network configuration dictionary.

        """
        first: bool = True

        for key in net_dict:
            if "layer" in key:
                idx: str = str(self.extract_index(key) + 1)
                layer_config = net_dict[key]
                neuron_model_str = layer_config["neuron_model"]

                try:
                    neuron_model = NeuronModel(neuron_model_str)
                except ValueError:
                    supported = ", ".join(m.value for m in NeuronModel)
                    msg = (
                        f"Invalid neuron model '{neuron_model_str}'. "
                        f"Choose from: {supported}"
                    )
                    raise NeuronModelError(msg) from None

                if first:
                    self.layers[f"{_LAYER_FC}{idx}"] = nn.Linear(
                        in_features=net_dict["n_inputs"],
                        out_features=layer_config["n_neurons"],
                        bias=False,
                    )

                    n_inputs_next: int = layer_config["n_neurons"]

                    first = False

                else:
                    self.layers[f"{_LAYER_FC}{idx}"] = nn.Linear(
                        in_features=n_inputs_next,
                        out_features=layer_config["n_neurons"],
                        bias=False,
                    )

                    n_inputs_next = layer_config["n_neurons"]

                name = f"{neuron_model.value}{idx}"

                match neuron_model:
                    case NeuronModel.IF:
                        self.layers[name] = snn.Leaky(
                            beta=0.0,
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

                    case NeuronModel.LIF:
                        self.layers[name] = snn.Leaky(
                            beta=layer_config["beta"],
                            learn_beta=layer_config["learn_beta"],
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

                    case NeuronModel.SYN:
                        self.layers[name] = snn.Synaptic(
                            alpha=layer_config["alpha"],
                            learn_alpha=layer_config["learn_alpha"],
                            beta=layer_config["beta"],
                            learn_beta=layer_config["learn_beta"],
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

                    case NeuronModel.RIF:
                        self.layers[name] = snn.RLeaky(
                            linear_features=layer_config["n_neurons"],
                            beta=0.0,
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

                    case NeuronModel.RLIF:
                        self.layers[name] = snn.RLeaky(
                            linear_features=layer_config["n_neurons"],
                            beta=layer_config["beta"],
                            learn_beta=layer_config["learn_beta"],
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

                    case NeuronModel.RSYN:
                        self.layers[name] = snn.RSynaptic(
                            linear_features=layer_config["n_neurons"],
                            alpha=layer_config["alpha"],
                            learn_alpha=layer_config["learn_alpha"],
                            beta=layer_config["beta"],
                            learn_beta=layer_config["learn_beta"],
                            threshold=layer_config["threshold"],
                            learn_threshold=layer_config["learn_threshold"],
                            reset_mechanism=layer_config["reset_mechanism"],
                        )

    def reset(self) -> None:
        """Reset all neuron states and recordings."""
        for layer in self.layers:
            idx = str(self.extract_index(layer))

            if _LAYER_FC not in layer:
                self.mem_rec[layer] = []
                self.syn_rec[layer] = []
                self.spk_rec[layer] = []

                if layer in {f"{_LAYER_IF}{idx}", f"{_LAYER_LIF}{idx}"}:
                    self.mem[layer] = self.layers[layer].reset_mem()

                elif layer == f"{_LAYER_SYN}{idx}":
                    self.syn[layer], self.mem[layer] = self.layers[layer].reset_mem()

                elif layer in {f"{_LAYER_RIF}{idx}", f"{_LAYER_RLIF}{idx}"}:
                    self.spk[layer], self.mem[layer] = self.layers[layer].reset_mem()

                elif layer == f"{_LAYER_RSYN}{idx}":
                    self.spk[layer], self.syn[layer], self.mem[layer] = self.layers[
                        layer
                    ].reset_mem()

    def record(self, layer: str) -> None:
        """Record neuron states for a layer.

        Args:
            layer: Name of the layer to record.

        """
        if _LAYER_FC not in layer:
            self.mem_rec[layer].append(self.mem[layer])
            self.spk_rec[layer].append(self.spk[layer])

            if _LAYER_SYN in layer:
                self.syn_rec[layer].append(self.syn[layer])

    def stack_rec(self) -> None:
        """Stack recordings into tensors."""
        for layer in self.layers:
            if _LAYER_FC not in layer:
                self.mem_rec[layer] = torch.stack(self.mem_rec[layer], dim=0)
                self.spk_rec[layer] = torch.stack(self.spk_rec[layer], dim=0)

                if _LAYER_SYN in layer:
                    self.syn_rec[layer] = torch.stack(self.syn_rec[layer], dim=0)

    def extract_index(self, layer_name: str) -> int:
        """Extract numeric index from layer name.

        Args:
            layer_name: Name of the layer (e.g., 'layer_0', 'fc1').

        Returns:
            Numeric index extracted from the layer name.

        Raises:
            ValueError: If the layer name does not contain exactly one integer.

        """
        index = re.findall(r"\d+", layer_name)

        if len(index) != 1:
            msg = (
                f"Invalid layer name: {layer_name}. "
                f'Use "layer_" + <integer layer index>'
            )
            raise ValueError(msg)
        return int(index[0])

    def forward(self, input_spikes: torch.Tensor) -> None:
        """Forward pass through the network.

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
                "vhdl generator",
            )

        for step in range(input_spikes.shape[0]):
            first: bool = True
            prev_layer: str | None = None

            for layer in self.layers:
                idx: str = str(self.extract_index(layer))

                if _LAYER_FC in layer:
                    if first:
                        cur[layer] = self.layers[layer](input_spikes[step])
                        first = False
                        prev_layer = layer

                    else:
                        cur[layer] = self.layers[layer](self.spk[prev_layer])
                        prev_layer = layer

                elif layer in {f"{_LAYER_IF}{idx}", f"{_LAYER_LIF}{idx}"}:
                    self.spk[layer], self.mem[layer] = self.layers[layer](
                        cur[prev_layer],
                        self.mem[layer],
                    )
                    prev_layer = layer

                elif layer == f"{_LAYER_SYN}{idx}":
                    self.spk[layer], self.syn[layer], self.mem[layer] = self.layers[
                        layer
                    ](cur[prev_layer], self.syn[layer], self.mem[layer])
                    prev_layer = layer

                elif layer in {f"{_LAYER_RIF}{idx}", f"{_LAYER_RLIF}{idx}"}:
                    self.spk[layer], self.mem[layer] = self.layers[layer](
                        cur[prev_layer],
                        self.spk[layer],
                        self.mem[layer],
                    )
                    prev_layer = layer

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

                self.record(layer)

        self.stack_rec()


class NetBuilder:
    """Network builder for spiking neural networks.

    Provides configuration parsing and validation for building
    spiking neural networks with various neuron models.
    """

    def __init__(self, net_dict: dict[str, Any]) -> None:
        """Initialize network builder.

        Args:
            net_dict: Network configuration dictionary.

        """
        self.default_dict: dict[str, Any] = {
            "n_cycles": 73,
            "n_inputs": 40,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 128,
                "alpha": 0.9,
                "learn_alpha": False,
                "beta": 0.9375,
                "learn_beta": False,
                "threshold": 1.0,
                "learn_threshold": False,
                "reset_mechanism": "subtract",
            },
            "layer_1": {
                "neuron_model": "lif",
                "n_neurons": 10,
                "alpha": 0.9,
                "learn_alpha": False,
                "beta": 0.9375,
                "learn_beta": False,
                "threshold": 1.0,
                "learn_threshold": False,
                "reset_mechanism": "none",
            },
        }

        self.net_allowed_keys: list[str] = self.select_keys()
        self.supported_models: list[str] = [model.value for model in NeuronModel]

        self.has_alpha: dict[str, bool] = {
            NeuronModel.IF.value: False,
            NeuronModel.LIF.value: False,
            NeuronModel.SYN.value: True,
            NeuronModel.RIF.value: False,
            NeuronModel.RLIF.value: False,
            NeuronModel.RSYN.value: True,
        }

        self.has_beta: dict[str, bool] = {
            NeuronModel.IF.value: False,
            NeuronModel.LIF.value: True,
            NeuronModel.SYN.value: True,
            NeuronModel.RIF.value: False,
            NeuronModel.RLIF.value: True,
            NeuronModel.RSYN.value: True,
        }

        self.supported_resets: list[str] = [
            mechanism.value for mechanism in ResetMechanism
        ]

        self.net_dict: dict[str, Any] = self.parse_config(net_dict)

    def build(self) -> SNN:
        """Build and return the spiking neural network.

        Returns:
            Built SNN instance.

        """
        snn = SNN(self.net_dict)

        logging.info("Network ready: %s", snn)

        return snn

    def select_keys(self) -> list[str]:
        """Extract allowed configuration keys from default dictionary.

        Returns:
            List of allowed configuration keys.

        """
        keywords = self.default_dict.keys()

        allowed_keys: list[str] = []

        for k in keywords:
            k = re.sub(r"\d+", "", k)

            if k not in allowed_keys:
                allowed_keys.append(k)

        return allowed_keys

    def _parse_global_config(
        self,
        net_dict: dict[str, Any],
        parsed_dict: dict[str, Any],
    ) -> None:
        """Parse global network configuration parameters."""
        for key in net_dict:
            if (
                any(allowed in key for allowed in self.net_allowed_keys)
                and "layer" not in key
            ):
                if not isinstance(net_dict[key], int):
                    msg = f"{key} must be an integer value"
                    raise LayerConfigError(msg)

                parsed_dict[key] = net_dict[key]

    def _parse_layer_config(
        self,
        net_dict: dict[str, Any],
        parsed_dict: dict[str, Any],
    ) -> None:
        """Parse layer configuration parameters."""
        for key in net_dict:
            if (
                any(allowed in key for allowed in self.net_allowed_keys)
                and "layer" in key
            ):
                parsed_dict[key] = {}
                layer: dict[str, Any] = net_dict[key]

                self._parse_layer_neurons(layer, parsed_dict[key])
                self._parse_layer_model(layer, parsed_dict[key])
                self._parse_layer_threshold(layer, parsed_dict[key])
                self._parse_layer_reset(layer, parsed_dict[key])
                self._parse_layer_alpha(layer, parsed_dict[key])
                self._parse_layer_beta(layer, parsed_dict[key])

    def _parse_layer_neurons(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse neuron count for a layer."""
        if "n_neurons" in layer:
            if not isinstance(layer["n_neurons"], int):
                msg = "Number of neurons must be integer"
                raise LayerConfigError(msg)

            parsed_layer["n_neurons"] = layer["n_neurons"]
        else:
            parsed_layer["n_neurons"] = self.default_dict["layer_0"]["n_neurons"]

    def _parse_layer_model(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse neuron model for a layer."""
        if "neuron_model" in layer:
            if layer["neuron_model"] not in self.supported_models:
                supported = self.supported_models
                msg = f"Unsupported neuron model. Choose from: {supported}"
                raise NeuronModelError(msg)

            parsed_layer["neuron_model"] = layer["neuron_model"]
        else:
            parsed_layer["neuron_model"] = self.default_dict["layer_0"]["neuron_model"]

    def _parse_layer_threshold(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse threshold configuration for a layer."""
        if "threshold" in layer:
            if not isinstance(layer["threshold"], (int, float)):
                msg = "Threshold must be numeric"
                raise LayerConfigError(msg)

            parsed_layer["threshold"] = layer["threshold"]
        else:
            parsed_layer["threshold"] = self.default_dict["layer_0"]["threshold"]

        if "learn_threshold" in layer:
            if not isinstance(layer["learn_threshold"], bool):
                msg = "learn_threshold must be boolean"
                raise LayerConfigError(msg)

            parsed_layer["learn_threshold"] = layer["learn_threshold"]
        else:
            parsed_layer["learn_threshold"] = self.default_dict["layer_0"][
                "learn_threshold"
            ]

    def _parse_layer_reset(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse reset mechanism for a layer."""
        if "reset_mechanism" in layer:
            if layer["reset_mechanism"] not in self.supported_resets:
                supported = self.supported_resets
                msg = f"Invalid reset mechanism. Choose from: {supported}"
                raise LayerConfigError(msg)

            parsed_layer["reset_mechanism"] = layer["reset_mechanism"]
        else:
            parsed_layer["reset_mechanism"] = self.default_dict["layer_0"][
                "reset_mechanism"
            ]

    def _parse_layer_alpha(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse alpha decay for a layer."""
        if self.has_alpha[parsed_layer["neuron_model"]]:
            if "alpha" in layer:
                if not isinstance(layer["alpha"], float):
                    msg = "Alpha decay must be float"
                    raise LayerConfigError(msg)

                if layer["alpha"] < 0.0 or layer["alpha"] > 1.0:
                    msg = "Alpha decay must be between 0 and 1"
                    raise LayerConfigError(msg)

                parsed_layer["alpha"] = layer["alpha"]
            else:
                parsed_layer["alpha"] = self.default_dict["layer_0"]["alpha"]

            if "learn_alpha" in layer:
                if not isinstance(layer["learn_alpha"], bool):
                    msg = "learn_alpha must be boolean"
                    raise LayerConfigError(msg)

                parsed_layer["learn_alpha"] = layer["learn_alpha"]
            else:
                parsed_layer["learn_alpha"] = self.default_dict["layer_0"][
                    "learn_alpha"
                ]

    def _parse_layer_beta(
        self,
        layer: dict[str, Any],
        parsed_layer: dict[str, Any],
    ) -> None:
        """Parse beta decay for a layer."""
        if self.has_beta[parsed_layer["neuron_model"]]:
            if "beta" in layer:
                if not isinstance(layer["beta"], float):
                    msg = "Beta decay must be float"
                    raise LayerConfigError(msg)

                if layer["beta"] < 0.0 or layer["beta"] > 1.0:
                    msg = "Beta decay must be between 0 and 1"
                    raise LayerConfigError(msg)

                parsed_layer["beta"] = layer["beta"]
            else:
                parsed_layer["beta"] = self.default_dict["layer_0"]["beta"]

            if "learn_beta" in layer:
                if not isinstance(layer["learn_beta"], bool):
                    msg = "learn_beta must be boolean"
                    raise LayerConfigError(msg)

                parsed_layer["learn_beta"] = layer["learn_beta"]
            else:
                parsed_layer["learn_beta"] = self.default_dict["layer_0"]["learn_beta"]

    def parse_config(self, net_dict: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate network configuration.

        Args:
            net_dict: Raw network configuration dictionary.

        Returns:
            Validated and parsed network configuration dictionary.

        """
        parsed_dict: dict[str, Any] = {}

        self._parse_global_config(net_dict, parsed_dict)
        self._parse_layer_config(net_dict, parsed_dict)

        if "n_cycles" not in parsed_dict:
            parsed_dict["n_cycles"] = self.default_dict["n_cycles"]

        if "n_inputs" not in parsed_dict:
            parsed_dict["n_inputs"] = self.default_dict["n_inputs"]

        at_least_one_layer: bool = False
        for key in parsed_dict:
            if "layer_" in key:
                at_least_one_layer = True

        if not at_least_one_layer:
            for key in self.default_dict:
                if "layer_" in key:
                    parsed_dict[key] = self.default_dict[key]

        log_message: str = "Network configured: \n"
        log_message += json.dumps(parsed_dict, indent=4) + "\n"

        logging.info(log_message)

        return parsed_dict


if __name__ == "__main__":
    from net_dict import net_dict

    logging.basicConfig(level=logging.INFO)

    net_builder = NetBuilder(net_dict)

    snn = net_builder.build()
