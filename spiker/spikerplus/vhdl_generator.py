"""VHDL generator for spiking neural networks.

This module provides functionality to generate VHDL code for
hardware implementation of spiking neural networks.
"""

from math import log2
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .vhdl.layer import Layer
from .vhdl.network import FullAccelerator, Network


class VhdlGenerator:
    """VHDL code generator for spiking neural networks.

    Generates synthesizable VHDL code for hardware deployment
    of trained spiking neural networks.
    """

    def __init__(self, net: nn.Module, optim_config: dict[str, Any]) -> None:
        """Initialize VHDL generator.

        Args:
            net: Trained neural network.
            optim_config: Optimization configuration dictionary.
        """
        self.net = net
        self.optim_config = optim_config

        self.input_size: int = self.input_size(next(iter(self.net.layers)))
        self.output_size: int = self.output_size(list(self.net.layers)[-2])

    def generate(
        self, functional: bool = True, interface: bool = False, debug: bool = False,
    ) -> Network | FullAccelerator:
        """Generate VHDL code for the network.

        Args:
            functional: Whether to generate functional VHDL code.
            interface: Whether to generate the full accelerator with interface.
            debug: Whether to include debug signals.

        Returns:
            Network or FullAccelerator VHDL code object.
        """
        vhdl_net = Network(self.net.n_cycles, debug=debug)
        self.functional = functional

        for layer in self.net.layers:
            if "fc" in layer:
                ff_w = self.extract_weights(layer)

            else:
                vhdl_net.add(self.init_layer(layer, ff_w))

        if not interface:
            return vhdl_net

        return FullAccelerator(vhdl_net, self.input_size, self.output_size)

    def input_size(self, layer: str) -> int:
        """Compute input size for a layer.

        Args:
            layer: Layer name.

        Returns:
            Number of input neurons for the layer.
        """
        if "fc" in layer:
            ff_w = self.extract_weights(layer)

            return ff_w.shape[1]

        raise ValueError("Cannot compute size. I need a linear layer")

    def output_size(self, layer: str) -> int:
        """Compute output size for a layer.

        Args:
            layer: Layer name.

        Returns:
            Number of output neurons for the layer.
        """
        if "fc" in layer:
            ff_w = self.extract_weights(layer)

            return ff_w.shape[0]

        raise ValueError("Cannot compute size. I need a linear layer")

    def init_layer(self, layer: str, ff_w: npt.NDArray[np.float64]) -> Layer:
        """Initialize a VHDL layer from network layer.

        Args:
            layer: Layer name.
            ff_w: Feed-forward weights array.

        Returns:
            Initialized VHDL Layer object.
        """
        th = np.repeat(self.extract_threshold(layer), ff_w.shape[0])
        beta_shift = self.extract_beta(layer)
        reset = self.extract_reset(layer)
        fb_w = self.extract_weights(layer)

        if not fb_w:
            fb_w = torch.zeros((ff_w.shape[0], ff_w.shape[0])).numpy()

        return Layer(
            label=layer,
            w_exc=ff_w,
            w_inh=fb_w,
            v_th=th,
            bitwidth=self.optim_config["neurons_bw"],
            fp_decimals=self.optim_config["fp_dec"],
            w_inh_bw=self.optim_config["weights_bw"],
            w_exc_bw=self.optim_config["weights_bw"],
            shift=beta_shift,
            reset=reset,
            functional=self.functional,
        )

    def extract_weights(self, layer: str) -> npt.NDArray[np.float64] | None:
        """Extract weights from a network layer.

        Args:
            layer: Layer name.

        Returns:
            Weight array or None if not available.
        """
        layer_obj = self.net.layers[layer]

        if hasattr(layer_obj, "weight"):
            return layer_obj.weight.data.cpu().numpy()

        if hasattr(layer_obj, "recurrent"):
            return layer_obj.recurrent.weight.data.cpu().numpy()

        return None

    def extract_threshold(self, layer: str) -> npt.NDArray[np.float64] | None:
        """Extract threshold from a network layer.

        Args:
            layer: Layer name.

        Returns:
            Threshold array or None if not available.
        """
        layer_obj = self.net.layers[layer]

        if hasattr(layer_obj, "threshold"):
            return np.array([layer_obj.threshold.data.item()])

        return None

    def extract_reset(self, layer: str) -> str | None:
        """Extract reset mechanism from a network layer.

        Args:
            layer: Layer name.

        Returns:
            Reset mechanism string or None if not available.
        """
        layer_obj = self.net.layers[layer]

        if hasattr(layer_obj, "reset_mechanism"):
            reset = layer_obj.reset_mechanism

            if reset == "subtract":
                return "subtractive"

            if reset == "zero":
                return "fixed"

            if reset == "none":
                return "none"

            msg = f"Reset type not supported: {reset}"
            raise ValueError(msg)

        return None

    def extract_alpha(self, layer: str) -> int:
        """Extract alpha decay parameter from a network layer.

        Args:
            layer: Layer name.

        Returns:
            Alpha shift value for VHDL.

        Raises:
            ValueError: If alpha is not a valid float or not in [0,1].
        """
        if hasattr(self.net.layers[layer], "alpha"):
            alpha = self.net.layers[layer].alpha.data.item()

            if not isinstance(alpha, float):
                raise ValueError("Alpha decay must be float")

            if alpha < 0.0 or alpha > 1.0:
                raise ValueError("Alpha decay must be between 0 and 1")

            return self.pow2_shift(1 - alpha)
        raise ValueError("Layer does not have alpha attribute")

    def extract_beta(self, layer: str) -> int:
        """Extract beta decay parameter from a network layer.

        Args:
            layer: Layer name.

        Returns:
            Beta shift value for VHDL.

        Raises:
            ValueError: If beta is not a valid float or not in [0,1].
        """
        if hasattr(self.net.layers[layer], "beta"):
            beta = self.net.layers[layer].beta.data.item()

            if not isinstance(beta, float):
                raise ValueError("Beta decay must be float")

            if beta < 0.0 or beta > 1.0:
                raise ValueError("Beta decay must be between 0 and 1")

            return self.pow2_shift(1 - beta)
        raise ValueError("Layer does not have beta attribute")

    def pow2_shift(self, value: float) -> int:
        """Calculate power-of-2 shift for fixed-point representation.

        Args:
            value: Value to compute shift for.

        Returns:
            Shift amount (log2 of value rounded).

        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError("Value must be positive for log2 calculation")
        return round(log2(value))
