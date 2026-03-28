"""Comprehensive tests for optimizer module."""

import pytest
import torch

from spikerplus.net_builder import NetBuilder
from spikerplus.optimizer import Optimizer, Quantizer, QuantSNN


class TestQuantizer:
    """Test suite for Quantizer class."""

    def test_fixed_point(self):
        """Test fixed-point quantization."""
        quantizer = Quantizer()

        value = torch.tensor([1.5, -0.5, 0.25])
        fp_dec = 2
        bitwidth = 8

        result = quantizer.fixed_point(value, fp_dec, bitwidth)

        assert result is not None
        assert result.dtype == torch.float32

    def test_saturated_int(self):
        """Test saturated integer conversion."""
        quantizer = Quantizer()

        value = torch.tensor([100.0, -100.0, 50.0])
        bitwidth = 8

        result = quantizer.saturated_int(value, bitwidth)

        assert result is not None
        assert result.dtype == torch.float32

    def test_saturate_tensor(self):
        """Test saturation for tensor values."""
        quantizer = Quantizer()

        value = torch.tensor([200.0, -200.0, 50.0])
        bitwidth = 8

        result = quantizer.saturate(value, bitwidth)

        # Values should be saturated to [-128, 127]
        assert torch.all(result <= 127.0)
        assert torch.all(result >= -128.0)

    def test_saturate_scalar(self):
        """Test saturation for scalar values."""
        quantizer = Quantizer()

        # Test value above max
        result = quantizer.saturate(200.0, 8)
        assert result == 127.0

        # Test value below min
        result = quantizer.saturate(-200.0, 8)
        assert result == -128.0

        # Test value in range
        result = quantizer.saturate(50.0, 8)
        assert result == 50.0

    def test_to_int_tensor(self):
        """Test integer conversion for tensor."""
        quantizer = Quantizer()

        value = torch.tensor([1.5, 2.7, 3.2])
        result = quantizer.to_int(value)

        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_to_int_scalar(self):
        """Test integer conversion for scalar."""
        quantizer = Quantizer()

        result = quantizer.to_int(1.5)
        assert result == 1.0

        result = quantizer.to_int(2.7)
        assert result == 2.0


class TestQuantSNN:
    """Test suite for QuantSNN class."""

    def test_quant_snn_initialization(self):
        """Test QuantSNN initialization."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "learn_beta": False,
                "learn_threshold": False,
                "reset_mechanism": "subtract",
            },
        }

        quant_snn = QuantSNN(net_dict, neurons_bw=8)

        assert quant_snn.neurons_bw == 8
        assert quant_snn.quantizer is not None

    def test_quant_snn_forward(self):
        """Test QuantSNN forward pass."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "learn_beta": False,
                "learn_threshold": False,
                "reset_mechanism": "subtract",
            },
        }

        quant_snn = QuantSNN(net_dict, neurons_bw=8)

        input_spikes = torch.randn(10, 2, 5)
        quant_snn(input_spikes)

        assert hasattr(quant_snn, "spk_rec")
        assert hasattr(quant_snn, "mem_rec")


class TestOptimizer:
    """Test suite for Optimizer class."""

    def _create_simple_network(self):
        """Helper to create a simple network for testing."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
            "layer_1": {
                "neuron_model": "lif",
                "n_neurons": 3,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "none",
            },
        }

        net_builder = NetBuilder(net_dict)
        return net_builder.build(), net_dict

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        snn, net_dict = self._create_simple_network()

        optim_config = {
            "weights_bw": {"min": 4, "max": 8},
            "neurons_bw": {"min": 4, "max": 10},
            "fp_dec": {"min": 2, "max": 3},
        }

        optimizer = Optimizer(snn, net_dict, optim_config)

        assert optimizer.net == snn
        assert optimizer.net_dict is not None
        assert optimizer.optim_config is not None

    def test_optimizer_default_config(self):
        """Test optimizer with default config values."""
        snn, net_dict = self._create_simple_network()

        optim_config = {
            "weights_bw": {"min": 4},
            "neurons_bw": {"max": 10},
            "fp_dec": {},
        }

        optimizer = Optimizer(snn, net_dict, optim_config)

        # Should use default values for missing keys
        assert 4 in optimizer.optim_config["weights_bw"]
        assert 10 in optimizer.optim_config["neurons_bw"]
        assert 2 in optimizer.optim_config["fp_dec"]  # Default min

    def test_optimizer_invalid_min_type(self):
        """Test that non-integer min raises ValueError."""
        snn, net_dict = self._create_simple_network()

        optim_config = {
            "weights_bw": {"min": "4"},  # Should be int
            "neurons_bw": {"min": 4, "max": 10},
            "fp_dec": {"min": 2, "max": 3},
        }

        with pytest.raises(ValueError, match="Range specifiers must be integers"):
            Optimizer(snn, net_dict, optim_config)

    def test_optimizer_invalid_max_type(self):
        """Test that non-integer max raises ValueError."""
        snn, net_dict = self._create_simple_network()

        optim_config = {
            "weights_bw": {"min": 4, "max": "8"},  # Should be int
            "neurons_bw": {"min": 4, "max": 10},
            "fp_dec": {"min": 2, "max": 3},
        }

        with pytest.raises(ValueError, match="Range specifiers must be integers"):
            Optimizer(snn, net_dict, optim_config)

    def test_build_quant_snn(self):
        """Test building quantized SNN."""
        snn, net_dict = self._create_simple_network()

        optim_config = {
            "weights_bw": {"min": 4, "max": 8},
            "neurons_bw": {"min": 4, "max": 10},
            "fp_dec": {"min": 2, "max": 3},
        }

        optimizer = Optimizer(snn, net_dict, optim_config)

        # Build quantized network
        optimizer.build_quant_snn(weights_bw=8, neurons_bw=8, fp_dec=2)

        assert optimizer.net is not None
        assert isinstance(optimizer.net, QuantSNN)
