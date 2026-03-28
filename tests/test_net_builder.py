"""Comprehensive tests for net_builder module."""

import pytest
import torch

from spikerplus.net_builder import NetBuilder, SNN


class TestNetBuilder:
    """Test suite for NetBuilder class."""

    def test_basic_lif_network(self):
        """Test building a basic LIF network."""
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
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert hasattr(snn, "layers")
        assert len(snn.layers) == 2  # fc1 + lif1

    def test_multi_layer_network(self):
        """Test building a multi-layer network."""
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
                "n_neurons": 4,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
            "layer_2": {
                "neuron_model": "lif",
                "n_neurons": 2,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "none",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert len(snn.layers) == 6  # 3 fc + 3 lif

    def test_if_neuron_model(self):
        """Test building network with IF neuron model."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "if",
                "n_neurons": 8,
                "threshold": 1.0,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert "if1" in snn.layers

    def test_syn_neuron_model(self):
        """Test building network with Synaptic neuron model."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "syn",
                "n_neurons": 8,
                "threshold": 1.0,
                "alpha": 0.8,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert "syn1" in snn.layers

    def test_rif_neuron_model(self):
        """Test building network with RIF neuron model."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "rif",
                "n_neurons": 8,
                "threshold": 1.0,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert "rif1" in snn.layers

    def test_rlif_neuron_model(self):
        """Test building network with RLIF neuron model."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "rlif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert "rlif1" in snn.layers

    def test_rsyn_neuron_model(self):
        """Test building network with RSynaptic neuron model."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "rsyn",
                "n_neurons": 8,
                "threshold": 1.0,
                "alpha": 0.8,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert "rsyn1" in snn.layers

    def test_invalid_neuron_model(self):
        """Test that invalid neuron model raises NeuronModelError."""
        from spikerplus.exceptions import NeuronModelError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "invalid",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        with pytest.raises(NeuronModelError, match="Unsupported neuron model"):
            NetBuilder(net_dict)

    def test_invalid_n_neurons_type(self):
        """Test that non-integer n_neurons raises LayerConfigError."""
        from spikerplus.exceptions import LayerConfigError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": "8",  # Should be int
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        with pytest.raises(LayerConfigError, match="Number of neurons must be integer"):
            NetBuilder(net_dict)

    def test_invalid_threshold_type(self):
        """Test that non-numeric threshold raises LayerConfigError."""
        from spikerplus.exceptions import LayerConfigError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": "1.0",  # Should be numeric
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        with pytest.raises(LayerConfigError, match="Threshold must be numeric"):
            NetBuilder(net_dict)

    def test_invalid_beta_range(self):
        """Test that beta outside 0-1 range raises LayerConfigError."""
        from spikerplus.exceptions import LayerConfigError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 1.5,  # Should be between 0 and 1
                "reset_mechanism": "subtract",
            },
        }

        with pytest.raises(
            LayerConfigError, match="Beta decay must be between 0 and 1"
        ):
            NetBuilder(net_dict)

    def test_invalid_alpha_range(self):
        """Test that alpha outside 0-1 range raises LayerConfigError."""
        from spikerplus.exceptions import LayerConfigError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "syn",
                "n_neurons": 8,
                "threshold": 1.0,
                "alpha": -0.1,  # Should be between 0 and 1
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        with pytest.raises(
            LayerConfigError, match="Alpha decay must be between 0 and 1"
        ):
            NetBuilder(net_dict)

    def test_invalid_reset_mechanism(self):
        """Test that invalid reset mechanism raises LayerConfigError."""
        from spikerplus.exceptions import LayerConfigError

        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "invalid",
            },
        }

        with pytest.raises(LayerConfigError, match="Invalid reset mechanism"):
            NetBuilder(net_dict)

    def test_default_values(self):
        """Test that default values are used when not specified."""
        net_dict = {
            "layer_0": {
                "n_neurons": 8,
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None
        assert snn.n_cycles == 73  # Default value

    def test_learn_threshold(self):
        """Test learn_threshold parameter."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "learn_threshold": True,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None

    def test_learn_beta(self):
        """Test learn_beta parameter."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "learn_beta": True,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None

    def test_learn_alpha(self):
        """Test learn_alpha parameter."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "syn",
                "n_neurons": 8,
                "threshold": 1.0,
                "alpha": 0.8,
                "learn_alpha": True,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn is not None


class TestSNN:
    """Test suite for SNN class."""

    def test_forward_pass(self):
        """Test forward pass through the network."""
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
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        input_spikes = torch.randn(10, 2, 5)
        snn(input_spikes)

        assert hasattr(snn, "spk_rec")
        assert hasattr(snn, "mem_rec")
        assert len(snn.spk_rec) == 1
        assert len(snn.mem_rec) == 1

    def test_multi_layer_forward(self):
        """Test forward pass through multi-layer network."""
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
        snn = net_builder.build()

        input_spikes = torch.randn(10, 2, 5)
        snn(input_spikes)

        assert len(snn.spk_rec) == 2
        assert len(snn.mem_rec) == 2

    def test_reset_mechanisms(self):
        """Test different reset mechanisms."""
        reset_mechanisms = ["zero", "subtract", "none"]

        for reset in reset_mechanisms:
            net_dict = {
                "n_cycles": 10,
                "n_inputs": 5,
                "layer_0": {
                    "neuron_model": "lif",
                    "n_neurons": 8,
                    "threshold": 1.0,
                    "beta": 0.9,
                    "reset_mechanism": reset,
                },
            }

            net_builder = NetBuilder(net_dict)
            snn = net_builder.build()

            input_spikes = torch.randn(10, 2, 5)
            snn(input_spikes)

            assert snn is not None

    def test_extract_index(self):
        """Test index extraction from layer names."""
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
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        assert snn.extract_index("fc1") == 1
        assert snn.extract_index("lif1") == 1
        assert snn.extract_index("layer_0") == 0
        assert snn.extract_index("layer_10") == 10

    def test_invalid_layer_name(self):
        """Test that invalid layer name raises ValueError."""
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
        }

        net_builder = NetBuilder(net_dict)
        snn = net_builder.build()

        with pytest.raises(ValueError, match="Invalid layer name"):
            snn.extract_index("invalid_layer")
