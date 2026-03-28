"""Comprehensive tests for validation module."""

import pytest

from spikerplus.exceptions import LayerConfigError, NeuronModelError
from spikerplus.validation import (
    validate_type,
    validate_range,
    validate_choice,
    validate_layer_config,
    validate_net_config,
)


class TestValidateType:
    """Test suite for validate_type function."""

    def test_valid_type(self):
        """Test validation with correct type."""
        validate_type(5, int, "test_param")
        validate_type(5.0, float, "test_param")
        validate_type("test", str, "test_param")
        validate_type(True, bool, "test_param")

    def test_invalid_type(self):
        """Test validation with incorrect type."""
        with pytest.raises(LayerConfigError, match="must be of type int"):
            validate_type("5", int, "test_param")

        with pytest.raises(LayerConfigError, match="must be of type float"):
            validate_type(5, float, "test_param")

    def test_multiple_types(self):
        """Test validation with multiple allowed types."""
        validate_type(5, (int, float), "test_param")
        validate_type(5.0, (int, float), "test_param")

        with pytest.raises(LayerConfigError):
            validate_type("5", (int, float), "test_param")


class TestValidateRange:
    """Test suite for validate_range function."""

    def test_valid_range(self):
        """Test validation with value in range."""
        validate_range(5.0, 0.0, 10.0, "test_param")
        validate_range(0.0, 0.0, 10.0, "test_param")
        validate_range(10.0, 0.0, 10.0, "test_param")

    def test_below_range(self):
        """Test validation with value below range."""
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_range(-1.0, 0.0, 10.0, "test_param")

    def test_above_range(self):
        """Test validation with value above range."""
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_range(11.0, 0.0, 10.0, "test_param")

    def test_integer_range(self):
        """Test validation with integer values."""
        validate_range(5, 0, 10, "test_param")

        with pytest.raises(LayerConfigError):
            validate_range(-1, 0, 10, "test_param")


class TestValidateChoice:
    """Test suite for validate_choice function."""

    def test_valid_choice(self):
        """Test validation with valid choice."""
        validate_choice("a", ["a", "b", "c"], "test_param")
        validate_choice(1, [1, 2, 3], "test_param")

    def test_invalid_choice(self):
        """Test validation with invalid choice."""
        with pytest.raises(NeuronModelError, match="must be one of"):
            validate_choice("d", ["a", "b", "c"], "test_param")

        with pytest.raises(NeuronModelError):
            validate_choice(4, [1, 2, 3], "test_param")


class TestValidateLayerConfig:
    """Test suite for validate_layer_config function."""

    def test_valid_layer_config(self):
        """Test validation with valid layer configuration."""
        config = {
            "n_neurons": 8,
            "neuron_model": "lif",
            "threshold": 1.0,
            "learn_threshold": False,
            "reset_mechanism": "subtract",
            "beta": 0.9,
            "learn_beta": False,
        }
        validate_layer_config(config)

    def test_invalid_n_neurons_type(self):
        """Test validation with invalid n_neurons type."""
        config = {"n_neurons": "8"}
        with pytest.raises(LayerConfigError, match="must be of type int"):
            validate_layer_config(config)

    def test_invalid_n_neurons_value(self):
        """Test validation with invalid n_neurons value."""
        config = {"n_neurons": -1}
        with pytest.raises(LayerConfigError, match="must be positive"):
            validate_layer_config(config)

    def test_invalid_neuron_model(self):
        """Test validation with invalid neuron model."""
        config = {"neuron_model": "invalid"}
        with pytest.raises(NeuronModelError, match="must be one of"):
            validate_layer_config(config)

    def test_valid_neuron_models(self):
        """Test validation with all valid neuron models."""
        valid_models = ["if", "lif", "syn", "rif", "rlif", "rsyn"]
        for model in valid_models:
            config = {"neuron_model": model}
            validate_layer_config(config)

    def test_invalid_threshold_type(self):
        """Test validation with invalid threshold type."""
        config = {"threshold": "1.0"}
        with pytest.raises(LayerConfigError, match="must be one of types"):
            validate_layer_config(config)

    def test_invalid_reset_mechanism(self):
        """Test validation with invalid reset mechanism."""
        config = {"reset_mechanism": "invalid"}
        with pytest.raises(NeuronModelError, match="must be one of"):
            validate_layer_config(config)

    def test_valid_reset_mechanisms(self):
        """Test validation with all valid reset mechanisms."""
        valid_resets = ["zero", "subtract", "none"]
        for reset in valid_resets:
            config = {"reset_mechanism": reset}
            validate_layer_config(config)

    def test_invalid_alpha_type(self):
        """Test validation with invalid alpha type."""
        config = {"alpha": 0.8}  # Should be float
        validate_layer_config(config)

        config = {"alpha": "0.8"}
        with pytest.raises(LayerConfigError, match="must be of type float"):
            validate_layer_config(config)

    def test_invalid_alpha_range(self):
        """Test validation with alpha outside range."""
        config = {"alpha": -0.1}
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_layer_config(config)

        config = {"alpha": 1.1}
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_layer_config(config)

    def test_invalid_beta_type(self):
        """Test validation with invalid beta type."""
        config = {"beta": 0.9}  # Should be float
        validate_layer_config(config)

        config = {"beta": "0.9"}
        with pytest.raises(LayerConfigError, match="must be of type float"):
            validate_layer_config(config)

    def test_invalid_beta_range(self):
        """Test validation with beta outside range."""
        config = {"beta": -0.1}
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_layer_config(config)

        config = {"beta": 1.1}
        with pytest.raises(LayerConfigError, match="must be between"):
            validate_layer_config(config)


class TestValidateNetConfig:
    """Test suite for validate_net_config function."""

    def test_valid_net_config(self):
        """Test validation with valid network configuration."""
        config = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "n_neurons": 8,
                "neuron_model": "lif",
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
        }
        validate_net_config(config)

    def test_invalid_n_cycles_type(self):
        """Test validation with invalid n_cycles type."""
        config = {"n_cycles": "10"}
        with pytest.raises(LayerConfigError, match="must be of type int"):
            validate_net_config(config)

    def test_invalid_n_cycles_value(self):
        """Test validation with invalid n_cycles value."""
        config = {"n_cycles": -1}
        with pytest.raises(LayerConfigError, match="must be positive"):
            validate_net_config(config)

    def test_invalid_n_inputs_type(self):
        """Test validation with invalid n_inputs type."""
        config = {"n_inputs": "5"}
        with pytest.raises(LayerConfigError, match="must be of type int"):
            validate_net_config(config)

    def test_invalid_n_inputs_value(self):
        """Test validation with invalid n_inputs value."""
        config = {"n_inputs": -1}
        with pytest.raises(LayerConfigError, match="must be positive"):
            validate_net_config(config)

    def test_invalid_layer_in_net_config(self):
        """Test validation with invalid layer in network config."""
        config = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "n_neurons": "8",  # Should be int
            },
        }
        with pytest.raises(LayerConfigError):
            validate_net_config(config)
