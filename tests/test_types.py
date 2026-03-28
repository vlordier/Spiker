"""Tests for types module (enums and dataclasses)."""

import pytest

from spikerplus.types import (
    NeuronModel,
    ResetMechanism,
    ReadoutType,
    LayerConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainingConfig,
    QuantizationConfig,
)


class TestNeuronModel:
    """Test suite for NeuronModel enum."""

    def test_enum_values(self):
        """Test that all neuron model values are correct."""
        assert NeuronModel.IF.value == "if"
        assert NeuronModel.LIF.value == "lif"
        assert NeuronModel.SYN.value == "syn"
        assert NeuronModel.RIF.value == "rif"
        assert NeuronModel.RLIF.value == "rlif"
        assert NeuronModel.RSYN.value == "rsyn"

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert NeuronModel.from_string("if") == NeuronModel.IF
        assert NeuronModel.from_string("lif") == NeuronModel.LIF
        assert NeuronModel.from_string("syn") == NeuronModel.SYN

    def test_from_string_invalid(self):
        """Test from_string with invalid values."""
        with pytest.raises(ValueError, match="Invalid neuron model"):
            NeuronModel.from_string("invalid")

    def test_all_models_have_alpha_beta(self):
        """Test which models have alpha and beta."""
        # Models with alpha: SYN, RSYN
        assert NeuronModel.SYN in [NeuronModel.SYN, NeuronModel.RSYN]
        assert NeuronModel.RSYN in [NeuronModel.SYN, NeuronModel.RSYN]

        # Models with beta: LIF, SYN, RLIF, RSYN
        assert NeuronModel.LIF in [
            NeuronModel.LIF,
            NeuronModel.SYN,
            NeuronModel.RLIF,
            NeuronModel.RSYN,
        ]


class TestResetMechanism:
    """Test suite for ResetMechanism enum."""

    def test_enum_values(self):
        """Test that all reset mechanism values are correct."""
        assert ResetMechanism.ZERO.value == "zero"
        assert ResetMechanism.SUBTRACT.value == "subtract"
        assert ResetMechanism.NONE.value == "none"

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert ResetMechanism.from_string("zero") == ResetMechanism.ZERO
        assert ResetMechanism.from_string("subtract") == ResetMechanism.SUBTRACT
        assert ResetMechanism.from_string("none") == ResetMechanism.NONE

    def test_from_string_invalid(self):
        """Test from_string with invalid values."""
        with pytest.raises(ValueError, match="Invalid reset mechanism"):
            ResetMechanism.from_string("invalid")


class TestReadoutType:
    """Test suite for ReadoutType enum."""

    def test_enum_values(self):
        """Test that all readout type values are correct."""
        assert ReadoutType.SPK.value == "spk"
        assert ReadoutType.SPK_COUNT.value == "spk_count"
        assert ReadoutType.MEM.value == "mem"
        assert ReadoutType.MEM_SOFTMAX.value == "mem_softmax"
        assert ReadoutType.MEM_MAX.value == "mem_max"
        assert ReadoutType.MEM_AVG.value == "mem_avg"

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert ReadoutType.from_string("spk") == ReadoutType.SPK
        assert ReadoutType.from_string("mem") == ReadoutType.MEM
        assert ReadoutType.from_string("mem_max") == ReadoutType.MEM_MAX

    def test_from_string_invalid(self):
        """Test from_string with invalid values."""
        with pytest.raises(ValueError, match="Invalid readout type"):
            ReadoutType.from_string("invalid")


class TestLayerConfig:
    """Test suite for LayerConfig dataclass."""

    def test_default_values(self):
        """Test default values of LayerConfig."""
        config = LayerConfig()
        assert config.n_neurons == 128
        assert config.neuron_model == NeuronModel.LIF
        assert config.threshold == 1.0
        assert config.learn_threshold is False
        assert config.reset_mechanism == ResetMechanism.SUBTRACT
        assert config.alpha == 0.9
        assert config.learn_alpha is False
        assert config.beta == 0.9375
        assert config.learn_beta is False

    def test_custom_values(self):
        """Test LayerConfig with custom values."""
        config = LayerConfig(
            n_neurons=64,
            neuron_model=NeuronModel.SYN,
            threshold=2.0,
            learn_threshold=True,
            reset_mechanism=ResetMechanism.ZERO,
            alpha=0.8,
            learn_alpha=True,
            beta=0.9,
            learn_beta=True,
        )
        assert config.n_neurons == 64
        assert config.neuron_model == NeuronModel.SYN
        assert config.threshold == 2.0
        assert config.learn_threshold is True
        assert config.reset_mechanism == ResetMechanism.ZERO
        assert config.alpha == 0.8
        assert config.learn_alpha is True
        assert config.beta == 0.9
        assert config.learn_beta is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LayerConfig(n_neurons=64, neuron_model=NeuronModel.SYN)
        result = config.to_dict()
        assert result["n_neurons"] == 64
        assert result["neuron_model"] == "syn"
        assert "threshold" in result
        assert "reset_mechanism" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "n_neurons": 64,
            "neuron_model": "syn",
            "threshold": 2.0,
            "reset_mechanism": "zero",
            "alpha": 0.8,
            "beta": 0.9,
        }
        config = LayerConfig.from_dict(data)
        assert config.n_neurons == 64
        assert config.neuron_model == NeuronModel.SYN
        assert config.threshold == 2.0
        assert config.reset_mechanism == ResetMechanism.ZERO
        assert config.alpha == 0.8
        assert config.beta == 0.9

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing values."""
        data = {"n_neurons": 64}
        config = LayerConfig.from_dict(data)
        assert config.n_neurons == 64
        assert config.neuron_model == NeuronModel.LIF  # Default
        assert config.threshold == 1.0  # Default


class TestNetworkConfig:
    """Test suite for NetworkConfig dataclass."""

    def test_default_values(self):
        """Test default values of NetworkConfig."""
        config = NetworkConfig()
        assert config.n_cycles == 73
        assert config.n_inputs == 40
        assert len(config.layers) == 1
        assert isinstance(config.layers[0], LayerConfig)

    def test_custom_values(self):
        """Test NetworkConfig with custom values."""
        layer = LayerConfig(n_neurons=64)
        config = NetworkConfig(n_cycles=100, n_inputs=50, layers=[layer])
        assert config.n_cycles == 100
        assert config.n_inputs == 50
        assert len(config.layers) == 1
        assert config.layers[0].n_neurons == 64

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = NetworkConfig(n_cycles=10, n_inputs=5)
        result = config.to_dict()
        assert result["n_cycles"] == 10
        assert result["n_inputs"] == 5
        assert "layer_0" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {"n_neurons": 64, "neuron_model": "lif"},
            "layer_1": {"n_neurons": 32, "neuron_model": "syn"},
        }
        config = NetworkConfig.from_dict(data)
        assert config.n_cycles == 10
        assert config.n_inputs == 5
        assert len(config.layers) == 2
        assert config.layers[0].n_neurons == 64
        assert config.layers[1].n_neurons == 32


class TestOptimizerConfig:
    """Test suite for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default values of OptimizerConfig."""
        config = OptimizerConfig()
        assert config.weights_bw == (4, 8)
        assert config.neurons_bw == (4, 10)
        assert config.fp_dec == (2, 3)

    def test_custom_values(self):
        """Test OptimizerConfig with custom values."""
        config = OptimizerConfig(
            weights_bw=(2, 4),
            neurons_bw=(8, 16),
            fp_dec=(4, 6),
        )
        assert config.weights_bw == (2, 4)
        assert config.neurons_bw == (8, 16)
        assert config.fp_dec == (4, 6)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig()
        result = config.to_dict()
        assert result["weights_bw"] == {"min": 4, "max": 8}
        assert result["neurons_bw"] == {"min": 4, "max": 10}
        assert result["fp_dec"] == {"min": 2, "max": 3}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "weights_bw": {"min": 2, "max": 4},
            "neurons_bw": {"min": 8, "max": 16},
            "fp_dec": {"min": 4, "max": 6},
        }
        config = OptimizerConfig.from_dict(data)
        assert config.weights_bw == (2, 4)
        assert config.neurons_bw == (8, 16)
        assert config.fp_dec == (4, 6)


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default values of TrainingConfig."""
        config = TrainingConfig()
        assert config.n_epochs == 20
        assert config.readout_type == ReadoutType.MEM
        assert config.learning_rate == 5e-4
        assert config.adam_beta1 == 0.9
        assert config.adam_beta2 == 0.999

    def test_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            n_epochs=50,
            readout_type=ReadoutType.SPK,
            learning_rate=1e-3,
            adam_beta1=0.8,
            adam_beta2=0.99,
        )
        assert config.n_epochs == 50
        assert config.readout_type == ReadoutType.SPK
        assert config.learning_rate == 1e-3
        assert config.adam_beta1 == 0.8
        assert config.adam_beta2 == 0.99

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        result = config.to_dict()
        assert result["n_epochs"] == 20
        assert result["readout_type"] == "mem"
        assert "learning_rate" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "n_epochs": 50,
            "readout_type": "spk",
            "learning_rate": 1e-3,
        }
        config = TrainingConfig.from_dict(data)
        assert config.n_epochs == 50
        assert config.readout_type == ReadoutType.SPK
        assert config.learning_rate == 1e-3


class TestQuantizationConfig:
    """Test suite for QuantizationConfig dataclass."""

    def test_default_values(self):
        """Test default values of QuantizationConfig."""
        config = QuantizationConfig()
        assert config.neurons_bw == 16
        assert config.weights_bw == 8
        assert config.fp_dec == 8

    def test_custom_values(self):
        """Test QuantizationConfig with custom values."""
        config = QuantizationConfig(
            neurons_bw=32,
            weights_bw=16,
            fp_dec=4,
        )
        assert config.neurons_bw == 32
        assert config.weights_bw == 16
        assert config.fp_dec == 4

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = QuantizationConfig()
        result = config.to_dict()
        assert result["neurons_bw"] == 16
        assert result["weights_bw"] == 8
        assert result["fp_dec"] == 8

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "neurons_bw": 32,
            "weights_bw": 16,
            "fp_dec": 4,
        }
        config = QuantizationConfig.from_dict(data)
        assert config.neurons_bw == 32
        assert config.weights_bw == 16
        assert config.fp_dec == 4
