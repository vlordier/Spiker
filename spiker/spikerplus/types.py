"""Type definitions, enums, and dataclasses for Spiker framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict


class NeuronModel(Enum):
    """Supported neuron models in Spiker framework."""

    IF = "if"  # Integrate-and-Fire
    LIF = "lif"  # Leaky Integrate-and-Fire
    SYN = "syn"  # Synaptic
    RIF = "rif"  # Recurrent Integrate-and-Fire
    RLIF = "rlif"  # Recurrent Leaky Integrate-and-Fire
    RSYN = "rsyn"  # Recurrent Synaptic

    @classmethod
    def from_string(cls, value: str) -> "NeuronModel":
        """Convert string to NeuronModel enum.

        Args:
            value: String representation of neuron model.

        Returns:
            The corresponding NeuronModel enum.

        Raises:
            ValueError: If the string doesn't match any neuron model.
        """
        for model in cls:
            if model.value == value:
                return model
        valid_models = [m.value for m in cls]
        msg = f"Invalid neuron model '{value}'. Valid options: {valid_models}"
        raise ValueError(msg)


class ResetMechanism(Enum):
    """Supported reset mechanisms in Spiker framework."""

    ZERO = "zero"  # Reset membrane potential to zero
    SUBTRACT = "subtract"  # Subtract threshold from membrane potential
    NONE = "none"  # No reset

    @classmethod
    def from_string(cls, value: str) -> "ResetMechanism":
        """Convert string to ResetMechanism enum.

        Args:
            value: String representation of reset mechanism.

        Returns:
            The corresponding ResetMechanism enum.

        Raises:
            ValueError: If the string doesn't match any reset mechanism.
        """
        for mechanism in cls:
            if mechanism.value == value:
                return mechanism
        valid_mechanisms = [m.value for m in cls]
        msg = f"Invalid reset mechanism '{value}'. Valid options: {valid_mechanisms}"
        raise ValueError(msg)


class ReadoutType(Enum):
    """Supported readout types for training."""

    SPK = "spk"  # Spike readout
    SPK_COUNT = "spk_count"  # Spike count readout
    MEM = "mem"  # Membrane potential readout
    MEM_SOFTMAX = "mem_softmax"  # Membrane potential with softmax
    MEM_MAX = "mem_max"  # Maximum membrane potential
    MEM_AVG = "mem_avg"  # Average membrane potential

    @classmethod
    def from_string(cls, value: str) -> "ReadoutType":
        """Convert string to ReadoutType enum.

        Args:
            value: String representation of readout type.

        Returns:
            The corresponding ReadoutType enum.

        Raises:
            ValueError: If the string doesn't match any readout type.
        """
        for readout in cls:
            if readout.value == value:
                return readout
        valid_readouts = [r.value for r in cls]
        msg = f"Invalid readout type '{value}'. Valid options: {valid_readouts}"
        raise ValueError(msg)


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a single network layer.

    Attributes:
        n_neurons: Number of neurons in the layer.
        neuron_model: Type of neuron model to use.
        threshold: Firing threshold for neurons.
        learn_threshold: Whether threshold is learnable.
        reset_mechanism: How to reset membrane potential after firing.
        alpha: Decay factor for synaptic neurons (0-1).
        learn_alpha: Whether alpha is learnable.
        beta: Decay factor for leaky neurons (0-1).
        learn_beta: Whether beta is learnable.
    """

    n_neurons: int = 128
    neuron_model: NeuronModel = NeuronModel.LIF
    threshold: float = 1.0
    learn_threshold: bool = False
    reset_mechanism: ResetMechanism = ResetMechanism.SUBTRACT
    alpha: float = 0.9
    learn_alpha: bool = False
    beta: float = 0.9375
    learn_beta: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "n_neurons": self.n_neurons,
            "neuron_model": self.neuron_model.value,
            "threshold": self.threshold,
            "learn_threshold": self.learn_threshold,
            "reset_mechanism": self.reset_mechanism.value,
            "alpha": self.alpha,
            "learn_alpha": self.learn_alpha,
            "beta": self.beta,
            "learn_beta": self.learn_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerConfig":
        """Create LayerConfig from dictionary.

        Args:
            data: Dictionary with layer configuration.

        Returns:
            LayerConfig instance.
        """
        return cls(
            n_neurons=data.get("n_neurons", cls.n_neurons),
            neuron_model=NeuronModel.from_string(
                data.get("neuron_model", cls.neuron_model.value),
            ),
            threshold=data.get("threshold", cls.threshold),
            learn_threshold=data.get("learn_threshold", cls.learn_threshold),
            reset_mechanism=ResetMechanism.from_string(
                data.get("reset_mechanism", cls.reset_mechanism.value),
            ),
            alpha=data.get("alpha", cls.alpha),
            learn_alpha=data.get("learn_alpha", cls.learn_alpha),
            beta=data.get("beta", cls.beta),
            learn_beta=data.get("learn_beta", cls.learn_beta),
        )


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the entire network.

    Attributes:
        n_cycles: Number of time steps for the network.
        n_inputs: Number of input neurons.
        layers: List of layer configurations.
    """

    n_cycles: int = 73
    n_inputs: int = 40
    layers: list[LayerConfig] = field(default_factory=lambda: [LayerConfig()])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result: dict[str, Any] = {
            "n_cycles": self.n_cycles,
            "n_inputs": self.n_inputs,
        }
        for i, layer in enumerate(self.layers):
            result[f"layer_{i}"] = layer.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkConfig":
        """Create NetworkConfig from dictionary.

        Args:
            data: Dictionary with network configuration.

        Returns:
            NetworkConfig instance.
        """
        layers = [
            LayerConfig.from_dict(data[key])
            for key in sorted(data.keys())
            if key.startswith("layer_")
        ]

        return cls(
            n_cycles=data.get("n_cycles", cls.n_cycles),
            n_inputs=data.get("n_inputs", cls.n_inputs),
            layers=layers or [LayerConfig()],
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
        weights_bw: Bit-width range for weights.
        neurons_bw: Bit-width range for neurons.
        fp_dec: Fixed-point decimal range.
    """

    weights_bw: tuple[int, int] = (4, 8)
    neurons_bw: tuple[int, int] = (4, 10)
    fp_dec: tuple[int, int] = (2, 3)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "weights_bw": {
                "min_val": self.weights_bw[0],
                "max_val": self.weights_bw[1],
            },
            "neurons_bw": {
                "min_val": self.neurons_bw[0],
                "max_val": self.neurons_bw[1],
            },
            "fp_dec": {"min_val": self.fp_dec[0], "max_val": self.fp_dec[1]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        """Create OptimizerConfig from dictionary.

        Args:
            data: Dictionary with optimizer configuration.

        Returns:
            OptimizerConfig instance.
        """
        weights_bw = data.get("weights_bw", {})
        neurons_bw = data.get("neurons_bw", {})
        fp_dec = data.get("fp_dec", {})

        return cls(
            weights_bw=(
                weights_bw.get("min", cls.weights_bw[0]),
                weights_bw.get("max", cls.weights_bw[1]),
            ),
            neurons_bw=(
                neurons_bw.get("min", cls.neurons_bw[0]),
                neurons_bw.get("max", cls.neurons_bw[1]),
            ),
            fp_dec=(
                fp_dec.get("min", cls.fp_dec[0]),
                fp_dec.get("max", cls.fp_dec[1]),
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training.

    Attributes:
        n_epochs: Number of training epochs.
        readout_type: Type of readout to use.
        learning_rate: Learning rate for optimizer.
        adam_beta1: Beta1 parameter for Adam optimizer.
        adam_beta2: Beta2 parameter for Adam optimizer.
    """

    n_epochs: int = 20
    readout_type: ReadoutType = ReadoutType.MEM
    learning_rate: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "n_epochs": self.n_epochs,
            "readout_type": self.readout_type.value,
            "learning_rate": self.learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from dictionary.

        Args:
            data: Dictionary with training configuration.

        Returns:
            TrainingConfig instance.
        """
        return cls(
            n_epochs=data.get("n_epochs", cls.n_epochs),
            readout_type=ReadoutType.from_string(
                data.get("readout_type", cls.readout_type.value),
            ),
            learning_rate=data.get("learning_rate", cls.learning_rate),
            adam_beta1=data.get("adam_beta1", cls.adam_beta1),
            adam_beta2=data.get("adam_beta2", cls.adam_beta2),
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for quantization.

    Attributes:
        neurons_bw: Bit-width for neurons.
        weights_bw: Bit-width for weights.
        fp_dec: Number of fixed-point decimal bits.
    """

    neurons_bw: int = 16
    weights_bw: int = 8
    fp_dec: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "neurons_bw": self.neurons_bw,
            "weights_bw": self.weights_bw,
            "fp_dec": self.fp_dec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationConfig":
        """Create QuantizationConfig from dictionary.

        Args:
            data: Dictionary with quantization configuration.

        Returns:
            QuantizationConfig instance.
        """
        return cls(
            neurons_bw=data.get("neurons_bw", cls.neurons_bw),
            weights_bw=data.get("weights_bw", cls.weights_bw),
            fp_dec=data.get("fp_dec", cls.fp_dec),
        )


class LayerDict(TypedDict, total=False):
    """Typed dictionary for layer configuration."""

    n_neurons: int
    neuron_model: str
    threshold: float
    learn_threshold: bool
    reset_mechanism: str
    alpha: float
    learn_alpha: bool
    beta: float
    learn_beta: bool


class OptimizerRangeDict(TypedDict, total=False):
    """Typed dictionary for optimizer range configuration."""

    min_val: int
    max_val: int


class OptimizerDict(TypedDict, total=False):
    """Typed dictionary for optimizer configuration."""

    weights_bw: OptimizerRangeDict
    neurons_bw: OptimizerRangeDict
    fp_dec: OptimizerRangeDict


class NetworkDict(TypedDict, total=False):
    """Typed dictionary for network configuration."""

    n_cycles: int
    n_inputs: int
    layer_0: LayerDict
    layer_1: LayerDict
    layer_2: LayerDict
    layer_3: LayerDict
    layer_4: LayerDict


class TrainingDict(TypedDict, total=False):
    """Typed dictionary for training configuration."""

    n_epochs: int
    readout_type: str
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
