"""Spiker: Spiking Neural Network Framework.

A framework for building, training, and deploying spiking neural networks
with support for hardware generation (VHDL) and quantization.

This package provides:
- Network building with various neuron models (LIF, IF, Synaptic, etc.)
- Training with multiple readout strategies
- Quantization-aware optimization
- VHDL generation for hardware deployment
- Type-safe configuration with enums and dataclasses
- Input validation and custom exceptions
"""

from .device import get_device as get_device
from .device import to_device as to_device
from .exceptions import (
    DataLoaderError,
    LayerConfigError,
    NetworkConfigError,
    NeuronModelError,
    QuantizationError,
    SpikerError,
    TrainingError,
    VHDLGenerationError,
)
from .net_builder import NetBuilder
from .optimizer import Optimizer
from .trainer import Trainer
from .types import (
    LayerConfig,
    LayerDict,
    NetworkConfig,
    NetworkDict,
    NeuronModel,
    OptimizerConfig,
    OptimizerDict,
    OptimizerRangeDict,
    QuantizationConfig,
    ReadoutType,
    ResetMechanism,
    TrainingConfig,
    TrainingDict,
)
from .validation import (
    validate_choice,
    validate_layer_config,
    validate_net_config,
    validate_range,
    validate_type,
)
from .vhdl_generator import VhdlGenerator

__all__ = [
    # Device
    "get_device",
    "to_device",
    # Exceptions
    "DataLoaderError",
    "LayerConfigError",
    "NetworkConfigError",
    "NeuronModelError",
    "QuantizationError",
    "SpikerError",
    "TrainingError",
    "VHDLGenerationError",
    # Core
    "NetBuilder",
    "Optimizer",
    "Trainer",
    "VhdlGenerator",
    # Types
    "LayerConfig",
    "LayerDict",
    "NetworkConfig",
    "NetworkDict",
    "NeuronModel",
    "OptimizerConfig",
    "OptimizerDict",
    "OptimizerRangeDict",
    "QuantizationConfig",
    "ReadoutType",
    "ResetMechanism",
    "TrainingConfig",
    "TrainingDict",
    # Validation
    "validate_choice",
    "validate_layer_config",
    "validate_net_config",
    "validate_range",
    "validate_type",
]
