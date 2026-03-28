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

from .exceptions import (
    DataLoaderError as DataLoaderError,
)
from .exceptions import (
    LayerConfigError as LayerConfigError,
)
from .exceptions import (
    NetworkConfigError as NetworkConfigError,
)
from .exceptions import (
    NeuronModelError as NeuronModelError,
)
from .exceptions import (
    QuantizationError as QuantizationError,
)
from .exceptions import (
    SpikerError as SpikerError,
)
from .exceptions import (
    TrainingError as TrainingError,
)
from .exceptions import (
    VHDLGenerationError as VHDLGenerationError,
)
from .net_builder import NetBuilder as NetBuilder
from .optimizer import Optimizer as Optimizer
from .trainer import Trainer as Trainer
from .types import (
    LayerConfig as LayerConfig,
)
from .types import (
    NetworkConfig as NetworkConfig,
)
from .types import (
    NeuronModel as NeuronModel,
)
from .types import (
    OptimizerConfig as OptimizerConfig,
)
from .types import (
    QuantizationConfig as QuantizationConfig,
)
from .types import (
    ReadoutType as ReadoutType,
)
from .types import (
    ResetMechanism as ResetMechanism,
)
from .types import (
    TrainingConfig as TrainingConfig,
)
from .validation import (
    validate_choice as validate_choice,
)
from .validation import (
    validate_layer_config as validate_layer_config,
)
from .validation import (
    validate_net_config as validate_net_config,
)
from .validation import (
    validate_range as validate_range,
)
from .validation import (
    validate_type as validate_type,
)
from .vhdl_generator import VhdlGenerator as VhdlGenerator
