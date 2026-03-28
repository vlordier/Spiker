"""Custom exceptions for Spiker framework."""


class SpikerError(Exception):
    """Base exception for Spiker framework."""



class NetworkConfigError(SpikerError):
    """Raised when network configuration is invalid."""



class NeuronModelError(NetworkConfigError):
    """Raised when neuron model configuration is invalid."""



class LayerConfigError(NetworkConfigError):
    """Raised when layer configuration is invalid."""



class VHDLGenerationError(SpikerError):
    """Raised when VHDL generation fails."""



class QuantizationError(SpikerError):
    """Raised when quantization fails."""



class TrainingError(SpikerError):
    """Raised when training fails."""



class DataLoaderError(SpikerError):
    """Raised when data loading fails."""

