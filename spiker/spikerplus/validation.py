"""Input validation decorators and helpers for Spiker framework."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from .exceptions import LayerConfigError, NeuronModelError
from .types import NeuronModel, ResetMechanism


def validate_type(value: Any, expected_type: type, param_name: str) -> None:
    """Validate that a value is of the expected type.

    Args:
        value: The value to validate.
        expected_type: The expected type.
        param_name: The name of the parameter (for error messages).

    Raises:
        LayerConfigError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = ", ".join(t.__name__ for t in expected_type)
            msg = f"{param_name} must be one of types: {type_names}"
        else:
            msg = f"{param_name} must be of type {expected_type.__name__}"
        raise LayerConfigError(msg)


def validate_range(
    value: float, min_val: float, max_val: float, param_name: str
) -> None:
    """Validate that a numeric value is within a specified range.

    Args:
        value: The value to validate.
        min_val: The minimum allowed value (inclusive).
        max_val: The maximum allowed value (inclusive).
        param_name: The name of the parameter (for error messages).

    Raises:
        LayerConfigError: If the value is outside the specified range.
    """
    if value < min_val or value > max_val:
        msg = f"{param_name} must be between {min_val} and {max_val}"
        raise LayerConfigError(msg)


def validate_choice(value: Any, valid_choices: list[Any], param_name: str) -> None:
    """Validate that a value is one of the valid choices.

    Args:
        value: The value to validate.
        valid_choices: The list of valid choices.
        param_name: The name of the parameter (for error messages).

    Raises:
        NeuronModelError: If the value is not in the valid choices.
    """
    if value not in valid_choices:
        msg = f"{param_name} must be one of {valid_choices}"
        raise NeuronModelError(msg)


def validate_config_param(
    param_name: str,
    expected_type: type | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    valid_choices: list[Any] | None = None,
    required: bool = True,
) -> Callable:
    """Decorator to validate configuration parameters.

    Args:
        param_name: The name of the parameter to validate.
        expected_type: The expected type of the parameter.
        min_val: The minimum allowed value (for numeric types).
        max_val: The maximum allowed value (for numeric types).
        valid_choices: The list of valid choices.
        required: Whether the parameter is required.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = _extract_parameter_value(func, param_name, args, kwargs)

            _validate_required(value, required, param_name)

            if value is None:
                return func(*args, **kwargs)

            _validate_type_if_needed(value, expected_type, param_name)
            _validate_range_if_needed(value, min_val, max_val, param_name)
            _validate_choices_if_needed(value, valid_choices, param_name)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _extract_parameter_value(
    func: Callable, param_name: str, args: tuple, kwargs: dict
) -> Any:
    """Extract parameter value from function arguments."""
    value = kwargs.get(param_name)
    if value is None and len(args) > 0:
        import inspect

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        if param_name in param_names:
            param_index = param_names.index(param_name)
            if param_index < len(args):
                value = args[param_index]
    return value


def _validate_required(value: Any, required: bool, param_name: str) -> None:
    """Validate that required parameter is present."""
    if required and value is None:
        msg = f"{param_name} is required"
        raise LayerConfigError(msg)


def _validate_type_if_needed(
    value: Any, expected_type: type | None, param_name: str
) -> None:
    """Validate type if expected_type is specified."""
    if expected_type is not None:
        validate_type(value, expected_type, param_name)


def _validate_range_if_needed(
    value: float, min_val: float | None, max_val: float | None, param_name: str
) -> None:
    """Validate range if min_val and max_val are specified."""
    if min_val is not None and max_val is not None:
        validate_range(value, min_val, max_val, param_name)


def _validate_choices_if_needed(
    value: Any, valid_choices: list[Any] | None, param_name: str
) -> None:
    """Validate choices if valid_choices is specified."""
    if valid_choices is not None:
        validate_choice(value, valid_choices, param_name)


def _validate_neurons_param(layer_dict: dict[str, Any]) -> None:
    """Validate n_neurons parameter."""
    if "n_neurons" in layer_dict:
        validate_type(layer_dict["n_neurons"], int, "n_neurons")
        if layer_dict["n_neurons"] <= 0:
            msg = "n_neurons must be positive"
            raise LayerConfigError(msg)


def _validate_model_param(layer_dict: dict[str, Any]) -> None:
    """Validate neuron_model parameter."""
    if "neuron_model" in layer_dict:
        valid_models = [model.value for model in NeuronModel]
        validate_choice(layer_dict["neuron_model"], valid_models, "neuron_model")


def _validate_threshold_param(layer_dict: dict[str, Any]) -> None:
    """Validate threshold and learn_threshold parameters."""
    if "threshold" in layer_dict:
        validate_type(layer_dict["threshold"], (int, float), "threshold")

    if "learn_threshold" in layer_dict:
        validate_type(layer_dict["learn_threshold"], bool, "learn_threshold")


def _validate_reset_param(layer_dict: dict[str, Any]) -> None:
    """Validate reset_mechanism parameter."""
    if "reset_mechanism" in layer_dict:
        valid_resets = [mechanism.value for mechanism in ResetMechanism]
        validate_choice(layer_dict["reset_mechanism"], valid_resets, "reset_mechanism")


def _validate_alpha_param(layer_dict: dict[str, Any]) -> None:
    """Validate alpha and learn_alpha parameters."""
    if "alpha" in layer_dict:
        validate_type(layer_dict["alpha"], float, "alpha")
        validate_range(layer_dict["alpha"], 0.0, 1.0, "alpha")

    if "learn_alpha" in layer_dict:
        validate_type(layer_dict["learn_alpha"], bool, "learn_alpha")


def _validate_beta_param(layer_dict: dict[str, Any]) -> None:
    """Validate beta and learn_beta parameters."""
    if "beta" in layer_dict:
        validate_type(layer_dict["beta"], float, "beta")
        validate_range(layer_dict["beta"], 0.0, 1.0, "beta")

    if "learn_beta" in layer_dict:
        validate_type(layer_dict["learn_beta"], bool, "learn_beta")


def validate_layer_config(layer_dict: dict[str, Any]) -> None:
    """Validate a layer configuration dictionary.

    Args:
        layer_dict: The layer configuration dictionary.

    Raises:
        LayerConfigError: If the configuration is invalid.
    """
    _validate_neurons_param(layer_dict)
    _validate_model_param(layer_dict)
    _validate_threshold_param(layer_dict)
    _validate_reset_param(layer_dict)
    _validate_alpha_param(layer_dict)
    _validate_beta_param(layer_dict)


def validate_net_config(net_dict: dict[str, Any]) -> None:
    """Validate a network configuration dictionary.

    Args:
        net_dict: The network configuration dictionary.

    Raises:
        LayerConfigError: If the configuration is invalid.
    """
    # Validate n_cycles
    if "n_cycles" in net_dict:
        validate_type(net_dict["n_cycles"], int, "n_cycles")
        if net_dict["n_cycles"] <= 0:
            msg = "n_cycles must be positive"
            raise LayerConfigError(msg)

    # Validate n_inputs
    if "n_inputs" in net_dict:
        validate_type(net_dict["n_inputs"], int, "n_inputs")
        if net_dict["n_inputs"] <= 0:
            msg = "n_inputs must be positive"
            raise LayerConfigError(msg)

    # Validate each layer
    for key in net_dict:
        if "layer" in key:
            validate_layer_config(net_dict[key])
