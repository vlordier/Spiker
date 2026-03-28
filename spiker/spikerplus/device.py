"""Device management utilities for Spiker framework."""

import logging

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device for computation.

    Args:
        prefer_gpu: Whether to prefer GPU over other accelerators.

    Returns:
        torch.device: The selected device (cuda, mps, or cpu).
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(
                "Using CUDA device: %s (%.1fGB total memory)", device_name, memory
            )
            return device

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple Silicon MPS device")
            return device

    device = torch.device("cpu")
    logging.info("Using CPU device")
    return device


def to_device(
    data: torch.Tensor | list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
    """Move tensor(s) to the specified device.

    Args:
        data: Tensor or list of tensors to move.
        device: Target device.

    Returns:
        Tensor or list of tensors on the target device.
    """
    if isinstance(data, list):
        return [to_device(t, device) for t in data]
    return data.to(device)


def get_device_name(device: torch.device) -> str:
    """Get human-readable device name.

    Args:
        device: PyTorch device.

    Returns:
        String representation of the device.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon MPS"
    return "CPU"
