"""Spiking Heidelberg Digits dataset loader for Spiker framework."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tonic
import torch
from torch.utils.data import DataLoader


class ShdDL:
    """Spiking Heidelberg Digits dataset loader.

    Loads and preprocesses the Spiking Heidelberg Digits (SHD) dataset
    for spike-based neural networks.
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable | str = "default",
        download: bool = True,
        num_steps: int = 100,
    ) -> None:
        """Initialize SHD dataset loader.

        Args:
            data_dir: Directory to store/load the dataset.
            transform: Transform to apply to spikes, or "default" for
                standard preprocessing.
            download: Whether to download the dataset if not present.
            num_steps: Number of time bins for frame conversion.
        """
        _ = download  # Reserved for future use

        if transform == "default":
            self.transform: Callable = tonic.transforms.Compose(
                [
                    tonic.transforms.ToFrame(
                        sensor_size=tonic.datasets.hsd.SHD.sensor_size,
                        n_time_bins=num_steps,
                    ),
                    Squeeze(dim=1),
                    ToFloatTensor(),
                ]
            )

        else:
            self.transform = transform

        self.num_cpu_cores: int | None = os.cpu_count()

        self.train_set = tonic.datasets.hsd.SHD(
            save_to=data_dir, train=True, transform=self.transform
        )

        self.test_set = tonic.datasets.hsd.SHD(
            save_to=data_dir, train=False, transform=self.transform
        )

    def load(
        self,
        train_drop_last: bool = True,
        train_shuffle: bool = True,
        test_drop_last: bool = True,
        test_shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Load train and test dataloaders.

        Args:
            train_drop_last: Whether to drop last incomplete training batch.
            train_shuffle: Whether to shuffle training data.
            test_drop_last: Whether to drop last incomplete test batch.
            test_shuffle: Whether to shuffle test data.
            batch_size: Batch size for dataloaders.
            num_workers: Number of worker processes for data loading.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        if not num_workers:
            num_workers = self.num_cpu_cores

        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            drop_last=train_drop_last,
        )

        test_loader = DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=test_shuffle,
            num_workers=num_workers,
            drop_last=test_drop_last,
        )

        return train_loader, test_loader


class Squeeze:
    """Squeeze transform that removes a dimension of size 1."""

    def __init__(self, dim: int) -> None:
        """Initialize squeeze transform.

        Args:
            dim: Dimension to squeeze.
        """
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply squeeze transform.

        Args:
            tensor: Input tensor.

        Returns:
            Squeezed tensor.

        Raises:
            ValueError: If the dimension doesn't have size 1.
        """
        if tensor.shape[self.dim] == 1:
            return tensor.squeeze(self.dim)

        actual = tensor.shape[self.dim]
        msg = f"Expected size 1 at dimension {self.dim}, got {actual}"
        raise ValueError(msg)


class ToFloatTensor:
    """Convert numpy array to float tensor."""

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        """Convert array to float tensor.

        Args:
            array: Input numpy array.

        Returns:
            Float tensor.
        """
        return torch.tensor(array, dtype=torch.float32)
