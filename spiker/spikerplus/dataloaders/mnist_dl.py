"""MNIST dataset loader for Spiker framework."""

import os
from collections.abc import Callable
from pathlib import Path

import torch
from snntorch import spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistDL:
    """MNIST dataset loader.

    Loads and preprocesses the MNIST dataset for spike-based neural networks.
    Converts images to spike trains using rate coding.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        transform: Callable | str = "default",
        download: bool = True,
        image_width: int = 28,
        image_height: int = 28,
        num_steps: int = 100,
        gain: int = 1,
    ) -> None:
        """Initialize MNIST dataset loader.

        Args:
            data_dir: Directory to store/load the dataset.
            transform: Transform to apply to images, or "default" for
                standard preprocessing.
            download: Whether to download the dataset if not present.
            image_width: Width to resize images to.
            image_height: Height to resize images to.
            num_steps: Number of time steps for spike encoding.
            gain: Gain factor for spike rate encoding.

        """
        self.spike_transform = SpikeTransform(num_steps=num_steps, gain=gain)

        if transform == "default":
            self.transform: Callable = transforms.Compose(
                [
                    transforms.Resize((image_width, image_height)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,)),
                    self.spike_transform,
                ],
            )
        else:
            self.transform = transform

        self.num_cpu_cores: int | None = os.cpu_count()

        self.train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=self.transform,
        )

        self.test_set = datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=self.transform,
        )

    def load(
        self,
        *,
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


class SpikeTransform:
    """Spike rate encoding transform.

    Converts images to spike trains using rate coding.
    """

    def __init__(self, num_steps: int = 100, gain: int = 1) -> None:
        """Initialize spike transform.

        Args:
            num_steps: Number of time steps for spike encoding.
            gain: Gain factor for spike rate.

        """
        self.num_steps = num_steps
        self.gain = gain

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply spike rate encoding.

        Args:
            img: Input image tensor.

        Returns:
            Spike-encoded tensor.

        """
        img = img.reshape(img.shape[1] * img.shape[2])

        return spikegen.rate(img, num_steps=self.num_steps, gain=self.gain)
