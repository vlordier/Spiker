"""Audio MNIST dataset loader for Spiker framework."""

import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as fn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split


class AudioMnistDL:
    """Audio MNIST dataset loader.

    Loads and preprocesses the Audio MNIST dataset for spike-based neural networks.
    Converts audio waveforms to mel spectrograms and optionally spikifies them.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        fft_window: float = 25e-3,  # s
        hop_length_s: float = 10e-3,  # s
        n_channels: int = 40,
        spiking_thresh: float = 0.9,
        transform: Callable | str = "default",
        train_size: float = 0.8,
    ) -> None:
        """Initialize Audio MNIST dataset loader.

        Args:
            data_dir: Directory containing the dataset.
            fft_window: Short Term Fourier Transform window size in seconds.
            hop_length_s: Step size between windows in seconds.
            n_channels: Number of mel filterbank channels.
            spiking_thresh: Threshold for converting to spike trains.
            transform: Transform to apply to waveforms, or "default" for mel filterbank.
            train_size: Proportion of data to use for training (0-1).

        """
        # Input data sample rate
        self.sample_rate: float = 48e3  # Hz

        # Short Term Fourier Transform (STFT) window
        self.fft_window = fft_window

        # Step from one window to the other (controls overlap)
        self.hop_length_s = hop_length_s

        # Number of input channels: filters in the mel bank
        self.n_mels = n_channels

        # Spiking threshold
        self.spiking_thresh = spiking_thresh

        if transform == "default":
            self.transform: Callable = MelFilterbank(
                sample_rate=self.sample_rate,
                fft_window=self.fft_window,
                hop_length_s=self.hop_length_s,
                n_mels=self.n_mels,
                db=True,
                normalize=True,
                spikify=True,
                spiking_thresh=self.spiking_thresh,
            )

        else:
            self.transform = transform

        self.dataset = CustomDataset(root_dir=data_dir, transform=self.transform)

        self.num_cpu_cores: int | None = os.cpu_count()

        # Train/test split
        train_len = int(train_size * len(self.dataset))
        test_len = len(self.dataset) - train_len

        # Split the dataset into training and validation sets
        self.train_set, self.test_set = random_split(
            self.dataset,
            [train_len, test_len],
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


class CustomDataset(Dataset):
    """Custom dataset for Audio MNIST.

    Loads WAV files from a directory structure where each subdirectory
    corresponds to a different speaker/user.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        max_length: int = 35000,
    ) -> None:
        """Initialize custom dataset.

        Args:
            root_dir: Directory containing subdirectories, one for each user.
            transform: Optional transform to apply to waveforms.
            max_length: Maximum waveform length (samples will be padded/truncated).

        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length

        self.data: list[tuple[str, int]] = []

        root_path = Path(root_dir)
        for user_path in root_path.iterdir():
            if not user_path.is_dir():
                continue
            for file_path in user_path.iterdir():
                if file_path.suffix == ".wav":
                    label = int(file_path.stem.split("_")[0])
                    self.data.append((str(file_path), label))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (waveform, label).

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path, label = self.data[idx]
        waveform, _ = torchaudio.load(file_path)

        # Pad or truncate the waveform to match max_length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, : self.max_length]

        elif waveform.size(1) < self.max_length:
            pad_size = self.max_length - waveform.size(1)
            waveform = fn.pad(waveform, (0, pad_size))

        if self.transform:
            waveform = self.transform(waveform)

        # --- If converting to snnTorch the part under this can be
        # modified ---

        # Reshape and return to make it compatible with sparch
        waveform = waveform.squeeze(dim=0).permute(1, 0)

        return waveform, label


class MelFilterbank:
    """Mel filterbank transform for audio preprocessing.

    Converts audio waveforms to mel spectrograms with optional
    dB scaling, normalization, and spike conversion.
    """

    def __init__(
        self,
        sample_rate: float = 48e3,
        *,
        fft_window: float = 25e-3,
        hop_length_s: float = 10e-3,
        n_mels: int = 40,
        db: bool = False,
        normalize: bool = False,
        spikify: bool = False,
        spiking_thresh: float = 0.9,
    ) -> None:
        """Initialize mel filterbank transform.

        Args:
            sample_rate: Audio sample rate in Hz.
            fft_window: FFT window size in seconds.
            hop_length_s: Hop length between windows in seconds.
            n_mels: Number of mel filterbank channels.
            db: Whether to convert to dB scale.
            normalize: Whether to normalize the spectrogram.
            spikify: Whether to convert to spike trains.
            spiking_thresh: Threshold for spike conversion.

        """
        self.sample_rate = sample_rate
        self.n_fft = int(fft_window * sample_rate)
        self.hop_length = int(hop_length_s * sample_rate)
        self.n_mels = n_mels

        self.db = db

        if self.db:
            # Convert the Mel Spectrogram to dB scale
            self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.normalize = normalize
        self.spikify = spikify
        self.spiking_thresh = spiking_thresh

        # Define the MelSpectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply mel filterbank transform.

        Args:
            waveform: Input audio waveform tensor.

        Returns:
            Mel spectrogram tensor.

        """
        # Apply the Mel Spectrogram transform
        mel_spec = self.mel_spectrogram(waveform)

        if self.db:
            # Convert the Mel Spectrogram to dB scale
            mel_spec = self.db_transform(mel_spec)

        if self.normalize:
            # Normalize mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        if self.spikify:
            # Convert spectrogram into spike trains
            mel_spec = (mel_spec > self.spiking_thresh).float()

        return mel_spec
