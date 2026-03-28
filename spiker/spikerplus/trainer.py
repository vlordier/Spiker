"""Trainer for spiking neural networks.

This module provides training and evaluation functionality for spiking
neural networks with various readout strategies.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.utils.data import DataLoader

from .types import ReadoutType


class Trainer:
    """Trainer for spiking neural networks.

    Provides training and evaluation functionality with support for
    multiple readout strategies and configurable loss functions.
    """

    def __init__(
        self,
        net: nn.Module,
        readout_type: str | ReadoutType = ReadoutType.MEM,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            net: Neural network to train.
            readout_type: Type of readout to use (spk, mem, etc.).
            optimizer: Optimizer for training, or None for default Adam.
            loss_fn: Loss function, or None for CrossEntropyLoss.
        """

        self.supported_readouts = [readout.value for readout in ReadoutType]

        self.net = net

        if isinstance(readout_type, ReadoutType):
            self.readout_type = readout_type.value
        else:
            if readout_type not in self.supported_readouts:
                msg = f"Invalid readout type. Choose from: {self.supported_readouts}"
                raise ValueError(msg)

            self.readout_type = readout_type

        if optimizer is None:
            adam_beta1: float = 0.9
            adam_beta2: float = 0.999
            lr: float = 5e-4

            self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
                self.net.parameters(), lr=lr, betas=(adam_beta1, adam_beta2)
            )
        else:
            self.optimizer = optimizer

        if loss_fn is None:
            self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)

        log_message = "Training set-up: \n"
        log_message += "Readout type: " + str(self.readout_type) + "\n"
        log_message += "Optimizer: " + str(self.optimizer) + "\n"
        log_message += "Loss function: " + str(self.loss_fn) + "\n"
        log_message += "Device: " + str(self.device) + "\n"

        logging.info(log_message)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 20,
        store: bool = False,
        output_dir: str = "Trained",
    ) -> None:

        train_loss = torch.zeros(n_epochs)
        val_loss = torch.zeros(n_epochs)

        batch_size = next(iter(train_loader))[0].shape[0]

        log_message = "Epochs: " + str(n_epochs) + "\n"
        log_message += "Batch size: " + str(batch_size) + "\n"
        log_message += "Training batches: " + str(len(train_loader)) + "\n"
        log_message += "Training samples: "
        log_message += str(len(train_loader) * batch_size) + "\n"
        log_message += "Validation batches: " + str(len(val_loader)) + "\n"
        log_message += "Validation samples: "
        log_message += str(len(val_loader) * batch_size) + "\n\n"
        log_message += "Begin training\n\n"

        logging.info(log_message)

        start_time = time.time()

        for epoch in range(n_epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.log(epoch, train_loss, val_loss, train_acc, val_acc, start_time)

        if store:
            self.store(output_dir)

    def train_one_epoch(self, dataloader: DataLoader) -> tuple[float, float]:

        accuracy = 0
        batch_count = 0

        # Iterate over the dataloader
        for _batch_idx, (data, labels) in enumerate(dataloader):
            data = data.permute(1, 0, 2).to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            self.net.train()

            self.net(data)

            out_rec, targets = self.readout(labels)

            # Compute the loss over all time steps at once
            loss_val = self.loss_fn(out_rec, targets)

            # Gradient calculation + weight update
            loss_val.backward()
            self.optimizer.step()

            accuracy += self.compute_accuracy(labels)
            batch_count += 1

        accuracy /= batch_count

        return loss_val.item(), accuracy

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:

        # Test set
        with torch.no_grad():
            self.net.eval()

            accuracy = 0
            batch_count = 0

            # Iterate over the dataloader
            for _batch_idx, (data, labels) in enumerate(dataloader):
                data = data.permute(1, 0, 2).to(self.device)
                labels = labels.to(self.device)

                self.net(data)

                out_rec, targets = self.readout(labels)

                # Compute the loss over all time steps at once
                loss_val = self.loss_fn(out_rec, targets)

                accuracy += self.compute_accuracy(labels)
                batch_count += 1

            accuracy /= batch_count

        return loss_val.item(), accuracy

    def readout(self, labels):

        if "mem" in self.readout_type:
            _, out_rec = list(self.net.mem_rec.items())[-1]

            if self.readout_type == "mem_max":
                out_rec, _ = torch.max(out_rec, dim=0, keepdim=True)

            elif self.readout_type == "mem_avg":
                out_rec = torch.mean(out_rec, dim=0, keepdim=True)

            elif self.readout_type == "mem_softmax":
                out_rec = fn.softmax(out_rec, dim=-1)

        else:
            _, out_rec = list(self.net.spk_rec.items())[-1]

            if self.readout_type == "spk_count":
                out_rec = torch.sum(out_rec, dim=0, keepdim=True)

        return out_rec.reshape(-1, out_rec.shape[-1]), labels.repeat(out_rec.shape[0])

    def compute_accuracy(self, labels: torch.Tensor) -> float:
        if "mem" in self.readout_type:
            _, out_rec = list(self.net.mem_rec.items())[-1]
            _, idx = torch.mean(out_rec, dim=0).max(dim=1)
        else:
            _, out_rec = list(self.net.spk_rec.items())[-1]
            _, idx = torch.sum(out_rec, dim=0).max(dim=1)
        return torch.mean((labels == idx).float().detach().cpu()).item()

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc, start_time=None):

        log_message = ""

        epoch = str(epoch)
        log_message += "Epoch " + epoch + "\n"

        if start_time:
            elapsed = time.time() - start_time
            elapsed = f"{elapsed:.2f}" + "s"
            log_message += "Elapsed time " + elapsed + "\n"

        train_loss = f"{train_loss:.2f}"
        val_loss = f"{val_loss:.2f}"
        log_message += "Train loss: " + train_loss + "\n"
        log_message += "Validation loss: " + val_loss + "\n"

        train_acc = str(train_acc * 100) + "%"
        val_acc = str(val_acc * 100) + "%"
        log_message += "Train accuracy: " + train_acc + "\n"
        log_message += "Validation accuracy: " + val_acc + "\n"

        logging.info(log_message)

    def store(self, out_dir, out_file=None):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not out_file:
            out_file = "trained_state_dict.pt"

        out_path = out_dir + "/" + out_file

        torch.save(self.net.state_dict(), out_path)


if __name__ == "__main__":
    import sys

    from net_builder import NetBuilder
    from net_dict import net_dict
    from torch.utils.data import DataLoader, random_split

    audio_mnist_dir = "../../SnnModels/SnnTorch/AudioMnist/"

    if audio_mnist_dir not in sys.path:
        sys.path.append(audio_mnist_dir)

    from audio_mnist import CustomDataset, MelFilterbank

    logging.basicConfig(level=logging.INFO)

    root_dir = audio_mnist_dir + "/Data"
    batch_size = 128

    sample_rate = 48e3

    # Short Term Fourier Transform (STFT) window
    fft_window = 25e-3  # s

    # Step from one window to the other (controls overlap)
    hop_length_s = 10e-3  # s

    # Number of input channels: filters in the mel bank
    n_mels = 40

    # Spiking threshold
    spiking_thresh = 0.9

    transform = MelFilterbank(
        sample_rate=sample_rate,
        fft_window=fft_window,
        hop_length_s=hop_length_s,
        n_mels=n_mels,
        db=True,
        normalize=True,
        spikify=True,
        spiking_thresh=spiking_thresh,
    )

    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    net_builder = NetBuilder(net_dict)

    snn = net_builder.build()

    trainer = Trainer(snn)

    trainer.train(train_loader, test_loader)
