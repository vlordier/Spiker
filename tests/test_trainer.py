"""Comprehensive tests for trainer module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spikerplus.net_builder import NetBuilder
from spikerplus.trainer import Trainer


class TestTrainer:
    """Test suite for Trainer class."""

    def _create_simple_network(self):
        """Helper to create a simple network for testing."""
        net_dict = {
            "n_cycles": 10,
            "n_inputs": 5,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 8,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "subtract",
            },
            "layer_1": {
                "neuron_model": "lif",
                "n_neurons": 3,
                "threshold": 1.0,
                "beta": 0.9,
                "reset_mechanism": "none",
            },
        }

        net_builder = NetBuilder(net_dict)
        return net_builder.build()

    def _create_dummy_dataloader(self, n_samples=20, batch_size=4):
        """Helper to create a dummy dataloader for testing."""
        # Create dummy spike data: [n_samples, time_steps, n_inputs]
        data = torch.randint(0, 2, (n_samples, 10, 5)).float()
        # Create dummy labels: [n_samples]
        labels = torch.randint(0, 3, (n_samples,))

        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_trainer_initialization(self):
        """Test trainer initialization with default parameters."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        assert trainer.net == snn
        assert trainer.readout_type == "mem"
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None

    def test_trainer_with_custom_optimizer(self):
        """Test trainer initialization with custom optimizer."""
        snn = self._create_simple_network()
        custom_optimizer = torch.optim.SGD(snn.parameters(), lr=0.01)

        trainer = Trainer(snn, readout_type="mem", optimizer=custom_optimizer)

        assert trainer.optimizer == custom_optimizer

    def test_trainer_with_custom_loss(self):
        """Test trainer initialization with custom loss function."""
        snn = self._create_simple_network()
        custom_loss = nn.MSELoss()

        trainer = Trainer(snn, readout_type="mem", loss_fn=custom_loss)

        assert trainer.loss_fn == custom_loss

    def test_invalid_readout_type(self):
        """Test that invalid readout type raises ValueError."""
        snn = self._create_simple_network()

        with pytest.raises(ValueError, match="Invalid readout type"):
            Trainer(snn, readout_type="invalid")

    def test_valid_readout_types(self):
        """Test all valid readout types."""
        snn = self._create_simple_network()
        valid_readouts = [
            "spk",
            "spk_count",
            "mem",
            "mem_softmax",
            "mem_max",
            "mem_avg",
        ]

        for readout_type in valid_readouts:
            trainer = Trainer(snn, readout_type=readout_type)
            assert trainer.readout_type == readout_type

    def test_train_one_epoch(self):
        """Test training for one epoch."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        train_loader = self._create_dummy_dataloader(n_samples=8, batch_size=4)

        loss, acc = trainer.train_one_epoch(train_loader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_evaluate(self):
        """Test evaluation on a dataset."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        val_loader = self._create_dummy_dataloader(n_samples=8, batch_size=4)

        loss, acc = trainer.evaluate(val_loader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_train_multiple_epochs(self):
        """Test training for multiple epochs."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        train_loader = self._create_dummy_dataloader(n_samples=16, batch_size=4)
        val_loader = self._create_dummy_dataloader(n_samples=8, batch_size=4)

        # Training should not raise any errors
        trainer.train(train_loader, val_loader, n_epochs=2, store=False)

    def test_readout_mem(self):
        """Test memory readout."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        out_rec, targets = trainer.readout(labels)

        assert out_rec.shape[0] == 2 * 10  # batch_size * time_steps
        assert out_rec.shape[1] == 3  # n_neurons
        assert targets.shape[0] == 2 * 10

    def test_readout_mem_max(self):
        """Test max memory readout."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem_max")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        out_rec, targets = trainer.readout(labels)

        assert out_rec.shape[0] == 2  # batch_size
        assert out_rec.shape[1] == 3  # n_neurons

    def test_readout_mem_avg(self):
        """Test average memory readout."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem_avg")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        out_rec, targets = trainer.readout(labels)

        assert out_rec.shape[0] == 2  # batch_size
        assert out_rec.shape[1] == 3  # n_neurons

    def test_readout_spk(self):
        """Test spike readout."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="spk")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        out_rec, targets = trainer.readout(labels)

        assert out_rec.shape[0] == 2 * 10  # batch_size * time_steps
        assert out_rec.shape[1] == 3  # n_neurons

    def test_readout_spk_count(self):
        """Test spike count readout."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="spk_count")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        out_rec, targets = trainer.readout(labels)

        assert out_rec.shape[0] == 2  # batch_size
        assert out_rec.shape[1] == 3  # n_neurons

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        input_spikes = torch.randn(10, 2, 5).to(trainer.device)
        snn(input_spikes)

        labels = torch.randint(0, 3, (2,)).to(trainer.device)
        accuracy = trainer.compute_accuracy(labels)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_store_model(self, tmp_path):
        """Test model storage."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        out_dir = str(tmp_path / "test_output")
        out_file = "test_model.pt"

        trainer.store(out_dir, out_file)

        import os

        assert os.path.exists(os.path.join(out_dir, out_file))

    def test_store_model_default_filename(self, tmp_path):
        """Test model storage with default filename."""
        snn = self._create_simple_network()
        trainer = Trainer(snn, readout_type="mem")

        out_dir = str(tmp_path / "test_output")

        trainer.store(out_dir)

        import os

        assert os.path.exists(os.path.join(out_dir, "trained_state_dict.pt"))
