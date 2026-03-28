# Spiker: A Framework for Efficient Spiking Neural Network FPGA Accelerators

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/spikerplus.svg)](https://badge.fury.io/py/spikerplus)
[![Docs: Documentation](https://img.shields.io/badge/Docs-Documentation-blue)](https://github.com/smilies-polito/Spiker#readme)

Spiker is a comprehensive framework for generating efficient, low-power, and low-area customized Spiking Neural Networks (SNN) accelerators on FPGA for inference at the edge. It presents a library of highly efficient neuron architectures and a design framework, enabling the development of complex neural network accelerators with minimal Python code.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tutorials](#tutorials)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🔍 Overview

Spiker provides an end-to-end workflow for designing SNN hardware accelerators:
1. Build and train SNN models using PyTorch and snnTorch
2. Optimize network parameters for hardware efficiency
3. Generate synthesizable VHDL code for FPGA implementation
4. Validate functionality through simulation

## ⭐ Features

- **Multiple Neuron Models**: Support for IF, LIF, Synaptic, RIF, RLIF, and RSynaptic neurons
- **Flexible Network Configuration**: Easy specification of network architecture through dictionaries
- **Hardware-Aware Optimization**: Parameters tuned for efficient FPGA implementation
- **Automatic VHDL Generation**: From trained PyTorch models to synthesizable VHDL
- **FPGA-Optimized**: Designed for resource-constrained edge devices
- **Well-Documented**: Extensive documentation and tutorials
- **Modern Python**: Requires Python 3.12+ with type hints and modern tooling (ruff, pre-commit)

## 📁 Project Structure

```
Spiker/
├── spiker/                 # Main Python package
│   ├── __init__.py         # Package exports
│   ├── setup.py            # Installation script (legacy)
│   └── spikerplus/         # Core implementation
│       ├── __init__.py     # Module exports
│       ├── net_builder.py  # SNN construction utilities
│       ├── trainer.py      # Training framework
│       ├── optimizer.py    # Optimization utilities
│       ├── vhdl_generator.py # VHDL code generation
│       ├── dataloaders/    # Data loading utilities
│       └── vhdl/           # VHDL templates and components
├── Tutorials/              # Jupyter notebooks and examples
├── Doc/                    # Documentation files
├── README.md               # This file
└── pyproject.toml          # Modern build configuration (PEP 621)
```

## 🚀 Installation

### From PyPI (Recommended)
```bash
pip install spikerplus
```

### Using uv (Alternative, faster installer)
```bash
uv pip install spikerplus
```

### From Source
```bash
git clone https://github.com/smilies-polito/Spiker.git
cd Spiker/spiker
pip install .
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/smilies-polito/Spiker.git
cd Spiker

# Install the package in development mode with development dependencies
uv pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## 📖 Usage

### Basic SNN Creation
```python
from spikerplus.net_builder import NetBuilder
import torch

# Define network configuration
net_dict = {
    "n_cycles": 73,
    "n_inputs": 40,
    "layer_0": {
        "neuron_model": "lif",
        "n_neurons": 128,
        "threshold": 1.0,
        "beta": 0.9375,
        "reset_mechanism": "subtract"
    },
    "layer_1": {
        "neuron_model": "lif",
        "n_neurons": 10,
        "threshold": 1.0,
        "beta": 0.9375,
        "reset_mechanism": "none"
    }
}

# Build the network
net_builder = NetBuilder(net_dict)
snn = net_builder.build()

# Forward pass
input_spikes = torch.randn(73, 32, 40)  # [time_steps, batch_size, n_inputs]
output = snn(input_spikes)
```

### Training
```python
from spikerplus.trainer import Trainer

trainer = Trainer(snn, readout_type="mem")
trainer.train(train_loader, val_loader, n_epochs=20)
```

### VHDL Generation
```python
from spikerplus.vhdl_generator import VhdlGenerator

optim_config = {
    "neurons_bw": 16,
    "fp_dec": 8,
    "weights_bw": 8
}

vhdl_gen = VhdlGenerator(snn, optim_config)
vhdl_code = vhdl_gen.generate(functional=True, interface=False)
```

## 🎥 Tutorials

Comprehensive video tutorials are available on YouTube:
[Spiker Tutorial Series](https://www.youtube.com/watch?v=y3OvFHBXrDE&list=PLkIAXI4vJ8EgfZki2WRh2Da_h-w6gKbsd)

The tutorials cover:
- Network definition and building
- Training procedures
- Hardware optimization
- VHDL code generation
- FPGA implementation

## 📚 Citation

If you use Spiker in your research, please cite:

### Spiker+ (Latest Version)
```bibtex
@article{carpegna_spiker_2024,
    title = {Spiker+: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge},
    author = {Carpegna, Alessio and Savino, Alessandro and Di Carlo, Stefano},
    journal = {IEEE Transactions on Emerging Topics in Computing},
    year = {2024},
    doi = {10.1109/TETC.2024.3511676}
}
```

### Original Spiker
```bibtex
@inproceedings{carpegna_spiker_2022,
    title = {Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks},
    author = {Carpegna, Alessio and Savino, Alessandro and Di Carlo, Stefano},
    booktitle = {2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)},
    pages = {14--19},
    year = {2022},
    doi = {10.1109/ISVLSI54635.2022.00016}
}
```

## 🙏 Acknowledgements

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070238. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

Special thanks to:
- [Neuropuls](https://neuropuls.eu/) for their support
- Domenico Elia Sabella for valuable assistance in revising and cleaning the code
- The developers of [rftafas/hdltools](https://github.com/rftafas/hdltools) whose code was adapted for the VHDL utilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.