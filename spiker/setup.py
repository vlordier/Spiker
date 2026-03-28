from setuptools import setup, find_packages
import os


def read_long_description():
    """Read the README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


setup(
    name="spikerplus",
    version="2.0.3",
    author="Alessio Carpegna",
    author_email="alessio.carpegna@polito.it",
    description="Build, train, optimize and generate hardware accelerators for Spiking Neural Networks using VHDL",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/smilies-polito/Spiker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.20",
        "torch>=1.12",
        "snntorch>=0.9.1",
        "tabulate>=0.9.0",
    ],
    keywords=[
        "spiking neural networks",
        "FPGA",
        "VHDL",
        "neuromorphic computing",
        "edge AI",
    ],
)
