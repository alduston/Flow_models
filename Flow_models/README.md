# Flow Models Research Repository

This repository contains research scripts for flow-based generative models, including Critical Gate (CSEM) and VAE Reflow implementations.

## Project Structure

```
Flow_models/
├── CSEM/
│   └── Scripts/
│       ├── jupyter/
│       │   └── crit_gate.ipynb
│       └── python/
│           └── crit_gate.py
├── VAE_reflow/
│   └── Scripts/
│       ├── jupyter/
│       │   └── VAE_reflow2d .ipynb
│       └── python/
│           └── vae_reflow2d.py
├── requirements.txt
└── README.md
```

## Requirements

### Python Version
- **Python 3.10 or later** (required for union type syntax `int | None`)

### Hardware Requirements
- **GPU recommended**: CUDA-compatible GPU for optimal performance
- **CPU fallback**: Scripts will work on CPU but may be significantly slower

## Setup Instructions

### 1. Create a Virtual Environment

#### Using venv (recommended):
```bash
# Create virtual environment
python -m venv flow_models_env

# Activate virtual environment
# On Windows:
flow_models_env\Scripts\activate
# On macOS/Linux:
source flow_models_env/bin/activate
```

#### Using conda:
```bash
# Create conda environment with Python 3.10+
conda create -n flow_models python=3.10
conda activate flow_models
```

### 2. Install Dependencies

#### Option A: Install from requirements.txt (recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Install packages individually
```bash
# Core dependencies
pip install numpy scipy matplotlib scikit-learn

# PyTorch (choose appropriate version for your CUDA setup)
pip install torch torchvision

# GPU acceleration (optional - requires CUDA)
pip install cupy-cuda12x  # For CUDA 12.x
# OR for CUDA 11.x:
# pip install cupy-cuda11x
```

### 3. Verify Installation

Test that all dependencies are properly installed:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn
try:
    import cupy as cp
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    print("CuPy not available - running on CPU only")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

### Running Python Scripts

#### CSEM Critical Gate:
```bash
cd CSEM/Scripts/python
python crit_gate.py
```

#### VAE Reflow 2D:
```bash
cd VAE_reflow/Scripts/python
python vae_reflow2d.py
```

### Running Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the desired notebook:
   - `CSEM/Scripts/jupyter/crit_gate.ipynb`
   - `VAE_reflow/Scripts/jupyter/VAE_reflow2d .ipynb`

## Key Dependencies

- **NumPy**: Numerical computing
- **PyTorch**: Deep learning framework
- **CuPy**: GPU-accelerated NumPy (optional)
- **Matplotlib**: Plotting and visualization
- **Scikit-learn**: Machine learning utilities (PCA, KernelDensity)
- **SciPy**: Scientific computing

## Notes

- The scripts were originally developed in Google Colab and converted to standalone Python files
- GPU acceleration via CuPy is optional but recommended for performance
- Some functions may require significant computational resources
- The code includes both CPU and GPU fallback implementations

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
1. Ensure you have the correct CUDA version installed
2. Install the appropriate CuPy version for your CUDA version
3. The scripts will automatically fall back to CPU if GPU is unavailable

### Memory Issues
For large-scale experiments:
1. Reduce batch sizes in the scripts
2. Use CPU-only mode if GPU memory is insufficient
3. Consider using smaller datasets for testing

### Import Errors
If you encounter import errors:
1. Verify all dependencies are installed: `pip list`
2. Check Python version: `python --version`
3. Ensure virtual environment is activated
