# Robo-106 Project Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation Steps](#installation-steps)
4. [Troubleshooting](#troubleshooting)
5. [Running the Project](#running-the-project)

## Prerequisites

Before you begin, ensure you have the following installed:
- Anaconda or Miniconda
- Git
- Python 3.10

## Environment Setup

### 1. Create Conda Environment

Create a new Conda environment specifically for Robo-106:

```bash
# Create the environment
conda create --name r106 python=3.10

# Activate the environment
conda activate r106
```

### 2. Install Dependencies

You have two methods to install dependencies:

#### Method 1: Using requirements.txt

```bash
# Install dependencies from requirements file
pip install -r requirements.txt
```

#### Method 2: Using environment.yaml (Backup Method)

```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate r106
```

## Configuration

### Set Mode in local_config.py

Open `local_config.py` and set the MODE:

```python
MAIN = {'MODE': 0}  # 0 for running simulations
```

## Troubleshooting Common Dependencies

If you encounter issues with specific libraries, install them manually:

```bash
# Upgrade pip
pip install --upgrade pip

# Individual library installations
pip install matplotlib
pip install shapely
pip install opencv-python
pip install Pillow
pip install scikit-learn
pip install trimesh
pip install imageio
pip install pythreejs
```

### PyVista and VTK Installation

If you have trouble with PyVista:

```bash
# Try specific version
pip install pyvista==0.34.0

# Alternative: Conda installation
conda install -c conda-forge pyvista vtk
```

### VTK and HDF5 Specific Installation

```bash
# Install via Conda
conda install -c conda-forge vtk hdf5
```

## Running the Project

Once all dependencies are installed:

```bash
# Run the main script
python sockets.py
```

## Additional Troubleshooting

### Library Path Issues

If you encounter library path problems:

```bash
# Check HDF5 libraries
ldconfig -p | grep hdf5

# Add library path (example path, adjust as needed)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/lib
```

### Python Version Check

```bash
# Verify Python version
python --version
```

## Common Fixes

1. If `shapely` causes issues, you can comment out its import in `plan.py`:
   ```python
   # Temporarily comment out: from shapely.geometry import Polygon
   ```

2. For persistent VTK errors, comment out specific lines in `vtkmodules/all.py`

## Final Notes

- Always ensure you're in the `r106` Conda environment before running the project
- If a step doesn't work, carefully read the error message and revisit the corresponding section
- When in doubt, reinstall the specific problematic library

## Support

If you continue to experience issues:
- Double-check your Python and library versions
- Ensure all dependencies are compatible
- Consider creating a new Conda environment from scratch

---

**Disclaimer**: This guide is based on the original setup documentation. Your specific environment might require slight modifications.
