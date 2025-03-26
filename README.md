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

Open `local_config.py` and configure the following:

1. Set the MODE in the MAIN dictionary:
```python
MAIN = {'MODE': 0}  # 0 for running simulations
```

2. Configure ZIP dictionary:
```python
ZIP = {
    'EXAMPLE': [
        1742534789699400500, 
        1740234697756020900, 
        1742534789699400500, 
        1742534700701697500, 
        1742561006057666900
    ]
}
```

## Troubleshooting Common Dependencies

If you encounter issues with specific libraries, install them manually:

```bash
# Matplotlib
pip install matplotlib>=3.2.2

# Shapely
pip install shapely==2.0.6

# Plotly
pip install plotly==5.24.1

# Boto3
pip install boto3==1.35.28

# Pythreejs
pip install pythreejs==2.4.2

# Requests
pip install requests==2.32.3

# Scikit-learn
pip install scikit-learn==1.5.0

# Trimesh
pip install trimesh==4.5.3

# OpenCV
pip install opencv-python>=4.6.0

# Additional dependencies
pip install imageio
```

If a specific library fails to install:

```bash
# Individual library installation without any version mention
pip install <library-name>

# Example
pip install shapely
pip install trimesh
```

### PyVista and VTK Installation

If you have trouble with PyVista:

```bash
# Try specific version
pip install pyvista==0.44.1

# If issues persist, try alternative methods:
pip install --upgrade setuptools wheel

# Conda installation
conda install -c conda-forge pyvista vtk

# Install HDF5
conda install -c conda-forge hdf5

# Build from source (advanced users)
git clone https://gitlab.kitware.com/vtk/vtk.git
cd vtk
# Follow VTK documentation for building

# If still facing issues, try:
pip install git+https://github.com/pyvista/pyvista.git
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

2. For persistent VTK errors, comment out specific lines from the error message in `vtkmodules/all.py`

## Final Notes

- Always ensure you're in the `r106` Conda environment before running the project
- If a step doesn't work, carefully read the error message and revisit the corresponding section
- When in doubt, reinstall the specific problematic library

## Support

If you continue to experience issues:
- Double-check your Python and library versions. I have used python 3.10 to run this.
- Ensure all dependencies are compatible
- Consider creating a new Conda environment from scratch

---

**Disclaimer**: This guide is based on the original setup documentation. Your specific environment might require slight modifications.
