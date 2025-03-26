# Robo-106 Project Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dependency Installation](#dependency-installation)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Running the Project](#running-the-project)

## Prerequisites

Before you begin, ensure you have the following installed:
- Anaconda or Miniconda
- Git
- Python 3.10

## Environment Setup

### 1. Create Conda Environment

```bash
# Create the environment
conda create --name r106 python=3.10

# Activate the environment
conda activate r106
```

## Dependency Installation

### Specific Dependency Versions

Install the following dependencies with their specific versions:

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

### PyVista Installation

```bash
# Try specific version
pip install pyvista==0.44.1

# If issues persist, try alternative methods:
pip install --upgrade setuptools wheel

# Conda installation
conda install -c conda-forge pyvista vtk

# If still facing issues, try:
pip install git+https://github.com/pyvista/pyvista.git
```

## Configuration

### Local Configuration Setup

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

## Troubleshooting

### Library Installation Troubleshooting

If a specific library fails to install:

```bash
# Individual library installation
pip install <library-name>

# Example
pip install shapely
pip install trimesh
```

### Library Path and Compatibility

```bash
# Check HDF5 libraries
ldconfig -p | grep hdf5

# Add library path (adjust path as needed)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/lib
```

### Python Version Check

```bash
# Verify Python version
python --version
```

## Running the Project

### Initial Run

```bash
# Run the main sockets script
python sockets.py
```

### Potential Fixes

1. If `shapely` causes issues, temporarily comment out its import in `plan.py`:
   ```python
   # Temporarily comment out: from shapely.geometry import Polygon
   ```

2. For persistent VTK errors, comment out specific lines in `vtkmodules/all.py`

## Advanced Troubleshooting

### VTK and PyVista Installation

```bash
# Conda-based installation
conda install -c conda-forge vtk pyvista

# Install HDF5
conda install -c conda-forge hdf5

# Build from source (advanced users)
git clone https://gitlab.kitware.com/vtk/vtk.git
cd vtk
# Follow VTK documentation for building
```

## Final Recommendations

- Always activate the `r106` Conda environment before running the project
- Carefully read error messages to diagnose specific issues
- Ensure compatibility between library versions
- Consider creating a fresh Conda environment if persistent issues occur

---

**Disclaimer**: This guide provides a comprehensive setup process. Your specific environment might require slight adjustments.

**Troubleshooting Tips**:
- If a library doesn't install, try:
  1. Upgrading pip
  2. Installing a specific version
  3. Using Conda instead of pip
- Check your Python version for compatibility
- Verify library dependencies and conflicts

## Support

For unresolved issues:
- Check library documentation
- Consult project maintainers
- Review error logs carefully
