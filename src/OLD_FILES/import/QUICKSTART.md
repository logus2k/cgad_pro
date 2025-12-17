# ISS Mesh Generator - Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip install trimesh gmsh numpy pandas cupy-cuda12x scipy openpyxl

# 2. Verify installation
python test_mesh_generator.py
```

## Get ISS Model (2 options)

### Option A: NASA Direct Download
1. Visit: https://nasa3d.arc.nasa.gov/
2. Search: "International Space Station"
3. Download: ISS.glb
4. Save to: `data/models/iss.glb`

### Option B: Use prepare script
```bash
python prepare_iss_model.py --download
python prepare_iss_model.py --validate data/models/iss.glb
```

## Generate Your First Mesh (1 command)

```bash
python iss_mesh_generator.py data/models/iss.glb --elements 5000 --output my_first_mesh
```

Creates:
- `my_first_mesh.xlsx` - Use with your FEM solver
- `my_first_mesh.json` - Use with Three.js

## Use in Your FEM Solver

```python
import pandas as pd

# Load mesh
coord = pd.read_excel("my_first_mesh.xlsx", sheet_name="coord", header=None)
conec = pd.read_excel("my_first_mesh.xlsx", sheet_name="conec", header=None)

x = coord.iloc[:, 0].to_numpy(dtype=float)
y = coord.iloc[:, 1].to_numpy(dtype=float)
z = coord.iloc[:, 2].to_numpy(dtype=float)

quad8 = conec.to_numpy(dtype=int) - 1  # 0-based indexing

# Continue with your existing quad8_gpu.py code
```

## Next Steps

1. Generate multiple resolutions:
   ```bash
   python iss_mesh_generator.py data/models/iss.glb --elements 20000 --output iss_20k
   python iss_mesh_generator.py data/models/iss.glb --elements 50000 --output iss_50k
   ```

2. Run examples:
   ```bash
   python example_usage.py
   ```

3. Integrate with Three.js:
   - Use `my_first_mesh.json`
   - Load in your GAIA globe application
   - Map temperatures from FEM results to vertex colors

## Troubleshooting

**"No module named gmsh"**
```bash
pip install gmsh
```

**"No Quad-8 elements found"**
```bash
python prepare_iss_model.py --optimize data/models/iss.glb
# Then regenerate mesh
```

**Out of memory**
```bash
# Use smaller mesh
python iss_mesh_generator.py data/models/iss.glb --elements 1000 --output iss_small
```

## Files Included

- `iss_mesh_generator.py` - Main mesh generator
- `prepare_iss_model.py` - Download/validate ISS model
- `example_usage.py` - Usage examples
- `test_mesh_generator.py` - Verification tests
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation
- `QUICKSTART.md` - This file

## Support

Check README.md for:
- Detailed API documentation
- Advanced configuration
- Quality validation
- Performance benchmarks
