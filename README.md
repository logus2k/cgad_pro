# FEMulator Pro

**High-Performance GPU-Accelerated Finite Element Analysis**

*Project for High Performance Graphical Computing*

A platform for benchmarking and comparing different computational strategies for Finite Element Method (FEM) solvers, featuring real-time 3D visualization and comprehensive performance analysis tools.

## About

FEMulator Pro solves the 2D Laplace equation using the Finite Element Method with 8-node quadrilateral (Quad-8) elements. The platform implements multiple solver backends to demonstrate and compare different parallelization strategies on modern hardware.

The Laplace equation governs a wide range of physical phenomena:

| Application | Physical Interpretation |
|-------------|------------------------|
| Incompressible irrotational flow | Velocity potential |
| Steady-state heat conduction | Temperature field |
| Electrostatics | Electric potential |
| Diffusion (steady-state) | Concentration field |

## Features

- **Multiple Solver Implementations**: CPU, GPU (CuPy), Numba JIT, Numba CUDA, Threaded, and Multiprocess
- **Real-time 3D Visualization**: Interactive mesh and solution field rendering with Three.js
- **Live Progress Streaming**: Monitor solver convergence in real-time via WebSocket
- **Automated Benchmarking**: Statistical analysis across solver/mesh combinations
- **Report Generation**: Performance reports with interactive charts, exportable to DOCX/PDF
- **Mesh Gallery**: Pre-built test geometries (Backward-Facing Step, Elbow, S-Bend, T-Junction, Venturi, Y-Shaped)

## Solver Implementations

| Implementation | Technology | Best For |
|----------------|------------|----------|
| **CPU Baseline** | NumPy / SciPy | Reference, debugging |
| **CPU Threaded** | ThreadPoolExecutor | Light parallelism |
| **CPU Multiprocess** | multiprocessing.Pool | Multi-core CPU utilization |
| **Numba CPU** | Numba JIT | Production without GPU |
| **Numba CUDA** | Numba CUDA kernels | Custom GPU control |
| **GPU (CuPy)** | CuPy + cuSPARSE | Maximum performance |

### Performance Characteristics

Based on benchmark results across multiple hardware configurations:

- **Small meshes (<10K nodes)**: Numba CPU recommended (GPU transfer overhead dominates)
- **Large meshes (>100K nodes)**: GPU achieves **20-30x speedup** over CPU baseline
- **GPU bottleneck**: Iterative solver (CG) consumes 80-90% of total time on large meshes

## Installation

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| **Windows** | Windows 10+, Docker Desktop |
| **Linux** | Docker Engine, Docker Compose plugin |
| **GPU Support** | NVIDIA GPU + drivers (optional) |

---

### Option 1: DockerHub (Quickest)

Pull and run the pre-built image directly:

```yaml
# docker-compose.yml
services:
  femulator:
    image: logus2k/femulator:latest
    container_name: femulator
    hostname: femulator
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [ gpu ]    
    ports:
      - "5868:5868"
    networks:
      - femulator_network

networks:
  femulator_network:
    driver: bridge
```

```bash
docker-compose up -d
```

Access the application at `http://localhost:5868`

---

### Option 2: Helper Scripts (Recommended for Development)

The repository includes platform-specific scripts that wrap all Docker operations:

| Purpose | Windows | Linux |
|---------|---------|-------|
| Start application | `start.bat` | `./start.sh` |
| Stop application | `stop.bat` | `./stop.sh` |
| Update application | `update.bat` | `./update.sh` |
| Full rebuild | `rebuild_all.bat` | `./rebuild_all.sh` |
| Full removal | `remove_all.bat` | `./remove_all.sh` |

**First-time setup (Linux only):**
```bash
chmod +x *.sh
```

**Start the application:**
```bash
# Windows
start.bat

# Linux
./start.sh
```

The scripts automatically detect GPU availability and select CPU or GPU mode accordingly.

---

### Option 3: Manual Installation (Without Docker)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install \
    python-socketio uvicorn fastapi \
    numpy pandas scipy matplotlib h5py \
    psutil weasyprint playwright pypandoc

# Install Playwright browser
playwright install chromium

# GPU support (requires CUDA 13.x)
pip install cupy-cuda13x numba-cuda cuda-python

# Start the server
cd src/app/server
python main.py
```

Access the application at `http://localhost:5867`

## Usage

### Web Interface

1. Select a mesh from the gallery
2. Choose a solver implementation
3. Click **Run Simulation**
4. View real-time solver progress and 3D visualization
5. Explore solution fields (potential, velocity, pressure)

### Running Benchmarks

```bash
cd src/app/server/automated_benchmark

# Full benchmark suite
python run_benchmark.py

# Resume interrupted run
python run_benchmark.py --resume

# Specific solver or model
python run_benchmark.py --solver gpu --model "Y-Shaped"
```

### Generating Reports

```bash
# Generate markdown report from benchmark results
python -m report_generator benchmark_results.json

# Export to Word/PDF
python markdown_to_report.py report.md --pdf
```

## Technical Background

### Finite Element Formulation

The solver uses Quad-8 isoparametric elements with:
- 8 nodes per element (4 corner + 4 mid-edge)
- Quadratic shape functions
- 3×3 Gauss-Legendre integration

### Linear Solver

All implementations use the **Conjugate Gradient (CG)** method with Jacobi preconditioning:
- Symmetric positive-definite system from elliptic PDE
- Memory-efficient (no explicit factorization)
- Predictable convergence behavior
- Data-parallel operations suitable for GPU

### Boundary Conditions

- **Robin BC** (inlet): Prescribed flux-potential combination
- **Dirichlet BC** (outlet): Fixed potential values

## Mesh Format

Meshes are stored in HDF5 format:

```
mesh.h5
├── x        # Node X coordinates (float64)
├── y        # Node Y coordinates (float64)
└── quad8    # Element connectivity (int32, N×8)
```

Also supports `.npz` (NumPy) and `.xlsx` (Excel) formats.

## Project Structure

```
src/
├── app/
│   ├── client/           # Web frontend (HTML/JS/CSS)
│   └── server/           # FastAPI backend
│       └── automated_benchmark/
├── cpu/                  # CPU solver
├── gpu/                  # CuPy GPU solver
├── numba/                # Numba JIT solver
├── numba_cuda/           # Numba CUDA solver
├── cpu_threaded/         # Threaded solver
├── cpu_multiprocess/     # Multiprocess solver
└── shared/               # Common utilities
```

## License

Apache 2.0 License

## Acknowledgments

- [SciPy](https://scipy.org/) for sparse matrix operations and conjugate gradient solver
- [CuPy](https://cupy.dev/) for GPU-accelerated computing
- [Three.js](https://threejs.org/) for 3D visualization
- [ECharts](https://echarts.apache.org/) for benchmark visualizations
