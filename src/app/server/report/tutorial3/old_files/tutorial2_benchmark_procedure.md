# FEMulator Pro - Benchmark Testing Procedure

## Tutorial #2: High Performance Graphical Computing

**Document Version:** 1.0  
**Date:** January 2026  
**Purpose:** Standardized benchmark testing protocol for performance comparison across solver implementations

---

## 1. Overview

This document describes the systematic testing procedure for benchmarking the six FEM solver implementations across multiple mesh configurations and server hardware. Following this protocol ensures consistent, comparable results across all test environments.

### 1.1 Objectives

- Measure execution time for each pipeline stage (assembly, solve, post-processing)
- Compare performance across solver implementations
- Evaluate scaling behavior with mesh size
- Assess GPU utilization across different hardware configurations

### 1.2 Test Matrix Summary

| Dimension | Values | Count |
|-----------|--------|-------|
| **Servers** | RTX 4090, RTX 5060, RTX 5090 | 3 |
| **Meshes** | Y-Shaped, Venturi, Backward-Facing Step | 3 |
| **Mesh Sizes** | Small, Medium, Large, Extra-Large | 4 |
| **Solvers** | CPU Baseline, CPU Threaded, CPU Multiprocess, Numba JIT, Numba CUDA, GPU CuPy | 6 |

**Total test configurations:** 3 × 3 × 4 × 6 = **216 runs** (plus warmup runs)

---

## 2. Prerequisites

### 2.1 Software Requirements

Ensure the following are installed and functional on the test server:

| Component | Minimum Version | Verification Command |
|-----------|-----------------|---------------------|
| Python | 3.10+ | `python --version` |
| NumPy | 1.24+ | `python -c "import numpy; print(numpy.__version__)"` |
| SciPy | 1.10+ | `python -c "import scipy; print(scipy.__version__)"` |
| Numba | 0.57+ | `python -c "import numba; print(numba.__version__)"` |
| CuPy | 12.0+ | `python -c "import cupy; print(cupy.__version__)"` |
| CUDA Toolkit | 11.8+ | `nvcc --version` |

### 2.2 GPU Verification

Before starting tests, verify GPU is recognized:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Verify CuPy can access GPU
python -c "import cupy as cp; print(f'GPU: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')"
```

### 2.3 Mesh Files

Ensure all mesh files are available in the data directory:

```
data/input/
├── y_shaped_small.h5
├── y_shaped_medium.h5
├── y_shaped_large.h5
├── y_shaped_xlarge.h5
├── venturi_small.h5
├── venturi_medium.h5
├── venturi_large.h5
├── venturi_xlarge.h5
├── backward_step_small.h5
├── backward_step_medium.h5
├── backward_step_large.h5
└── backward_step_xlarge.h5
```

---

## 3. Server Configuration

### 3.1 Identify Your Server

Before running tests, identify and record your server configuration:

| Server ID | GPU Model | VRAM | CUDA Cores | Expected Use |
|-----------|-----------|------|------------|--------------|
| **SRV-4090** | NVIDIA RTX 4090 | 24 GB | 16,384 | Reference high-end |
| **SRV-5060** | NVIDIA RTX 5060 | TBD | TBD | Mid-range comparison |
| **SRV-5090** | NVIDIA RTX 5090 | TBD | TBD | Next-gen comparison |

### 3.2 Record Server Details

Fill in the following for your server:

```
Server ID:          ____________________
GPU Model:          ____________________
GPU VRAM:           ____________________ GB
CUDA Cores:         ____________________
Driver Version:     ____________________
CUDA Version:       ____________________
CPU Model:          ____________________
CPU Cores:          ____________________
RAM:                ____________________ GB
Tester Name:        ____________________
Test Date:          ____________________
```

---

## 4. Mesh Specifications

### 4.1 Mesh Size Reference

Record the actual node/element counts for each mesh file:

#### Y-Shaped Mesh

| Size | File | Nodes | Elements | Notes |
|------|------|-------|----------|-------|
| Small | `y_shaped_small.h5` | | | |
| Medium | `y_shaped_medium.h5` | | | |
| Large | `y_shaped_large.h5` | | | |
| X-Large | `y_shaped_xlarge.h5` | | | |

#### Venturi Mesh

| Size | File | Nodes | Elements | Notes |
|------|------|-------|----------|-------|
| Small | `venturi_small.h5` | | | |
| Medium | `venturi_medium.h5` | | | |
| Large | `venturi_large.h5` | | | |
| X-Large | `venturi_xlarge.h5` | | | |

#### Backward-Facing Step Mesh

| Size | File | Nodes | Elements | Notes |
|------|------|-------|----------|-------|
| Small | `backward_step_small.h5` | | | |
| Medium | `backward_step_medium.h5` | | | |
| Large | `backward_step_large.h5` | | | |
| X-Large | `backward_step_xlarge.h5` | | | |

---

## 5. Solver Implementations

### 5.1 Solver Reference

| ID | Solver Name | Source File | Execution Model |
|----|-------------|-------------|-----------------|
| **S1** | CPU Baseline | `quad8_cpu_v3.py` | Sequential NumPy |
| **S2** | CPU Threaded | `quad8_cpu_threaded.py` | ThreadPoolExecutor |
| **S3** | CPU Multiprocess | `quad8_cpu_multiprocess.py` | ProcessPoolExecutor |
| **S4** | Numba JIT | `quad8_numba.py` | JIT + prange |
| **S5** | Numba CUDA | `quad8_numba_cuda.py` | @cuda.jit kernels |
| **S6** | GPU CuPy | `quad8_gpu_v3.py` | RawKernel CUDA C |

### 5.2 Execution Order

**IMPORTANT:** Always execute solvers in the following order:

1. CPU Baseline (S1)
2. CPU Threaded (S2)
3. CPU Multiprocess (S3)
4. Numba JIT (S4)
5. Numba CUDA (S5)
6. GPU CuPy (S6)

This order progresses from least to most GPU-dependent, ensuring proper warmup and cache behavior.

---

## 6. Warmup Protocol

### 6.1 Purpose

Warmup runs are essential to:

- Trigger JIT compilation (Numba)
- Populate GPU kernel caches (CUDA)
- Stabilize memory allocation patterns
- Ensure consistent timing measurements

### 6.2 Warmup Procedure

**Before switching to a new solver implementation:**

1. Run the solver **once** with the **Small** mesh of any geometry
2. **Discard** the timing results from this warmup run
3. Wait **5 seconds** before starting recorded runs
4. Proceed with the full test sequence for that solver

### 6.3 Warmup Checklist

For each solver, confirm warmup completed:

| Solver | Warmup Completed | Warmup Mesh Used | Notes |
|--------|------------------|------------------|-------|
| S1 - CPU Baseline | ☐ | | |
| S2 - CPU Threaded | ☐ | | |
| S3 - CPU Multiprocess | ☐ | | |
| S4 - Numba JIT | ☐ | | |
| S5 - Numba CUDA | ☐ | | |
| S6 - GPU CuPy | ☐ | | |

---

## 7. Test Execution

### 7.1 Execution Sequence

For each solver (after warmup), execute tests in this order:

```
For each MESH (Y-Shaped → Venturi → Backward-Step):
    For each SIZE (Small → Medium → Large → X-Large):
        Run solver
        Record all metrics
        Wait 3 seconds before next run
```

### 7.2 Command Template

```bash
# General format
python run_benchmark.py \
    --solver <solver_type> \
    --mesh <mesh_file> \
    --output <results_file>

# Example: CPU Baseline with Y-Shaped Medium mesh
python run_benchmark.py \
    --solver cpu \
    --mesh data/input/y_shaped_medium.h5 \
    --output results/cpu_baseline_y_shaped_medium.json
```

### 7.3 Using the Web Interface

Alternatively, use the FEMulator Pro web interface:

1. Open the application in browser
2. Navigate to **Gallery** → Select mesh
3. Navigate to **Settings** → Select solver type
4. Click **Run Simulation**
5. Wait for completion
6. Navigate to **Benchmark** → **Export Results**

### 7.4 Execution Checklist

Use this checklist to track progress. Mark each cell when completed:

#### Y-Shaped Mesh

| Solver | Small | Medium | Large | X-Large |
|--------|-------|--------|-------|---------|
| S1 - CPU Baseline | ☐ | ☐ | ☐ | ☐ |
| S2 - CPU Threaded | ☐ | ☐ | ☐ | ☐ |
| S3 - CPU Multiprocess | ☐ | ☐ | ☐ | ☐ |
| S4 - Numba JIT | ☐ | ☐ | ☐ | ☐ |
| S5 - Numba CUDA | ☐ | ☐ | ☐ | ☐ |
| S6 - GPU CuPy | ☐ | ☐ | ☐ | ☐ |

#### Venturi Mesh

| Solver | Small | Medium | Large | X-Large |
|--------|-------|--------|-------|---------|
| S1 - CPU Baseline | ☐ | ☐ | ☐ | ☐ |
| S2 - CPU Threaded | ☐ | ☐ | ☐ | ☐ |
| S3 - CPU Multiprocess | ☐ | ☐ | ☐ | ☐ |
| S4 - Numba JIT | ☐ | ☐ | ☐ | ☐ |
| S5 - Numba CUDA | ☐ | ☐ | ☐ | ☐ |
| S6 - GPU CuPy | ☐ | ☐ | ☐ | ☐ |

#### Backward-Facing Step Mesh

| Solver | Small | Medium | Large | X-Large |
|--------|-------|--------|-------|---------|
| S1 - CPU Baseline | ☐ | ☐ | ☐ | ☐ |
| S2 - CPU Threaded | ☐ | ☐ | ☐ | ☐ |
| S3 - CPU Multiprocess | ☐ | ☐ | ☐ | ☐ |
| S4 - Numba JIT | ☐ | ☐ | ☐ | ☐ |
| S5 - Numba CUDA | ☐ | ☐ | ☐ | ☐ |
| S6 - GPU CuPy | ☐ | ☐ | ☐ | ☐ |

---

## 8. Results Recording

### 8.1 Metrics to Capture

For each test run, record the following metrics:

| Metric | Unit | Description |
|--------|------|-------------|
| `load_mesh` | seconds | Time to load mesh from file |
| `assemble_system` | seconds | Time to assemble global stiffness matrix |
| `apply_bc` | seconds | Time to apply boundary conditions |
| `solve_system` | seconds | Time for iterative solver |
| `compute_derived` | seconds | Time for post-processing (velocity, pressure) |
| `total_workflow` | seconds | Total simulation time |
| `iterations` | count | CG iterations to convergence |
| `converged` | boolean | Whether solver converged |
| `peak_memory` | MB | Peak memory usage (if available) |

### 8.2 Results Template

For each run, record results in this format:

```
=== TEST RUN ===
Date/Time:      ____________________
Server ID:      ____________________
Solver:         ____________________
Mesh:           ____________________
Mesh Size:      ____________________
Nodes:          ____________________
Elements:       ____________________

--- Timing (seconds) ---
load_mesh:          ________
assemble_system:    ________
apply_bc:           ________
solve_system:       ________
compute_derived:    ________
total_workflow:     ________

--- Solver ---
converged:          ☐ Yes  ☐ No
iterations:         ________
final_residual:     ________

--- Memory ---
peak_memory_mb:     ________

--- Notes ---
________________________________
________________________________
```

### 8.3 JSON Export Format

If exporting via the application, results will be in this JSON format:

```json
{
  "server_id": "SRV-4090",
  "server_hash": "abc123",
  "solver_type": "cpu",
  "mesh_file": "y_shaped_medium.h5",
  "mesh_info": {
    "nodes": 50000,
    "elements": 48000
  },
  "timing_metrics": {
    "load_mesh": 0.234,
    "assemble_system": 5.678,
    "apply_bc": 0.123,
    "solve_system": 12.345,
    "compute_derived": 1.234,
    "total_workflow": 19.614
  },
  "solution_stats": {
    "converged": true,
    "iterations": 1234,
    "final_residual": 1.23e-9
  },
  "memory": {
    "peak_mb": 512.5
  },
  "timestamp": "2026-01-03T12:34:56Z"
}
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| CUDA out of memory | Mesh too large for GPU | Use smaller mesh or check for memory leaks |
| Solver not converging | Poor conditioning | Check boundary conditions, try different mesh |
| Numba compilation slow | First run | This is expected; use warmup protocol |
| Inconsistent timings | Background processes | Close other applications, disable updates |
| GPU not detected | Driver issue | Reinstall CUDA drivers, check `nvidia-smi` |

### 9.2 Validation Checks

After each run, verify:

1. **Convergence:** Solver reached target tolerance
2. **Solution range:** `u` values are physically reasonable
3. **Timing consistency:** No anomalous spikes (re-run if >20% deviation)
4. **Iteration count:** Should be similar across solvers for same mesh

### 9.3 Re-run Criteria

Re-run a test if:

- Solver did not converge
- Timing varies >20% from expected range
- System interrupted during execution
- Error messages in console output

---

## 10. Results Submission

### 10.1 File Naming Convention

Name result files using this pattern:

```
benchmark_<SERVER>_<SOLVER>_<MESH>_<SIZE>_<DATE>.json
```

Examples:
- `benchmark_4090_cpu_baseline_y_shaped_medium_20260103.json`
- `benchmark_5090_gpu_cupy_venturi_xlarge_20260103.json`

### 10.2 Submission Package

Submit a ZIP archive containing:

```
<YOUR_NAME>_benchmark_results/
├── server_info.txt          # Server configuration details
├── results/                  # All JSON result files
│   ├── benchmark_*.json
│   └── ...
├── logs/                     # Console output logs (if any)
│   └── ...
└── notes.txt                 # Any observations or issues encountered
```

### 10.3 Submission Method

Send results to: **[coordinator email/shared drive location]**

**Deadline:** [specify deadline]

### 10.4 Contact

For questions or issues during testing:

- **Technical issues:** [contact]
- **Procedure questions:** [contact]
- **Results submission:** [contact]

---

## 11. Quick Reference Card

### Execution Order (per solver)

```
1. WARMUP: Run once with any Small mesh → Discard results → Wait 5s
2. TEST: Y-Shaped    (Small → Medium → Large → X-Large)
3. TEST: Venturi     (Small → Medium → Large → X-Large)
4. TEST: Back-Step   (Small → Medium → Large → X-Large)
5. NEXT SOLVER: Repeat from step 1
```

### Solver Order

```
S1 → S2 → S3 → S4 → S5 → S6
CPU    CPU      CPU         Numba   Numba    GPU
Base   Thread   Multi       JIT     CUDA     CuPy
```

### Key Metrics

```
Primary:    assemble_system, solve_system, total_workflow
Secondary:  iterations, peak_memory
Validation: converged, final_residual
```

---

## Appendix A: Results Summary Sheet

**Tester:** __________________ **Server:** __________________ **Date:** __________________

### Timing Summary (seconds)

| Mesh | Size | S1 | S2 | S3 | S4 | S5 | S6 |
|------|------|----|----|----|----|----|----|
| Y-Shaped | Small | | | | | | |
| Y-Shaped | Medium | | | | | | |
| Y-Shaped | Large | | | | | | |
| Y-Shaped | X-Large | | | | | | |
| Venturi | Small | | | | | | |
| Venturi | Medium | | | | | | |
| Venturi | Large | | | | | | |
| Venturi | X-Large | | | | | | |
| Back-Step | Small | | | | | | |
| Back-Step | Medium | | | | | | |
| Back-Step | Large | | | | | | |
| Back-Step | X-Large | | | | | | |

### Speedup vs Baseline (S1 = 1.0×)

| Mesh | Size | S2 | S3 | S4 | S5 | S6 |
|------|------|----|----|----|----|
| Y-Shaped | Small | | | | | |
| Y-Shaped | Medium | | | | | |
| Y-Shaped | Large | | | | | |
| Y-Shaped | X-Large | | | | | |
| Venturi | Small | | | | | |
| Venturi | Medium | | | | | |
| Venturi | Large | | | | | |
| Venturi | X-Large | | | | | |
| Back-Step | Small | | | | | |
| Back-Step | Medium | | | | | |
| Back-Step | Large | | | | | |
| Back-Step | X-Large | | | | | |

---

**End of Document**
