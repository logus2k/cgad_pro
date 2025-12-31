# FEM Solver Performance Benchmark Report

## Executive Summary

This report presents comprehensive performance benchmarks comparing six FEM solver implementations across four mesh sizes, demonstrating the progression from sequential CPU execution to fully GPU-accelerated computation.

### Key Results at a Glance

| Implementation | XL Mesh Time | Speedup vs Baseline |
|----------------|--------------|---------------------|
| CPU Baseline | [placeholder] | 1.0x |
| CPU Threaded | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] |
| **CuPy GPU** | **[placeholder]** | **[placeholder]** |

**Maximum Speedup Achieved:** [placeholder]x (CuPy GPU vs CPU Baseline on XL mesh)

---

## 1. Test Configuration

### 1.1 Hardware Environment

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i9-13900K (24 cores / 32 threads) |
| RAM | 94.3 GB DDR5 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| CUDA Version | 13.0 |
| OS | Linux (WSL2) 6.6.87.2-microsoft-standard |
| Python | 3.10.19 |

### 1.2 Test Meshes

All benchmarks use the **Y-Shaped Tube** geometry - a 2D longitudinal cross-section representing a flow channel that starts as a single rectangular inlet and splits into two parallel outlet branches.

| Size Label | Mesh File | Nodes | Elements | Matrix NNZ (est.) |
|------------|-----------|-------|----------|-------------------|
| XS (Tiny) | `y_tube1_53.h5` | 202 | 53 | ~3.4K |
| M (Medium) | `y_tube1_195k.h5` | 195,853 | 46,607 | ~3.0M |
| L (Large) | `y_tube1_772k.h5` | 772,000 | 192,000 | ~12.3M |
| XL (Extra Large) | `y_tube1_1_3m.h5` | 1,357,953 | 338,544 | ~21.7M |

### 1.3 Solver Configuration

| Parameter | Value |
|-----------|-------|
| Problem Type | 2D Potential Flow (Laplace Equation) |
| Element Type | Quad-8 (8-node serendipity quadrilateral) |
| Linear Solver | Conjugate Gradient |
| Tolerance | 1e-8 (relative) |
| Max Iterations | 50,000 |
| Preconditioner | Jacobi (diagonal) |
| Equilibration | Diagonal scaling |
| Boundary Conditions | Robin (inlet), Dirichlet (outlet) |

### 1.4 Benchmark Protocol

| Parameter | Value |
|-----------|-------|
| Warm-up Runs | 1 (discarded) |
| Measured Runs | 3 (averaged) |
| Timing Method | `time.perf_counter()` (high-resolution) |
| JIT Compilation | Included in warm-up, excluded from measurements |

**Note:** The warm-up run ensures JIT compilation (Numba) and GPU kernel compilation (CuPy RawKernel) overhead is excluded from timing measurements, providing a fair comparison of steady-state performance.

---

## 2. Implementations Tested

| # | Implementation | File | Parallelism Strategy |
|---|----------------|------|----------------------|
| 1 | CPU Baseline | `quad8_cpu_v3.py` | Sequential Python loops |
| 2 | CPU Threaded | `quad8_cpu_threaded.py` | ThreadPoolExecutor (GIL-limited) |
| 3 | CPU Multiprocess | `quad8_cpu_multiprocess.py` | multiprocessing.Pool |
| 4 | Numba CPU | `quad8_numba.py` | @njit + prange |
| 5 | Numba CUDA | `quad8_numba_cuda.py` | @cuda.jit kernels |
| 6 | CuPy GPU | `quad8_gpu_v3.py` | CUDA C RawKernels |

---

## 3. Results: Total Workflow Time

### 3.1 Summary Table (All Mesh Sizes)

Total workflow time in seconds (average of 3 runs):

| Implementation | XS (202) | M (196K) | L (772K) | XL (1.36M) |
|----------------|----------|----------|----------|------------|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

### 3.2 Speedup vs CPU Baseline

| Implementation | XS | M | L | XL |
|----------------|----|----|----|----|
| CPU Baseline | 1.0x | 1.0x | 1.0x | 1.0x |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

### 3.3 Visual: Total Time Comparison (XL Mesh)

```
Implementation      Time (seconds)                                    Speedup
─────────────────────────────────────────────────────────────────────────────
CPU Baseline   │████████████████████████████████████████████████│ [xxx.x]s  1.0x
CPU Threaded   │████████████████████████████████████████████    │ [xxx.x]s  [x.x]x
CPU Multiproc  │████████████████████████████                    │ [xxx.x]s  [x.x]x
Numba CPU      │████████████████                                │ [xx.x]s   [x.x]x
Numba CUDA     │████                                            │ [xx.x]s   [xx]x
CuPy GPU       │██                                              │ [x.x]s    [xx]x
               └────────────────────────────────────────────────┘
```

---

## 4. Results: Stage-by-Stage Breakdown

### 4.1 XL Mesh (1.36M nodes) - Detailed Timing

Time in seconds per stage (average of 3 runs):

| Stage | CPU Baseline | CPU Threaded | CPU Multiproc | Numba CPU | Numba CUDA | CuPy GPU |
|-------|--------------|--------------|---------------|-----------|------------|----------|
| `load_mesh` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `assemble_system` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `apply_bc` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `solve_system` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `compute_derived` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| **total_workflow** | **[placeholder]** | **[placeholder]** | **[placeholder]** | **[placeholder]** | **[placeholder]** | **[placeholder]** |

### 4.2 Stage Speedup (XL Mesh, vs CPU Baseline)

| Stage | Threaded | Multiproc | Numba CPU | Numba CUDA | CuPy GPU |
|-------|----------|-----------|-----------|------------|----------|
| `assemble_system` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `apply_bc` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `solve_system` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| `compute_derived` | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| **Total** | **[placeholder]** | **[placeholder]** | **[placeholder]** | **[placeholder]** | **[placeholder]** |

### 4.3 Time Distribution (% of Total)

Where time is spent in each implementation (XL mesh):

| Implementation | Assembly | BC | Solve | Post-Proc | Other |
|----------------|----------|-------|-------|-----------|-------|
| CPU Baseline | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |
| CPU Threaded | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |
| CPU Multiprocess | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |
| Numba CPU | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |
| Numba CUDA | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |
| CuPy GPU | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% | [placeholder]% |

**Key Insight:** As assembly is optimized, the solve stage becomes the dominant bottleneck.

---

## 5. Scaling Analysis

### 5.1 How Performance Scales with Problem Size

Total time vs mesh size (shows scaling behavior):

| Implementation | XS→M | M→L | L→XL | Scaling Factor |
|----------------|------|-----|------|----------------|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] | O(n^[x]) |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | O(n^[x]) |

### 5.2 Speedup vs Problem Size

Does GPU advantage increase with larger problems?

| Mesh Size | Nodes | CuPy GPU Speedup |
|-----------|-------|------------------|
| XS | 202 | [placeholder]x |
| M | 195,853 | [placeholder]x |
| L | 772,000 | [placeholder]x |
| XL | 1,357,953 | [placeholder]x |

**Observation:** [Placeholder for observation about speedup scaling with problem size]

---

## 6. Convergence Verification

All implementations must produce numerically identical results to validate correctness.

### 6.1 Solver Convergence (XL Mesh)

| Implementation | Converged | Iterations | Final Residual |
|----------------|-----------|------------|----------------|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] |

### 6.2 Solution Statistics (XL Mesh)

| Implementation | u_min | u_max | u_mean | u_std |
|----------------|-------|-------|--------|-------|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

### 6.3 Cross-Implementation Verification

| Metric | Value |
|--------|-------|
| Maximum |u_baseline - u_gpu| | [placeholder] |
| All iterations match | [placeholder] |
| All converged | [placeholder] |

**Conclusion:** [Placeholder - confirm all implementations produce identical results within numerical precision]

---

## 7. Computational Efficiency Metrics

### 7.1 Throughput (XL Mesh)

| Implementation | Elements/sec (Assembly) | DOFs/sec (Solve) |
|----------------|-------------------------|------------------|
| CPU Baseline | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] |

*Elements/sec = Nels / assemble_time; DOFs/sec = Nnds × iterations / solve_time*

### 7.2 Hardware Utilization

| Implementation | CPU Threads Active | GPU Utilization |
|----------------|--------------------|-----------------| 
| CPU Baseline | 1 | N/A |
| CPU Threaded | 24 (GIL-limited) | N/A |
| CPU Multiprocess | 24 | N/A |
| Numba CPU | 24 | N/A |
| Numba CUDA | 1 | ~85% |
| CuPy GPU | 1 | ~92% |

---

## 8. Analysis & Discussion

### 8.1 Bottleneck Evolution

As optimizations progress, the computational bottleneck shifts:

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|--------------------|
| CPU Baseline | Assembly ([x]%) | Solve ([x]%) |
| CPU Threaded | Assembly ([x]%) | Solve ([x]%) |
| CPU Multiprocess | Assembly ([x]%) | Solve ([x]%) |
| Numba CPU | Assembly ([x]%) | Solve ([x]%) |
| Numba CUDA | Solve ([x]%) | BC Application ([x]%) |
| CuPy GPU | Solve ([x]%) | BC Application ([x]%) |

### 8.2 Why Each Optimization Helps

| Transition | Speedup | Reason |
|------------|---------|--------|
| Baseline → Threaded | [x]x | Limited by Python GIL; threads only help for I/O |
| Threaded → Multiprocess | [x]x | Bypasses GIL via separate processes; IPC overhead limits gains |
| Multiprocess → Numba CPU | [x]x | JIT compilation eliminates interpreter overhead; true parallel loops |
| Numba CPU → Numba CUDA | [x]x | GPU parallelism: thousands of threads vs dozens of CPU cores |
| Numba CUDA → CuPy GPU | [x]x | CUDA C kernels more optimized than Numba-generated PTX; vectorized COO on GPU |

### 8.3 Amdahl's Law Observations

With assembly highly optimized on GPU, the solve stage becomes the limiting factor:

- **CPU Baseline:** Assembly is [x]% of total → optimizing it yields large gains
- **CuPy GPU:** Assembly is only [x]% of total → further assembly optimization has diminishing returns
- **Future optimization target:** The iterative solver (CG) is now the bottleneck

### 8.4 Small Mesh Considerations

For the XS mesh (202 nodes), GPU implementations may show reduced speedup or even slowdown due to:
- Kernel launch overhead (~5-20μs per kernel)
- Data transfer latency
- Insufficient parallelism to saturate GPU

**Observed:** [Placeholder for observation about XS mesh performance]

---

## 9. Conclusions

### 9.1 Key Findings

1. **Maximum Speedup:** CuPy GPU achieves [placeholder]x speedup over CPU Baseline on the XL mesh.

2. **Threading is Ineffective:** CPU Threaded shows only [placeholder]x speedup due to Python's GIL.

3. **JIT Compilation is Transformative:** Numba CPU delivers [placeholder]x speedup by eliminating interpreter overhead.

4. **GPU Parallelism Dominates:** GPU implementations (Numba CUDA, CuPy) provide [placeholder]x speedup through massive parallelism.

5. **Speedup Increases with Problem Size:** GPU advantage grows from [placeholder]x (XS) to [placeholder]x (XL).

6. **Solve Becomes the New Bottleneck:** On GPU, the iterative solver consumes [placeholder]% of total time.

### 9.2 Recommendations

| Use Case | Recommended Implementation |
|----------|---------------------------|
| Development/debugging | CPU Baseline or Numba CPU |
| Production (no GPU) | Numba CPU |
| Production (with GPU) | CuPy GPU |
| Small meshes (<10K nodes) | Numba CPU (GPU overhead not worthwhile) |
| Large meshes (>100K nodes) | CuPy GPU |

### 9.3 Future Optimization Opportunities

1. **GPU-Accelerated Solver:** Implement a custom GPU CG kernel or use cuSOLVER
2. **Robin BC on GPU:** Move edge detection and Robin assembly to GPU
3. **Mixed Precision:** Use FP32 for assembly, FP64 only for solve
4. **Multi-GPU:** Distribute large meshes across multiple GPUs

---

## 10. Reproducibility

### 10.1 Benchmark Metadata

| Field | Value |
|-------|-------|
| Benchmark Date | [placeholder] |
| Report Generated | [placeholder] |
| Server Hash | [placeholder] |
| Total Runs Executed | [placeholder] |

### 10.2 Raw Data Location

All benchmark records are stored in `benchmark_results.json` with the following comparison run identifiers:

| Mesh | Comparison ID |
|------|---------------|
| XS | `y_tube_xs_6impl_[date]` |
| M | `y_tube_m_6impl_[date]` |
| L | `y_tube_l_6impl_[date]` |
| XL | `y_tube_xl_6impl_[date]` |

### 10.3 How to Reproduce

```bash
# Run benchmarks (example command)
python run_benchmarks.py --mesh y_tube1_1_3m.h5 --implementations all --warmup 1 --runs 3
```

---

## Appendix A: Detailed Run Data

### A.1 XS Mesh (202 nodes, 53 elements)

#### Run Statistics

| Implementation | Run 1 | Run 2 | Run 3 | Mean | Std Dev |
|----------------|-------|-------|-------|------|---------|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

### A.2 M Mesh (195,853 nodes, 46,607 elements)

[Same table structure as A.1]

### A.3 L Mesh (772,000 nodes, 192,000 elements)

[Same table structure as A.1]

### A.4 XL Mesh (1,357,953 nodes, 338,544 elements)

[Same table structure as A.1]

---

## Appendix B: Stage Timing Details (All Meshes)

### B.1 Assembly Time

| Implementation | XS | M | L | XL |
|----------------|----|----|----|----|
| CPU Baseline | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Threaded | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CPU Multiprocess | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Numba CUDA | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| CuPy GPU | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

### B.2 Solve Time

[Same table structure as B.1]

### B.3 BC Application Time

[Same table structure as B.1]

### B.4 Post-Processing Time

[Same table structure as B.1]

---

## Appendix C: Memory Usage

**Note:** Per-run peak memory tracking is not currently implemented. This section will be populated when memory instrumentation is added to the solver implementations.

### C.1 Estimated Memory Requirements

| Mesh | Nodes | Matrix NNZ | Est. RAM (CPU) | Est. VRAM (GPU) |
|------|-------|------------|----------------|-----------------|
| XS | 202 | ~3.4K | < 100 MB | < 100 MB |
| M | 195,853 | ~3.0M | ~1.5 GB | ~0.8 GB |
| L | 772,000 | ~12.3M | ~6 GB | ~3 GB |
| XL | 1,357,953 | ~21.7M | ~12 GB | ~6 GB |

---

*Report generated: [placeholder]*

*FEMulator Pro - GPU-Accelerated Finite Element Analysis*
