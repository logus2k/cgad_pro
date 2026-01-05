## 4. Implementation 4 — Numba JIT CPU (Compiled Shared-Memory Parallelism)

### 4.1 What was done and why it matters
The Numba JIT implementation was introduced to establish a CPU baseline that is *already optimized*. This is a methodological requirement: claiming GPU speedups against naïve Python would be misleading and would not reflect genuine architectural advantage. Implementation 4 therefore defines the **CPU performance ceiling** within a single node using realistic engineering practices.

### 4.2 Technical mechanism (how acceleration is obtained)
Numba compiles Python functions into native code through LLVM. In practice, the project used:

- `@njit(cache=True)` for core numerical routines (element computations, BC computations, post-processing),
- `@njit(parallel=True, cache=True)` + `prange` to parallelize element loops.

Conceptually:
**Python source → Numba IR → LLVM IR → optimized machine code**.

Critical implications:
- **Interpreter overhead is removed**, making tight loops C-like in performance.
- **GIL is not a limitation** for `prange`, enabling real multi-threaded execution.
- **Compiler optimizations** become available (loop unrolling, SIMD, bounds-check elimination, inlining).

### 4.3 Key design decision: explicit loops over NumPy vectorization
A defining choice in this implementation is replacing some vectorized NumPy operations with explicit loops. This is a non-trivial performance engineering decision: in JIT-compiled contexts, explicit loops often outperform vectorization because:
- vectorization can allocate temporaries and increase memory traffic,
- loops allow the compiler to fuse operations and optimize instruction flow.

This is a strong indicator of technical maturity, because it shows alignment with the compiler model rather than “Python style”.

### 4.4 Where speedups are expected—and where they are not
The project accelerated element-wise stages but did not accelerate the sparse solver (SciPy CG), which remains outside the JIT boundary.

**Table 4.1 — Numba JIT CPU: stage-level impact**

| Solver Stage | Execution | Expected Impact | Why |
|---|---|---|---|
| Element stiffness computation | JIT + parallel | High | CPU cores + compiled loops |
| Global assembly (COO vals) | JIT + parallel | Moderate–High | Structured writes, reduced overhead |
| Post-processing (velocity/fields) | JIT + parallel | High | Element-independent computations |
| Sparse solve (CG, SciPy) | CPU library | None | Sparse ops not JIT-accelerated |

**Critical interpretation:** once assembly becomes fast, the runtime profile shifts so that **the sparse solve dominates**, especially as mesh size grows. This is precisely the point at which CPU-only acceleration saturates and motivates GPU migration.

### 4.5 Benchmark-relevant constraints (what must be stated in the report)
Numba introduces first-run compilation overhead. This explains the observed phenomenon that the “second run is much faster”.

**Table 4.2 — Why first-run ≠ steady-state (Numba JIT)**

| Effect | Impact | Required benchmark action |
|---|---|---|
| JIT compilation | Inflates run #1 | Exclude warm-up |
| Cached compilation (`cache=True`) | Fast subsequent runs | Report steady-state timing |
| Thread scaling depends on bandwidth | Diminishing returns | Record core count and threads |

---

## 5. Implementation 5 — Numba CUDA (GPU Offloading with Python Kernels)

### 5.1 Motivation: moving beyond CPU bandwidth limits
Implementation 5 moves the element loop to the GPU, targeting two limitations revealed in Implementation 4:
1) CPU core count is small compared to the parallelism available in FEM element loops,  
2) sparse operations and assembly patterns become increasingly bandwidth-dominated as mesh size increases.

The GPU offers:
- thousands of concurrent threads,
- far higher memory bandwidth,
- hardware scheduling designed for throughput.

### 5.2 Mapping strategy: one thread per element
The kernel design adopts **one GPU thread per element**, which is the natural mapping for independent element stiffness computations.

**Table 5.1 — Numba CUDA mapping and configuration**

| Component | Choice | Rationale |
|---|---|---|
| Parallel mapping | 1 thread / element | Independent work, simple scaling |
| Typical block size | 128 threads/block | Balance occupancy vs register pressure |
| Thread-private arrays | `cuda.local.array` | No synchronization inside element |
| Global outputs | COO values + global force | Assembly result storage |

This mapping yields predictable scaling and is easy to validate for correctness, which is critical in early GPU iterations.

### 5.3 Atomics: correctness vs performance
Global force vector assembly requires `cuda.atomic.add` because multiple elements contribute to shared nodes.

This is a key performance trade-off:
- atomics guarantee correctness,
- but they introduce contention and serialization where many threads update the same node.

**Critical note:** even if force contributions are small (or sometimes zero), the pattern itself is important to report because it explains scaling behavior and can be a bottleneck for other load cases.

### 5.4 Hybrid pipeline: what remains on CPU and why it matters
In Implementation 5, COO row/col indices were generated on CPU. This is a pragmatic choice for development, but it introduces:
- CPU–GPU synchronization,
- extra transfers,
- and pipeline fragmentation.

**Table 5.2 — Implementation 5: hybrid pipeline implications**

| Stage | Location | Consequence |
|---|---|---|
| Element Ke computation | GPU | Strong parallel speedup |
| Force accumulation | GPU (atomics) | Contention-dependent overhead |
| COO index generation | CPU | Sync point + transfer cost |
| Sparse solve | GPU (CuPy) | Bandwidth advantage |
| Boundary logic | often CPU | Additional sync (if present) |

### 5.5 Why “second run is faster” is expected here
GPU runs often become faster after the first execution due to:
- CUDA context initialization,
- kernel JIT compilation,
- memory allocator warm-up/caching.

This must be stated explicitly; otherwise, benchmark results may be misinterpreted.

---

## 6. Implementation 6 — CuPy RawKernel (CUDA C) + Fully GPU-Resident Pipeline

### 6.1 Motivation: eliminate the remaining overhead sources
Implementation 6 is the “performance endpoint”. It was introduced to address the two dominant inefficiencies remaining in Implementation 5:
1) hybrid CPU–GPU stages causing synchronization and transfers,  
2) limited kernel control and optimization depth in Python-defined kernels.

The guiding principle is: **once data enters the GPU, it stays there until the computation ends**.

### 6.2 RawKernel: why CUDA C matters
CuPy RawKernel compiles CUDA C kernels embedded as strings. This provides:
- full access to CUDA features (shared memory, warp primitives, fine-grained control),
- maximum performance potential,
- reduced overhead and fewer restrictions compared to Python kernels.

**Critical angle:** RawKernel is not “just faster”; it enables a different *engineering posture*: performance tuning becomes feasible at the memory and instruction level.

### 6.3 Architectural upgrade: COO indices generated on GPU
A major improvement is moving COO index generation to the GPU using vectorized CuPy operations. This avoids CPU-side loops and prevents CPU–GPU synchronization.

**Table 6.1 — The architectural delta that matters (Impl. 5 vs Impl. 6)**

| Feature | Numba CUDA (Impl. 5) | CuPy RawKernel (Impl. 6) | Why it matters |
|---|---|---|---|
| Kernel language | Python | CUDA C | control + peak perf |
| COO indices | CPU | GPU | removes sync + transfer |
| Pipeline residency | Hybrid | Fully GPU | amortizes overhead |
| Sparse ops | GPU | GPU | enables solver dominance regime |
| Tuning depth | limited | high | register/shared memory tuning |

### 6.4 Solver dominance: the real bottleneck is sparse linear algebra
For large meshes, the CG/GMRES stage dominates runtime. This is expected and must be framed correctly: once assembly is optimized, further assembly tuning has diminishing returns because the solver is bandwidth-bound and called repeatedly (SpMV per iteration).

**Table 6.2 — Expected runtime profile (large meshes)**

| Stage | Expected share | Interpretation |
|---|---:|---|
| Assembly kernel | 5–15% | highly parallel, amortized |
| Sparse build (COO→CSR) | 5–10% | structural overhead |
| Sparse solve | 60–80% | bandwidth-bound dominant cost |
| Post-processing | 5–10% | highly parallel |
| Transfers | minimal | only at boundaries |

This analysis gives the benchmark section technical credibility: performance is explained by algorithmic structure, not guesswork.

### 6.5 Robustness: CG fallback to GMRES
The solver includes a fallback mechanism (GMRES if CG fails). This is relevant because:
- it shows engineering robustness,
- it prevents “benchmark failures” from being confused with performance issues,
- it supports consistent experiments across GPUs.

---

## 7. Performance Characteristics

# 7. Cross-GPU Performance Comparison

This section presents a cross-GPU performance comparison aimed at evaluating how the different solver implementations scale across heterogeneous GPU architectures. While the previous sections focused on *algorithmic acceleration* and *implementation strategy* (Numba JIT CPU, Numba CUDA, and CuPy RawKernel), the purpose here is to isolate and analyse the **impact of hardware characteristics** on performance, scalability, and efficiency.

Rather than reporting results for a single device, this study adopts a comparative approach across multiple GPUs, enabling a clearer distinction between gains derived from software optimisation and gains attributable to GPU architecture. This is particularly relevant for GPU computing, where memory bandwidth, core count, and VRAM capacity can significantly influence observed speedups.

---

## 7.1 Objective of the Cross-GPU Analysis

The primary objectives of this comparative analysis are:

- To assess how solver performance scales across GPUs with different compute capabilities;
- To identify which solver stages benefit most from higher-end hardware;
- To evaluate whether performance gains scale proportionally with theoretical hardware improvements;
- To detect architectural bottlenecks (e.g. memory bandwidth, VRAM limits, kernel launch overheads);
- To validate the robustness and portability of the GPU implementations across devices.

This analysis complements the implementation-focused discussion by demonstrating that the developed GPU solvers are not tuned to a single device, but instead behave consistently across a range of architectures.

---

## 7.2 Evaluated GPU Architectures

The comparison is conducted using three NVIDIA GPUs representing distinct performance tiers:

- **NVIDIA GeForce RTX 5090** (high-end, latest-generation architecture),
- **NVIDIA GeForce RTX 4090** (high-end, previous-generation flagship),
- **NVIDIA GeForce RTX 4060 Ti** (mid-range consumer GPU).

These devices differ substantially in terms of:
- number of CUDA cores,
- memory bandwidth,
- available VRAM,
- cache hierarchy and SM configuration.

This diversity makes them well suited for analysing how solver efficiency and acceleration potential evolve from mid-range to flagship hardware.

---

## 7.3 Experimental Control and Reproducibility

To ensure a fair and meaningful comparison, all experiments follow a strictly controlled setup:

- The **same application framework** is used across all devices;
- The **same solver implementations** (CPU baseline, Numba JIT, Numba CUDA, CuPy RawKernel) are executed without modification;
- Identical **geometries**, **mesh resolutions**, and **boundary conditions** are applied;
- Solver parameters such as convergence tolerance, maximum iterations, and numerical schemes are kept constant;
- Each benchmark is executed in **steady-state mode**, excluding first-run overheads such as:
  - JIT compilation (Numba),
  - CUDA context initialisation,
  - kernel compilation (CuPy RawKernel).

By enforcing this controlled methodology, observed performance differences can be attributed exclusively to hardware characteristics rather than implementation or configuration variability.

---

## 7.4 Performance Metrics Collected

For each solver and GPU, the following metrics are collected and reported:

- **Total execution time**, measured in steady-state conditions;
- **Stage-level timing breakdown**, including:
  - assembly,
  - sparse matrix construction,
  - linear solve,
  - post-processing;
- **Solver iteration counts**, to confirm numerical equivalence across devices;
- **GPU memory usage**, including peak VRAM consumption where applicable;
- **Relative speedups**, computed with respect to the optimized CPU baseline (Numba JIT).

This combination of metrics allows both *quantitative* and *qualitative* performance analysis, linking execution time to architectural behaviour.

---

## 7.5 Results Organisation and Reporting Strategy

To avoid unnecessary fragmentation and to maintain clarity, results are structured as follows:

- **One table per GPU**, summarising solver performance across all tested geometries and mesh sizes;
- Each table includes all solver variants, enabling direct intra-GPU comparison;
- A **final cross-GPU summary table** presents normalized results for identical workloads, highlighting relative performance differences between GPUs.

This organisation strikes a balance between completeness and readability, ensuring that trends are visible without excessive table proliferation.

---

## 7.6 Scope and Interpretation Guidelines

It is important to note that:

- Performance scaling is not expected to be linear with theoretical peak FLOPS;
- Smaller meshes may show limited GPU advantage due to kernel launch and transfer overheads;
- Larger meshes increasingly expose memory bandwidth and VRAM limitations, particularly on mid-range GPUs;
- Differences in solver dominance (assembly vs. sparse solve) may shift with problem size and hardware.

These factors are explicitly considered in the analysis presented in the following subsections.


## 7.7 GPU Utilization Analysis

| Metric | Typical Value | Optimization Target |
|--------|---------------|---------------------|
| Occupancy | 50-75% | Register pressure |
| Memory throughput | 70-85% peak | Coalescing |
| Compute utilization | 60-80% | Algorithm efficiency |

### 7.7.1 Performance Breakdown

Expected time distribution for large problems:

| Stage | Time Fraction | Notes |
|-------|---------------|-------|
| Mesh loading | <5% | I/O bound |
| Assembly kernel | 5-15% | Highly parallel |
| Matrix construction | 5-10% | CuPy sparse ops |
| Linear solve | 60-80% | Memory bandwidth bound |
| Post-processing | 5-10% | Highly parallel |
| Data transfer | <5% | PCIe overhead |

### 7.7.2 Scaling Characteristics

| Problem Size | GPU Advantage | Notes |
|--------------|---------------|-------|
| <10K elements | Minimal | Transfer overhead dominates |
| 10K-100K | Significant (5-20×) | Good GPU utilization |
| 100K-1M | Maximum (20-100×) | Full GPU saturation |
| >1M | Memory limited | May require multi-GPU |

### 7.7.3 Comparison with CPU Implementations

| Aspect | CPU Baseline | Numba CPU | GPU CuPy |
|--------|--------------|-----------|----------|
| Assembly | O(N) sequential | O(N/P) parallel | O(N/T) massively parallel |
| Threads/cores | 1 | 4-32 | 1000s |
| Memory bandwidth | 50-100 GB/s | 50-100 GB/s | 500-900 GB/s |
| Latency | Low | Low | Higher (PCIe) |
| Throughput | Moderate | Good | Excellent |

---

**Table 7.8 — Per-GPU results table (template)**

| Geometry | Mesh | Nodes | Elements | Solver | Total (ms) | Assembly (ms) | Solve (ms) | Iter | Speedup vs Numba JIT |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|

**Table 7.9 — Cross-GPU normalized comparison (template)**

| Geometry | Mesh | Solver | 5090 Total | 4090 Total | 4060Ti Total | 4090/5090 | 4060Ti/5090 |
|---|---|---|---:|---:|---:|---:|---:|


---
