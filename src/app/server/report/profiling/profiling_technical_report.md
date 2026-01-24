![](images/documents/cover/cover_page_web.png)

# Progressive Optimization Through Profiling

This section documents the systematic optimization journey of the GPU-accelerated Finite Element Method (FEM) solver, progressing from a serial CPU baseline through parallel CPU variants to fully optimized GPU implementations. Each section presents the implementation's technical approach, the profiling data that revealed its bottlenecks, and the insights that guided subsequent optimization efforts.

The solver was validated across a comprehensive test matrix of 144 configurations: six mesh geometries (Y-Shaped channel, Venturi tube, S-Bend, T-Junction, Backward-Facing Step, and 90° Elbow), each at four refinement levels ranging from approximately 200 to 1.3 million nodes, executed on all six solver implementations.

We present a detailed analysis for the Y-Shaped channel geometry, selected as representative of the overall performance patterns observed across all configurations. Results focus on the smallest mesh (201 nodes, 52 elements) and largest mesh (1,357,953 nodes, 338,544 elements) to clearly illustrate the contrast between fixed-overhead behavior and production-scale performance; intermediate mesh sizes confirmed smooth scaling transitions consistent with these boundary cases.

All profiling data was collected using NVIDIA Nsight Systems with NVTX annotations, enabling precise phase-level timing analysis. The hardware utilized for profiling tests was a Intel i9-12900K CPU and a NVidia RTX 4090 GPU.

## 1. CPU Baseline Implementation

### Technical Approach

The baseline implementation (`quad8_cpu_v4.py`) establishes a serial reference using NumPy for numerics and SciPy for sparse matrix operations. The solver follows the standard FEM workflow: mesh loading, stiffness matrix assembly, boundary condition application, iterative solving, and post-processing.

**Assembly** iterates sequentially over all elements:

```python
for e in range(Nels):
    Ke, fe = Elem_Quad8(coords, connect[e], k_iso)
    # Accumulate to COO format
    data_K.extend(Ke.flatten())
    row_K.extend(...)
    col_K.extend(...)
```

Each element computation (`Elem_Quad8`) performs 9-point Gauss-Legendre quadrature, computing shape functions via `Shape_N_Der8` and using `np.linalg.det` and `np.linalg.inv` for Jacobian operations at each integration point.

**Sparse matrix construction** uses SciPy's COO format with subsequent conversion to CSR for efficient row-slicing during boundary condition application. The **solver** employs SciPy's conjugate gradient (`scipy.sparse.linalg.cg`) with an optional Jacobi preconditioner, monitoring convergence via residual callbacks every 10 iterations.

### Profiling Results

| Mesh | Total | Assembly | Solve | Assembly % | Solve % |
|------|-------|----------|-------|------------|---------|
| y_tube_201 | 141.5 ms | ~13 ms | ~7 ms | 9% | 5% |
| y_tube_1_3m | 506.3 s | 85.7 s | 395.9 s | 17% | 78% |

![](images/documents/profiling_report/y_tube_201_cpu.png)

***Figure 1a**: CPU Baseline timeline for y_tube_201 (141.5 ms total) — mesh loading dominates at small scale.*

![](images/documents/profiling_report/y_tube_1_3m_cpu.png)

***Figure 1b**: CPU Baseline timeline for y_tube_1_3m (506.3s total) showing assembly (85.7s) and solve (395.9s) phases.*

### Analysis

The profiling reveals two distinct bottleneck patterns depending on mesh scale. At small scale, mesh loading dominates (121.9 ms, 86% of runtime), while assembly and solve are negligible. At production scale, the solve phase consumes 78% of total runtime, with assembly accounting for 17%.

The serial assembly loop suffers from Python interpreter overhead on each of the 338,544 iterations. Within each iteration, `np.linalg.det` and `np.linalg.inv` incur function call overhead for small 2x2 matrices where explicit formulas would be faster. The list `.extend()` operations for COO accumulation cause repeated memory reallocations.

These observations establish the optimization targets: parallelize assembly to address the per-element overhead, and ultimately accelerate the solve phase which dominates at scale.

## 2. CPU Threaded Implementation

### Technical Approach

The threaded implementation (`quad8_cpu_threaded_v2.py`) introduces parallelism via Python's `concurrent.futures.ThreadPoolExecutor`. Elements are processed in batches of 1000 to amortize thread management overhead.

```python
with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    futures = [executor.submit(_process_element_batch_threaded, batch_start, batch_end, ...)
               for batch_start, batch_end in batches]
    for future in as_completed(futures):
        data, rows, cols = future.result()
        data_K.extend(data)  # Serial aggregation
```

Each worker executes the same `Elem_Quad8` logic as the baseline, but multiple batches execute concurrently. Post-processing (velocity and pressure computation) is similarly parallelized.

### Profiling Results

| Mesh | Total | Assembly | Solve | vs Baseline |
|------|-------|----------|-------|-------------|
| y_tube_201 | 165.9 ms | ~9 ms | ~7 ms | 0.85x (slower) |
| y_tube_1_3m | 447.3 s | 37.4 s | 381.7 s | 1.13x |

![](images/documents/profiling_report/y_tube_201_cpu_threaded.png)

***Figure 2a**: CPU Threaded timeline for y_tube_201 (165.9 ms total) — threading overhead exceeds gains at small scale.*

![](images/documents/profiling_report/y_tube_1_3m_cpu_threaded.png)

***Figure 2b**: CPU Threaded timeline for y_tube_1_3m (447.3s total) showing reduced assembly phase (37.4s) but similar solve duration (381.7s).*

### Analysis

At production scale, assembly time improved from 85.7s to 37.4s (2.3x speedup), demonstrating that the element computations benefit from concurrent execution. However, total speedup is only 1.13x because the solve phase (381.7s) remains essentially unchanged.

The threading gains are limited by Python's Global Interpreter Lock (GIL). While NumPy's BLAS operations release the GIL during matrix computations, the surrounding Python code (loop control, function calls, list operations) still contends for the lock. The serial aggregation step (`data_K.extend()`) after each batch completion further limits scalability.

At small scale, threading actually degrades performance (165.9 ms vs 141.5 ms). The overhead of creating the thread pool, dispatching work, and collecting results exceeds the time saved on the trivial 52-element workload.

The key insight: threading provides modest assembly improvements but cannot address the dominant solve bottleneck, and introduces overhead that hurts small-scale performance.

## 3. CPU Multiprocess Implementation

### Technical Approach

The multiprocess implementation (`quad8_cpu_multiprocess_v3.py`) bypasses the GIL entirely by using `multiprocessing.Pool`. Each worker process has its own Python interpreter and memory space, enabling true parallel execution.

To minimize inter-process communication (IPC) overhead, large arrays are broadcast once during worker initialization rather than serialized per batch:

```python
def _init_assembly_worker(coords_, connect_, ...):
    global _w_coords, _w_connect, ...
    _w_coords = coords_
    _w_connect = connect_

with mp.Pool(processes=self.num_workers, 
             initializer=_init_assembly_worker, 
             initargs=(...)) as pool:
    results = pool.map(_process_element_batch_mp, batch_ranges)
```

### Profiling Results

| Mesh | Total | Assembly | Solve | vs Baseline | vs Threaded |
|------|-------|----------|-------|-------------|-------------|
| y_tube_201 | 262.2 ms | 53.2 ms | ~5 ms | 0.54x (slower) | 0.63x (slower) |
| y_tube_1_3m | 427.1 s | 43.1 s | 378.8 s | 1.19x | 1.05x |

![](images/documents/profiling_report/y_tube_201_cpu_multiprocess.png)

***Figure 3a**: CPU Multiprocess timeline for y_tube_201 (262.2 ms total) — process spawn overhead dominates at small scale.*

![](images/documents/profiling_report/y_tube_1_3m_cpu_multiprocess.png)

***Figure 3b**: CPU Multiprocess timeline for y_tube_1_3m (427.1s total) showing assembly (43.1s) and solve (378.8s) phases with process coordination overhead.*

### Analysis

Contrary to expectations, multiprocessing performed worse than threading for assembly at production scale (43.1s vs 37.4s). This counterintuitive result stems from the overhead costs of process-based parallelism:

1. **Process spawn overhead**: Creating worker processes costs 1-2 seconds, compared to near-instantaneous thread creation.
2. **IPC serialization**: Despite the initializer optimization, results must still be pickled and transmitted back to the main process after each batch.
3. **Memory duplication**: Each process maintains its own copy of working data, increasing memory pressure.

At small scale, the overhead is catastrophic: 262.2 ms versus 141.5 ms baseline, an 85% slowdown. The process pool creation alone exceeds the entire baseline runtime.

The marginal total improvement (1.19x) comes primarily from slightly better solve performance (378.8s vs 395.9s), likely due to reduced memory contention after processes terminate.

Additional practical considerations include Windows compatibility — the implementation uses `spawn` rather than `fork` to work across platforms, which incurs higher startup costs but ensures portability. Memory pressure is also a concern: each worker process maintains its own copy of mesh data, which can strain RAM for very large meshes approaching system memory limits.

This implementation demonstrates that bypassing the GIL is not sufficient for performance gains; the communication and coordination costs of process-based parallelism can outweigh the benefits of true concurrency, particularly when the workload involves returning substantial data from workers.

## 4. Numba JIT Implementation

### Technical Approach

The Numba JIT implementation (`quad8_numba_v2.py`) takes a fundamentally different approach: rather than parallelizing Python code, it compiles the computational kernels to machine code using Numba's `@njit` decorator.

**Shape function computation** uses explicit scalar operations that Numba compiles efficiently:

```python
@njit(cache=True)
def shape_n_der8(xi, eta):
    N = np.empty(8)
    dNdxi = np.empty(8)
    dNdeta = np.empty(8)
    # Explicit per-node formulas - no NumPy broadcasting overhead
    N[0] = (xi - 1) * (eta + xi + 1) * (1 - eta) / 4
    # ...
```

**Jacobian operations** are implemented as explicit 2x2 formulas rather than library calls:

```python
det_J = J[0,0]*J[1,1] - J[0,1]*J[1,0]
inv_J[0,0] = J[1,1] / det_J
inv_J[0,1] = -J[0,1] / det_J
# ...
```

**Assembly** uses `prange` for automatic parallelization with pre-allocated output arrays:

```python
@njit(parallel=True, cache=True)
def assemble_system_numba(coords, connect, ...):
    all_data = np.empty(Nels * 64)  # Pre-allocated, no list growth
    all_rows = np.empty(Nels * 64, dtype=np.int64)
    
    for e in prange(Nels):  # Parallel iteration
        # Each thread writes to deterministic slice [e*64 : (e+1)*64]
```

### Profiling Results

| Mesh | Total | Assembly | Solve | vs Baseline |
|------|-------|----------|-------|-------------|
| y_tube_201 | 219.4 ms | 86.8 ms | ~5 ms | 0.65x (slower) |
| y_tube_1_3m | 390.3 s | ~4.6 s | 385.6 s | 1.30x |

![](images/documents/profiling_report/y_tube_201_numba.png)

***Figure 4a**: Numba JIT timeline for y_tube_201 (219.4 ms total) — JIT compilation overhead visible on first run.*

![](images/documents/profiling_report/y_tube_1_3m_numba.png)

***Figure 4b**: Numba JIT timeline for y_tube_1_3m (390.3s total) showing dramatically reduced assembly (~4.6s) but unchanged solve phase (385.6s).*

### Analysis

At production scale, Numba JIT achieves a dramatic 18.6x speedup for assembly (85.7s → 4.6s), matching the assembly performance of the GPU implementations while remaining entirely on CPU. This demonstrates the power of JIT compilation combined with `prange` auto-parallelization:

1. **Eliminated interpreter overhead**: The compiled code runs at native speed without per-iteration Python overhead.
2. **Efficient memory access**: Pre-allocated contiguous arrays with deterministic indexing enable cache-friendly access patterns.
3. **No IPC overhead**: Unlike multiprocessing, `prange` threads share memory without serialization.
4. **Optimized small operations**: Explicit 2x2 matrix formulas compile to a few machine instructions, versus function call overhead for `np.linalg.inv`.

However, total speedup is only 1.30x because the solve phase (385.6s) remains unchanged. SciPy's iterative solver cannot be JIT-compiled, and it already uses optimized BLAS internally. The solve now accounts for 99% of runtime.

At small scale, the 219.4 ms runtime reflects JIT compilation overhead on first execution — Numba must compile each `@njit` function to machine code before it can run. The `cache=True` directive mitigates this for subsequent runs by persisting the compiled code to disk. On repeated executions with warm cache, the small-mesh overhead drops significantly, making Numba JIT competitive with the baseline even at small scale. This first-run versus subsequent-run distinction is important for interactive applications where startup latency matters.

The key insight from Numba JIT: CPU assembly can be optimized to match GPU assembly performance, but the solve phase becomes an insurmountable bottleneck. Further improvements require GPU-accelerated solving.

## 5. Numba CUDA Implementation

### Technical Approach

The Numba CUDA implementation (`quad8_numba_cuda.py` with `kernels_numba_cuda.py`) marks the transition to GPU computing. CUDA kernels are written in Python syntax using Numba's `@cuda.jit` decorator, with one thread processing one element.

**Kernel structure** uses local memory for per-element working arrays:

```python
@cuda.jit
def quad8_assembly_kernel(x, y, quad8, xp, wp, vals_out, fg_out):
    e = cuda.grid(1)
    if e >= quad8.shape[0]:
        return
    
    # Per-thread local arrays (registers/local memory)
    edofs = cuda.local.array(8, dtype=np.int32)
    Ke = cuda.local.array((8, 8), dtype=np.float64)
    # ... computation identical to CPU but parallelized across GPU threads
    
    # Output: deterministic indexing, no race conditions
    base_idx = e * 64
    for i in range(8):
        for j in range(8):
            vals_out[base_idx + k] = Ke[i, j]
    
    # Atomic update for force vector (nodes shared between elements)
    for i in range(8):
        cuda.atomic.add(fg_out, edofs[i], fe[i])
```

**Sparse matrix construction** occurs on CPU after transferring values back from GPU:

```python
vals_host = d_vals.copy_to_host()  # Device-to-host transfer
K = sp.coo_matrix((vals_host, (rows, cols)), shape=(Nnds, Nnds)).tocsr()
```

**Solve** uses CuPy's GPU-accelerated conjugate gradient with Jacobi preconditioning.

### Profiling Results

| Mesh | Total | Assembly | Apply BC | Solve | Kernels | MemCpy |
|------|-------|----------|----------|-------|---------|--------|
| y_tube_201 | 1.18 s | 399.4 ms | 123.9 ms | ~200 ms | 2,135 | 146 |
| y_tube_1_3m | 14.94 s | 4.59 s | 3.47 s | 5.92 s | 179,115 | 10,097 |

![](images/documents/profiling_report/y_tube_201_numba_cuda.png)

***Figure 5a**: Numba CUDA timeline for y_tube_201 (1.18s total) — GPU initialization and kernel launch overhead dominate at small scale.*

![](images/documents/profiling_report/y_tube_1_3m_numba_cuda.png)

***Figure 5b**: Numba CUDA timeline for y_tube_1_3m (14.94s total) showing assembly (4.59s), BC application (3.47s), and solve (5.92s) phases with visible kernel and memory transfer activity.*

### Analysis

The Numba CUDA implementation achieves 33.9x total speedup over the CPU baseline at production scale (506.3s → 14.94s). The GPU-accelerated solve phase is the breakthrough: 395.9s → 5.92s (66.9x speedup).

Profiling reveals the phase breakdown:

- **Assembly (4.59s, 31%)**: Similar to Numba JIT, as both use compiled code with parallel execution. The GPU's massive thread count doesn't provide additional benefit here because Numba JIT already saturates CPU cores.
- **BC Application (3.47s, 23%)**: Robin boundary conditions involve edge integration loops that aren't fully optimized for GPU execution. The profile shows this as a significant new bottleneck.
- **Solve (5.92s, 40%)**: The CuPy CG solver runs entirely on GPU, eliminating the CPU solve bottleneck.

The memory transfer pattern shows substantial device-to-host activity during solve (the large green D2H bar), indicating data moving back to CPU repeatedly during iterations.

At small scale, the GPU implementation is dramatically slower (1.18s vs 141.5ms baseline, 8.3x slowdown). The 449.5 ms mesh loading phase and 399.4 ms assembly phase include CUDA context initialization, kernel compilation, and memory allocation overhead that cannot be amortized over the trivial 52-element workload.

Key observations: GPU acceleration finally breaks through the solve bottleneck, but BC application emerges as a new optimization target, and fixed GPU overhead makes this unsuitable for small meshes.

## 6. CuPy GPU Implementation

### Technical Approach

The CuPy implementation (`quad8_gpu_v3.py`) optimizes the GPU pipeline using CuPy's `RawKernel` for hand-written CUDA C kernels, providing finer control than Numba's Python-to-PTX compilation.

**Assembly kernel** in CUDA C with explicit optimizations:

```c
extern "C" __global__
void assembly_kernel(const double* __restrict__ x,
                     const double* __restrict__ y,
                     const int* __restrict__ quad8,
                     /* ... */) {
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= Nels) return;
    
    double Ke[64];  // 8x8 in registers
    // ... computation with potential compiler unrolling
}
```

The `__restrict__` qualifier informs the compiler that pointers don't alias, enabling aggressive optimization.

**Robin BC kernel** is now fully GPU-resident:

```c
extern "C" __global__
void robin_bc_kernel(const double* x, const double* y,
                     const int* robin_edges, /* ... */) {
    int edge_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Full edge integration on GPU - no CPU fallback
}
```

**Sparse matrix construction** stays on GPU using CuPy's sparse module:

```python
rows_gpu = cp.array(rows)
cols_gpu = cp.array(cols)
K_gpu = cupyx.scipy.sparse.coo_matrix((vals_gpu, (rows_gpu, cols_gpu)))
K_csr = K_gpu.tocsr()  # Conversion on GPU
```

### Profiling Results

| Mesh | Total | Assembly | Apply BC | Solve | Kernels | MemCpy |
|------|-------|----------|----------|-------|---------|--------|
| y_tube_201 | 658.6 ms | ~50 ms | 133.7 ms | ~70 ms | 3,361 | 240 |
| y_tube_1_3m | 7.55 s | 0.58 s | 1.23 s | 5.68 s | 222,066 | 14,547 |

![](images/documents/profiling_report/y_tube_201_gpu.png)

***Figure 6a**: CuPy GPU timeline for y_tube_201 (658.6 ms total) — GPU overhead still significant but reduced compared to Numba CUDA.*

![](images/documents/profiling_report/y_tube_1_3m_gpu.png)

***Figure 6b**: CuPy GPU timeline for y_tube_1_3m (7.55s total) showing optimized assembly (~0.58s), BC application (1.23s), and solve (5.68s) phases.*

### Analysis

The CuPy implementation achieves 67.1x total speedup over the CPU baseline (506.3s → 7.55s) and 2.0x improvement over Numba CUDA (14.94s → 7.55s).

Phase-by-phase comparison with Numba CUDA:

| Phase | Numba CUDA | CuPy GPU | Improvement |
|-------|------------|----------|-------------|
| Assembly | 4.59 s | 0.58 s | 7.9x |
| Apply BC | 3.47 s | 1.23 s | 2.8x |
| Solve | 5.92 s | 5.68 s | 1.04x |

**Assembly optimization (7.9x)**: The RawKernel approach provides multiple advantages over Numba CUDA:
- CUDA C compiler (`nvcc`) applies more aggressive optimizations than Numba's PTX generation
- `__restrict__` qualifiers enable better memory access optimization
- Sparse matrix construction remains on GPU, eliminating the D2H transfer for values that Numba CUDA required

**BC optimization (2.8x)**: The fully GPU-resident Robin BC kernel eliminates CPU-GPU synchronization points that existed in the Numba CUDA hybrid approach.

**Solve (1.04x)**: Both implementations use CuPy's CG solver, so performance is nearly identical. The solve phase now accounts for 75% of total runtime, indicating it has become the new bottleneck.

At small scale, CuPy (658.6 ms) is faster than Numba CUDA (1.18 s) but still 4.7x slower than the CPU baseline. The reduced overhead comes from CuPy's pre-compiled kernels versus Numba's JIT compilation, but GPU context initialization still dominates.

## 7. Conclusions

### Performance Summary

The optimization journey achieved a 67x speedup from CPU baseline to the final CuPy GPU implementation at production scale:

| Solver | Assembly | Solve | Total | vs Baseline |
|--------|----------|-------|-------|-------------|
| CPU Baseline | 85.7 s | 395.9 s | 506.3 s | 1.0x |
| CPU Threaded | 37.4 s | 381.7 s | 447.3 s | 1.1x |
| CPU Multiprocess | 43.1 s | 378.8 s | 427.1 s | 1.2x |
| Numba JIT | 4.6 s | 385.6 s | 390.3 s | 1.3x |
| Numba CUDA | 4.59 s + 3.47 s BC | 5.92 s | 14.94 s | 33.9x |
| CuPy GPU | 0.58 s + 1.23 s BC | 5.68 s | 7.55 s | 67.1x |

### Key Insights

**Profiling-driven optimization is essential.** Each implementation's bottlenecks were revealed through NVTX-annotated profiling:
- CPU baseline showed assembly and solve as co-dominant bottlenecks
- CPU parallel variants revealed that solve cannot be accelerated through CPU parallelism
- Numba JIT proved assembly could match GPU speed on CPU, isolating solve as the true barrier
- GPU implementations shifted bottlenecks to BC application, then to solve again

**CPU parallelization has fundamental limits.** Threading achieved only 2.3x assembly speedup due to GIL contention. Multiprocessing, despite bypassing the GIL, performed worse due to IPC overhead. The solve phase, which uses SciPy's already-optimized BLAS, showed negligible improvement across all CPU variants.

**JIT compilation is highly effective for CPU-bound kernels.** Numba JIT achieved 18.6x assembly speedup through compilation alone, demonstrating that interpreter overhead, not algorithmic complexity, was the CPU assembly bottleneck.

**GPU acceleration requires workload-appropriate scale.** At 201 nodes, the CPU baseline (141.5 ms) outperformed all GPU implementations (658.6 ms best case) by 4.7x. GPU overhead (context initialization, kernel launch latency, memory transfers) only amortizes at scale.

**Memory transfer optimization is critical for GPU performance.** The 7.9x assembly improvement from Numba CUDA to CuPy came largely from keeping sparse matrix construction on GPU, eliminating a device-to-host transfer.

### Scaling Behavior

The ratio of large-mesh to small-mesh runtime reveals each solver's scaling efficiency:

| Solver | Small (201) | Large (1.36M) | Ratio |
|--------|-------------|---------------|-------|
| CPU Baseline | 141.5 ms | 506.3 s | 3,578x |
| Numba JIT | 219.4 ms | 390.3 s | 1,779x |
| Numba CUDA | 1.18 s | 14.94 s | 12.7x |
| CuPy GPU | 658.6 ms | 7.55 s | 11.5x |

GPU solvers exhibit dramatically better scaling: a 6,750x increase in problem size (201 → 1.36M nodes) results in only 11-13x runtime increase. This sub-linear scaling reflects the GPU's massive parallelism absorbing the increased workload.

### Limitations

**Single GPU constraint.** The current implementation targets a single GPU. Meshes exceeding GPU memory would require either out-of-core methods or multi-GPU distribution.

**Solver algorithm.** The conjugate gradient solver with Jacobi preconditioning, while effective, is not optimal for all problem types. Ill-conditioned systems may require algebraic multigrid or incomplete factorization preconditioners.

**Profiling overhead.** NVTX annotations and Nsight Systems instrumentation introduce minor overhead (~1-3%) that affects absolute timings, though relative comparisons remain valid.

### Future Work

**Solver optimization.** The solve phase now dominates at 75% of runtime. Potential improvements include:
- Mixed-precision solving (FP32 for iterations, FP64 for final refinement)
- Batched small-matrix operations for preconditioner application
- Alternative preconditioners (algebraic multigrid, sparse approximate inverse)

**Adaptive solver selection.** Implement automatic selection between CPU baseline (small meshes) and GPU solvers (large meshes) based on problem size, with the crossover point determined empirically around 10,000-50,000 elements.

**Memory optimization.** Explore unified memory (CUDA managed memory) to simplify the programming model and potentially improve performance for meshes near GPU memory limits.

**Multi-GPU scaling.** For very large meshes, domain decomposition with multi-GPU execution — potentially leveraging Dask for distributed computation — would extend the solver's capability beyond single-GPU memory constraints.
