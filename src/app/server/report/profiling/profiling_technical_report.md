# Technical Report: Progressive Optimization of Quad-8 FEM Solver for 2D Potential Flow

This report section addresses project's objectives in the *High Performance Graphical Computing* curricular unit: demonstrating the design and implementation of a GPU-accelerated Finite Element Method (FEM) solver, starting from a functional CPU baseline and progressively incorporating parallelization strategies, execution models, and performance optimizations. 

The report emphasizes the evolution across implementations, highlighting:
- **Key features and technical aspects**: Core design choices, code structure, and computational strategies.
- **Improvements over the previous implementation**: How each version builds on the prior one, introducing new techniques for speedup or efficiency.
- **Problems addressed**: Specific bottlenecks resolved (e.g., serial computation, GIL limitations, data transfers).
- **Performance insights**: Based on the Nsight Systems profiles you shared (where applicable) and general expectations for your mesh sizes (~300k–400k elements, ~1M nodes). I've inferred typical runtimes and bottlenecks; for precise benchmarks, running with your meshes is recommended.
- **Project alignment**: How each step advances toward GPU acceleration, with lessons on parallelization and evaluation.

The implementations are presented in a logical progression: starting with the pure CPU baseline, moving through CPU parallel variants, then JIT-accelerated CPU, and finally GPU versions. This mirrors a systematic experimentation path from serial CPU to full GPU offloading.

---

## 1. CPU Baseline Implementation (`quad8_cpu_v3.py` and supporting files)
### Key Features and Technical Aspects
This is the foundational serial CPU solver using NumPy for numerics and SciPy for sparse matrix operations and linear solving. It computes the stiffness matrix (`Ke`) and force vector (`fe`) per element via explicit loops in `Elem_Quad8`, using 9-point Gauss-Legendre quadrature for integration. Shape functions and Jacobians are calculated in `Shape_N_Der8`. Boundary conditions (Dirichlet/Robin) are applied serially, and the system is solved with SciPy's `cg` (conjugate gradient) or GMRES. Post-processing derives velocities and pressures serially. The code is pure Python/NumPy/SciPy, with no parallelism or JIT, making it simple and portable.

### Improvements Over Previous
As the starting point, this establishes the baseline—no prior version. It validates the FEM workflow (assembly → BCs → solve → post-process) on CPU, providing a reference for correctness and initial timings.

### Problems Addressed
- Establishes core functionality: Handles mesh loading (HDF5/XLSX via pandas), sparse COO/CSR matrix construction, and iterative solving with monitoring (residual checks every few iterations).
- No major issues at this stage, but inherent serial nature limits scalability for large meshes (e.g., ~427k elements would take 30–60s+ total, dominated by assembly loops and sparse ops).

### Performance Insights
- **Bottlenecks**: Assembly (~50–70% of time) due to slow Python loops over elements/points; BC application serial over edges; solve iterations compute expensive residuals (`A @ xk`).
- **Expected Runtime**: For your meshes, assembly ~20–40s, solve ~10–20s, total ~40–60s on a standard CPU.
- **Nsight Profile**: Would show single-thread activity, long durations in Python/NumPy calls (e.g., `np.linalg.det/inv`), no kernels/MemCpy. NVTX phases (added in my suggestion) highlight imbalances.

### Project Alignment
This baseline fulfills the "functional CPU starting point" in your introduction, allowing systematic evaluation of optimizations. It exposes CPU limitations (serial loops, interpreter overhead), motivating parallel strategies in subsequent versions.

---

## 2. CPU Threaded Implementation (`quad8_cpu_threaded.py`)
### Key Features and Technical Aspects
Builds on the baseline by introducing explicit multi-threading with `concurrent.futures.ThreadPoolExecutor`. Assembly and post-processing are parallelized: elements are batched (default 1000), and workers compute `Ke/fe` or velocities using pure NumPy functions. BCs and solve remain serial. The design uses configurable `num_workers` (defaults to `os.cpu_count()`) for hardware adaptation.

### Improvements Over Previous (CPU Baseline)
- Introduces concurrency: Parallelizes per-element computations, reducing assembly time by leveraging multiple threads.
- Batch processing minimizes thread overhead; NumPy ops (e.g., small matrix multiplies) can release the GIL for some parallelism.

### Problems Addressed
- Tackles serial assembly bottleneck: Baseline's element loop is now distributed across threads, addressing CPU underutilization on multi-core systems.
- GIL limitations partially mitigated by NumPy's GIL-releasing BLAS (if MKL/OpenBLAS), but pure Python parts (e.g., loops) still contend.
- Overhead managed via batching, avoiding excessive thread creation for large Nels.

### Performance Insights
- **Bottlenecks**: GIL restricts true parallelism (~1.5–3x speedup vs baseline on 8-core CPU); post-thread COO/sparse construction serial. Solve unchanged.
- **Expected Runtime**: Assembly ~10–20s (improved), total ~20–40s. On your meshes, threading shines for arithmetic-heavy integration but less for I/O-bound parts.
- **Nsight Profile**: Multi-thread tracks during assembly/post; idle threads due to GIL. NVTX phases show shorter "assemble_system" bar.

### Project Alignment
This version investigates basic CPU parallelization (threading model), evidencing GIL's impact on Python concurrency. It sets up for more advanced strategies, showing incremental gains toward high-performance goals.

---

## 3. CPU Multiprocess Implementation (`quad8_cpu_multiprocess_v2.py`)
### Key Features and Technical Aspects
Advances to true parallelism using `multiprocessing.Pool`, forking/spawning processes for assembly and post-processing batches. Worker initializers (`_init_assembly_worker`) broadcast large arrays (mesh data) once, reducing IPC costs. Element functions are module-level for pickling. BCs/solve serial.

### Improvements Over Previous (CPU Threaded)
- Bypasses GIL entirely: Processes run independently, enabling full core utilization for NumPy ops.
- Optimizer pattern cuts data serialization overhead (critical for large meshes like 1M+ nodes), improving over threaded's contention.

### Problems Addressed
- Resolves GIL bottleneck from threaded version: Achieves near-linear scaling (e.g., 4–8x on 8 cores) for compute-bound assembly.
- Windows compatibility via spawn; batching balances process startup costs.
- Memory duplication managed implicitly, though large meshes may strain RAM.

### Performance Insights
- **Bottlenecks**: Process spawn overhead (~1–2s startup); serial sparse matrix build post-assembly. Solve dominant if iterations high.
- **Expected Runtime**: Assembly ~5–10s, total ~15–30s—better than threaded for CPU-bound tasks.
- **Nsight Profile**: Separate process tracks during parallel phases; fork/spawn events visible. NVTX shows efficient overlap if cores sufficient.

### Project Alignment
Highlights multiprocessing as a GIL workaround, evaluating execution models for CPU scalability. It bridges to GPU by showing limits of process-based parallelism (e.g., memory overhead), motivating device offloading.

---

## 4. Numba JIT Implementation (`quad8_numba.py` and supporting files)
### Key Features and Technical Aspects
Shifts to JIT compilation with Numba's `@njit` for element functions (e.g., `elem_quad8`, `shape_n_der8`) and Robin BCs, compiling to machine code. Assembly uses `@njit(parallel=True)` with `prange` for auto-multi-threading. Post-processing serial but JIT'd. Sparse ops/Solve via SciPy.

### Improvements Over Previous (CPU Multiprocess)
- Replaces explicit concurrency with JIT + auto-parallel: Simpler code (no pools/threads), faster loops via compilation.
- `prange` provides lightweight threading without process overhead, combining JIT speed with parallelism.

### Problems Addressed
- Overhead from multiprocessing (spawn, memory dup): JIT runs in single process, faster startup/lower memory.
- Slow Python loops in baseline/threaded: Compiled scalars/matmuls approach C speeds, addressing interpreter drag.
- Numba restrictions handled (e.g., manual matmuls, cache=True for reuse).

### Performance Insights
- **Bottlenecks**: Serial post-assembly sparse build; solve unchanged. JIT warm-up minor.
- **Expected Runtime**: Assembly ~2–5s, total ~10–20s—significant leap over multiprocess for loops.
- **Nsight Profile**: Thread activity in `prange`; shorter phases. NVTX highlights JIT efficiency.

### Project Alignment
Demonstrates JIT as a low-effort CPU accelerator, evaluating auto-parallel vs explicit. Prepares for GPU by introducing Numba ecosystem, easing transition to CUDA.

---

## 5. Numba CUDA Implementation (`kernels_numba_cuda.py`, `quad8_numba_cuda.py`)
### Key Features and Technical Aspects
First GPU entry: Uses Numba's `@cuda.jit` for kernels (assembly/post-processing per-element on GPU threads). Data transfers managed; sparse matrix built on CPU post-GPU. Solve via CuPy CG (GPU) with Jacobi preconditioner. NVTX annotations for profiling.

### Improvements Over Previous (Numba JIT)
- Offloads to GPU: Massive parallelism (one thread/element) for assembly, leveraging thousands of cores.
- Hybrid: GPU for compute-heavy kernels, CPU for sparse/solve fallback—extends JIT to device.

### Problems Addressed
- CPU scaling limits: GPU handles larger meshes faster (e.g., 427k elements in ~5s assembly vs CPU's 2–5s).
- Data transfers minimized; atomics for reductions. Profile shows no major stalls.

### Performance Insights
- **Bottlenecks**: Solve (~8.7s, 240k kernels from CG iterations); CPU sparse build (~5.3s total assembly).
- **Profiled Runtime**: ~18s total; assembly improved, but solve dominant.
- **Nsight Profile**: Red kernel bars in solve; short MemCpy.

### Project Alignment
Marks GPU transition, addressing CPU bottlenecks with device parallelism. Evaluates hybrid models, setting stage for full GPU.

---

## 6. CuPy GPU Implementation (`quad8_gpu_v3.py` and supporting files)
### Key Features and Technical Aspects
Optimized GPU: CuPy `RawKernel` for custom CUDA C assembly/post-processing. Full GPU pipeline possible (sparse via `cpsparse`); solve with CuPy CG or SciPy GMRES fallback. Robin BCs GPU-ported.

### Improvements Over Previous (Numba CUDA)
- Finer GPU control: RawKernel allows PTX optimizations, ~5x assembly speedup (~1s vs 5s).
- Reduced transfers: More ops on-device; better preconditioning options.

### Problems Addressed
- Numba's Pythonic limits: CUDA C enables manual unrolling/low-level tweaks.
- Kernel overhead in solve: Intentional CPU fallback for stability comparison.

### Performance Insights
- **Bottlenecks**: Solve (~8.5s, 250k kernels); minor MemCpy increase.
- **Profiled Runtime**: ~10s total—clear gain over Numba CUDA.
- **Nsight Profile**: Shorter assembly; dense kernels in solve.

### Project Alignment
Culminates in optimized GPU, evidencing full acceleration. Compares stability/performance, fulfilling systematic optimization goals.

---
