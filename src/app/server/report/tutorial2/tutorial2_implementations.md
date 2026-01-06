# Implementations

## Execution Models

This section presents multiple implementations of the same Finite Element Method (FEM) problem using different execution models on CPU and GPU. All implementations share an identical numerical formulation, discretization, boundary conditions, and solver configuration; observed differences arise exclusively from the execution strategy and computational backend.

The implementations cover sequential CPU execution, shared-memory and process-based CPU parallelism, just-in-time compiled CPU execution using Numba, and GPU-based execution using Numba CUDA and CuPy with custom raw kernels. Together, these approaches span execution models ranging from interpreter-driven execution to compiled and accelerator-based computation.

Numerical equivalence is preserved across all implementations, enabling direct and fair comparison of execution behavior, performance, and scalability under consistent numerical conditions.

---

# Implementation 1: CPU Baseline

## 1. Overview
The CPU baseline implementation serves as the reference against which all other CPU and GPU implementations are evaluated. It prioritizes correctness, algorithmic clarity, and reproducibility over performance, establishing both the functional specification and the performance floor for the project.

| Attribute | Description |
|---------|-------------|
| Technology | Python (NumPy, SciPy) |
| Execution Model | Sequential, single-process |
| Role | Correctness reference and performance baseline |
| Dependencies | NumPy, SciPy, pandas, h5py |

---

## 2. Technology Background

### 2.1 NumPy and SciPy Ecosystem
The baseline implementation is built on Python’s scientific computing ecosystem:

- **NumPy** provides N-dimensional arrays and vectorized operations backed by optimized BLAS/LAPACK libraries.  
- **SciPy** supplies sparse matrix data structures and iterative solvers for large linear systems.  
- **h5py and pandas** support efficient binary input/output for mesh and result data.  

This stack enables concise algorithm expression while delegating computationally intensive kernels to compiled numerical libraries.

### 2.2 Execution Characteristics
Execution is performed within the CPython interpreter and is therefore subject to the Global Interpreter Lock (GIL). While NumPy and SciPy release the GIL during computational kernels, Python-level control flow remains serialized.

For FEM workloads, this results in a mixed execution model:

- **Element loops** execute sequentially at the Python level with the GIL held.  
- **Dense linear algebra operations** are executed in optimized BLAS/LAPACK routines with the GIL released.  
- **Sparse iterative solvers** execute predominantly in compiled SciPy code, also releasing the GIL during major operations.  

### 2.3 Relevance for FEM
The sequential CPU baseline fulfills several essential roles in the FEM workflow:

- Provides a clear and traceable mapping between the mathematical formulation and the implementation  
- Serves as a correctness reference for validating parallel implementations  
- Enables early identification of computational bottlenecks through profiling  
- Establishes a minimum performance bound for speedup evaluation  

---

## 3. Implementation Strategy

### 3.1 Mesh Loading
Mesh data is loaded primarily from binary HDF5 files. This choice minimizes parsing overhead and ensures that input/output costs remain negligible relative to computation, even for large meshes.

### 3.2 System Assembly
The global stiffness matrix and load vector are assembled using a classical element-by-element FEM approach:

1. The global sparse matrix is initialized in a format optimized for incremental insertion.  
2. Elements are processed sequentially.  
3. For each element, an 8×8 local stiffness matrix and corresponding load contributions are computed using numerical quadrature.  
4. Local contributions are scattered into the global sparse matrix.  

After assembly, the global matrix is converted to a compressed sparse format optimized for sparse matrix–vector products during the solution phase. This two-phase strategy balances insertion efficiency during assembly with arithmetic efficiency during iterative solution.

### 3.3 Boundary Condition Application
Boundary conditions are applied after assembly using standard FEM techniques:

- **Robin boundary conditions (inlet)** are enforced through numerical integration of boundary contributions.  
- **Dirichlet boundary conditions (outlet)** are imposed using the penalty method for implementation simplicity.  

The computational cost of boundary condition application is small relative to assembly and solution phases.

### 3.4 Linear System Solution
The resulting linear system is solved using the Conjugate Gradient (CG) method provided by SciPy. To ensure robust and consistent convergence:

- The system is diagonally equilibrated to improve numerical conditioning.  
- A Jacobi (diagonal) preconditioner is applied.  

The same solver configuration and convergence criteria are used across all implementations, ensuring identical iteration counts and comparable numerical behavior.

### 3.5 Post-Processing
Post-processing computes derived quantities such as velocity fields and pressure from the solved potential field. These operations involve additional element-level loops and are executed sequentially.

While not dominant, post-processing introduces a measurable overhead for large meshes.

---

## 4. Optimization Techniques Applied

### 4.1 Sparse Matrix Format Selection
Different sparse matrix formats are employed at different stages of the computation:

| Format | Insertion | SpMV | Memory | Usage |
|--------|-----------|------|--------|-------|
| LIL (List of Lists) | O(1) amortized | O(nnz) | Higher | Assembly |
| CSR (Compressed Sparse Row) | O(n) | O(nnz) optimal | Lower | Solve |

This separation minimizes assembly overhead while ensuring efficient memory access during iterative solution.

### 4.2 Diagonal Equilibration
Prior to solving, the linear system is diagonally equilibrated to improve conditioning. This scaling reduces sensitivity to variations in element size and improves convergence behavior, particularly for large or heterogeneous meshes.

### 4.3 Preconditioning Strategy
A Jacobi (diagonal) preconditioner is employed within the Conjugate Gradient solver. Despite its simplicity, this preconditioner provides a favorable trade-off between implementation complexity and convergence robustness, ensuring stable and reproducible iteration counts.

### 4.4 Vectorized Inner Operations
Within each element computation, dense linear algebra operations are expressed using NumPy array operations. These operations are executed in optimized compiled libraries, partially mitigating Python interpreter overhead at the inner-kernel level.

---

## 5. Challenges and Limitations

### 5.1 Sequential Element Loop
The assembly phase relies on an explicit Python loop over all elements. For large meshes, this results in linear scaling dominated by interpreter overhead rather than arithmetic intensity.

### 5.2 Global Interpreter Lock (GIL) Constraints
Although numerical kernels release the GIL, Python-level control flow and sparse matrix indexing remain serialized. As a result, multi-threaded execution provides limited benefit for this implementation.

### 5.3 Sparse Matrix Insertion Overhead
Incremental updates to the global sparse matrix incur significant overhead due to dynamic memory allocation, object management, and indirect indexing. These costs dominate assembly time for large problem sizes.

### 5.4 Memory Access Patterns
Element assembly involves scattered reads of nodal data and scattered writes to the global sparse matrix. This access pattern exhibits poor spatial locality, leading to cache inefficiencies and increased memory traffic.

### 5.5 Observed Execution Behavior

#### 5.5.1 Element-Level Execution and Interpreter Overhead
Assembly follows a strictly element-by-element execution model aligned with the FEM formulation. Performance is dominated by Python loop execution and sparse matrix indexing rather than floating-point computation, resulting in interpreter-bound behavior.

#### 5.5.2 Sparse Matrix Format Trade-offs
Sparse matrix assembly is performed using a format optimized for incremental insertion, followed by conversion to a compressed format optimized for sparse matrix–vector operations.

This conversion introduces additional overhead but is required for efficient solver execution. In the baseline implementation, conversion overhead is amortized over multiple solver iterations.

#### 5.5.3 Impact of Preconditioning on Convergence
Solver convergence is highly sensitive to preconditioning. In the absence of preconditioning, the Conjugate Gradient method exhibits significantly increased iteration counts, sensitivity to problem scaling, and potential convergence failure.

The Jacobi preconditioner improves numerical conditioning and stabilizes convergence with negligible computational overhead, ensuring consistent iteration counts across problem sizes.

#### 5.5.4 Residual Evaluation and Solver Diagnostics
Convergence monitoring is based on explicit evaluation of the true residual norm rather than solver-internal estimates. This provides a consistent convergence criterion across implementations and enables early detection of numerical anomalies.

The additional cost of residual evaluation is limited to a sparse matrix–vector product per monitoring step and is negligible relative to overall solver runtime.

---

## 6. Performance Characteristics and Baseline Role

### 6.1 Expected Scaling
From an algorithmic perspective, the CPU baseline exhibits the following computational complexity:

| Stage | Complexity | Dominant Factor |
|------|-----------|-----------------|
| Mesh loading | O(N_nodes) | I/O bandwidth |
| Assembly | O(N_elements × 64 × 9) | Python loop overhead |
| Boundary condition application | O(N_boundary) | Minor relative cost |
| Linear system solution | O(iterations × nnz) | SpMV memory bandwidth |
| Post-processing | O(N_elements × 8 × 4) | Python loop overhead |

The constant factors reflect the fixed size of element stiffness matrices and the numerical quadrature scheme employed.

### 6.2 Profiling Observations
For large meshes, the expected distribution of execution time is:

- **Assembly:** approximately 50–70% of total runtime  
- **Solve:** approximately 20–40%, governed by sparse matrix–vector products and iteration count  
- **Post-processing:** approximately 5–15%  
- **Mesh I/O and boundary conditions:** typically below 5%  

### 6.3 Baseline Role
The CPU baseline establishes the following reference points:

- **Correctness reference:** All alternative implementations must produce numerically equivalent results.  
- **Performance floor:** Any parallel CPU or GPU-based approach must improve upon this execution time.  
- **Solver behavior reference:** Convergence behavior and iteration counts are expected to remain consistent across implementations.  

This implementation therefore defines the reference execution profile for all reported speedups, scalability analyses, and efficiency metrics.

---

## 7. Summary
The CPU baseline provides a clear, correct, and reproducible reference for all subsequent implementations. While intentionally limited in scalability, it establishes a shared algorithmic foundation, a correctness benchmark, and a performance floor for comparative evaluation.

Key observations include:

- Assembly is interpreter-bound and dominates runtime.  
- Python-level overhead outweighs arithmetic cost for element-level operations.  
- The iterative solver is primarily memory-bound.  

Subsequent implementations address these limitations through parallel execution models, JIT compilation, and GPU offloading, while preserving numerical equivalence with this baseline.

# Implementation 2: CPU Threaded

## 1. Overview

The CPU Threaded implementation extends the CPU baseline by introducing parallelism through Python’s `concurrent.futures.ThreadPoolExecutor`. The objective is to evaluate whether multi-threading can accelerate FEM assembly and post-processing despite the presence of Python’s Global Interpreter Lock (GIL).

Unlike the baseline, which executes all element-level operations sequentially, this implementation partitions the mesh into batches processed concurrently by multiple threads. The approach relies on the fact that NumPy releases the GIL during computational kernels, allowing partial overlap of execution across threads.

| Attribute | Description |
|-----------|-------------|
| Technology | Python ThreadPoolExecutor (`concurrent.futures`) |
| Execution Model | Multi-threaded with GIL constraints |
| Role | Evaluate benefits and limits of threading on CPU |
| Dependencies | NumPy, SciPy, concurrent.futures (stdlib) |

---

## 2. Technology Background

### 2.1 Python Threading and the Global Interpreter Lock

Python’s Global Interpreter Lock (GIL) enforces serialized execution of Python bytecode, preventing true parallel execution of CPU-bound workloads across threads. This simplifies memory management but significantly constrains scalability for numerical applications implemented at the Python level.

However, many NumPy operations release the GIL during execution, including:

- Vectorized array arithmetic  
- Dense linear algebra routines (BLAS/LAPACK)  
- Element-wise mathematical kernels  

This behavior enables limited concurrency when the computation is structured to maximize time spent inside GIL-released NumPy kernels, while minimizing Python-level control flow.

### 2.2 ThreadPoolExecutor Execution Model

The `ThreadPoolExecutor` abstraction provides a pool of reusable worker threads and a future-based execution model.

Key characteristics include:

- Persistent worker threads, reducing creation overhead  
- Asynchronous task submission via `Future` objects  
- Automatic synchronization and cleanup through context management  
- Dynamic scheduling that enables basic load balancing  

This abstraction simplifies parallel orchestration while preserving shared-memory access to NumPy arrays.

### 2.3 Implications for FEM Workloads

Relative to the CPU baseline, the expected impact of threading on FEM operations is mixed:

| Operation | GIL Released | Expected Benefit |
|----------|--------------|------------------|
| Python loop iteration | No | None |
| Sparse matrix indexing | No | None |
| NumPy dense kernels | Yes | Moderate |
| Element-wise NumPy ops | Yes | Moderate |

The overall benefit therefore depends on increasing the ratio of GIL-free numerical computation relative to GIL-held Python coordination.

---

## 3. Implementation Strategy

### 3.1 Batch-Based Parallelization

To amortize threading overhead and reduce GIL contention, elements are grouped into fixed-size batches. Each batch is processed by a single thread, enabling coarse-grained parallelism:

```
┌─────────────────────────────────────────────────────┐
│                    Element Range                     │
│  [0, 1000) [1000, 2000) [2000, 3000) ... [N-1000, N) │
└──────┬──────────┬───────────┬──────────────┬────────┘
       │          │           │              │
       ▼          ▼           ▼              ▼
   ┌───────┐  ┌───────┐  ┌───────┐      ┌───────┐
   │Thread │  │Thread │  │Thread │ ...  │Thread │
   │  0    │  │  1    │  │  2    │      │  N-1  │
   └───┬───┘  └───┬───┘  └───┬───┘      └───┬───┘
       │          │           │              │
       ▼          ▼           ▼              ▼
   ┌───────────────────────────────────────────────┐
   │              COO Data Aggregation              │
   │        rows[], cols[], vals[] per batch        │
   └───────────────────────────────────────────────┘
                          │
                          ▼
   ┌───────────────────────────────────────────────┐
   │         COO → CSR Matrix Construction          │
   └───────────────────────────────────────────────┘
```



Each thread operates independently on a contiguous range of elements, computing local stiffness contributions and storing results in thread-local buffers.

### 3.2 Element Batch Processing

Each batch computes stiffness matrices and load contributions for a subset of elements and stores results in pre-allocated arrays using COO (Coordinate) format.

Key steps include:

1. Pre-allocation of output arrays for rows, columns, and values  
2. Sequential processing of elements within the batch  
3. Computation of local stiffness matrices using NumPy operations  
4. Storage of local contributions in thread-local COO arrays  

This design avoids shared writes during assembly and minimizes synchronization.

### 3.3 Parallel Assembly Orchestration

The main assembly routine dispatches batches to worker threads using a thread pool. Results are collected asynchronously, allowing faster threads to return without blocking on slower batches. After all threads complete, individual COO arrays are concatenated and converted to CSR format.

### 3.4 COO-Based Global Assembly

Unlike the baseline implementation, which performs incremental insertion into a LIL matrix, this implementation assembles the global stiffness matrix using COO format:

| Aspect | Baseline (LIL) | Threaded (COO) |
|------|----------------|----------------|
| Thread safety | Not thread-safe | Naturally thread-safe |
| Insertion pattern | Incremental | Batched |
| Duplicate handling | Explicit | Automatic on CSR conversion |
| Parallel suitability | Poor | High |

The final `COO → CSR` conversion automatically merges duplicate entries arising from shared nodes between elements.

### 3.5 Post-Processing Parallelization

Derived field computation (velocity and magnitude) follows the same batch-based threading strategy. Each thread processes a disjoint subset of elements and writes results into non-overlapping regions of the output arrays, avoiding data races.

### 3.6 Linear System Solution

The linear solver is identical to the CPU baseline. SciPy’s Conjugate Gradient solver is used with the same preconditioning and convergence criteria. No Python-level threading is applied to the solver phase, as SciPy internally manages optimized numerical kernels and threading via BLAS libraries.

---

## 4. Optimization Techniques Applied

### 4.1 Batch Size Selection

Batch size is a critical tuning parameter controlling the balance between coordination overhead and load balance. Empirical testing indicates that batch sizes between 500 and 2000 elements provide the best trade-off for typical problem sizes.

### 4.2 Pre-allocation of Thread-Local Buffers

Each batch allocates fixed-size arrays once per thread invocation, avoiding repeated dynamic memory allocation within inner loops. This reduces overhead and improves cache locality.

### 4.3 Inlined Element Computation

Element stiffness computation is implemented directly within the batch function to minimize function call overhead and maximize time spent in GIL-released NumPy kernels.

### 4.4 Shared Read-Only Data

Mesh coordinates, connectivity, and quadrature data are shared across threads as read-only NumPy arrays. This avoids memory duplication while maintaining thread safety.

---

## 5. Challenges and Limitations

### 5.1 GIL Contention

Despite NumPy releasing the GIL during numerical kernels, a substantial fraction of execution time remains GIL-bound due to Python loops, indexing, and sparse data manipulation. This fundamentally limits scalability.

### 5.2 Memory Bandwidth Saturation

All threads share the same memory subsystem, leading to contention and diminishing returns beyond a modest number of threads.

### 5.3 Thread Management Overhead

Task submission, scheduling, and result aggregation introduce non-negligible overhead, which dominates execution time for small problem sizes.

### 5.4 Limited Solver Parallelism

The solver phase remains effectively sequential at the Python level. While underlying BLAS libraries may use threads, overall solver performance is memory-bound and does not benefit significantly from additional Python threading.

---

## 6. Performance Characteristics and Role

### 6.1 Expected Scaling Behavior

Thread-level parallelism yields sub-linear speedup governed by Amdahl’s Law. Only portions of the assembly and post-processing phases benefit from concurrent execution.

### 6.2 Practical Speedup Regime

Empirical behavior typically shows:

- Modest gains with 2–4 threads  
- Diminishing returns beyond 4–8 threads  
- Potential slowdowns when contention outweighs parallel benefits  

### 6.3 Role in the Implementation Suite

This implementation serves as an intermediate reference between the sequential CPU baseline and more aggressive parallelization strategies. It highlights the structural limitations imposed by the GIL and motivates approaches that bypass it entirely.

---

## 7. Summary

The CPU Threaded implementation demonstrates both the potential and limitations of Python threading for numerical computation:

**Achievements:**

- Introduced parallelism without external dependencies
- Developed batch processing pattern reusable in other implementations
- Identified COO assembly as thread-safe alternative to LIL
- Established baseline for comparing more aggressive parallelization

**Limitations:**

- GIL contention limits achievable speedup
- Memory bandwidth shared across threads
- Python-level overhead remains significant
- Scaling plateaus at modest thread counts

**Key Insight:** For FEM workloads with significant per-element Python overhead, threading provides limited benefit. True parallelism requires either bypassing the GIL (multiprocessing, Numba) or offloading to hardware with native parallelism (GPU).

The batch processing architecture developed here, however, establishes a pattern that transfers to more effective parallelization strategies in subsequent implementations.

---

# Implementation 3: CPU Multiprocess

## 1. Overview

The CPU Multiprocess implementation achieves true parallelism by using process-based parallel execution. Unlike threading, multiprocessing bypasses the Global Interpreter Lock (GIL) entirely, enabling genuine concurrent execution across CPU cores. This comes at the cost of increased inter-process communication (IPC) and memory duplication.

| Attribute | Description |
|-----------|-------------|
| Technology | multiprocessing.Pool (Python stdlib) |
| Execution Model | Multi-process, separate memory spaces |
| Role | True CPU parallelism and GIL bypass demonstration |
| Dependencies | NumPy, SciPy, multiprocessing (stdlib) |

---

## 2. Technology Background

### 2.1 Python Multiprocessing

The multiprocessing execution model spawns multiple independent worker processes. Each worker runs its own Python interpreter with an isolated memory space and its own Global Interpreter Lock.

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Worker  │  │ Worker  │  │ Worker  │  │ Worker  │        │
│  │Process 0│  │Process 1│  │Process 2│  │Process 3│        │
│  │         │  │         │  │         │  │         │        │
│  │Own GIL  │  │Own GIL  │  │Own GIL  │  │Own GIL  │        │
│  │Own Heap │  │Own Heap │  │Own Heap │  │Own Heap │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                          │                                   │
│                    Pickle/IPC                                │
└─────────────────────────────────────────────────────────────┘
```


Key characteristics:

- **Separate memory**: Each process has isolated address space  
- **Independent GIL**: No GIL contention between processes  
- **IPC required**: Data must be serialized (pickled) for transfer  
- **Higher overhead**: Process creation and coordination are more expensive than threads  

### 2.2 multiprocessing.Pool

The Pool abstraction manages a fixed number of worker processes and distributes work among them using mapping primitives.

| Method | Behavior | Ordering |
|--------|----------|----------|
| `map()` | Blocking, returns list | Preserved |
| `map_async()` | Non-blocking | Preserved |
| `imap()` | Lazy iterator | Preserved |
| `imap_unordered()` | Lazy iterator | Arbitrary |

Compared to the threaded implementation, which can collect results asynchronously, `map()` returns results in submission order, simplifying aggregation.


### 2.3 Pickle Serialization

Inter-process communication relies on pickle serialization:

- All input arguments are serialized and sent to workers  
- Return values are serialized and sent back to the main process  
- Worker functions must be defined at module level  
- Large arrays incur significant serialization overhead  


### 2.4 Relevance for FEM

Relative to threading, multiprocessing offers true parallelism but introduces additional overheads:

| Aspect | Threading | Multiprocessing |
|------|-----------|-----------------|
| GIL impact | Serializes Python bytecode | None |
| Memory | Shared | Duplicated per process |
| Startup cost | Low | High |
| Communication | Direct memory access | Pickle serialization |
| Scalability | Limited by GIL | Limited by cores and IPC |

For FEM assembly with element-independent computation, multiprocessing can approach near-linear speedup if IPC overhead is amortized.

---

## 3. Implementation Strategy

### 3.1 Module-Level Function Requirement

A critical constraint of multiprocessing is that worker logic must be defined at module level to be serializable. This imposes structural constraints compared to class-centric designs.

All computational kernels and batch-processing logic must therefore reside at top-level scope.


### 3.2 Batch Processing Architecture

The global element set is partitioned into contiguous batches. Each batch is processed independently by a worker process.

Each batch contains:

- Element index range  
- Coordinate data  
- Connectivity information  
- Quadrature data  

Batching amortizes IPC overhead and reduces scheduling frequency.


### 3.3 Data Serialization Implications

Unlike threading, multiprocessing requires explicit data transfer per batch:

```
Main Process                    Worker Process
     │                               │
     │  pickle(x, y, quad8)          │
     ├──────────────────────────────▶│
     │                               │  (compute element matrices)
     │  pickle(rows, cols, vals)     │
     │◀──────────────────────────────┤
     │                               │
```


For large meshes, serialization frequency and volume become dominant performance constraints.


### 3.4 COO Assembly Strategy

As in the threaded implementation, assembly uses a coordinate-based sparse representation:

- Workers generate independent COO contributions  
- The main process concatenates all partial results  
- COO → CSR conversion merges duplicates automatically  

This avoids concurrent updates to shared sparse structures.


### 3.5 Post-Processing Parallelization

Derived field computation follows the same batching strategy. The solution field must also be serialized and transmitted to workers, increasing IPC overhead during post-processing.


### 3.6 Linear System Solution

The linear solver is executed in the main process using the same configuration as other implementations, ensuring consistent convergence behavior and numerical equivalence.

---

## 4. Optimization Techniques Applied

### 4.1 Batch Size for IPC Amortization

Larger batches reduce IPC frequency but limit load balancing flexibility:

| Batch Size | Batches (100K elements) | IPC Transfers | IPC Overhead |
|------------|--------------------------|---------------|--------------|
| 100 | 1000 | 2000 | Very High |
| 1000 | 100 | 200 | Medium |
| 5000 | 20 | 40 | Low |
| 10000 | 10 | 20 | Very Low |

### 4.2 Tuple-Based Argument Packing

All data required for batch processing is grouped and transmitted together. This simplifies orchestration but increases serialization cost per task.


### 4.3 COO Assembly for Parallel Safety

Independent per-batch output generation avoids shared-state mutation. Duplicate summation is deferred to the final sparse matrix conversion.


### 4.4 Worker Count Configuration

Worker count typically matches available CPU cores. While this maximizes parallelism, it also increases memory duplication and IPC traffic.

---

## 5. Challenges and Limitations

### 5.1 Serialization Overhead

Serialization dominates overhead:

- Input data is serialized for each batch  
- Output data is serialized back to the main process  
- Small batch sizes exacerbate overhead  

### 5.2 Memory Duplication

Each worker process holds a private copy of input data:

```
Total Memory ≈ Main Process + N_workers × (coord arrays + connectivity)
```

For a 100K node mesh with 8 workers:

- Main process: ~10 MB  
- Workers: ~80 MB  
- **Total:** ~90 MB (vs. ~10 MB for threading)


### 5.3 Process Startup Cost

Process creation introduces fixed overhead:

| Component | Typical Time |
|----------|--------------|
| Fork/spawn | 10–50 ms per process |
| Interpreter initialization | 50–100 ms per process |
| Module imports | Variable |
| Pool creation (4 workers) | 200–500 ms |


### 5.4 Limited Shared State

Workers cannot directly modify shared data. All results must be merged in the main process, introducing a sequential aggregation phase.

### 5.5 Pickle Constraints

Serialization requirements restrict code structure and increase implementation complexity.

---

## 6. Performance Characteristics

### 6.1 Scaling Model

Multiprocessing performance can be approximated as:

\[
T_{parallel} = \frac{T_{serial}}{N} + T_{overhead}
\]

where:

- \(T_{serial}\): Sequential computation time  
- \(N\): Number of worker processes  
- \(T_{overhead}\): IPC and process management overhead  


### 6.2 Break-Even Analysis

Multiprocessing becomes beneficial when computation dominates overhead:

| Elements | Computation Time | Overhead (8 workers) | Benefit |
|----------|------------------|----------------------|---------|
| 1,000 | ~0.1 s | ~0.5 s | Negative |
| 10,000 | ~1 s | ~0.5 s | Marginal |
| 50,000 | ~5 s | ~0.6 s | Good |
| 100,000 | ~10 s | ~0.7 s | Excellent |


### 6.3 Memory Bandwidth Considerations

All processes share the same memory subsystem. Bandwidth saturation and NUMA effects can limit scaling on multi-socket systems.

### 6.4 Comparison with Threading

Relative to threading:

- Better scalability for large problems  
- Worse performance for small problems  
- Higher memory consumption  


## 7. Summary

The CPU Multiprocess implementation demonstrates true parallel execution by bypassing Python's GIL through process-based parallelism:

**Achievements:**

- Genuine concurrent execution across CPU cores
- Near-linear speedup for large problems
- Validated COO assembly pattern for parallel safety
- Identified serialization as the primary overhead

**Limitations:**

- High memory usage from data duplication
- Significant IPC overhead for small/medium problems
- Code constraints from pickle requirements
- Process startup latency

**Key Insight:** Multiprocessing trades memory and communication overhead for true parallelism. It excels for large, compute-bound problems where the element loop dominates, but the overhead makes it less suitable for smaller problems or memory-constrained environments.

**Comparison with Threading:**

| Criterion | Winner |
|-----------|--------|
| Small problems (<10K elements) | Threading |
| Large problems (>50K elements) | Multiprocessing |
| Memory efficiency | Threading |
| Maximum speedup potential | Multiprocessing |

The next implementation (Numba JIT) explores an alternative approach: instead of working around the GIL through separate processes, it compiles Python to native code that releases the GIL during execution, combining the benefits of shared memory with true parallelism.

---