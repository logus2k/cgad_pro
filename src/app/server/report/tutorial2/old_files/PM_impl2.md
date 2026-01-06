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