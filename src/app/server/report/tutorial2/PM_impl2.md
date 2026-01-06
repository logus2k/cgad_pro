# Implementation 2: CPU Threaded

## 1. Overview
The CPU Threaded implementation introduces shared-memory parallelism using Python’s `ThreadPoolExecutor`. It evaluates the extent to which multi-threading can accelerate FEM assembly and post-processing under CPython’s Global Interpreter Lock (GIL), leveraging the fact that NumPy and SciPy release the GIL during computational kernels.

| Attribute | Description |
|---------|-------------|
| Technology | Python (ThreadPoolExecutor, NumPy, SciPy) |
| Execution Model | Multi-threaded, shared memory |
| Role | Assess threading benefits and limitations under GIL constraints |
| Dependencies | NumPy, SciPy, concurrent.futures |

---

## 2. Technology Background

### 2.1 Python Threading and the Global Interpreter Lock
Python threading is constrained by the Global Interpreter Lock (GIL), which enforces mutual exclusion on Python bytecode execution. As a result, Python-level loops, indexing operations, and object manipulation remain serialized even when multiple threads are active.

However, many NumPy and SciPy operations release the GIL during execution in optimized C/Fortran kernels. This enables concurrent execution of numerical kernels across threads, provided that computation is structured to maximize time spent in GIL-released regions.

For FEM workloads, this leads to a hybrid execution profile:

- **Python control flow:** serialized due to the GIL  
- **Dense numerical kernels (BLAS/LAPACK):** potentially parallel  
- **Sparse matrix construction and indexing:** GIL-held  

The effectiveness of threading therefore depends on increasing the ratio of GIL-released computation relative to Python-level coordination.

### 2.2 ThreadPoolExecutor Execution Model
The `ThreadPoolExecutor` provides a fixed pool of worker threads and a future-based interface for task submission and result collection. In this implementation, it is used to process batches of elements concurrently while maintaining shared access to read-only mesh data.

Key characteristics relevant to this implementation include:

- Thread reuse to reduce creation overhead  
- Dynamic task scheduling and load balancing  
- Shared-memory access to NumPy arrays without data duplication  

---

## 3. Implementation Strategy

### 3.1 Batch-Oriented Assembly Architecture
Instead of assigning individual elements to threads, the implementation groups contiguous ranges of elements into batches. Each batch is processed by a single thread, increasing computational granularity and amortizing Python-level coordination overhead.

The resulting execution architecture is illustrated below:

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


This architecture enables thread-safe parallel assembly by ensuring that each thread operates exclusively on private batch-local data structures during the parallel phase.

### 3.2 Parallel System Assembly
Assembly proceeds in two distinct phases:

1. **Parallel batch processing:**  
   Each thread computes element stiffness matrices for its assigned batch and stores contributions in local COO-format arrays.

2. **Global aggregation:**  
   Batch-level COO arrays are concatenated in the main thread and converted to CSR format. Duplicate entries resulting from shared nodes are automatically summed during conversion.

This approach avoids concurrent writes to shared sparse matrices and preserves numerical correctness.

### 3.3 Boundary Condition Application
Boundary conditions are applied after global assembly using the same formulation as the baseline implementation. Their computational cost is negligible relative to assembly and is not parallelized.

### 3.4 Linear System Solution
The linear system is solved using the same Conjugate Gradient configuration as the baseline, including diagonal equilibration and Jacobi preconditioning.

The solver itself is not parallelized at the Python level. While internal BLAS routines may exploit multi-threading, solver control flow remains serialized.


### 3.5 Threaded Post-Processing
Post-processing operations are parallelized using the same batch-based threading strategy. Each thread processes a disjoint element range and writes results to non-overlapping output regions, avoiding synchronization overhead.

---

## 4. Optimization Techniques Applied

### 4.1 Batch Size Selection
Batch size controls the trade-off between coordination overhead and load balance. Larger batches improve cache locality and amortize Python-level overhead, while excessively large batches may reduce thread utilization.


### 4.2 COO-Based Sparse Assembly
Using COO aggregation instead of incremental insertion enables thread-safe assembly and eliminates contention on shared sparse data structures.


### 4.3 Pre-allocation of Batch Buffers
Batch-level arrays are pre-allocated to avoid repeated memory allocation inside inner loops, reducing allocator pressure and improving cache efficiency.


### 4.4 Shared Read-Only Data Access
Mesh coordinates, connectivity, and integration data are shared across threads by reference, avoiding memory duplication while maintaining thread safety.

---

## 5. Challenges and Limitations

### 5.1 GIL Contention
A significant fraction of runtime remains in GIL-held Python code, including loop iteration, indexing, and sparse index generation. This fundamentally limits scalability.


### 5.2 Memory Bandwidth Saturation
All threads share the same memory subsystem, leading to contention for memory bandwidth and cache resources as thread count increases.


### 5.3 Sparse Construction Overhead
Although COO aggregation avoids unsafe writes, constructing and concatenating large coordinate arrays introduces overhead that scales sub-linearly with thread count.


### 5.4 Thread Coordination Overhead
Task scheduling, future management, and result aggregation introduce fixed overheads that dominate performance for small problem sizes.


### 5.5 Solver-Level Serialization
Solver control flow remains serialized at the Python level, constraining end-to-end speedup even when assembly scales moderately well.

---

## 6. Performance Characteristics and Baseline Role

### 6.1 Expected Scaling
Relative to the CPU baseline:

- **Assembly:** sub-linear speedup due to GIL-held control flow and sparse construction overhead  
- **Solve:** similar behavior to baseline, dominated by memory-bound SpMV operations  
- **Post-processing:** moderate speedup possible for large meshes  

Overall speedup is bounded by the serial fraction of Python-level coordination.


### 6.2 Baseline Role
This implementation defines the upper bound of performance achievable with shared-memory threading in CPython and serves as a reference point between the sequential baseline and GIL-bypassing approaches.

---

## 7. Summary
The CPU Threaded implementation preserves numerical equivalence with the baseline while introducing batch-based shared-memory parallelism. It demonstrates that:

- Threading can accelerate FEM workloads when computation occurs in GIL-released kernels  
- GIL contention and memory bandwidth fundamentally limit scalability  
- Speedup saturates at modest thread counts  

This implementation clarifies the practical limits of Python threading for FEM assembly and motivates the use of multiprocessing, JIT compilation, or GPU offloading for higher scalability.
