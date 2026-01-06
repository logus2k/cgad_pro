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