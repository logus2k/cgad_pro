# Implementation 3: CPU Multiprocess

## 1. Overview
The CPU Multiprocess implementation achieves true parallelism by using Python's `multiprocessing.Pool`, which spawns separate Python interpreter processes. Unlike threading, multiprocessing bypasses the Global Interpreter Lock (GIL), enabling concurrent execution at the cost of inter-process communication (IPC) and increased memory usage.

| Attribute | Description |
|---------|-------------|
| Technology | multiprocessing.Pool (Python stdlib) |
| Execution Model | Multi-process, separate memory spaces |
| Role | True CPU parallelism, GIL bypass demonstration |
| Dependencies | NumPy, SciPy, multiprocessing (stdlib) |

---

## 2. Technology Background

### 2.1 Python Multiprocessing
The `multiprocessing` module provides process-based parallelism that sidesteps the GIL entirely. Each worker process runs its own Python interpreter with its own memory space:


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

- **Separate memory:** each process has an isolated memory space  
- **Independent GIL:** no GIL contention between processes  
- **IPC required:** data must be serialized (pickled) for transfer  
- **Higher overhead:** process creation is more expensive than threads  

### 2.2 multiprocessing.Pool
The `Pool` class manages a pool of worker processes and distributes work using map-style execution. The implementation uses `map()` semantics (blocking, ordered results), which simplifies aggregation in the main process.

### 2.3 Pickle Serialization
Inter-process communication relies on pickle serialization:

- function arguments are pickled and sent to workers  
- return values are pickled and sent back to the main process  
- worker functions must be defined at module level (not nested / not lambda)  
- large NumPy arrays incur significant serialization overhead  

### 2.4 Theoretical Expectations for FEM
For element-independent FEM computations, multiprocessing can achieve near-linear speedup in the compute portion, offset by IPC and process management overhead:

| Aspect | Threading | Multiprocessing |
|--------|-----------|-----------------|
| GIL impact | Serializes Python bytecode | None (separate interpreters) |
| Memory | Shared | Duplicated per process |
| Startup cost | Low | High (fork/spawn) |
| Communication | Direct memory access | Pickle serialization |
| Scalability | Limited by GIL | Limited by cores and IPC |

---

## 3. Implementation Strategy

### 3.1 Module-Level Function Requirement
A key constraint is that worker functions must be defined at module level so they can be pickled and executed by worker processes. This requires restructuring compared to purely class-method designs, and applies to element kernels as well as batch processing functions.

### 3.2 Batch Processing Architecture
The batch structure mirrors the threaded implementation: elements are grouped into contiguous ranges and dispatched to workers. Each batch encapsulates:

- element range indices  
- coordinate arrays and connectivity (pickled/copied)  
- integration points and weights  

Batching is essential to amortize IPC overhead by increasing computation per task.

### 3.3 Data Serialization Implications
Unlike threading (shared memory), multiprocessing requires explicit data transfer between processes. For large meshes, the cost of serializing and transferring arrays becomes a first-order performance factor.

For a mesh with 100,000 nodes (illustrative scale):

- coordinate arrays: ~1.6 MB per array  
- connectivity: ~3.2 MB  
- each batch sends multiple MB and receives COO results back  
- total IPC increases with number of batches  

### 3.4 Pool.map Semantics
The use of `pool.map()` returns ordered results and blocks until completion, simplifying aggregation. This reduces orchestration complexity compared to asynchronous futures, but implies that overall completion time is limited by the slowest batch.

### 3.5 Post-Processing Parallelization
Velocity computation follows the same batch execution pattern. In this stage, the solution vector must also be transmitted to each worker, increasing the communication footprint relative to assembly.

---

## 4. Optimization Techniques Applied

### 4.1 Batch Size for IPC Amortization
Batch size directly controls IPC frequency. Larger batches reduce the number of transfers and amortize serialization costs, at the risk of reduced load balancing.

| Batch Size | Batches (100K elements) | IPC Transfers | IPC Overhead |
|------------|--------------------------|---------------|--------------|
| 100 | 1000 | 2000 | Very High |
| 1000 | 100 | 200 | Medium |
| 5000 | 20 | 40 | Low |
| 10000 | 10 | 20 | Very Low |

### 4.2 Avoiding Repeated Serialization
Integration points can be computed once and reused across batches. This reduces redundant computation at the expense of transferring small constant arrays per batch (typically negligible compared to mesh-scale data).

### 4.3 COO Assembly for Parallel Safety
As in threading, COO format enables independent batch processing. Each worker produces private COO arrays, and duplicates are resolved during COO→CSR conversion in the main process.

### 4.4 Worker Count Configuration
Worker processes typically scale with core count, but optimal settings depend on memory capacity and IPC overhead. Increasing worker count can saturate memory bandwidth and amplify duplication costs.

---

## 5. Challenges and Limitations

### 5.1 Serialization Overhead
The dominant overhead in multiprocessing is pickle serialization:

- data sent to each worker includes mesh-scale arrays and batch descriptors  
- data returned includes COO indices/values and batch-level outputs  

For some regimes, serialization can exceed computation time.

### 5.2 Memory Duplication
Each worker process maintains its own memory space, increasing peak memory usage approximately with the number of workers:

- duplicated coordinate/connectivity data  
- additional batch buffers per process  

This can become a practical constraint for large meshes and high worker counts.

### 5.3 Process Startup Cost
Worker process startup and interpreter initialization introduces fixed overhead. This cost is amortized only for large workloads, making multiprocessing less effective for small or medium meshes.

### 5.4 Limited Shared State
Workers cannot directly modify shared sparse matrices or other mutable structures in the main process. All intermediate results must be returned and aggregated centrally.

### 5.5 Pickle Constraints
Pickle imposes structural constraints (e.g., module-level functions), which can affect code organization and increases implementation complexity compared to threading.

---

## 6. Performance Characteristics and Baseline Role

### 6.1 Scaling Model
A simplified execution-time model captures the trade-off between compute speedup and overhead:

\[
T_{parallel} = \frac{T_{serial}}{N} + T_{overhead}
\]

where \(N\) is the number of worker processes and \(T_{overhead}\) includes:

- pool creation and process startup  
- per-batch serialization and IPC  
- centralized aggregation and CSR conversion  

### 6.2 Break-Even Regime
Multiprocessing becomes beneficial when computation dominates overhead:

\[
T_{computation} > T_{overhead}
\]

This typically occurs for sufficiently large element counts, where batch computation time amortizes process startup and per-batch IPC.

### 6.3 Memory Bandwidth Considerations
Although processes have separate address spaces, they still share the same memory subsystem. Memory bandwidth can therefore limit scaling, particularly on high core counts or NUMA systems.

### 6.4 Baseline Role
This implementation demonstrates true parallelism by bypassing the GIL, and provides a reference point for comparing:

- shared-memory approaches limited by the GIL (threading)  
- approaches that aim to retain shared memory while achieving true parallelism (e.g., JIT compilation)  

---

## 7. Summary
The CPU Multiprocess implementation bypasses the GIL through process-based parallelism, enabling genuine concurrent execution of element-level FEM loops. Its performance is driven by a trade-off:

- **Strength:** near-linear compute scaling is possible for large workloads  
- **Costs:** IPC serialization, memory duplication, process startup, and centralized aggregation  

Overall, multiprocessing is most effective for large, compute-heavy meshes where batch computation dominates communication overhead, while smaller problems are often limited by fixed process and IPC costs.
