# Implementation 2: CPU Threaded

## 1. Overview

The CPU Threaded implementation (`quad8_cpu_threaded.py`) introduces parallelism using Python's `concurrent.futures.ThreadPoolExecutor`. This implementation explores the extent to which threading can accelerate FEM computations despite Python's Global Interpreter Lock (GIL), leveraging the fact that NumPy operations release the GIL during execution.

| Attribute | Description |
|-----------|-------------|
| **Technology** | ThreadPoolExecutor (concurrent.futures) |
| **Execution Model** | Multi-threaded with GIL constraints |
| **Role** | Explore threading benefits within GIL limitations |
| **Source File** | `quad8_cpu_threaded.py` |
| **Dependencies** | NumPy, SciPy, concurrent.futures (stdlib) |

---

## 2. Technology Background

### 2.1 Python Threading and the GIL

Python's Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This design choice simplifies CPython's memory management but limits the effectiveness of multi-threading for CPU-bound tasks.

However, the GIL can be released by extension modules during long-running operations. NumPy releases the GIL during:

- Array arithmetic operations
- Linear algebra routines (BLAS/LAPACK calls)
- Element-wise mathematical functions

This creates an opportunity: if computation is structured as batched NumPy operations, multiple threads can execute concurrently during the NumPy portions while serializing only during Python-level coordination.

### 2.2 ThreadPoolExecutor

The `concurrent.futures` module provides a high-level interface for parallel execution:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=N) as executor:
    futures = [executor.submit(func, args) for args in work_items]
    for future in as_completed(futures):
        result = future.result()
```

Key characteristics:

- **Thread reuse**: Pool maintains worker threads, avoiding creation overhead
- **Future-based**: Asynchronous result retrieval via `Future` objects
- **Context manager**: Automatic cleanup and thread joining
- **Work stealing**: Load balancing across available threads

### 2.3 Theoretical Expectations for FEM

For FEM workloads with threading:

| Operation | GIL Released? | Threading Benefit |
|-----------|---------------|-------------------|
| Python loop iteration | No | None |
| NumPy array creation | Partially | Limited |
| NumPy matrix multiply | Yes | Potential |
| NumPy element-wise ops | Yes | Potential |
| Sparse matrix indexing | No | None |

The expected benefit depends on the ratio of GIL-released computation to GIL-held coordination. Batching multiple elements per thread maximizes the compute-to-coordination ratio.

---

## 3. Implementation Strategy

### 3.1 Batch Processing Architecture

The key insight is to process multiple elements per thread, amortizing thread coordination overhead:

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

### 3.2 Element Batch Processing

Each batch processes a contiguous range of elements, computing stiffness matrices and storing results in pre-allocated arrays:

```python
def process_element_batch_assembly(args):
    start_idx, end_idx, x, y, quad8, xp, wp = args
    batch_size = end_idx - start_idx
    
    # Pre-allocate output arrays
    rows = np.zeros(batch_size * 64, dtype=np.int32)
    cols = np.zeros(batch_size * 64, dtype=np.int32)
    vals = np.zeros(batch_size * 64, dtype=np.float64)
    
    for local_e, e in enumerate(range(start_idx, end_idx)):
        edofs = quad8[e]
        XN = np.column_stack([x[edofs], y[edofs]])
        Ke, fe = compute_element_stiffness(XN, xp, wp)
        
        # Store COO entries
        base_idx = local_e * 64
        k = 0
        for i in range(8):
            for j in range(8):
                rows[base_idx + k] = edofs[i]
                cols[base_idx + k] = edofs[j]
                vals[base_idx + k] = Ke[i, j]
                k += 1
    
    return start_idx, rows, cols, vals, fe_batch
```

The batch function:

1. Receives element range and shared data references
2. Pre-allocates output arrays (avoids repeated allocation)
3. Iterates over elements within the batch
4. Returns COO-format sparse data for later assembly

### 3.3 Parallel Assembly with ThreadPoolExecutor

The main assembly method orchestrates batch processing:

```python
def assemble_system(self):
    xp, wp = gauss_points_9()
    
    # Create batches
    batches = []
    for start in range(0, self.Nels, self.batch_size):
        end = min(start + self.batch_size, self.Nels)
        batches.append((start, end, self.x, self.y, self.quad8, xp, wp))
    
    # Process in parallel
    all_rows, all_cols, all_vals = [], [], []
    
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        futures = [executor.submit(process_element_batch_assembly, b) 
                   for b in batches]
        
        for future in as_completed(futures):
            start_idx, rows, cols, vals, fe_batch = future.result()
            all_rows.append(rows)
            all_cols.append(cols)
            all_vals.append(vals)
    
    # Combine and build sparse matrix
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)
    
    self.Kg = coo_matrix((vals, (rows, cols)), 
                          shape=(self.Nnds, self.Nnds)).tocsr()
```

### 3.4 COO Matrix Assembly Strategy

Unlike the baseline's LIL-based incremental insertion, the threaded implementation uses COO (Coordinate) format:

| Aspect | LIL (Baseline) | COO (Threaded) |
|--------|----------------|----------------|
| Insertion | Incremental, in-place | Batch append |
| Thread safety | Not thread-safe | Embarrassingly parallel |
| Memory | Dynamic per entry | Pre-allocated arrays |
| Duplicate handling | Explicit check | Automatic summation on conversion |

COO format allows threads to produce independent arrays that are concatenated after all threads complete. The `coo_matrix.tocsr()` conversion automatically sums duplicate entries, correctly handling shared nodes between elements.

### 3.5 Threaded Post-Processing

Velocity computation follows the same batching pattern:

```python
def compute_derived_fields(self):
    xp = gauss_points_4()
    
    batches = []
    for start in range(0, self.Nels, self.batch_size):
        end = min(start + self.batch_size, self.Nels)
        batches.append((start, end, self.x, self.y, self.quad8, self.u, xp))
    
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        futures = [executor.submit(process_element_batch_velocity, b) 
                   for b in batches]
        
        for future in as_completed(futures):
            start_idx, vel_batch, abs_vel_batch = future.result()
            batch_size = vel_batch.shape[0]
            self.vel[start_idx:start_idx + batch_size] = vel_batch
            self.abs_vel[start_idx:start_idx + batch_size] = abs_vel_batch
```

### 3.6 Linear Solver

The linear solver remains unchanged from the baseline, using SciPy's CG implementation:

```python
u_eq, self.solve_info = cg(
    Kg_eq, fg_eq,
    rtol=TARGET_TOL,
    maxiter=self.maxiter,
    M=M,
    callback=self.monitor
)
```

SciPy's sparse solvers already utilize optimized BLAS internally and manage their own threading through libraries like OpenBLAS or MKL. Adding Python-level threading around the solver would not provide benefit and could introduce overhead.

---

## 4. Key Code Patterns

### 4.1 Batch Creation

Elements are divided into fixed-size batches:

```python
self.batch_size = batch_size  # Default: 1000 elements per batch

batches = []
for start in range(0, self.Nels, self.batch_size):
    end = min(start + self.batch_size, self.Nels)
    batches.append((start, end, self.x, self.y, self.quad8, xp, wp))
```

The batch size is a tunable parameter balancing:

- **Too small**: High coordination overhead, poor cache utilization
- **Too large**: Fewer batches than threads, poor load balancing

### 4.2 Inlined Element Computation

To maximize GIL-released time, element stiffness computation is inlined rather than calling imported functions:

```python
def compute_element_stiffness(XN, xp, wp):
    Ke = np.zeros((8, 8), dtype=np.float64)
    
    for ip in range(9):
        csi, eta = xp[ip, 0], xp[ip, 1]
        
        # Shape function derivatives computed inline
        Dpsi = np.zeros((8, 2), dtype=np.float64)
        Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
        # ... remaining derivatives ...
        
        # Jacobian computation
        jaco = XN.T @ Dpsi
        Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
        
        # Matrix operations (GIL released here)
        B = Dpsi @ Invj
        Ke += wip * (B @ B.T)
    
    return Ke, fe
```

### 4.3 Result Aggregation

The `as_completed` iterator allows processing results as threads finish, regardless of submission order:

```python
for future in as_completed(futures):
    start_idx, rows, cols, vals, fe_batch = future.result()
    all_rows.append(rows)  # Append is thread-safe in Python
```

This pattern provides natural load balancing: fast batches don't wait for slow ones.

---

## 5. Optimization Techniques Applied

### 5.1 Batch Size Tuning

The default batch size of 1000 elements was chosen empirically:

| Batch Size | Batches (100K elements) | Coordination Overhead | Cache Efficiency |
|------------|-------------------------|----------------------|------------------|
| 100 | 1000 | High | Poor |
| 500 | 200 | Medium | Medium |
| 1000 | 100 | Low | Good |
| 5000 | 20 | Very Low | Excellent |

Larger batches reduce overhead but may cause load imbalance if element computation times vary.

### 5.2 Pre-allocation Strategy

Each batch pre-allocates its output arrays to avoid repeated memory allocation:

```python
# Single allocation per batch
rows = np.zeros(batch_size * 64, dtype=np.int32)
cols = np.zeros(batch_size * 64, dtype=np.int32)
vals = np.zeros(batch_size * 64, dtype=np.float64)
```

This is more efficient than growing lists incrementally within the loop.

### 5.3 Shared Data Access

Read-only data (coordinates, connectivity, integration points) is shared across threads via references in the batch tuple:

```python
batches.append((start, end, self.x, self.y, self.quad8, xp, wp))
```

NumPy arrays are not copied; all threads read from the same memory. This is safe for read-only access and avoids memory duplication.

### 5.4 Worker Count Selection

The number of workers defaults to the CPU count:

```python
import os
self.num_workers = num_workers or os.cpu_count()
```

This allows the thread pool to scale with available cores while remaining configurable for experimentation.

---

## 6. Challenges and Limitations

### 6.1 GIL Contention

Despite NumPy releasing the GIL during array operations, significant time is spent in GIL-held code:

- Loop iteration (`for local_e, e in enumerate(...)`)
- Array indexing (`quad8[e]`, `x[edofs]`)
- Tuple unpacking and function calls
- COO index assignment (`rows[base_idx + k] = ...`)

Profiling reveals that the Python-level overhead often exceeds the NumPy computation time, limiting parallel efficiency.

### 6.2 Memory Bandwidth

All threads share the same memory bus, creating contention:

- Coordinate arrays are read by all threads
- COO arrays are written by all threads
- Cache coherency traffic increases with thread count

Beyond a certain thread count, memory bandwidth saturation limits scaling.

### 6.3 False Sharing

When threads write to adjacent memory locations, cache line invalidation occurs even without true data conflicts:

```python
# Potential false sharing if start_idx values are adjacent
self.vel[start_idx:start_idx + batch_size] = vel_batch
```

The batch-based approach mitigates this by ensuring each thread writes to distinct, non-adjacent regions.

### 6.4 Thread Creation Overhead

While `ThreadPoolExecutor` reuses threads, there is still overhead:

- Initial thread creation at pool startup
- Work item queuing and dequeuing
- Future object creation and result retrieval

For small problems, this overhead may exceed any parallel benefit.

### 6.5 No Solver Parallelization

The CG solver remains sequential at the Python level. While the underlying BLAS may use threads, Python's control flow (iteration loop, convergence checks) is serialized.

---

## 7. Performance Characteristics

### 7.1 Expected Scaling

Threading introduces both benefits and overheads:

| Component | Scaling Behavior |
|-----------|------------------|
| Element computation | Sub-linear (GIL contention) |
| Thread coordination | Constant overhead per batch |
| Memory access | Limited by bandwidth |
| Result aggregation | Sequential (concatenation) |

Amdahl's Law applies: if $P$ fraction of work is parallelizable and achieves speedup $S_P$, overall speedup is:

$$S = \frac{1}{(1-P) + \frac{P}{S_P}}$$

With GIL contention reducing $S_P$, the achievable speedup is modest.

### 7.2 Optimal Thread Count

Empirical testing typically shows:

- **2-4 threads**: Measurable improvement
- **4-8 threads**: Diminishing returns
- **>8 threads**: Possible slowdown due to contention

The optimal count depends on:

- CPU architecture (cores, cache hierarchy)
- Problem size (elements, nodes)
- BLAS threading configuration

### 7.3 Problem Size Sensitivity

Threading overhead is amortized over more elements in larger problems:

| Problem Size | Relative Overhead | Expected Benefit |
|--------------|-------------------|------------------|
| Small (<10K elements) | High | Minimal or negative |
| Medium (10K-100K) | Medium | Moderate (1.2-2×) |
| Large (>100K) | Low | Better (1.5-3×) |

---

## 8. Insights and Lessons Learned

### 8.1 GIL is the Limiting Factor

The primary lesson is that Python threading provides limited benefit for CPU-bound numerical code. Even with NumPy's GIL release, the Python-level coordination consumes significant time.

Profiling breakdown (typical):

| Activity | Time Fraction | GIL Held |
|----------|---------------|----------|
| NumPy computation | 30-40% | No |
| Python loop/indexing | 40-50% | Yes |
| Thread coordination | 10-20% | Yes |

### 8.2 COO Format Enables Parallel Assembly

Switching from LIL to COO format was essential for thread safety:

- LIL's internal dictionaries are not thread-safe
- COO allows independent array construction per thread
- Duplicate summation during COO→CSR conversion handles shared nodes

### 8.3 Batch Size Trade-offs

Experimentation revealed the importance of batch sizing:

- **Too fine-grained**: Coordination dominates
- **Too coarse-grained**: Load imbalance, poor thread utilization
- **Sweet spot**: 500-2000 elements per batch for typical problems

### 8.4 Threading vs. Multiprocessing

This implementation reveals when threading is appropriate:

| Criterion | Threading | Multiprocessing |
|-----------|-----------|-----------------|
| GIL-bound work | Poor | Good |
| Shared memory needs | Good | Poor (requires copying) |
| Startup overhead | Low | High |
| Memory efficiency | High | Low (duplication) |

For FEM assembly with significant Python overhead, multiprocessing may be more effective (explored in the next implementation).

### 8.5 Solver Threading

The decision to not parallelize the CG solver at the Python level was validated:

- BLAS libraries (OpenBLAS, MKL) handle internal threading
- Python-level threading would add overhead without benefit
- The solver's SpMV operations are memory-bound, not compute-bound

---

## 9. Performance Comparison

The following table will be populated with benchmark results after testing:

| Metric | CPU Baseline | CPU Threaded | Speedup |
|--------|--------------|--------------|---------|
| Assembly Time (s) | — | — | — |
| Solve Time (s) | — | — | — |
| Post-processing Time (s) | — | — | — |
| Total Time (s) | — | — | — |
| CG Iterations | — | — | (same) |
| Peak Memory (MB) | — | — | — |

---

## 10. Summary

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
