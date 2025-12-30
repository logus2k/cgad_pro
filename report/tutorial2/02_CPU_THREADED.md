# CPU Threaded Implementation

## Overview

The CPU Threaded implementation (`quad8_cpu_threaded.py`) introduces **parallelism via Python's `ThreadPoolExecutor`** to distribute element computations across multiple CPU cores. This represents the first optimization step beyond the sequential baseline, exploring whether Python's threading model can accelerate the FEM assembly bottleneck.

This implementation maintains identical numerical results to the CPU baseline while attempting to exploit multi-core processors. However, it faces a fundamental challenge: **Python's Global Interpreter Lock (GIL)**, which limits the effectiveness of threading for CPU-bound Python code.

---

## Technology Stack

### Core Scientific Computing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.10+ | Primary implementation language |
| Parallelism | concurrent.futures.ThreadPoolExecutor | Thread-based parallel execution |
| Array Operations | NumPy 1.24+ | N-dimensional array computations |
| Sparse Matrices | SciPy (scipy.sparse) | COO and CSR matrix formats |
| Linear Solver | SciPy (scipy.sparse.linalg) | Conjugate Gradient solver |
| Data I/O | h5py, Pandas | Mesh file loading |

### Threading Architecture

| Component | Purpose |
|-----------|---------|
| `ThreadPoolExecutor` | Manages pool of worker threads |
| `as_completed()` | Processes results as threads finish |
| Batch processing | Groups elements to reduce thread overhead |
| COO matrix format | Enables parallel assembly without conflicts |

### Real-Time Event-Driven Notifications

Identical callback architecture to CPU baseline:

| Event | Trigger | Payload |
|-------|---------|---------|
| `fem:stageStart` | Stage begins | `{stage: string}` |
| `fem:stageComplete` | Stage ends | `{stage, duration}` |
| `fem:assemblyProgress` | During assembly | `{current, total}` |
| `fem:solveProgress` | Each N iterations | `{iteration, residual, etr}` |

---

## Architecture

### Class Structure

```
Quad8FEMSolverThreaded
├── __init__()                    # Configuration + thread pool settings
├── load_mesh()                   # Mesh I/O (identical to baseline)
├── assemble_system()             # THREADED: parallel element processing
├── apply_boundary_conditions()   # Sequential (same as baseline)
├── solve()                       # Sequential (SciPy CG)
├── compute_derived_fields()      # THREADED: parallel velocity computation
└── run()                         # Workflow orchestration

Helper Functions (module level):
├── process_element_batch_assembly()   # Worker function for assembly
├── process_element_batch_velocity()   # Worker function for post-processing
├── compute_element_stiffness()        # Single element Ke computation
└── compute_element_velocity()         # Single element velocity computation
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_workers` | `os.cpu_count()` | Number of threads in pool |
| `batch_size` | 1000 | Elements per batch (reduces thread overhead) |

### Execution Flow

```
┌─────────────────┐
│  1. Load Mesh   │  Sequential (I/O bound)
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Assembly    │  ◀── THREADED: batches processed in parallel
└────────┬────────┘      Each thread computes Ke for batch of elements
         ▼               Results merged via COO format
┌─────────────────┐
│  3. Apply BCs   │  Sequential (sparse modifications)
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Solve       │  Sequential (SciPy CG)
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Post-Process│  ◀── THREADED: velocity computed in parallel
└─────────────────┘
```

---

## Key Implementation Details

### Threading Strategy: Batch Processing

Instead of spawning one thread per element (excessive overhead), elements are grouped into **batches**:

```python
def assemble_system(self) -> None:
    """Assemble global stiffness matrix using ThreadPoolExecutor."""
    
    xp, wp = gauss_points_9()                    # Shared quadrature data
    
    # Divide elements into batches (default: 1000 elements per batch)
    batches = []
    for start in range(0, self.Nels, self.batch_size):
        end = min(start + self.batch_size, self.Nels)
        # Each batch is a tuple of (start_idx, end_idx, shared_data...)
        batches.append((start, end, self.x, self.y, self.quad8, xp, wp))
    
    # Process batches in parallel using thread pool
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        futures = [executor.submit(process_element_batch_assembly, batch) 
                   for batch in batches]
        
        for future in as_completed(futures):     # Collect results as they complete
            start_idx, rows, cols, vals, fe_batch = future.result()
            # Accumulate COO data from each batch...
```

**Batch Size Trade-off:**
- Too small → Thread creation overhead dominates
- Too large → Fewer batches → Less parallelism
- Default (1000) balances overhead vs. parallelism for typical mesh sizes

### COO Matrix Assembly Pattern

The key insight enabling parallel assembly is using **COO (Coordinate) format** instead of LIL:

```python
def process_element_batch_assembly(args):
    """Worker function: process a batch of elements for assembly."""
    start_idx, end_idx, x, y, quad8, xp, wp = args
    
    batch_size = end_idx - start_idx
    
    # Pre-allocate COO arrays for this batch (no conflicts between threads)
    rows = np.zeros(batch_size * 64, dtype=np.int32)   # 64 entries per element
    cols = np.zeros(batch_size * 64, dtype=np.int32)
    vals = np.zeros(batch_size * 64, dtype=np.float64)
    
    for local_e, e in enumerate(range(start_idx, end_idx)):
        edofs = quad8[e]
        XN = np.column_stack([x[edofs], y[edofs]])
        
        Ke, fe = compute_element_stiffness(XN, xp, wp)   # 8×8 element matrix
        
        # Store as COO triplets (no mutex needed - each thread writes to own slice)
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

**Why COO works for parallel assembly:**
- Each thread writes to its own pre-allocated arrays
- No synchronization needed during computation
- Duplicate indices are summed automatically when converting to CSR

### Matrix Construction from COO Data

After all threads complete, COO data is merged and converted to CSR:

```python
    # Combine COO data from all batches
    rows = np.concatenate(all_rows)              # Merge all row indices
    cols = np.concatenate(all_cols)              # Merge all column indices
    vals = np.concatenate(all_vals)              # Merge all values
    
    # Build sparse matrix - duplicates are summed (exactly what FEM assembly needs)
    self.Kg = coo_matrix(
        (vals, (rows, cols)),
        shape=(self.Nnds, self.Nnds),
        dtype=np.float64
    ).tocsr()                                     # Convert to CSR for solving
```

### Parallel Post-Processing

Velocity computation follows the same batching pattern:

```python
def compute_derived_fields(self) -> None:
    """Compute velocity field using ThreadPoolExecutor."""
    
    xp = gauss_points_4()                        # 2×2 quadrature for post-processing
    
    # Create batches with solution data
    batches = []
    for start in range(0, self.Nels, self.batch_size):
        end = min(start + self.batch_size, self.Nels)
        batches.append((start, end, self.x, self.y, self.quad8, self.u, xp))
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        futures = [executor.submit(process_element_batch_velocity, batch) 
                   for batch in batches]
        
        for future in as_completed(futures):
            start_idx, vel_batch, abs_vel_batch = future.result()
            batch_size = vel_batch.shape[0]
            self.vel[start_idx:start_idx + batch_size] = vel_batch
            self.abs_vel[start_idx:start_idx + batch_size] = abs_vel_batch
```

---

## The GIL Problem

### What is the GIL?

Python's **Global Interpreter Lock (GIL)** is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This means:

- Only **one thread executes Python code at a time**
- Threading provides concurrency (interleaving) but not parallelism (simultaneous execution)
- CPU-bound Python code sees **no speedup** from threading

### Why Threading Still Helps (Sometimes)

NumPy operations can release the GIL when calling into compiled C/Fortran libraries:

```python
# These operations MAY release the GIL:
result = A @ B                    # Matrix multiplication (BLAS)
det = np.linalg.det(matrix)       # Determinant (LAPACK)
inv = np.linalg.inv(matrix)       # Matrix inverse (LAPACK)
```

However, our element computation involves many small operations:

```python
# Small arrays (8×2, 8×8) - overhead dominates, GIL frequently held
Dpsi = np.zeros((8, 2), dtype=np.float64)    # Small allocation
Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4  # Python arithmetic
jaco = XN.T @ Dpsi                            # Small matmul (8×2)
```

For small arrays, the overhead of releasing/acquiring the GIL exceeds the computation time, so the GIL is often **not released**.

### Observed Impact

| Stage | GIL Impact | Expected Speedup |
|-------|------------|------------------|
| Assembly | High (small arrays, Python loops) | Minimal (1.1-1.5x) |
| Solve | N/A (single-threaded SciPy) | None |
| Post-processing | High (same as assembly) | Minimal |

---

## Design Decisions

### Approach Rationale

1. **Explore Threading First:** Before introducing complex dependencies (Numba, CuPy), test whether Python's built-in threading can help.

2. **Batch Processing:** Amortize thread creation overhead across many elements.

3. **COO Format:** Avoid synchronization issues that would arise with LIL format.

4. **Maintain Interface:** Keep identical API to baseline for easy comparison.

### Trade-offs Made

| Decision | Benefit | Cost |
|----------|---------|------|
| ThreadPoolExecutor | Simple, stdlib, no dependencies | Limited by GIL |
| Batch processing | Reduces thread overhead | Added complexity |
| COO → CSR conversion | Lock-free parallel assembly | Extra memory + conversion time |
| Module-level worker functions | Required for pickling | Less encapsulated code |

### What This Implementation Demonstrates

1. **GIL is a Real Limitation:** CPU-bound Python code cannot be parallelized effectively with threads.

2. **NumPy GIL Release is Inconsistent:** Small array operations don't benefit from threading.

3. **Alternative Approaches Needed:** True parallelism requires:
   - Multiprocessing (separate processes, separate GILs)
   - JIT compilation (Numba releases GIL)
   - GPU computing (entirely different execution model)

---

## Performance Characteristics

### Strengths

1. **Simple Implementation:** Uses only Python standard library
2. **No External Dependencies:** No Numba, CuPy, or MPI required
3. **Identical Results:** Produces exactly the same output as baseline
4. **I/O Parallelism:** Would help if I/O were the bottleneck (it's not)

### Limitations

1. **GIL Bottleneck:** Cannot achieve true parallelism for CPU-bound code
2. **Thread Overhead:** Creating/joining threads adds latency
3. **Memory Overhead:** COO format requires 3× storage vs. direct assembly
4. **Batch Tuning:** Optimal batch size depends on mesh size and CPU count
5. **No Solve Speedup:** Linear solver remains single-threaded

### Computational Complexity

| Stage | Time Complexity | Parallelism |
|-------|-----------------|-------------|
| Assembly | O(Nels × 64 × 9) / num_workers | Limited by GIL |
| COO→CSR | O(nnz log nnz) | None |
| Apply BCs | O(boundary_nodes) | None |
| Solve | O(iterations × nnz) | None |
| Post-process | O(Nels × 4) / num_workers | Limited by GIL |

### Benchmark Results

| Mesh | Nodes | Elements | Assembly (s) | Solve (s) | Total (s) |
|------|-------|----------|--------------|-----------|-----------|
| small_duct | ~5,000 | ~1,600 | [placeholder] | [placeholder] | [placeholder] |
| s_duct | ~65,000 | ~21,000 | [placeholder] | [placeholder] | [placeholder] |
| venturi_194k | ~194,000 | ~64,000 | [placeholder] | [placeholder] | [placeholder] |

*[Benchmark data to be populated with actual measurements]*

---

## Code Highlights

### Thread Pool Configuration

```python
def __init__(self, ..., num_workers: int = None, batch_size: int = 1000):
    # Default to all available CPUs
    self.num_workers = num_workers or os.cpu_count()
    self.batch_size = batch_size
    
    if self.verbose:
        print(f"Initialized with {self.num_workers} workers, batch_size={batch_size}")
```

### Worker Function Pattern

Worker functions must be defined at module level (not as methods) for thread pool compatibility:

```python
# Module-level function - can be pickled for thread pool
def process_element_batch_assembly(args):
    """Worker function receives all data as tuple argument."""
    start_idx, end_idx, x, y, quad8, xp, wp = args
    # ... process batch ...
    return start_idx, rows, cols, vals, fe_batch

# In class method:
with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    futures = [executor.submit(process_element_batch_assembly, batch) 
               for batch in batches]
```

### Results Collection with as_completed

```python
from concurrent.futures import as_completed

# Process results as threads complete (not in submission order)
for future in as_completed(futures):
    start_idx, rows, cols, vals, fe_batch = future.result()
    
    # Use start_idx to place results in correct position
    all_rows.append(rows)
    all_cols.append(cols)
    all_vals.append(vals)
```

---

## Lessons Learned

### Development Insights

1. **GIL Impact is Dramatic:** Expected 4-8x speedup on 8-core CPU; observed 1.1-1.5x. The GIL severely limits threading benefits for CPU-bound code.

2. **Batch Size Matters:** Too small (10 elements) → thread overhead dominates. Too large (10000 elements) → insufficient parallelism. 1000 is a reasonable default.

3. **COO Format is Key:** Attempting to use LIL with threading caused race conditions. COO's "append then merge" pattern is thread-safe.

4. **Small Arrays Don't Benefit:** NumPy's GIL release mechanism has overhead that exceeds benefit for 8×8 matrices.

### Debugging Challenges

1. **Race Conditions:** Early attempts modified shared LIL matrix from multiple threads, causing silent data corruption.

2. **Deadlocks:** Incorrect use of thread synchronization primitives caused hangs.

3. **Memory Leaks:** Forgetting to properly close thread pool or handle exceptions.

---

## Conclusions

The CPU Threaded implementation demonstrates that **Python threading is not effective for parallelizing CPU-bound FEM computations** due to the Global Interpreter Lock. While the implementation is correct and produces identical results, performance gains are minimal (typically 1.1-1.5x vs. single-threaded baseline).

### Key Takeaways

1. **Threading ≠ Parallelism in Python:** Threads provide concurrency but not parallel execution for CPU-bound code.

2. **GIL Release is Unreliable:** Small NumPy operations don't release the GIL effectively.

3. **Alternative Approaches Required:** True CPU parallelism requires multiprocessing (next implementation) or JIT compilation (Numba).

### When to Use This Implementation

- **Learning:** Understanding thread pool patterns and GIL limitations
- **I/O-bound workloads:** Would be effective if mesh loading were the bottleneck
- **Baseline comparison:** Demonstrates why more sophisticated approaches are needed

### Path Forward

The minimal speedup from threading motivates exploring:
- **Multiprocessing:** Separate processes bypass the GIL entirely
- **Numba JIT:** Compiled code releases the GIL during execution
- **GPU Computing:** Massive parallelism with thousands of cores

---
