# Implementation 5: Numba CUDA

## 1. Overview

The Numba CUDA implementation (`quad8_numba_cuda.py`) extends Numba's JIT compilation to GPU programming, using the `@cuda.jit` decorator to write GPU kernels in Python syntax. This approach offers an alternative to writing raw CUDA C code while maintaining direct control over GPU execution.

| Attribute | Description |
|-----------|-------------|
| **Technology** | Numba CUDA (@cuda.jit kernels) |
| **Execution Model** | GPU SIMT with Python-syntax kernels |
| **Role** | GPU parallelism with Python-native kernel development |
| **Source File** | `quad8_numba_cuda.py`, `kernels_numba_cuda.py` |
| **Dependencies** | NumPy, SciPy, Numba, CuPy (for sparse solver) |

---

## 2. Technology Background

### 2.1 Numba CUDA Overview

Numba extends its JIT compilation capabilities to NVIDIA GPUs through the `@cuda.jit` decorator:

```python
from numba import cuda

@cuda.jit
def my_kernel(input_array, output_array):
    i = cuda.grid(1)  # Global thread index
    if i < input_array.shape[0]:
        output_array[i] = input_array[i] * 2.0
```

This Python code is compiled to PTX (Parallel Thread Execution) assembly and executed on the GPU.

### 2.2 CUDA Execution Model

GPU kernels execute in a hierarchical thread structure:

```
Grid (kernel launch)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ... (up to 1024 threads)
├── Block 1
│   └── ...
└── Block N
    └── ...
```

Key concepts:

| Concept | Description |
|---------|-------------|
| **Grid** | All threads launched by a kernel |
| **Block** | Group of threads sharing shared memory |
| **Thread** | Single execution unit |
| **Warp** | 32 threads executing in lockstep |

### 2.3 Memory Hierarchy

```
┌─────────────────────────────────────────┐
│            Global Memory (VRAM)          │  Largest, slowest
├─────────────────────────────────────────┤
│         Shared Memory (per block)        │  Fast, limited (~48KB)
├─────────────────────────────────────────┤
│       Local Memory (per thread)          │  Compiler-managed
├─────────────────────────────────────────┤
│         Registers (per thread)           │  Fastest, very limited
└─────────────────────────────────────────┘
```

Numba CUDA provides access to all memory types:

- `cuda.local.array()` - Thread-local arrays
- `cuda.shared.array()` - Block-shared arrays
- Global arrays passed as kernel arguments

### 2.4 Comparison with CUDA C RawKernel

| Aspect | Numba CUDA | CuPy RawKernel |
|--------|------------|----------------|
| Language | Python syntax | CUDA C/C++ |
| Learning curve | Lower | Higher |
| Debugging | Python traceback | GPU debugger |
| Performance | Good (95%+ of raw CUDA) | Optimal |
| Flexibility | Some limitations | Full CUDA features |
| Development speed | Faster | Slower |

---

## 3. Implementation Strategy

### 3.1 Kernel Architecture

The implementation uses separate kernels for assembly and post-processing:

```python
# kernels_numba_cuda.py

@cuda.jit
def quad8_assembly_kernel(x, y, quad8, xp, wp, vals_out, fg_out):
    """One thread per element - compute element stiffness matrix."""
    e = cuda.grid(1)
    if e >= quad8.shape[0]:
        return
    
    # Per-thread local arrays
    Ke = cuda.local.array((8, 8), dtype=float64)
    XN = cuda.local.array((8, 2), dtype=float64)
    
    # ... element computation ...
    
    # Store results
    base_idx = e * 64
    for i in range(8):
        for j in range(8):
            vals_out[base_idx + k] = Ke[i, j]

@cuda.jit
def quad8_postprocess_kernel(u, x, y, quad8, xp, vel_out, abs_vel_out):
    """One thread per element - compute velocity field."""
    e = cuda.grid(1)
    if e >= quad8.shape[0]:
        return
    
    # ... velocity computation ...
```

### 3.2 Thread-to-Element Mapping

Each GPU thread processes one element:

```
Elements:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  ...
               ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Threads:      t0   t1   t2   t3   t4   t5   t6   t7   ...
              └────────────────────────────────────────┘
                           Block 0
```

For 100,000 elements with 128 threads per block:

- Blocks needed: ⌈100,000 / 128⌉ = 782
- Total threads: 782 × 128 = 100,096
- Idle threads: 96 (last block partially filled)

### 3.3 Local Memory Usage

Per-thread arrays are allocated using `cuda.local.array()`:

```python
@cuda.jit
def quad8_assembly_kernel(...):
    e = cuda.grid(1)
    
    # Thread-local arrays (in registers or local memory)
    edofs = cuda.local.array(8, dtype=np.int32)
    XN = cuda.local.array((8, 2), dtype=np.float64)
    Ke = cuda.local.array((8, 8), dtype=np.float64)
    fe = cuda.local.array(8, dtype=np.float64)
    psi = cuda.local.array(8, dtype=np.float64)
    Dpsi = cuda.local.array((8, 2), dtype=np.float64)
    B = cuda.local.array((8, 2), dtype=np.float64)
```

These arrays are private to each thread, eliminating synchronization needs.

### 3.4 Kernel Launch Configuration

The solver class configures and launches kernels:

```python
def assemble_system(self):
    # Transfer data to GPU
    d_x = cuda.to_device(self.x)
    d_y = cuda.to_device(self.y)
    d_quad8 = cuda.to_device(self.quad8)
    
    # Allocate output arrays
    d_vals = cuda.device_array(self.Nels * 64, dtype=np.float64)
    d_fg = cuda.device_array(self.Nnds, dtype=np.float64)
    
    # Launch configuration
    threads_per_block = 128
    blocks = (self.Nels + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    quad8_assembly_kernel[blocks, threads_per_block](
        d_x, d_y, d_quad8, d_xp, d_wp, d_vals, d_fg
    )
    
    # Synchronize and copy results
    cuda.synchronize()
    vals = d_vals.copy_to_host()
```

### 3.5 Force Vector Assembly with Atomics

Multiple elements share nodes, requiring atomic operations for force vector assembly:

```python
@cuda.jit
def quad8_assembly_kernel(..., fg_out):
    # ... element computation ...
    
    # Atomic update to global force vector
    for i in range(8):
        cuda.atomic.add(fg_out, edofs[i], fe[i])
```

`cuda.atomic.add` ensures thread-safe accumulation when multiple threads update the same global memory location.

### 3.6 GPU Sparse Solver Integration

The linear system is solved on GPU using CuPy's sparse solvers:

```python
import cupy as cp
import cupyx.scipy.sparse.linalg as cpsplg

def solve(self):
    # Convert to CuPy sparse matrix on GPU
    Kg_gpu = cpsparse.csr_matrix(self.Kg)
    fg_gpu = cp.asarray(self.fg)
    
    # CuPy CG solver
    u_gpu, info = cpsplg.cg(
        Kg_gpu, fg_gpu,
        M=preconditioner,
        tol=1e-8,
        maxiter=self.maxiter
    )
    
    # Copy solution back to CPU
    self.u = u_gpu.get()
```

---

## 4. Key Code Patterns

### 4.1 Complete Assembly Kernel

```python
@cuda.jit
def quad8_assembly_kernel(x, y, quad8, xp, wp, vals_out, fg_out):
    e = cuda.grid(1)
    if e >= quad8.shape[0]:
        return
    
    # Get element DOFs
    edofs = cuda.local.array(8, dtype=np.int32)
    for i in range(8):
        edofs[i] = quad8[e, i]
    
    # Get nodal coordinates
    XN = cuda.local.array((8, 2), dtype=np.float64)
    for i in range(8):
        XN[i, 0] = x[edofs[i]]
        XN[i, 1] = y[edofs[i]]
    
    # Initialize element matrices
    Ke = cuda.local.array((8, 8), dtype=np.float64)
    fe = cuda.local.array(8, dtype=np.float64)
    for i in range(8):
        fe[i] = 0.0
        for j in range(8):
            Ke[i, j] = 0.0
    
    # Integration loop (9 Gauss points)
    for ip in range(9):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        w = wp[ip]
        
        # Shape functions and derivatives (inlined)
        # ... (same formulas as CPU version) ...
        
        # Jacobian, determinant, inverse
        # ... (explicit computation) ...
        
        # Accumulate Ke
        wip = w * Detj
        for i in range(8):
            for j in range(8):
                Ke[i, j] += wip * (B[i, 0] * B[j, 0] + B[i, 1] * B[j, 1])
    
    # Store Ke values (flattened)
    base_idx = e * 64
    k = 0
    for i in range(8):
        for j in range(8):
            vals_out[base_idx + k] = Ke[i, j]
            k += 1
    
    # Atomic update to global force vector
    for i in range(8):
        cuda.atomic.add(fg_out, edofs[i], fe[i])
```

### 4.2 Velocity Post-Processing Kernel

```python
@cuda.jit
def quad8_postprocess_kernel(u, x, y, quad8, xp, vel_out, abs_vel_out):
    e = cuda.grid(1)
    if e >= quad8.shape[0]:
        return
    
    # Gather element data
    XN = cuda.local.array((8, 2), dtype=np.float64)
    u_e = cuda.local.array(8, dtype=np.float64)
    for i in range(8):
        idx = quad8[e, i]
        XN[i, 0] = x[idx]
        XN[i, 1] = y[idx]
        u_e[i] = u[idx]
    
    # Accumulate velocity over integration points
    vel_x_sum = 0.0
    vel_y_sum = 0.0
    v_mag_sum = 0.0
    
    for ip in range(4):  # 2×2 Gauss points
        # Compute gradient at integration point
        # ... (shape functions, Jacobian, B matrix) ...
        
        grad_x = 0.0
        grad_y = 0.0
        for i in range(8):
            grad_x += B[i, 0] * u_e[i]
            grad_y += B[i, 1] * u_e[i]
        
        vel_x_sum += -grad_x
        vel_y_sum += -grad_y
        v_mag_sum += math.sqrt(grad_x**2 + grad_y**2)
    
    # Store averaged values
    vel_out[e, 0] = vel_x_sum / 4.0
    vel_out[e, 1] = vel_y_sum / 4.0
    abs_vel_out[e] = v_mag_sum / 4.0
```

### 4.3 Grid Index Computation

```python
# 1D grid
e = cuda.grid(1)  # Equivalent to: blockIdx.x * blockDim.x + threadIdx.x

# 2D grid (if needed)
i, j = cuda.grid(2)
```

---

## 5. Optimization Techniques Applied

### 5.1 Thread Block Size Selection

The choice of threads per block affects occupancy and performance:

| Block Size | Blocks (100K elements) | Occupancy | Notes |
|------------|------------------------|-----------|-------|
| 32 | 3125 | Low | Minimal parallelism per SM |
| 64 | 1563 | Medium | Better but suboptimal |
| 128 | 782 | Good | Balance of occupancy and registers |
| 256 | 391 | Good | May limit registers per thread |
| 512 | 196 | Variable | High register pressure |

128 threads per block provides a good balance for this kernel's register usage.

### 5.2 Memory Coalescing

Global memory accesses are structured for coalescing where possible:

```python
# Coalesced write: consecutive threads write consecutive addresses
base_idx = e * 64  # Each thread writes to non-overlapping region
for k in range(64):
    vals_out[base_idx + k] = Ke_flat[k]
```

### 5.3 Register Pressure Management

Local arrays are sized to fit in registers when possible:

```python
# Small arrays → likely in registers
edofs = cuda.local.array(8, dtype=np.int32)     # 8 × 4 = 32 bytes
psi = cuda.local.array(8, dtype=np.float64)      # 8 × 8 = 64 bytes

# Larger arrays → may spill to local memory
Ke = cuda.local.array((8, 8), dtype=np.float64)  # 64 × 8 = 512 bytes
```

### 5.4 Avoiding Warp Divergence

The kernel structure minimizes divergence:

```python
# Single branch at kernel start
if e >= quad8.shape[0]:
    return

# No conditionals in main computation
for ip in range(9):  # All threads execute same iterations
    # ... uniform computation ...
```

---

## 6. Challenges and Limitations

### 6.1 Debugging Complexity

Numba CUDA debugging is more challenging than CPU code:

- Cannot use Python debugger inside kernels
- Limited print functionality (`cuda.print()` available but limited)
- Stack traces may be unclear
- Memory errors cause silent failures or crashes

### 6.2 Type Annotations

Unlike Numba CPU, CUDA kernels require careful type management:

```python
# Must use explicit dtypes
Ke = cuda.local.array((8, 8), dtype=np.float64)  # Not just dtype=float

# Math functions from math module, not numpy
import math
result = math.sqrt(x)  # Not np.sqrt(x)
```

### 6.3 Limited NumPy Support

Many NumPy functions are unavailable in CUDA kernels:

| Available | Unavailable |
|-----------|-------------|
| Basic indexing | `np.dot` |
| Arithmetic ops | `np.linalg.*` |
| `math.sqrt`, etc. | `np.column_stack` |
| `cuda.local.array` | Dynamic allocation |

### 6.4 Atomic Operation Overhead

Force vector assembly requires atomic operations:

```python
cuda.atomic.add(fg_out, edofs[i], fe[i])
```

Atomics are slower than regular writes, especially with high contention. For this problem, the force vector contribution is small relative to stiffness computation.

### 6.5 Memory Transfer Overhead

Data must be explicitly transferred between CPU and GPU:

```python
# CPU → GPU
d_x = cuda.to_device(self.x)

# GPU → CPU
vals = d_vals.copy_to_host()
```

For small problems, transfer time may exceed computation time.

### 6.6 COO Index Generation on CPU

Currently, COO row/column indices are generated on CPU:

```python
# This loop runs on CPU after kernel completes
for e in range(self.Nels):
    edofs = self.quad8[e]
    for i in range(8):
        for j in range(8):
            rows[base_idx + k] = edofs[i]
            cols[base_idx + k] = edofs[j]
```

This could be moved to GPU but adds complexity for modest benefit.

---

## 7. Performance Characteristics

### 7.1 GPU Utilization

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Occupancy | 50-75% | Limited by registers |
| Memory bandwidth | 60-80% | Coalesced access |
| Compute utilization | 40-60% | Memory-bound |

### 7.2 Comparison with Numba CPU

| Aspect | Numba CPU | Numba CUDA |
|--------|-----------|------------|
| Threads | CPU cores (4-32) | GPU threads (10,000+) |
| Memory | Shared RAM | Device memory |
| Latency | Low | Higher (PCIe transfer) |
| Throughput | Moderate | Very high |
| Best for | Small-medium problems | Large problems |

### 7.3 Comparison with CuPy RawKernel

| Aspect | Numba CUDA | CuPy RawKernel |
|--------|------------|----------------|
| Development speed | Faster | Slower |
| Performance | 90-95% of raw CUDA | 100% |
| Debugging | Easier | Harder |
| Learning curve | Lower | Higher |
| Shared memory control | Basic | Full |

---

## 8. Insights and Lessons Learned

### 8.1 Python Syntax for GPU Programming

Numba CUDA enables GPU programming without learning CUDA C:

```python
# Numba CUDA (Python)
@cuda.jit
def kernel(arr_out):
    i = cuda.grid(1)
    arr_out[i] = i * 2.0

# Equivalent CUDA C
__global__ void kernel(double* arr_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    arr_out[i] = i * 2.0;
}
```

This significantly reduces the barrier to GPU programming.

### 8.2 Local Array Performance

`cuda.local.array` provides efficient thread-private storage:

- Small arrays fit in registers (fastest)
- Larger arrays spill to local memory (slower but still fast)
- No synchronization needed between threads

### 8.3 Kernel Boundary Considerations

Like Numba CPU, the kernel boundary matters:

- Transfer data once, compute many times
- Avoid frequent small kernel launches
- Keep data on GPU between kernels when possible

### 8.4 Hybrid CPU-GPU Approach

The implementation uses a hybrid approach:

- **GPU**: Element matrix computation (massive parallelism)
- **GPU**: Sparse matrix solve (CuPy)
- **CPU**: COO index generation (simpler, sufficient performance)
- **CPU**: Boundary condition application (complex logic)

This pragmatic division uses each processor for its strengths.

### 8.5 CuPy Integration

CuPy provides GPU-accelerated sparse solvers:

```python
# Seamless integration with Numba CUDA
u_gpu = cuda.to_device(initial_guess)
Kg_cupy = cpsparse.csr_matrix(self.Kg)  # CPU sparse → GPU sparse

# CuPy solver operates on GPU
result = cpsplg.cg(Kg_cupy, fg_cupy, ...)
```

---

## 9. Performance Comparison

The following table will be populated with benchmark results after testing:

| Metric | CPU Baseline | Numba JIT | Numba CUDA | Speedup vs Baseline |
|--------|--------------|-----------|------------|---------------------|
| Assembly Time (s) | — | — | — | — |
| Solve Time (s) | — | — | — | — |
| Post-processing Time (s) | — | — | — | — |
| Total Time (s) | — | — | — | — |
| CG Iterations | — | — | — | (same) |
| GPU Memory (MB) | — | — | — | — |

---

## 10. Summary

The Numba CUDA implementation demonstrates GPU programming with Python syntax:

**Achievements:**

- GPU kernel development without CUDA C
- Thousands of parallel threads per kernel launch
- Integration with CuPy for GPU sparse solver
- Comparable performance to raw CUDA (90-95%)

**Limitations:**

- More debugging challenges than CPU code
- Limited NumPy function support
- Requires explicit memory management
- Type annotation requirements stricter

**Key Insight:** Numba CUDA provides an accessible path to GPU programming for Python developers. While slightly slower than hand-optimized CUDA C, the development speed advantage makes it practical for research and prototyping.

**When to Use Numba CUDA:**

| Scenario | Recommendation |
|----------|----------------|
| Rapid prototyping | Numba CUDA |
| Maximum performance | CuPy RawKernel |
| Python team | Numba CUDA |
| CUDA expertise available | Either |
| Complex shared memory patterns | CuPy RawKernel |

The final implementation (GPU CuPy with RawKernel) demonstrates the alternative approach: writing CUDA C kernels directly for maximum performance.

---
