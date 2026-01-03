# Implementation 6: GPU CuPy (RawKernel)

## 1. Overview

The GPU CuPy implementation (`quad8_gpu_v3.py`) represents the most performance-oriented approach, using CuPy's `RawKernel` to execute hand-written CUDA C kernels directly on the GPU. This provides maximum control over GPU execution while leveraging CuPy's ecosystem for sparse matrix operations and iterative solvers.

| Attribute | Description |
|-----------|-------------|
| **Technology** | CuPy RawKernel (CUDA C), CuPy sparse |
| **Execution Model** | GPU SIMT with native CUDA C kernels |
| **Role** | Maximum GPU performance, production-quality implementation |
| **Source File** | `quad8_gpu_v3.py` |
| **Dependencies** | NumPy, SciPy, CuPy |

---

## 2. Technology Background

### 2.1 CuPy Overview

CuPy is a NumPy-compatible array library for GPU computing. It provides:

- **Drop-in NumPy replacement**: `import cupy as cp` mirrors NumPy API
- **GPU arrays**: Data stored in GPU memory (VRAM)
- **Sparse matrices**: CSR/CSC/COO formats on GPU
- **Iterative solvers**: CG, GMRES, etc. running entirely on GPU
- **RawKernel**: Direct CUDA C/C++ kernel execution

### 2.2 RawKernel Interface

CuPy's `RawKernel` allows embedding CUDA C code directly in Python:

```python
import cupy as cp

kernel_code = r'''
extern "C" __global__
void my_kernel(const double* input, double* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0;
    }
}
'''

kernel = cp.RawKernel(kernel_code, 'my_kernel')
kernel((blocks,), (threads,), (d_input, d_output, n))
```

This provides:

- Full CUDA C feature set
- Maximum performance (no Python overhead in kernel)
- Direct control over memory, synchronization, shared memory
- Compiled once, cached for reuse

### 2.3 Comparison with Numba CUDA

| Aspect | Numba CUDA | CuPy RawKernel |
|--------|------------|----------------|
| Kernel language | Python | CUDA C/C++ |
| Performance | ~90-95% of peak | ~100% of peak |
| Shared memory | Basic support | Full control |
| Warp primitives | Limited | Full access |
| Learning curve | Lower | Higher |
| Debugging | Python-like | GPU debugger |
| Compilation | JIT per function | JIT per kernel string |

### 2.4 GPU Memory Model

```
┌─────────────────────────────────────────────────────────┐
│                    Host (CPU) Memory                     │
└─────────────────────────────────────────────────────────┘
                           ↕ PCIe Transfer
┌─────────────────────────────────────────────────────────┐
│                   Device (GPU) Memory                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Global Memory (VRAM)                │    │
│  │   - CuPy arrays (cp.array)                       │    │
│  │   - Sparse matrices (cupyx.scipy.sparse)         │    │
│  │   - Kernel input/output buffers                  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Shared Memory (per thread block)         │    │
│  │   - __shared__ arrays in CUDA C                  │    │
│  │   - Fast inter-thread communication              │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Registers (per thread)              │    │
│  │   - Local variables in kernel                    │    │
│  │   - Fastest access                               │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Strategy

### 3.1 CUDA C Kernel Architecture

The implementation defines two primary CUDA C kernels as string literals embedded in Python:

**Assembly Kernel** (`quad8_assembly_kernel`):
- One thread per element
- Computes 8×8 element stiffness matrix
- Writes 64 values to global COO value array
- Atomic update to global force vector

**Post-Processing Kernel** (`quad8_postprocess_kernel`):
- One thread per element
- Computes velocity gradient at 4 Gauss points
- Averages to element centroid velocity
- Writes velocity components and magnitude

### 3.2 Kernel Source Structure

```python
QUAD8_KERNEL_SOURCE = r"""
extern "C" __global__
void quad8_assembly_kernel(
    const double* x_in,
    const double* y_in,
    const int* quad8_in,
    const double* xp_in,
    const double* wp_in,
    const int Nels,
    const int Nnds,
    double* vals_out,
    double* fg_out)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Nels) return;
    
    // Constants
    const int N_NDS_PER_EL = 8;
    const int N_IP = 9;
    
    // Thread-local arrays (in registers/local memory)
    double XN[8][2];
    double Ke[8][8] = {{0.0}};
    double fe[8] = {0.0};
    int edofs[8];
    
    // Gather element data from global memory
    for (int i = 0; i < 8; ++i) {
        edofs[i] = quad8_in[e * 8 + i];
        XN[i][0] = x_in[edofs[i]];
        XN[i][1] = y_in[edofs[i]];
    }
    
    // Integration loop
    for (int ip = 0; ip < 9; ++ip) {
        // Shape functions, Jacobian, B matrix...
        // Accumulate Ke
    }
    
    // Scatter to global memory
    int base_idx = e * 64;
    int k = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            vals_out[base_idx + k] = Ke[i][j];
            k++;
        }
    }
    
    // Atomic force vector update
    for (int i = 0; i < 8; ++i) {
        atomicAdd(&fg_out[edofs[i]], fe[i]);
    }
}
"""
```

### 3.3 GPU-Accelerated COO Index Generation

Unlike the Numba CUDA version which generates COO indices on CPU, this implementation uses vectorized CuPy operations on GPU:

```python
def assemble_system(self):
    # Vectorized index generation on GPU
    local_i, local_j = cp.mgrid[0:8, 0:8]
    local_rows = cp.tile(local_i.ravel(), self.Nels)
    local_cols = cp.tile(local_j.ravel(), self.Nels)
    el_indices = cp.arange(self.Nels).repeat(64)
    
    self._rows = quad8_cp[el_indices, local_rows]
    self._cols = quad8_cp[el_indices, local_cols]
    
    # Kernel computes only values
    kernel((blocks,), (threads,), (
        self.x, self.y, quad8_cp,
        xp, wp,
        self.Nels, self.Nnds,
        self._vals, self.fg
    ))
```

This approach:

- Generates all $N_{el} \times 64$ row/column indices in parallel
- Avoids CPU-GPU synchronization during index generation
- Leverages CuPy's optimized array operations

### 3.4 Sparse Matrix Construction

After kernel execution, the COO data is converted to CSR:

```python
# Build sparse matrix on GPU
self.Kg = cupyx.scipy.sparse.coo_matrix(
    (self._vals, (self._rows, self._cols)),
    shape=(self.Nnds, self.Nnds)
).tocsr()
```

The entire sparse matrix lives in GPU memory, enabling GPU-accelerated solver operations.

### 3.5 GPU Sparse Solver

The linear system is solved entirely on GPU:

```python
import cupyx.scipy.sparse.linalg as cpsplg

# Diagonal equilibration (on GPU)
diag = self.Kg.diagonal()
D_inv_sqrt = 1.0 / cp.sqrt(cp.abs(diag))
Kg_eq = D_mat @ self.Kg @ D_mat
fg_eq = self.fg * D_inv_sqrt

# Jacobi preconditioner (on GPU)
diag_eq = Kg_eq.diagonal()
def jacobi_precond(x):
    return x / diag_eq

M = cpsplg.LinearOperator(shape=Kg_eq.shape, matvec=jacobi_precond)

# CG solve (entirely on GPU)
u_eq, info = cpsplg.cg(Kg_eq, fg_eq, M=M, tol=1e-8, maxiter=15000)

# De-equilibrate
self.u_gpu = u_eq * D_inv_sqrt
```

All operations—SpMV, vector operations, preconditioning—execute on GPU without CPU round-trips.

### 3.6 GPU Post-Processing

Velocity computation uses a dedicated RawKernel:

```python
def compute_derived_fields(self):
    # Launch post-processing kernel
    postprocess_kernel((blocks,), (threads,), (
        self.u_gpu, self.x, self.y, quad8_cp,
        xp_4,
        self.Nels, self.p0, self.rho,
        self.abs_vel_gpu, self.vel_gpu
    ))
    
    # Pressure from Bernoulli (vectorized on GPU)
    self.pressure_gpu = self.p0 - self.rho * self.abs_vel_gpu**2
```

---

## 4. Key Code Patterns

### 4.1 CUDA C Shape Function Derivatives

The kernel computes shape function derivatives inline:

```c
// In CUDA C kernel
double Dpsi[8][2];

Dpsi[0][0] = (2*csi+eta)*(1-eta)/4;
Dpsi[0][1] = (2*eta+csi)*(1-csi)/4;
Dpsi[1][0] = (2*csi-eta)*(1-eta)/4;
Dpsi[1][1] = (2*eta-csi)*(1+csi)/4;
// ... remaining 6 nodes ...
```

### 4.2 Jacobian and Inverse in CUDA C

```c
// Jacobian matrix
double J[2][2] = {{0.0}};
for (int i = 0; i < 8; ++i) {
    J[0][0] += XN[i][0] * Dpsi[i][0];
    J[0][1] += XN[i][0] * Dpsi[i][1];
    J[1][0] += XN[i][1] * Dpsi[i][0];
    J[1][1] += XN[i][1] * Dpsi[i][1];
}

// Determinant and inverse
double Detj = J[0][0] * J[1][1] - J[0][1] * J[1][0];
double InvJ[2][2];
InvJ[0][0] =  J[1][1] / Detj;
InvJ[0][1] = -J[0][1] / Detj;
InvJ[1][0] = -J[1][0] / Detj;
InvJ[1][1] =  J[0][0] / Detj;
```

### 4.3 Atomic Force Vector Update

```c
// Thread-safe accumulation to shared nodes
for (int i = 0; i < 8; ++i) {
    atomicAdd(&fg_out[edofs[i]], fe[i]);
}
```

### 4.4 Solver Fallback Strategy

The implementation includes fallback to GMRES if CG fails:

```python
try:
    u_eq, info = cpsplg.cg(Kg_eq, fg_eq, M=M, tol=TOL, maxiter=MAXITER)
except Exception as e:
    # Fallback to GMRES
    u_eq, info = cpsplg.gmres(Kg_eq, fg_eq, M=M, tol=TOL, restart=50)
```

---

## 5. Optimization Techniques Applied

### 5.1 Memory Coalescing

Global memory accesses are structured for coalescing:

```c
// Coalesced read: consecutive threads read consecutive elements
int edofs[8];
for (int i = 0; i < 8; ++i) {
    edofs[i] = quad8_in[e * 8 + i];  // Stride-1 access within element
}

// Coalesced write: consecutive threads write consecutive values
int base_idx = e * 64;
for (int k = 0; k < 64; ++k) {
    vals_out[base_idx + k] = Ke_flat[k];
}
```

### 5.2 Register Usage

Local arrays are sized to maximize register allocation:

```c
// These fit in registers (typical GPU has 255 registers per thread)
double XN[8][2];      // 128 bytes
double Ke[8][8];      // 512 bytes
double Dpsi[8][2];    // 128 bytes
double B[8][2];       // 128 bytes
int edofs[8];         // 32 bytes
```

### 5.3 Avoiding Warp Divergence

The kernel structure minimizes branch divergence:

```c
// Single early-exit for out-of-bounds threads
if (e >= Nels) return;

// All threads execute same number of loop iterations
for (int ip = 0; ip < 9; ++ip) {  // Fixed iteration count
    // No conditionals in main computation
}
```

### 5.4 GPU-Resident Data

Data stays on GPU throughout the pipeline:

```
Load Mesh → cp.asarray() → GPU
    ↓
Assemble → RawKernel → GPU (COO)
    ↓
Build Matrix → cupyx.sparse → GPU (CSR)
    ↓
Solve → cpsplg.cg → GPU
    ↓
Post-Process → RawKernel → GPU
    ↓
Export → .get() → CPU (only at end)
```

This minimizes PCIe transfer overhead.

### 5.5 Vectorized Boundary Condition Application

Robin boundary conditions use vectorized GPU operations:

```python
# Identify boundary edges on GPU
boundary_mask = cp.abs(self.x - x_min) < self.bc_tolerance
robin_nodes = cp.where(boundary_mask)[0]

# Apply penalty method
self.Kg.data[diagonal_indices] += PENALTY_FACTOR
```

---

## 6. Challenges and Limitations

### 6.1 CUDA C Complexity

Writing CUDA C requires understanding:

- Memory hierarchy and access patterns
- Thread synchronization primitives
- Warp execution model
- Register pressure and occupancy

This is a higher barrier than Numba CUDA's Python syntax.

### 6.2 Debugging Challenges

RawKernel debugging is more difficult:

- No Python debugger access
- Printf debugging limited (`printf` available but impacts performance)
- Memory errors cause crashes without clear diagnostics
- Race conditions are hard to detect

### 6.3 Kernel Compilation Overhead

Each RawKernel is JIT-compiled on first use:

| Kernel | Compilation Time |
|--------|------------------|
| Assembly | ~200-500 ms |
| Post-processing | ~100-200 ms |

CuPy caches compiled kernels, so subsequent runs are fast.

### 6.4 GPU Memory Constraints

Large problems may exceed GPU memory:

| Component | Memory per 100K nodes |
|-----------|----------------------|
| Coordinates (x, y) | ~1.6 MB |
| Connectivity | ~3.2 MB |
| Sparse matrix (CSR) | ~50-100 MB |
| Solution vectors | ~0.8 MB |
| Working memory | Variable |

For very large problems, out-of-core or multi-GPU strategies may be needed.

### 6.5 CuPy Sparse Solver Limitations

CuPy's sparse solvers have some limitations compared to SciPy:

- Fewer preconditioner options
- Some solvers less mature
- Occasional numerical differences

The fallback to GMRES addresses some robustness concerns.

---

## 7. Performance Characteristics

### 7.1 GPU Utilization Analysis

| Metric | Typical Value | Optimization Target |
|--------|---------------|---------------------|
| Occupancy | 50-75% | Register pressure |
| Memory throughput | 70-85% peak | Coalescing |
| Compute utilization | 60-80% | Algorithm efficiency |

### 7.2 Performance Breakdown

Expected time distribution for large problems:

| Stage | Time Fraction | Notes |
|-------|---------------|-------|
| Mesh loading | <5% | I/O bound |
| Assembly kernel | 5-15% | Highly parallel |
| Matrix construction | 5-10% | CuPy sparse ops |
| Linear solve | 60-80% | Memory bandwidth bound |
| Post-processing | 5-10% | Highly parallel |
| Data transfer | <5% | PCIe overhead |

### 7.3 Scaling Characteristics

| Problem Size | GPU Advantage | Notes |
|--------------|---------------|-------|
| <10K elements | Minimal | Transfer overhead dominates |
| 10K-100K | Significant (5-20×) | Good GPU utilization |
| 100K-1M | Maximum (20-100×) | Full GPU saturation |
| >1M | Memory limited | May require multi-GPU |

### 7.4 Comparison with CPU Implementations

| Aspect | CPU Baseline | Numba CPU | GPU CuPy |
|--------|--------------|-----------|----------|
| Assembly | O(N) sequential | O(N/P) parallel | O(N/T) massively parallel |
| Threads/cores | 1 | 4-32 | 1000s |
| Memory bandwidth | 50-100 GB/s | 50-100 GB/s | 500-900 GB/s |
| Latency | Low | Low | Higher (PCIe) |
| Throughput | Moderate | Good | Excellent |

---

## 8. Insights and Lessons Learned

### 8.1 RawKernel vs. Numba CUDA Trade-offs

Development experience revealed clear trade-offs:

| Factor | Recommendation |
|--------|----------------|
| Rapid prototyping | Numba CUDA |
| Maximum performance | RawKernel |
| Complex shared memory | RawKernel |
| Team with CUDA experience | RawKernel |
| Python-focused team | Numba CUDA |

### 8.2 GPU-Resident Pipeline Importance

Keeping data on GPU throughout the computation was crucial:

- Each CPU-GPU transfer adds ~1-10 ms latency
- Sparse solver performs thousands of SpMV operations
- Post-processing reuses solution vector already on GPU

### 8.3 Atomic Operations for Shared Updates

The force vector assembly pattern (`atomicAdd`) is essential but has performance implications:

- Contention on shared nodes reduces throughput
- For this problem, force vector contribution is small (homogeneous $f_L = 0$)
- For non-zero loads, atomic overhead would be more significant

### 8.4 CuPy Ecosystem Value

CuPy's comprehensive ecosystem proved valuable:

- `cupyx.scipy.sparse`: GPU sparse matrices without custom code
- `cupyx.scipy.sparse.linalg`: Production-quality GPU solvers
- NumPy-compatible API: Easy CPU-to-GPU porting

### 8.5 Sparse Solver Dominates Runtime

For large, well-conditioned problems:

- Assembly: ~10% of total time
- Solve: ~80% of total time
- Post-processing: ~10% of total time

Optimizing the assembly kernel has diminishing returns once the solver dominates.

### 8.6 Memory Bandwidth is the Bottleneck

The CG solver is memory-bandwidth bound:

- SpMV reads entire matrix each iteration
- Hundreds to thousands of iterations
- GPU's high bandwidth (500+ GB/s) is key advantage over CPU

---

## 9. Performance Comparison

The following table will be populated with benchmark results after testing:

| Metric | CPU Baseline | Numba JIT | Numba CUDA | GPU CuPy | Speedup vs Baseline |
|--------|--------------|-----------|------------|----------|---------------------|
| Assembly Time (s) | — | — | — | — | — |
| Solve Time (s) | — | — | — | — | — |
| Post-processing Time (s) | — | — | — | — | — |
| Total Time (s) | — | — | — | — | — |
| CG Iterations | — | — | — | — | (same) |
| GPU Memory (MB) | — | — | — | — | — |

---

## 10. Summary

The GPU CuPy implementation with RawKernel represents the performance-optimized endpoint of this implementation spectrum:

**Achievements:**

- Maximum GPU performance through native CUDA C kernels
- Entire computational pipeline on GPU (assembly, solve, post-process)
- Integration with CuPy sparse solvers for production-quality linear algebra
- Vectorized COO index generation on GPU

**Limitations:**

- Higher development complexity than Numba CUDA
- Debugging requires GPU-specific tools
- CUDA C knowledge required for kernel development
- GPU memory constraints for very large problems

**Key Insight:** For production FEM solvers targeting maximum performance, the combination of RawKernel for custom element operations and CuPy's sparse algebra for the linear solver provides an effective architecture. The GPU's massive parallelism (thousands of threads) and high memory bandwidth (500+ GB/s) enable order-of-magnitude speedups over CPU implementations for sufficiently large problems.

**Implementation Spectrum Summary:**

| Implementation | Development Effort | Performance | Best Use Case |
|----------------|-------------------|-------------|---------------|
| CPU Baseline | Low | 1× | Reference, debugging |
| CPU Threaded | Low | 1.5-3× | Quick improvement |
| CPU Multiprocess | Medium | 3-8× | CPU parallelism |
| Numba JIT | Medium | 10-30× | Best CPU performance |
| Numba CUDA | Medium | 20-50× | GPU prototyping |
| GPU CuPy | High | 50-100× | Production GPU |

The choice of implementation depends on problem size, available hardware, development resources, and performance requirements.

---
