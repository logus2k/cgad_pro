# Implementation 1: CPU Baseline

## 1. Overview

The CPU baseline implementation (`quad8_cpu_v3.py`) serves as the reference implementation against which all parallelized variants are measured. It provides a clear, sequential algorithm that prioritizes correctness and readability over performance, establishing both the functional specification and the performance baseline for the project.

| Attribute | Description |
|-----------|-------------|
| **Technology** | NumPy + SciPy |
| **Execution Model** | Single-threaded, sequential |
| **Role** | Reference implementation, correctness validation, performance baseline |
| **Source File** | `quad8_cpu_v3.py` |
| **Dependencies** | NumPy, SciPy, pandas, h5py |

---

## 2. Technology Background

### 2.1 NumPy and SciPy Ecosystem

The baseline implementation leverages Python's scientific computing stack:

- **NumPy**: Provides N-dimensional arrays and vectorized operations backed by optimized BLAS/LAPACK implementations
- **SciPy**: Offers sparse matrix data structures (`lil_matrix`, `csr_matrix`) and iterative solvers (Conjugate Gradient)
- **h5py/pandas**: Handle mesh I/O from HDF5 and Excel formats

### 2.2 Execution Characteristics

Python executes code through the CPython interpreter with the Global Interpreter Lock (GIL), which serializes execution of Python bytecode. However, NumPy operations release the GIL during computation, allowing underlying C/Fortran libraries to execute efficiently.

For FEM workloads, this means:

- **Element loops**: Execute sequentially in Python (GIL held)
- **Matrix operations**: Execute in optimized BLAS (GIL released during computation)
- **Sparse solver**: Executes in SciPy's compiled code (GIL released during major operations)

The baseline implementation makes no attempt to parallelize element loops, deliberately accepting sequential execution to establish a clear reference point.

### 2.3 Relevance for FEM

The sequential implementation serves several purposes:

1. **Algorithm clarity**: Each step is explicit and traceable
2. **Debugging reference**: Provides known-correct outputs for validation
3. **Profiling baseline**: Identifies computational bottlenecks before optimization
4. **Performance floor**: Establishes minimum expected performance

---

## 3. Implementation Strategy

### 3.1 Mesh Loading

Mesh data is loaded from HDF5 files using h5py, with fallback support for NPZ and Excel formats:

```python
with h5py.File(self.mesh_file, 'r') as f:
    self.x = np.array(f['x'], dtype=np.float64)
    self.y = np.array(f['y'], dtype=np.float64)
    self.quad8 = np.array(f['quad8'], dtype=np.int32)
```

HDF5 provides efficient binary I/O, minimizing mesh loading time relative to text-based formats.

### 3.2 System Assembly

Global matrix assembly follows the classical element-by-element approach:

1. Initialize sparse matrix in LIL (List of Lists) format for efficient incremental construction
2. Loop over all elements sequentially
3. For each element, compute the 8×8 stiffness matrix
4. Scatter element contributions to global matrix positions

```python
self.Kg = lil_matrix((self.Nnds, self.Nnds), dtype=np.float64)
self.fg = np.zeros(self.Nnds, dtype=np.float64)

for e in range(self.Nels):
    edofs = self.quad8[e]
    XN = np.column_stack((self.x[edofs], self.y[edofs]))
    Ke, fe = Elem_Quad8(XN, fL=0.0)
    
    for i in range(8):
        self.fg[edofs[i]] += fe[i]
        for j in range(8):
            self.Kg[edofs[i], edofs[j]] += Ke[i, j]
```

The LIL format is chosen for assembly because it supports efficient element insertion. After assembly completes, the matrix is converted to CSR format for efficient sparse matrix-vector products during the solve phase.

### 3.3 Boundary Condition Application

**Robin Boundary Conditions** (inlet):

Robin edges are identified geometrically by finding element edges where all three nodes lie on the minimum-x boundary:

```python
x_min = float(self.x.min())
boundary_nodes = set(np.where(np.abs(self.x - x_min) < self.bc_tolerance)[0])

for e in range(self.Nels):
    # Check each edge for boundary membership
    edges = [(n[0], n[4], n[1]), (n[1], n[5], n[2]), ...]
    for edge in edges:
        if all(k in boundary_nodes for k in edge):
            robin_edges.append(edge)
```

For each Robin edge, the `Robin_quadr` function computes the 3×3 boundary stiffness contribution and 3-element load contribution using 3-point Gauss-Legendre quadrature.

**Dirichlet Boundary Conditions** (outlet):

Fixed potential values at outlet nodes are enforced using the penalty method:

```python
PENALTY_FACTOR = 1.0e12
for n in exit_nodes:
    self.Kg[n, n] += PENALTY_FACTOR
```

The penalty method was chosen over row elimination for implementation simplicity, with the large penalty factor ($10^{12}$) ensuring the prescribed value is effectively enforced.

### 3.4 Linear System Solution

The solve stage employs several techniques to ensure robust convergence:

**Diagonal Equilibration**: The system is pre-scaled to improve conditioning:

$$\tilde{\mathbf{K}} = \mathbf{D}^{-1/2} \mathbf{K} \mathbf{D}^{-1/2}, \quad \tilde{\mathbf{f}} = \mathbf{D}^{-1/2} \mathbf{f}$$

where $\mathbf{D} = \text{diag}(\mathbf{K})$. After solving $\tilde{\mathbf{K}} \tilde{\mathbf{u}} = \tilde{\mathbf{f}}$, the solution is recovered as $\mathbf{u} = \mathbf{D}^{-1/2} \tilde{\mathbf{u}}$.

**Jacobi Preconditioner**: A diagonal preconditioner is applied within the CG iteration:

```python
diag_eq = Kg_eq.diagonal()
M_inv = 1.0 / diag_eq

def precond_jacobi(v):
    return M_inv * v

M = LinearOperator(Kg_eq.shape, precond_jacobi)
```

**SciPy CG Solver**: The `scipy.sparse.linalg.cg` function performs the iteration with callback-based progress monitoring:

```python
u_eq, self.solve_info = cg(
    Kg_eq, fg_eq,
    rtol=1e-8,
    maxiter=self.maxiter,
    M=M,
    callback=self.monitor
)
```

### 3.5 Post-Processing

Velocity fields are computed by evaluating the potential gradient at Gauss points within each element:

```python
for e in range(self.Nels):
    edofs = self.quad8[e]
    XN = np.column_stack((self.x[edofs], self.y[edofs]))
    
    for ip in range(4):  # 2×2 Gauss points
        B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
        grad = B.T @ self.u[edofs]
        vel_x[ip] = -grad[0]
        vel_y[ip] = -grad[1]
    
    self.vel[e] = [vel_x.mean(), vel_y.mean()]
    self.abs_vel[e] = v_ip.mean()
```

Pressure is then computed from Bernoulli's equation: $p = p_0 - \rho |\mathbf{v}|^2$.

---

## 4. Key Code Patterns

### 4.1 Element Stiffness Computation

The `Elem_Quad8` function encapsulates the core FEM computation:

```python
def Elem_Quad8(XN, fL):
    Ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)
    
    xp, wp = Genip2DQ(9)  # 9-point quadrature
    
    for ip in range(9):
        B, psi, Detj = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
        wip = wp[ip] * Detj
        Ke += wip * (B @ B.T)
        fe += fL * wip * psi
    
    return Ke, fe
```

This pattern—quadrature loop with shape function evaluation and matrix accumulation—is replicated across all implementations, with only the execution model varying.

### 4.2 Sparse Matrix Format Conversion

The transition from assembly to solve requires format conversion:

```python
# Assembly phase: LIL for efficient insertion
self.Kg = lil_matrix((self.Nnds, self.Nnds))
# ... assembly loop ...

# Solve phase: CSR for efficient SpMV
self.Kg = csr_matrix(self.Kg)
```

This two-phase approach balances insertion efficiency (LIL) against arithmetic efficiency (CSR).

### 4.3 Progress Monitoring

The `IterativeSolverMonitor` class tracks convergence and provides real-time feedback:

```python
class IterativeSolverMonitor:
    def __call__(self, xk):
        self.it += 1
        if self.it % self.every == 0:
            r = self.b - self.A @ xk
            res_norm = np.linalg.norm(r)
            # Report progress via callback
            if self.progress_callback:
                self.progress_callback.on_iteration(...)
```

This pattern enables the web interface to display live convergence curves regardless of solver implementation.

---

## 5. Optimization Techniques Applied

### 5.1 Sparse Matrix Selection

The choice of sparse formats reflects their computational characteristics:

| Format | Insertion | SpMV | Memory | Usage |
|--------|-----------|------|--------|-------|
| LIL (List of Lists) | O(1) amortized | O(nnz) | Higher | Assembly |
| CSR (Compressed Sparse Row) | O(n) | O(nnz) optimal | Lower | Solve |

### 5.2 Diagonal Equilibration

Pre-scaling the system matrix improves the condition number, reducing the number of CG iterations required for convergence. This is particularly important for problems with varying element sizes or material properties.

### 5.3 Vectorized Inner Operations

While the element loop is sequential, operations within each element leverage NumPy's vectorized arithmetic:

- `B @ B.T`: Matrix multiplication via BLAS
- `np.column_stack`: Efficient array construction
- `np.linalg.norm`: Vectorized norm computation

These operations execute in compiled code, partially offsetting the Python loop overhead.

---

## 6. Challenges and Limitations

### 6.1 Sequential Element Loop

The primary limitation is the sequential element loop during assembly. For a mesh with $N_{el}$ elements, the assembly stage requires $\mathcal{O}(N_{el})$ Python loop iterations, each involving:

- Array indexing and slicing
- Function call overhead (Python → NumPy → BLAS)
- Dictionary/sparse matrix insertion

This Python-level overhead becomes the dominant cost for large meshes.

### 6.2 GIL Constraints

Although NumPy releases the GIL during array operations, the GIL is held during:

- Loop iteration
- Function calls
- Sparse matrix indexing

Multi-threading within this implementation would provide minimal benefit due to GIL serialization.

### 6.3 Memory Traffic

The sequential access pattern prevents cache optimization. Each element computation:

1. Fetches nodal coordinates from scattered memory locations
2. Computes the element matrix
3. Scatters contributions to potentially distant global matrix positions

This access pattern has poor spatial locality, leading to cache misses on large problems.

### 6.4 Sparse Matrix Insertion Overhead

Despite LIL format's efficient insertion, Python-level indexing adds overhead:

```python
self.Kg[edofs[i], edofs[j]] += Ke[i, j]
```

Each insertion requires Python object manipulation, dictionary lookups, and dynamic memory allocation.

---

## 7. Performance Characteristics

### 7.1 Expected Scaling

The baseline implementation exhibits the following complexity:

| Stage | Complexity | Dominant Factor |
|-------|------------|-----------------|
| Load Mesh | O(N_nodes) | I/O bandwidth |
| Assemble | O(N_elements × 64 × 9) | Python loop overhead |
| Apply BC | O(N_boundary) | Small relative cost |
| Solve | O(iterations × nnz) | SpMV memory bandwidth |
| Post-process | O(N_elements × 8 × 4) | Python loop overhead |

### 7.2 Profiling Observations

Based on the implementation structure, the expected time distribution for large problems:

- **Assembly**: Dominant stage (50-70%) due to Python loop overhead
- **Solve**: Significant (20-40%) depending on iteration count
- **Post-processing**: Moderate (5-15%)
- **I/O and BC**: Minor (<5%)

### 7.3 Baseline Role

The CPU baseline establishes:

1. **Correctness reference**: All other implementations must produce identical results
2. **Performance floor**: Any parallel implementation must improve upon this timing
3. **Iteration count**: CG convergence behavior should be identical across implementations

---

## 8. Insights and Lessons Learned

### 8.1 Algorithm Clarity vs. Performance

The baseline prioritizes readability:

```python
for e in range(self.Nels):
    # Clear, traceable element processing
    edofs = self.quad8[e]
    XN = np.column_stack((self.x[edofs], self.y[edofs]))
    Ke, fe = Elem_Quad8(XN, fL=0.0)
```

This explicit structure made debugging straightforward and provided a clear specification for parallel implementations.

### 8.2 Sparse Format Trade-offs

The LIL→CSR conversion adds overhead but is essential:

- Direct CSR construction would require pre-computing the sparsity pattern
- COO assembly with duplicate summation is an alternative (used in parallel implementations)
- The conversion cost is amortized over many solver iterations

### 8.3 Preconditioning Importance

Initial experiments without preconditioning showed:

- 10-100× more iterations required
- Sensitivity to problem scaling
- Potential convergence failures

The Jacobi preconditioner, while simple, provides robust convergence with minimal implementation complexity.

### 8.4 Residual Monitoring

Computing the true residual ($\mathbf{r} = \mathbf{f} - \mathbf{K}\mathbf{u}$) rather than relying on the solver's internal estimate:

- Catches numerical issues early
- Provides consistent metrics across solver variants
- Adds minimal overhead (one SpMV per check)

---

## 9. Performance Comparison

The following table will be populated with benchmark results after testing:

| Metric | CPU Baseline | Notes |
|--------|--------------|-------|
| Assembly Time (s) | — | Reference |
| Solve Time (s) | — | Reference |
| Post-processing Time (s) | — | Reference |
| Total Time (s) | — | Reference |
| CG Iterations | — | Should match all implementations |
| Peak Memory (MB) | — | Reference |
| Speedup | 1.0× | Baseline definition |

---

## 10. Summary

The CPU baseline implementation provides a clear, correct, and measurable reference point for the project. Its sequential execution, while not optimal for performance, establishes:

- The algorithmic foundation shared by all implementations
- Correctness criteria for validation
- Performance floor for speedup calculations
- Profiling data identifying optimization targets

Key observations:

1. **Assembly dominates**: The element loop is the primary bottleneck
2. **Python overhead significant**: Function calls and indexing add measurable cost
3. **Vectorized operations help**: BLAS-backed array operations partially offset interpreted overhead
4. **Solver is memory-bound**: CG iteration time scales with sparse matrix memory traffic

The parallel implementations that follow address these limitations through threading, multiprocessing, JIT compilation, and GPU offloading, each with distinct trade-offs explored in subsequent sections.

---
