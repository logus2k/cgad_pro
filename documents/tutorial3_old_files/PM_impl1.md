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
