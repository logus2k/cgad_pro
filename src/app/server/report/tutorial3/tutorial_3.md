# High-Performance GPU-Accelerated Finite Element Analysis

**Project Tutorial #2 Report**

# 1. General Overview - Finite Element Method

The Finite Element Method (FEM) is a numerical technique widely used to approximate solutions of partial differential equations arising in engineering and scientific problems. Its main strength lies in its ability to handle complex geometries, heterogeneous materials, and general boundary conditions, which are often intractable using analytical approaches.

The fundamental idea of FEM is to replace a continuous problem by a discrete one. The physical domain is subdivided into a finite number of smaller regions, called elements, over which the unknown field is approximated using interpolation functions. By assembling the contributions of all elements, the original continuous problem is transformed into a system of algebraic equations that can be solved numerically.

![](images/documents/tutorial2/image1.png)



Because of this formulation, FEM naturally maps to linear algebra operations and therefore constitutes an ideal candidate for high-performance computing and parallel execution.

## 1.1. Classes of Problems Addressed by FEM

From a mathematical standpoint, FEM can be applied to several classes of partial differential equations, each associated with different physical phenomena and computational characteristics.

Elliptic problems describe steady-state systems in which no time dependence exists. Typical examples include heat conduction, electrostatics, diffusion, and potential flow. These problems lead to well-conditioned linear systems that are symmetric and positive definite, making them particularly suitable for iterative solvers.

Parabolic problems introduce time dependence and describe transient diffusion processes, such as heat propagation. Their numerical solution requires both spatial discretization and time integration, increasing computational complexity.

Hyperbolic problems arise in wave propagation and dynamic systems, such as structural vibrations and acoustics. These problems are often dominated by stability constraints and time-stepping considerations.

The present work focuses exclusively on elliptic problems, more specifically on the Laplace equation:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

where $u(x, y)$ represents the velocity potential field over a bounded domain $\Omega \subset \mathbb{R}^2$ with boundary $\Gamma = \partial\Omega$.

This equation governs a wide range of physical phenomena including steady-state heat transfer, electrostatics, mass diffusion, and incompressible potential flow. 

| Application Domain | Physical Interpretation of $u$ |
|-------------------|-------------------------------|
| Incompressible irrotational flow | Velocity potential |
| Steady-state heat conduction | Temperature field |
| Electrostatics | Electric potential |
| Diffusion (steady-state) | Concentration field |


The choice of Laplace's equation as the benchmark problem provides several advantages for performance analysis:

1. **Mathematical well-posedness**: Unique solution guaranteed under appropriate boundary conditions
2. **Symmetric positive-definite system**: Enables use of efficient iterative solvers (CG)
3. **Predictable convergence**: Facilitates consistent timing measurements
4. **Representative workload**: Assembly and solve patterns typical of broader FEM applications


This equation captures all the essential computational challenges of FEM and is therefore well suited for performance-oriented studies.

---

## 1.2. Mathematical Formulation


### 1.2.1. Spatial Discretization and Element Choice




In FEM, the continuous domain is discretized into a finite number of elements connected at nodes. Within each element, the unknown field is approximated using shape functions defined over the element’s geometry.

![](images/documents/tutorial2/image2.png)

Several element types exist, depending on dimensionality and interpolation order. In two dimensions, common choices include triangular and quadrilateral elements, with either linear or higher-order interpolation.

In this work, eight-node quadrilateral elements (Quad-8) are used. These elements employ quadratic interpolation functions, allowing higher accuracy compared to linear elements while preserving numerical stability. 



Each element comprises 8 nodes: 4 corner nodes and 4 mid-edge nodes.
Node numbering follows the standard convention with counter-clockwise ordering starting from the bottom-left corner.

| Node | Type | Parametric Coordinates $(\xi, \eta)$ |
|------|------|-------------------------------------|
| 1 | Corner | $(-1, -1)$ |
| 2 | Corner | $(+1, -1)$ |
| 3 | Corner | $(+1, +1)$ |
| 4 | Corner | $(-1, +1)$ |
| 5 | Mid-edge | $(0, -1)$ |
| 6 | Mid-edge | $(+1, 0)$ |
| 7 | Mid-edge | $(0, +1)$ |
| 8 | Mid-edge | $(-1, 0)$ |


The increased number of nodes per element leads to larger element matrices and higher arithmetic intensity during numerical integration, making them particularly suitable for performance evaluation on modern hardware.

The use of Quad-8 elements also provides a realistic representation of engineering-grade FEM simulations, where higher-order elements are commonly employed to improve solution accuracy.



### 1.2.2. Variational Formulation and Algebraic Representation

The FEM formulation begins by expressing the governing differential equation in weak form. For the Laplace equation, this leads to the variational problem,

$$\int_{\Omega} \nabla v \cdot \nabla \phi \, d\Omega = \int_{\Gamma} v\, q \, d\Gamma$$

where $\phi$ is the unknown scalar field, $v$ is a test function, and $q$ represents prescribed boundary fluxes.
After discretization using shape functions, the weak formulation results in a linear system of equations:

$$Ku=f $$

where:
- $\mathbf{K} \in \mathbb{R}^{N_{dof} \times N_{dof}}$ is the global stiffness matrix (sparse, symmetric, positive-definite)
- $\mathbf{u} \in \mathbb{R}^{N_{dof}}$ is the vector of nodal unknowns
- $\mathbf{f} \in \mathbb{R}^{N_{dof}}$ is the global load vector

The global stiffness matrix is then assembled from element-level contributions of the form:

$$\mathbf{K}^{(e)} = \int_{\Omega_e} (\nabla \mathbf{N})^T \mathbf{D} (\nabla \mathbf{N}) \, d\Omega$$ 

where $N$ denotes the shape functions and $D$ represents the material or conductivity matrix. The resulting global matrix is sparse, symmetric, and positive definite, which strongly influences solver choice and performance behavior.


![DESCRIPTION](images/documents/tutorial2/Sparse FEM Matrix.svg)


### 1.2.3. Boundary Conditions

Boundary conditions (BCs) specify the constraints and interactions imposed on the boundaries of a numerical model and are fundamental to obtaining a well-posed and solvable problem. They define how the system responds at its limits and ensure that the mathematical formulation admits a unique and physically consistent solution. In practical applications, boundary conditions are selected to reflect the real physical supports, loads, or environmental interactions acting on the domain. The most commonly identified categories of boundary conditions are essential (Dirichlet), natural (Neumann), and mixed (Robin) boundary conditions, and an appropriate combination of these is required to accurately represent the problem being analyzed.

#### 1.2.3.1. Dirichlet Boundary Conditions

Dirichlet boundary conditions specify fixed potential values at designated boundary nodes:

$$u = \bar{u} \quad \text{on } \Gamma_D$$

These are implemented using row/column elimination: for each constrained degree of freedom $i$ with prescribed value $\bar{u}_i$:

1. Set $K_{ii} = 1$ and $K_{ij} = K_{ji} = 0$ for $j \neq i$
2. Set $f_i = \bar{u}_i$
3. Modify $f_j \leftarrow f_j - K_{ji} \bar{u}_i$ for all $j \neq i$ (to preserve symmetry)

In the project context, Dirichlet conditions are applied at outlet boundaries where the potential is fixed.

#### 1.2.3.2. Robin Boundary Conditions

Robin boundary conditions combine flux and potential contributions at inlet boundaries:

$$p \cdot u + \frac{\partial u}{\partial n} = \gamma \quad \text{on } \Gamma_R$$

where $p$ is a coefficient and $\gamma$ represents the prescribed combination of flux and potential.



#### 1.2.3.3. Boundary Detection

Boundary nodes are identified geometrically based on coordinate tolerance. The implementation detects:

- **Inlet boundary**: Left edge of domain (minimum $x$ coordinate)
- **Outlet boundary**: Right edge of domain (maximum $x$ coordinate)

A tolerance parameter (`bc_tolerance = 1e-9`) handles floating-point precision in coordinate comparisons.

## 1.3. Linear Solver Strategy

### 1.3.1. Conjugate Gradient Method

All implementations use the Conjugate Gradient (CG) method for solving the linear system $\mathbf{K}\mathbf{u} = \mathbf{f}$. CG is particularly suitable for this application because:

1. **Symmetric positive-definite system**: The stiffness matrix $\mathbf{K}$ from elliptic PDEs satisfies the SPD requirement
2. **Memory efficiency**: Only matrix-vector products required, no explicit factorization
3. **Predictable convergence**: Error reduction bounded by condition number
4. **Parallelization potential**: Core operations (SpMV, dot products, axpy) are data-parallel

The CG algorithm generates a sequence of iterates $\mathbf{u}^{(k)}$ that minimize the $\mathbf{K}-norm$ Ajuof the error over a Krylov subspace of increasing dimension.

### 1.3.2. Jacobi Preconditioning

All implementations apply Jacobi (diagonal) preconditioning to accelerate convergence:

$$\mathbf{M} = \text{diag}(\mathbf{K})$$

The preconditioned system becomes:

$$\mathbf{M}^{-1/2} \mathbf{K} \mathbf{M}^{-1/2} \tilde{\mathbf{u}} = \mathbf{M}^{-1/2} \mathbf{f}$$

The Jacobi preconditioner was chosen deliberately for this performance study:

| Characteristic | Benefit |
|----------------|---------|
| Element-wise operations | Trivially parallelizable across all execution models |
| No fill-in | Memory footprint identical to diagonal extraction |
| No factorization | Setup cost $\mathcal{O}(n)$ |
| Implementation-independent | Does not favor any particular parallelization strategy |

More sophisticated preconditioners (ILU, AMG) might provide faster convergence but would introduce implementation-dependent performance variations that complicate fair comparison.

### 1.3.3. Solver Configuration

The following parameters are held constant across all implementations:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | Conjugate Gradient | Optimal for SPD systems |
| Preconditioner | Jacobi (diagonal) | Parallelizes uniformly |
| Relative tolerance | $10^{-8}$ | Engineering accuracy |
| Absolute tolerance | $0$ | Rely on relative criterion |
| Maximum iterations | No limit | Sufficient in order to simulate all test problems |
| Progress reporting | Every 50 iterations | Balance monitoring vs. overhead |

 ### 1.3.4. Convergence Monitoring

The solver monitors convergence using the relative residual norm:

$$\text{rel\_res} = \frac{\|\mathbf{r}^{(k)}\|_2}{\|\mathbf{b}\|_2} = \frac{\|\mathbf{f} - \mathbf{K}\mathbf{u}^{(k)}\|_2}{\|\mathbf{f}\|_2}$$

Convergence is declared when $\text{rel\_res} < 10^{-8}$ or the iteration count exceeds the maximum.

---

## 1.4. Post-Processing: Derived Fields

### 1.4.1. Velocity Field Computation

The velocity field is computed as the negative gradient of the potential:

$$\mathbf{v} = -\nabla u = -\begin{bmatrix} \frac{\partial u}{\partial x} \\ \frac{\partial u}{\partial y} \end{bmatrix}$$

For each element, the gradient is evaluated at 4 Gauss points and averaged:

$$\mathbf{v}_e = \frac{1}{4} \sum_{p=1}^{4} \left( -\mathbf{B}_p^T \mathbf{u}_e \right)$$

where $\mathbf{u}_e$ is the vector of nodal solution values for element $e$.

### 1.4.2. Velocity Magnitude

The velocity magnitude per element:

$$|\mathbf{v}|_e = \frac{1}{4} \sum_{p=1}^{4} \sqrt{v_{x,p}^2 + v_{y,p}^2}$$

### 1.4.3. Pressure Field

Pressure is computed from Bernoulli's equation for incompressible flow:

$$p = p_0 - \frac{1}{2} \rho |\mathbf{v}|^2$$

where:
- $p_0 = 101328.8$ Pa (reference pressure)
- $\rho = 0.6125$ kg/m³ (fluid density)

These constants are configurable parameters in the solver constructor.

---

## 1.5. Computational Pipeline of the Finite Element Method

From a computational perspective, the FEM workflow can be decomposed into a sequence of well-defined stages, each exhibiting distinct performance characteristics.

```java
1. Load mesh data

2. Initialize global stiffness matrix K ← 0

3. Initialize global load vector f ← 0

4. for each element e in mesh do
       Compute element stiffness matrix Ke
       Compute element load vector fe
       Assemble Ke into K
       Assemble fe into f
   end for

5. Apply boundary conditions to K and f

6. Solve linear system:
       K * u = f

7. Compute Derived

8. Export Results
```

The process begins with mesh loading, where nodal coordinates, element connectivity, and boundary information are read into memory. Although this stage is not computationally intensive, it defines data layout and memory access patterns for all subsequent steps.


Element-level assembly follows, during which local stiffness matrices and load vectors are computed using numerical integration. This stage involves a large number of floating-point operations and is inherently parallel, as each element can be processed independently. As such, it represents one of the most computationally intensive parts of the FEM pipeline.

```java
for each element e do
    Retrieve node coordinates
    Compute Jacobian and its determinant

    for each Gauss integration point gp do
        Evaluate shape functions N
        Evaluate derivatives ∇N
        Compute local stiffness contribution:
            Ke += (∇Nᵀ · D · ∇N) * det(J) * w_gp
    end for
end for
```

The local contributions are then assembled into the global sparse matrix. This step involves indirect memory accesses and accumulation of values at shared locations, making it sensitive to memory bandwidth and synchronization overheads. Efficient implementation of this phase is crucial for overall performance, particularly on GPU architectures.

```java
for each element e do
    for i = 1 to n_nodes_per_element do
        for j = 1 to n_nodes_per_element do
            I = global_index(e, i)
            J = global_index(e, j)
            K[I, J] += Ke[i, j]
        end for
    end for
end for
```

Boundary conditions are subsequently applied. Dirichlet conditions enforce prescribed values by modifying the system matrix and right-hand side, while Neumann conditions introduce additional contributions to the load vector. Although conceptually simple, this step must be carefully implemented to preserve numerical correctness.


```java
for each prescribed node i do
    K[i, :] = 0
    K[i, i] = 1
    f[i] = prescribed_value
end for
```

Once the system is fully assembled, the resulting linear system is solved using an iterative solver. This stage usually dominates execution time, as it involves repeated sparse matrix-vector multiplications and vector operations.

```java
Initialize u₀
r₀ = f - K u₀
p₀ = r₀

for k = 0 until convergence do
    α = (rᵀ r) / (pᵀ K p)
    u = u + α p
    r_new = r - α K p

    if ||r_new|| < tolerance then
        break
    end if

    β = (r_newᵀ r_new) / (rᵀ r)
    p = r_new + β p
    r = r_new
end for
```

Finally, post-processing is performed to reconstruct the solution field, compute derived quantities, and generate visualizations. While less computationally demanding, this step is essential for validating results and analyzing physical behavior.

### 1.5.1. Parallelization Targets

The assembly stage exhibits the highest parallelization potential because each element's stiffness matrix can be computed independently. The solve stage benefits from parallel SpMV but faces memory bandwidth constraints characteristic of sparse computations. Post-processing mirrors assembly in its parallel structure.

| Stage | Computational Pattern | Parallelization Opportunity |
|-------|----------------------|----------------------------|
| Load Mesh | I/O bound | Limited (disk/memory bandwidth) |
| Assemble System | Element-independent loops | **High** (embarrassingly parallel) |
| Apply BCs | Sequential modifications | Low (small fraction of runtime) |
| Solve System | Sparse matrix-vector products | **Medium** (memory bandwidth limited) |
| Compute Derived | Element-independent loops | **High** (embarrassingly parallel) |
| Export Results | I/O bound | Limited (disk bandwidth) |




---


# 2. Software Architecture

## 2.1 Solver Interface Contract

All solver classes implement a consistent interface, enabling the unified `SolverWrapper` to instantiate any implementation interchangeably:

```python
class Quad8FEMSolver:
    """Base interface implemented by all solver variants."""
    
    def __init__(
        self,
        mesh_file: Path | str,
        p0: float = 101328.8,        # Reference pressure
        rho: float = 0.6125,          # Fluid density
        gamma: float = 2.5,           # Robin BC coefficient
        rtol: float = 1e-8,           # CG relative tolerance
        maxiter: int = 15000,         # Maximum CG iterations
        bc_tolerance: float = 1e-9,   # BC detection tolerance
        cg_print_every: int = 50,     # Progress interval
        verbose: bool = True,         # Console output
        progress_callback = None      # Real-time monitoring
    ):
        """Initialize solver with mesh and parameters."""
        ...
    
    def run(
        self,
        output_dir: Path | str = None,
        export_file: Path | str = None
    ) -> Dict[str, Any]:
        """Execute complete FEM workflow and return results."""
        ...
```

This interface contract ensures that switching between CPU, GPU, Numba, and CUDA implementations requires only changing the solver type parameter, with no modifications to calling code.

## 2.2 SolverWrapper: Unified Factory

The `SolverWrapper` class provides a unified factory interface for solver instantiation:

```python
class SolverWrapper:
    """Unified interface for all solver implementations."""
    
    SOLVER_TYPES = [
        "cpu",           # NumPy/SciPy baseline
        "cpu_threaded",  # ThreadPoolExecutor
        "cpu_multiprocess", # ProcessPoolExecutor
        "numba",         # Numba JIT CPU
        "numba_cuda",    # Numba CUDA kernels
        "gpu",           # CuPy with RawKernel
        "auto"           # Auto-detect best available
    ]
    
    def __init__(self, solver_type: str, params: dict, progress_callback=None):
        # Instantiate appropriate solver based on type
        ...
    
    def run(self) -> Dict[str, Any]:
        # Execute solver with memory tracking
        ...
    
    @staticmethod
    def get_available_solvers() -> list:
        # Detect available implementations based on installed packages
        ...
```

The `auto` mode detects the best available solver by checking for GPU availability (CuPy import success).

## 2.3 Progress Callback System

Real-time monitoring is provided through a callback interface that all solvers invoke at consistent pipeline points:

```python
class ProgressCallback:
    """Interface for real-time solver monitoring."""
    
    def on_stage_start(self, stage: str) -> None:
        """Called when a pipeline stage begins."""
        ...
    
    def on_stage_complete(self, stage: str, duration: float) -> None:
        """Called when a pipeline stage completes."""
        ...
    
    def on_mesh_loaded(
        self, nodes: int, elements: int,
        coordinates: dict, connectivity: list
    ) -> None:
        """Called after mesh loading with mesh metadata."""
        ...
    
    def on_iteration(
        self, iteration: int, max_iterations: int,
        residual: float, relative_residual: float,
        elapsed_time: float, etr_seconds: float
    ) -> None:
        """Called during CG iterations with convergence data."""
        ...
    
    def on_solution_increment(
        self, iteration: int, solution: ndarray
    ) -> None:
        """Called periodically with partial solution for visualization."""
        ...
    
    def on_error(self, stage: str, message: str) -> None:
        """Called when an error occurs."""
        ...
```

This callback system enables the web interface to display live progress, convergence curves, and intermediate solution fields regardless of which solver implementation is executing.

## 2.4 Timing Instrumentation

Each solver records per-stage wall-clock time using high-resolution timers (`time.perf_counter()`). The timing dictionary structure is consistent across all implementations:

```python
timing_metrics = {
    'load_mesh': float,        # Mesh loading time (seconds)
    'assemble_system': float,  # Global assembly time
    'apply_bc': float,         # Boundary condition application
    'solve_system': float,     # Linear solver time
    'compute_derived': float,  # Post-processing time
    'total_workflow': float,   # Sum of above stages
    'total_program_time': float # Wall-clock from initialization
}
```

This granular timing enables identification of which stages benefit most from each parallelization strategy.

---

## 2.5 Result Format

All solvers return a standardized dictionary structure containing solution fields, convergence status, timing metrics, and metadata:

```python
results = {
    # Solution fields
    'u': ndarray,           # Nodal potential (Nnodes,)
    'vel': ndarray,         # Velocity vectors (Nelements, 2)
    'abs_vel': ndarray,     # Velocity magnitude (Nelements,)
    'pressure': ndarray,    # Pressure field (Nelements,)
    
    # Convergence status
    'converged': bool,      # True if tolerance achieved
    'iterations': int,      # CG iterations performed
    
    # Performance metrics
    'timing_metrics': {
        'load_mesh': float,
        'assemble_system': float,
        'apply_bc': float,
        'solve_system': float,
        'compute_derived': float,
        'total_workflow': float,
        'total_program_time': float,
    },
    
    # Solution statistics
    'solution_stats': {
        'u_range': [float, float],  # [min, max]
        'u_mean': float,
        'u_std': float,
        'final_residual': float,
        'relative_residual': float,
    },
    
    # Problem metadata
    'mesh_info': {
        'nodes': int,
        'elements': int,
        'matrix_nnz': int,
        'element_type': 'quad8',
        'nodes_per_element': 8,
    },
    
    # Solver configuration
    'solver_config': {
        'linear_solver': 'cg',
        'tolerance': float,
        'max_iterations': int,
        'preconditioner': 'jacobi',
    },
}
```

The `timing_metrics` dictionary is essential for performance analysis, providing per-stage timing that reveals which computational phases benefit most from each parallelization strategy.

---

## 2.6. Shared Computational Modules

### 2.6.1. Module Organization

Each implementation variant includes adapted versions of four core computational modules. While the mathematical operations are identical, each version is optimized for its execution model:

| Module | Purpose | CPU (NumPy) | Numba JIT | CuPy GPU | CUDA Kernel |
|--------|---------|-------------|-----------|----------|-------------|
| `shape_n_der8` | Shape functions, derivatives, Jacobian | `np.zeros`, `np.linalg` | `@njit`, explicit loops | `cp.zeros`, `cp.linalg` | Inlined in kernel |
| `genip2dq` | Gauss point coordinates and weights | `np.array` constants | `@njit`, return arrays | `cp.array` constants | Helper function |
| `elem_quad8` | Element stiffness matrix | `np.outer`, matrix ops | `@njit`, nested loops | `cp.outer`, matrix ops | Full kernel |
| `robin_quadr` | Robin BC edge integration | NumPy loops | `@njit` loops | CuPy loops | CPU fallback |

### 2.6.2. Implementation Adaptations

**NumPy (CPU Baseline)**

Uses vectorized operations and BLAS/LAPACK routines through NumPy:

```python
# Jacobian computation
jaco = XN.T @ Dpsi  # Matrix multiplication
Detj = np.linalg.det(jaco)
Invj = np.linalg.inv(jaco)
B = Dpsi @ Invj

# Stiffness accumulation
Ke += wip * (B @ B.T)  # Outer product
```

**Numba JIT**

Replaces NumPy operations with explicit loops for LLVM optimization:

```python
@njit(cache=True)
def shape_n_der8(XN, csi, eta):
    # Explicit Jacobian computation
    jaco = np.zeros((2, 2), dtype=np.float64)
    for i in range(8):
        jaco[0, 0] += XN[i, 0] * Dpsi[i, 0]
        jaco[0, 1] += XN[i, 0] * Dpsi[i, 1]
        jaco[1, 0] += XN[i, 1] * Dpsi[i, 0]
        jaco[1, 1] += XN[i, 1] * Dpsi[i, 1]
    
    # Explicit determinant
    Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
    ...
```

**CuPy GPU**

Mirrors NumPy API but executes on GPU memory:

```python
import cupy as cp

def Shape_N_Der8(XN, csi, eta):
    psi = cp.zeros(8)
    Dpsi = cp.zeros((8, 2))
    # ... same structure as NumPy
    jaco = XN.T @ Dpsi
    Detj = cp.linalg.det(jaco)
    ...
```

**CUDA Kernels (RawKernel and Numba CUDA)**

Inline all computations within the kernel to minimize memory transactions:

```c
// CuPy RawKernel (CUDA C)
__global__ void quad8_assembly_kernel(...) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Local arrays in registers/local memory
    double Ke[8][8] = {{0.0}};
    double XN[8][2];
    
    // All shape function, Jacobian, stiffness computation inlined
    ...
}
```

### 2.6.3. Mathematical Equivalence

Despite implementation differences, all versions compute mathematically identical results (within floating-point precision). This is verified by:

1. Comparing solution vectors across implementations
2. Checking that relative differences are within machine epsilon ( $\approx 10^{-15}$ )
3. Ensuring identical iteration counts for CG convergence

This equivalence is essential for valid performance comparisons: timing differences reflect execution model efficiency, not algorithmic variations.

---

## 2.7. Mesh Format and I/O

### 2.7.1. HDF5 Mesh Format

Meshes are stored in HDF5 format for efficient I/O operations:

```
mesh.h5
├── coordinates/
│   ├── x    (float64, shape: Nnodes)
│   └── y    (float64, shape: Nnodes)
└── connectivity/
    └── quad8 (int32, shape: Nelements × 8)
```

### 2.7.2. Format Advantages

HDF5 provides several advantages for this application:

| Feature | Benefit |
|---------|---------|
| Binary format | Faster I/O than text formats |
| Compression support | Reduced storage for large meshes |
| Memory mapping | Efficient access patterns |
| Platform independence | Cross-platform compatibility |
| Hierarchical structure | Organized data layout |
| Partial reads | Future extensibility for distributed computing |

### 2.7.3. Legacy Format Support

For compatibility, the solver also supports:

- **NPZ** (NumPy compressed archive): Fast binary format
- **Excel (.xlsx)**: Human-readable, useful for small test cases

All formats are converted to the internal NumPy array representation upon loading.

---

## 2.8. Summary

The common foundation described in this section ensures that all six solver implementations operate on identical mathematical and algorithmic ground. Key design decisions supporting fair performance comparison include:

1. **Identical FEM formulation**: Quad-8 elements, 9-point quadrature, Robin/Dirichlet BCs
2. **Uniform solver strategy**: Jacobi-preconditioned CG with fixed tolerance
3. **Consistent interfaces**: Same constructor signature, result format, callback system
4. **Equivalent computational modules**: Mathematically identical, adapted for each execution model
5. **Standardized timing**: Per-stage instrumentation with identical granularity

With this foundation established, the following sections examine how each implementation variant exploits parallelism within this common framework, and how the resulting performance characteristics differ across problem sizes and computational stages.

---

# 3. Implementations

## 3.1. Execution Models

This section presents multiple implementations of the same FEM problem using different execution models on CPU and GPU. All implementations share an identical numerical formulation, discretization, boundary conditions, and solver configuration; observed differences arise exclusively from the execution strategy and computational backend.

The implementations cover sequential CPU execution, shared-memory and process-based CPU parallelism, just-in-time compiled CPU execution using Numba, and GPU-based execution using Numba CUDA and CuPy with custom raw kernels. Together, these approaches span execution models ranging from interpreter-driven execution to compiled and accelerator-based computation.

Numerical equivalence is preserved across all implementations, enabling direct and fair comparison of execution behavior, performance, and scalability under consistent numerical conditions.

### 3.1.1. Pre-Implementation Phase

Before the development of the CPU and GPU execution models presented in this section, a dedicated pre-implementation phase was carried out to migrate an existing Finite Element Method (FEM) solver, previously developed in MATLAB by a member of the group, to the Python programming language.

The primary objective of this transition was to ensure that the original numerical formulation was fully preserved. In particular, the following aspects were maintained:

- The element types used (eight-node quadrilateral elements - Quad-8)
- The assembly procedures for stiffness matrices and load vectors
- The treatment of boundary conditions (Dirichlet and Robin conditions)
- The configuration of the linear solver and the corresponding convergence criteria

This phase was exclusively focused on functional and numerical validation of the Python implementation, and no performance optimization was performed. All computational kernels were rewritten using scientific Python libraries appropriate to the project objectives, thereby enabling, in a subsequent phase, the implementation of both CPU- and GPU-based solutions.

---

## 3.2. Implementation 1: CPU Baseline

###  3.2.1. Overview

The CPU baseline implementation serves as the reference against which all other CPU and GPU implementations are evaluated. It prioritizes correctness, algorithmic clarity, and reproducibility over performance, establishing both the functional specification and the performance floor for the project.

| Attribute | Description |
|---------|-------------|
| Technology | Python (NumPy, SciPy) |
| Execution Model | Sequential, single-process |
| Role | Correctness reference and performance baseline |
| Dependencies | NumPy, SciPy, pandas, h5py |

---

### 3.2.2. Technology Background

The baseline implementation is built on Python’s scientific computing ecosystem and executes on a sequential CPU model.

**Software ecosystem:**

- **NumPy** provides N-dimensional arrays and vectorized operations backed by optimized BLAS/LAPACK libraries.  
- **SciPy** supplies sparse matrix data structures and iterative solvers for large linear systems.  
- **h5py and pandas** support efficient binary input/output for mesh and result data.  
- This stack enables concise algorithm expression while delegating computationally intensive kernels to compiled numerical libraries.  

**Execution characteristics:**

- Execution is performed within the **CPython interpreter**, and is therefore subject to the **Global Interpreter Lock (GIL)**.  
- While NumPy and SciPy release the GIL during computational kernels, Python-level control flow remains serialized.  
- For FEM workloads, this results in a mixed execution model:
  - **Element loops** execute sequentially at the Python level with the GIL held.  
  - **Dense linear algebra operations** are executed in optimized BLAS/LAPACK routines with the GIL released.  
  - **Sparse iterative solvers** execute predominantly in compiled SciPy code, also releasing the GIL during major operations.  

**Relevance for FEM:**

- Provides a clear and traceable mapping between the mathematical formulation and the implementation.  
- Serves as a correctness reference for validating parallel implementations.  
- Enables early identification of computational bottlenecks through profiling.  
- Establishes a minimum performance bound for speedup evaluation.  

---

### 3.2.3. Implementation Strategy

The FEM workflow is organized into sequential stages to ensure correctness and consistent performance evaluation.

- **Mesh loading**: mesh data is loaded from binary HDF5 files to minimize parsing overhead and keep I/O negligible.  

- **System assembly** (element-by-element):
  1. initialize the global sparse matrix in an insertion-friendly format  
  2. compute per element an 8×8 local stiffness matrix + load terms using numerical quadrature  
  3. scatter local contributions into the global sparse system  
  4. convert the matrix to a compressed sparse format optimized for sparse matrix-vector products during solving  

- **Boundary conditions**:
  - Robin (inlet) enforced via numerical integration of boundary terms  
  - Dirichlet (outlet) imposed using a penalty method  
  - overall cost is small compared to assembly/solve  

- **Linear system solution**: solved using SciPy Conjugate Gradient (CG) with:
  - diagonal equilibration  
  - Jacobi (diagonal) preconditioning  
  - identical solver configuration across implementations for consistent convergence behavior  

- **Post-processing**: derived fields (e.g., velocity, pressure) computed via additional element-level loops; not dominant, but measurable for large meshes.
---

### 3.2.4. Optimization Techniques Applied

Several optimizations are applied to improve performance while preserving numerical equivalence and implementation simplicity.

- **Sparse matrix format selection**: different sparse formats are used depending on the computation stage, balancing assembly efficiency and solver performance:

| Format | Insertion | SpMV | Memory | Usage |
|--------|-----------|------|--------|-------|
| LIL (List of Lists) | O(1) amortized | O(nnz) | Higher | Assembly |
| CSR (Compressed Sparse Row) | O(n) | O(nnz) optimal | Lower | Solve |

  - The LIL → CSR strategy minimizes insertion overhead during assembly while ensuring optimal sparse matrix-vector products during iterative solving.

- **Diagonal equilibration**:
  - the linear system is diagonally equilibrated before solving to improve conditioning  
  - reduces sensitivity to element size variation and improves convergence, especially for large/heterogeneous meshes  

- **Preconditioning strategy**:
  - a Jacobi (diagonal) preconditioner is applied in the CG solver  
  - provides a good trade-off between simplicity and convergence robustness, ensuring stable iteration counts  

- **Vectorized inner operations**:
  - element-level dense operations are expressed using NumPy vectorized kernels 
  - this delegates inner computations to optimized compiled BLAS/LAPACK routines, mitigating Python interpreter overhead  

---

### 3.2.5. Challenges and Limitations

The sequential CPU baseline is mainly limited by Python interpreter overhead and sparse assembly costs, which restrict scalability for large meshes.

| Limitation | Impact |
|-----------|--------|
| Sequential Python element loop | Assembly becomes interpreter-bound and scales linearly with mesh size |
| GIL serialization | Limits any benefit from multi-threading at Python-level control flow |
| Incremental sparse insertion | High overhead from dynamic allocation and indirect indexing |
| Scattered memory access | Poor cache locality and increased memory traffic |

Additional observed behavior:

- **Sparse format trade-off**: assembly uses an insertion-friendly sparse format and converts to a solver-efficient compressed format; conversion adds overhead but is amortized over CG iterations.  
- **Preconditioning sensitivity**: CG convergence is highly dependent on preconditioning; Jacobi preconditioning stabilizes convergence with negligible cost.  
- **Residual monitoring**: convergence is evaluated using the true residual norm, ensuring consistent diagnostics across implementations with minimal overhead.  



---


### 3.2.6. Performance Characteristics and Baseline Role

The sequential CPU implementation defines the reference performance profile used to evaluate all parallel CPU and GPU approaches.

**Expected scaling**

| Stage | Complexity | Dominant Factor |
|------|-----------|-----------------|
| Mesh loading | O(N_nodes) | I/O bandwidth |
| Assembly | O(N_elements) | Python loop + sparse insertion overhead |
| Linear system solution | O(iterations × nnz) | SpMV memory bandwidth |
| Post-processing | O(N_elements) | Python loop overhead |

**Profiling observations (large meshes)**

- **Assembly:** ~50-70% of total runtime  
- **Solve:** ~20-40% (dominated by SpMV + iteration count)  
- **Post-processing:** ~5-15%  
- **Mesh I/O + boundary conditions:** typically <5%  

**Baseline role**

- **Correctness reference:** alternative implementations must match numerical results.  
- **Performance floor:** parallel methods must outperform this runtime.  
- **Solver reference:** convergence behavior and iteration counts should remain consistent.  

This baseline defines the reference execution profile for speedup and scalability analysis.


---

### 3.2.7. Summary
The CPU baseline provides a clear, correct, and reproducible reference for all subsequent implementations. While intentionally limited in scalability, it establishes a shared algorithmic foundation, a correctness benchmark, and a performance floor for comparative evaluation.

Key observations include:

- Assembly is interpreter-bound and dominates runtime.  
- Python-level overhead outweighs arithmetic cost for element-level operations.  
- The iterative solver is primarily memory-bound.  

Subsequent implementations address these limitations through parallel execution models, JIT compilation, and GPU offloading, while preserving numerical equivalence with this baseline.

## 3.3. Implementation 2: CPU Threaded

### 3.3.1. Overview

The CPU Threaded implementation extends the CPU baseline by introducing parallelism through Python’s `concurrent.futures.ThreadPoolExecutor`. The objective is to evaluate whether multi-threading can accelerate FEM assembly and post-processing despite the presence of Python’s Global Interpreter Lock (GIL).

Unlike the baseline, which executes all element-level operations sequentially, this implementation partitions the mesh into batches processed concurrently by multiple threads. The approach relies on the fact that NumPy releases the GIL during computational kernels, allowing partial overlap of execution across threads.

| Attribute | Description |
|-----------|-------------|
| Technology | Python ThreadPoolExecutor (`concurrent.futures`) |
| Execution Model | Multi-threaded with GIL constraints |
| Role | Evaluate benefits and limits of threading on CPU |
| Dependencies | NumPy, SciPy, concurrent.futures (stdlib) |

---

### 3.3.2. Technology Background

Python threading is limited by the Global Interpreter Lock (GIL), which serializes execution of Python bytecode and prevents true parallelism for CPU-bound workloads at the Python level. However, many NumPy kernels release the GIL, allowing partial concurrency.

**GIL and NumPy behavior**

- The GIL blocks parallel execution of Python-level code across threads.  
- NumPy releases the GIL in several operations:
  - Vectorized arithmetic
  - BLAS/LAPACK dense kernels 
  - Element-wise math kernels 

**ThreadPoolExecutor model**

- `ThreadPoolExecutor` provides reusable worker threads and a future-based execution model.  
- Key advantages:
  - Low overhead due to persistent threads  
  - Asynchronous submission via `Future`  
  - Dynamic scheduling (basic load balancing)  
  - Shared-memory access to NumPy arrays  

**Implications for FEM workloads**

| Operation | GIL Released | Expected Benefit |
|----------|--------------|------------------|
| Python loop iteration | No | None |
| Sparse matrix indexing | No | None |
| NumPy dense kernels | Yes | Moderate |
| Element-wise NumPy ops | Yes | Moderate |

- Speedup depends on maximizing time spent in GIL-free NumPy kernels and minimizing Python coordination.

---

### 3.3.3. Implementation Strategy

#### 3.3.3.1 Batch-Based Parallelization

To amortize threading overhead and reduce GIL contention, elements are grouped into fixed-size batches. Each batch is processed by a single thread, enabling coarse-grained parallelism:

![](images/documents/tutorial2/multithreading.png)

Each thread operates independently on a contiguous range of elements, computing local stiffness contributions and storing results in thread-local buffers.

#### 3.3.3.2 Element Batch Processing

Each batch computes stiffness matrices and load contributions for a subset of elements and stores results in pre-allocated arrays using COO (Coordinate) format.

Key steps include:

1. Pre-allocation of output arrays for rows, columns, and values  
2. Sequential processing of elements within the batch  
3. Computation of local stiffness matrices using NumPy operations  
4. Storage of local contributions in thread-local COO arrays  

This design avoids shared writes during assembly and minimizes synchronization.

#### 3.3.3.3 Parallel Assembly Orchestration

The main assembly routine dispatches batches to worker threads using a thread pool. Results are collected asynchronously, allowing faster threads to return without blocking on slower batches. After all threads complete, individual COO arrays are concatenated and converted to CSR format.

#### 3.3.3.4 COO-Based Global Assembly

Unlike the baseline implementation, which performs incremental insertion into a LIL matrix, this implementation assembles the global stiffness matrix using COO format:

| Aspect | Baseline (LIL) | Threaded (COO) |
|------|----------------|----------------|
| Thread safety | Not thread-safe | Naturally thread-safe |
| Insertion pattern | Incremental | Batched |
| Duplicate handling | Explicit | Automatic on CSR conversion |
| Parallel suitability | Poor | High |

The final `COO → CSR` conversion automatically merges duplicate entries arising from shared nodes between elements.

#### 3.3.3.5 Post-Processing Parallelization

Derived field computation (velocity and magnitude) follows the same batch-based threading strategy. Each thread processes a disjoint subset of elements and writes results into non-overlapping regions of the output arrays, avoiding data races.

#### 3.3.3.6 Linear System Solution

The linear solver is identical to the CPU baseline. SciPy’s Conjugate Gradient solver is used with the same preconditioning and convergence criteria. No Python-level threading is applied to the solver phase, as SciPy internally manages optimized numerical kernels and threading via BLAS libraries.

---

### 3.3.4. Optimization Techniques Applied

Several optimizations are applied to improve threaded performance by reducing overhead and maximizing time spent in GIL-free NumPy kernels.

- **Batch size selection**:
  - batch size controls the trade-off between scheduling overhead and load balance  
  - empirical testing shows best results for ~500-2000 elements per batch  

- **Pre-allocation of thread-local buffers**:
  - fixed-size arrays are allocated once per batch/thread invocation  
  - avoids repeated dynamic allocations inside inner loops, improving cache locality  

- **Inlined element computation**:
  - stiffness computation is implemented directly inside the batch function  
  - minimizes function call overhead and increases time spent in GIL-released NumPy kernels  

- **Shared read-only data**:
  - mesh coordinates, connectivity, and quadrature data are shared across threads as read-only arrays  
  - avoids memory duplication while ensuring thread safety  

---

### 3.3.5. Challenges and Limitations

The threaded implementation improves assembly throughput but remains fundamentally limited by GIL-bound coordination and shared-memory resource contention.

| Limitation | Impact |
|-----------|--------|
| GIL contention | Python loops, indexing, and sparse manipulation remain serialized, limiting scalability |
| Memory bandwidth saturation | Threads contend for the same memory subsystem, causing diminishing returns beyond a few threads |
| Thread management overhead | Task submission/scheduling/result aggregation becomes significant, especially for small meshes |
| Limited solver parallelism | Solver remains effectively sequential at Python level; BLAS threading offers limited gains due to memory-bound behavior |


---

### 3.3.6. Performance Characteristics and Role

Thread-level parallelism provides sub-linear speedup constrained by Amdahl’s Law, since only parts of assembly and post-processing can benefit from concurrency.

- **Expected scaling**:
  - speedup is limited by GIL-bound coordination and sequential solver phases  
  - only a fraction of total runtime is effectively parallelizable  

- **Practical speedup regime (empirical)**:
  - modest gains with 2-4 threads  
  - diminishing returns beyond 4-8 threads  
  - possible slowdowns when contention/overhead dominates  

- **Role in the implementation suite**:
  - serves as an intermediate step between the sequential baseline and stronger parallel approaches  
  - makes explicit the structural limitations imposed by the GIL, motivating designs that bypass it (multiprocessing or GPU)

---

### 3.3.7. Summary

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

## 3.4. Implementation 3: CPU Multiprocess

### 3.4.1. Overview

The CPU Multiprocess implementation achieves true parallelism by using process-based parallel execution. Unlike threading, multiprocessing bypasses the Global Interpreter Lock (GIL) entirely, enabling genuine concurrent execution across CPU cores. This comes at the cost of increased inter-process communication (IPC) and memory duplication.

| Attribute | Description |
|-----------|-------------|
| Technology | multiprocessing.Pool (Python stdlib) |
| Execution Model | Multi-process, separate memory spaces |
| Role | True CPU parallelism and GIL bypass demonstration |
| Dependencies | NumPy, SciPy, multiprocessing (stdlib) |

---

### 3.4.2. Technology Background

Python multiprocessing achieves parallelism by spawning multiple independent worker processes. Each process runs its own Python interpreter with an isolated memory space and its own GIL, avoiding GIL contention and enabling true CPU parallelism.

![Multiprocessing](images/documents/tutorial2/multiprocessing.png)

**Multiprocessing model**

- Separate memory: each process has an isolated address space  
- Independent GIL: no GIL contention between processes  
- IPC required: data must be transferred via serialization  
- Higher overhead: process creation and coordination are more expensive than threads  

**multiprocessing.Pool execution**

- `Pool` manages a fixed number of worker processes and distributes work using mapping primitives:

| Method | Behavior | Ordering |
|--------|----------|----------|
| `map()` | Blocking, returns list | Preserved |
| `map_async()` | Non-blocking | Preserved |
| `imap()` | Lazy iterator | Preserved |
| `imap_unordered()` | Lazy iterator | Arbitrary |

**Pickle serialization (IPC)**

- Data transfer relies on pickle serialization:
  - input arguments are serialized and sent to workers  
  - return values are serialized and returned to the main process  
- Large NumPy arrays can introduce significant overhead.  

**Relevance for FEM workloads**

| Aspect | Threading | Multiprocessing |
|------|-----------|-----------------|
| GIL impact | Serializes Python bytecode | None |
| Memory | Shared | Duplicated per process |
| Startup cost | Low | High |
| Communication | Direct memory access | Pickle serialization |
| Scalability | Limited by GIL | Limited by cores and IPC |

- For element-independent FEM assembly, multiprocessing can provide near-linear speedup if IPC costs are amortized.
---

### 3.4.3. Implementation Strategy

The multiprocessing implementation follows a batch-parallel execution model, where independent element batches are processed by separate worker processes. This enables true CPU parallelism but introduces additional constraints and IPC overhead.

- **Module-level function requirement**:
  - worker logic must be defined at module scope to be serializable (picklable)  
  - computational kernels and batch-processing logic are therefore implemented at top-level scope  

- **Batch processing architecture**:
  - the global element set is partitioned into contiguous batches  
  - each batch is processed independently by a worker process  
  - batches include the element range and required FEM data (coordinates, connectivity, quadrature)  
  - batching amortizes IPC overhead and reduces scheduling frequency  

- **Data serialization implications (IPC overhead)**:
  - unlike threading, multiprocessing requires explicit data transfer per batch  

![Multiprocessing Data Serialization](images/documents/tutorial2/multiprocessing_dataserial.png)

  - for large meshes, serialization volume and frequency become major performance constraints  

- **COO assembly strategy**:
  - workers produce thread/process-independent COO contributions  
  - the main process concatenates partial COO results  
  - COO → CSR conversion automatically merges duplicates  
  - avoids concurrent updates to shared sparse structures  

- **Post-processing**:
  - derived field computation uses the same batching strategy  
  - the solution field must also be serialized to workers, increasing IPC overhead  

- **Linear system solution**:
  - executed in the main process using the same solver configuration as other implementations  
  - ensures consistent convergence behavior and numerical equivalence  

---

### 3.4.4. Optimization Techniques Applied

The multiprocessing implementation focuses on reducing IPC overhead and ensuring safe parallel sparse assembly.

- **Batch size for IPC amortization**:
  - larger batches reduce IPC frequency but reduce load-balancing flexibility  

| Batch Size | Batches (100K elements) | IPC Transfers | IPC Overhead |
|------------|--------------------------|---------------|--------------|
| 100 | 1000 | 2000 | Very High |
| 1000 | 100 | 200 | Medium |
| 5000 | 20 | 40 | Low |
| 10000 | 10 | 20 | Very Low |

- **Tuple-based argument packing**:
  - all required batch data is packed and transferred together  
  - simplifies orchestration but increases serialization cost per task  

- **COO assembly for parallel safety**:
  - each worker generates independent COO outputs (no shared-state writes)  
  - duplicate handling is deferred to the final sparse matrix conversion  

- **Worker count configuration**:
  - worker count typically matches the number of CPU cores  
  - maximizes parallelism but increases memory duplication and IPC traffic  

---
### 3.4.5. Challenges and Limitations

The multiprocessing implementation enables true CPU parallelism but is heavily constrained by IPC, serialization, and memory duplication overhead.

| Limitation | Impact |
|-----------|--------|
| Serialization overhead (pickle) | Dominates runtime, as input/output must be serialized per batch (small batches worsen this) |
| Memory duplication | Each worker holds private copies of mesh data, significantly increasing total memory footprint |
| Process startup cost | High fixed overhead from spawning processes, interpreter initialization, and pool creation |
| Limited shared state | Results must be merged in the main process, introducing a sequential aggregation phase |
| Pickle constraints | Worker functions must be serializable, restricting structure and increasing implementation complexity |

**Memory duplication model**

Each worker process holds a private copy of input data:

```
Total Memory ≈ Main Process + N_workers × (coord arrays + connectivity)
```

For a 100K node mesh with 8 workers:

- Main process: ~10 MB  
- Workers: ~80 MB  
- Total: ~90 MB (vs. ~10 MB for threading)

**Startup overhead example**

| Component | Typical Time |
|----------|--------------|
| Fork/spawn | 10-50 ms per process |
| Interpreter initialization | 50-100 ms per process |
| Pool creation (4 workers) | 200-500 ms |

---

### 3.4.6. Performance Characteristics

Multiprocessing provides true CPU parallelism, but overall speedup depends on whether computation is large enough to amortize IPC and process management overhead.

**Scaling model**

\[
T_{parallel} = \frac{T_{serial}}{N} + T_{overhead}
\]

- \(T_{serial}\): sequential computation time  
- \(N\): number of worker processes  
- \(T_{overhead}\): IPC + process management overhead  

**Break-even behavior**

| Elements | Computation Time | Overhead (8 workers) | Benefit |
|----------|------------------|----------------------|---------|
| 1,000 | ~0.1 s | ~0.5 s | Negative |
| 10,000 | ~1 s | ~0.5 s | Marginal |
| 50,000 | ~5 s | ~0.6 s | Good |
| 100,000 | ~10 s | ~0.7 s | Excellent |

**Practical limitations**

- all processes share the same memory subsystem, so bandwidth saturation/NUMA effects can limit scaling  
- compared to threading:
  - better scalability for large problems  
  - worse performance for small problems  
  - higher memory consumption  
---

### 3.4.7. Summary

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

## 3.5. Implementation 4: Numba JIT CPU

### 3.5.1. Overview

The Numba JIT CPU implementation leverages Just-In-Time compilation to translate Python code into optimized native machine code at runtime. By compiling element-level FEM kernels and enabling parallel execution through Numba’s `prange` construct, this approach achieves true multi-threaded execution while preserving shared-memory semantics.

This implementation combines the low memory overhead of shared-memory execution with performance characteristics close to compiled languages, eliminating Python interpreter overhead from the dominant FEM assembly and post-processing phases.

| Attribute | Description |
|-----------|-------------|
| Technology | Numba JIT compiler with LLVM backend |
| Execution Model | JIT-compiled, multi-threaded shared memory |
| Role | High-performance CPU parallel execution |
| Dependencies | NumPy, SciPy, Numba |

---

### 3.5.2. Technology Background

Numba provides Just-In-Time (JIT) compilation by translating Python functions into optimized machine code using the LLVM infrastructure. This removes Python interpreter overhead and enables near-native CPU execution.

**Just-In-Time (JIT) compilation with Numba**

- Eliminates Python interpreter overhead  
- Native performance comparable to C/Fortran  
- Enables compiler optimizations (inlining, loop optimizations, SIMD via LLVM)  
- Executes without typical GIL constraints in compiled code  

**`@njit` compilation model**

- The implementation uses `@njit` to enforce *nopython* mode:
  - Python bytecode is bypassed  
  - Types are inferred at compile time  
  - Unsupported Python/NumPy features are disallowed  
- Compilation caching is enabled to amortize compilation cost across executions.  

**Parallel execution with `prange`**

- Loop parallelism is implemented using `prange`:
  - Execution occurs without GIL limitations  
  - Threads operate in shared memory  
  - OpenMP-style work distribution  
- Near-linear speedup is possible for independent iterations.  

**Relevance for FEM workloads**

- JIT compilation targets key FEM bottlenecks:
  - Element stiffness matrix computation
  - Element-level assembly loops
  - Derived field computation  
- Sparse matrix construction and solvers remain in SciPy, preserving numerical equivalence with previous approaches.


---

### 3.5.3. Implementation Strategy

The Numba implementation moves all element-level FEM computation into JIT-compiled kernels, minimizing interpreter overhead and enabling parallel execution through `prange`.

- **Function-level JIT compilation**:
  - all computational kernels are compiled with Numba in *nopython* mode  
  - stiffness computation, boundary contributions, and post-processing are implemented using loop-based formulations  
  - ensures element-level computation runs fully in compiled code (no Python interpreter overhead)  

- **Parallel element assembly (`prange`)**:
  - global assembly is performed as a parallel loop over elements  
  - each iteration:
    1. gathers element nodal coordinates  
    2. computes local stiffness matrix + load vector  
    3. writes contributions into pre-allocated COO arrays 
  - element independence enables safe parallelism and near-linear scaling  

- **Explicit loop-based kernels**:
  - operations are written as explicit loops (not vectorized NumPy) to maximize LLVM optimizations:
    - loop unrolling for small fixed-size loops  
    - inlining and reduced overhead  
    - fewer temporary allocations  
    - SIMD vectorization of inner loops  

- **Parallel post-processing**:
  - derived field computation follows the same compiled-parallel pattern  
  - each element evaluates gradients and stores results in element-wise output arrays  

- **Solver integration**:
  - Numba generates COO-format data, while SciPy performs sparse matrix construction and solution (CG)  
  - the JIT boundary is placed at the array level to preserve numerical equivalence with previous implementations


---

### 3.5.4. Optimization Techniques Applied

The Numba JIT implementation improves performance by eliminating interpreter overhead and enabling compiler-level optimizations.

- **Interpreter elimination**:
  - Python interpreter overhead is removed from element-level computation  
  - inner loops execute as native machine code  

- **Loop unrolling and inlining**:
  - small fixed-size loops are unrolled by LLVM  
  - nested function calls in compiled code are typically inlined, reducing call overhead  

- **SIMD vectorization**:
  - LLVM applies SIMD vectorization to inner arithmetic loops when possible  
  - enables multiple operations per CPU cycle  

- **Memory access optimization**:
  - COO output is written sequentially in element-major order  
  - improves cache locality and reduces write overhead  

- **Shared-memory parallelism**:
  - parallel execution uses shared memory without data duplication  
  - preserves memory efficiency compared to multiprocessing approaches  

---

### 3.5.5. Challenges and Limitations

While Numba JIT significantly accelerates element-level FEM computation, overall performance and usability are constrained by compilation cost, language limitations, and solver dominance at scale.

| Limitation | Impact |
|-----------|--------|
| JIT compilation overhead | First execution incurs hundreds of milliseconds of compilation time; amortized for large runs and reduced by caching |
| Limited NumPy/SciPy support | Only a subset of NumPy works in *nopython* mode; unsupported features must be rewritten using explicit loops |
| Debugging complexity | Debugging compiled code is harder; stack traces are less informative and debuggers cannot step into JIT regions |
| Allocations inside parallel loops | Memory allocation in parallel regions adds overhead; performance improves by minimizing allocations |
| Solver dominance at scale | As assembly accelerates, the sparse solver becomes the main runtime bottleneck, limiting further speedup |


---

### 3.5.6. Performance Characteristics and Role

The Numba JIT CPU implementation greatly accelerates element-level computation, shifting the runtime bottleneck toward the sparse solver.

**Expected scaling**

| Stage | Scaling Behavior | Dominant Factor |
|------|------------------|-----------------|
| Assembly | O(N_elements) | Compiled arithmetic |
| Post-processing | O(N_elements) | Compiled arithmetic |
| Linear system solution | O(iterations × nnz) | Sparse memory bandwidth |
| Boundary conditions | O(N_boundary) | Minor relative cost |

**Profiling observations (large meshes)**

- assembly and post-processing are reduced by **1-2 orders of magnitude**  
- the **solver becomes dominant** in total runtime  
- parallel efficiency remains high until **memory bandwidth saturation**  

**Role in the implementation suite**

- highest-performing CPU-based solution in this study  
- defines the practical upper bound for shared-memory CPU execution  
- serves as the main CPU reference when evaluating GPU implementations  


---

### 3.5.7. Summary

The Numba JIT CPU implementation eliminates Python interpreter overhead and enables true shared-memory parallelism for FEM assembly and post-processing.

Key observations include:

- Explicit loop-based kernels outperform vectorized NumPy formulations  
- True parallel execution is achieved without GIL constraints  
- Memory efficiency is preserved relative to multiprocessing  
- Sparse solver performance ultimately limits end-to-end speedup  

This implementation provides the most efficient CPU-based execution model in the study and forms a natural transition toward GPU-based acceleration.

## 3.6. Implementation 5: Numba CUDA

### 3.6.1. Overview

The Numba CUDA implementation extends the FEM solver to GPU execution using Numba’s `@cuda.jit` decorator, enabling the definition of GPU kernels using Python syntax. This approach provides access to massive GPU parallelism while avoiding direct CUDA C/C++ development, offering a balance between development productivity and performance.

Element-level FEM computations are offloaded to the GPU using a one-thread-per-element mapping, while sparse linear system solution is performed on the GPU using CuPy’s sparse solvers.

| Attribute | Description |
|-----------|-------------|
| Technology | Numba CUDA (`@cuda.jit`) |
| Execution Model | GPU SIMT execution |
| Role | GPU acceleration with Python-native kernels |
| Dependencies | NumPy, SciPy, Numba, CuPy |

---

### 3.6.2. Technology Background

Numba extends its JIT compilation framework to NVIDIA GPUs through the `@cuda.jit` decorator. CUDA kernels are compiled to PTX and executed on the GPU, enabling massive parallelism using the CUDA SIMT model (many lightweight threads executing the same kernel concurrently).

**Numba CUDA programming model**

- `@cuda.jit` compiles Python functions into GPU kernels (PTX code).  
- Execution follows the CUDA SIMT model, suited for thousands of parallel threads.  

**CUDA execution hierarchy**

- GPU kernels launch threads using a hierarchical structure:
  - Grid: all threads launched by a kernel  
  - Block: group of threads with cooperation via shared memory  
  - Thread: smallest execution unit  
  - Warp: 32 threads executing in lockstep  
- Threads are indexed with `cuda.grid(1)`, enabling direct mapping:
  - thread index ↔ FEM element index**  

**GPU memory hierarchy**
- Memory is organized in tiers:
  - Registers (fastest, thread-private)  
  - Local memory (thread-private, may spill)  
  - Shared memory (fast, block-shared)  
  - Global memory (large, high latency)  
- The implementation typically uses:
  - registers/local memory for element-level arrays  
  - global memory for mesh input data and assembled outputs  

**Relevance for FEM workloads**

- GPUs are effective for FEM with many independent elements.  
- Element stiffness computation has high arithmetic intensity and low dependency, making it well-suited for SIMT execution.

---

### 3.6.3. Implementation Strategy

The GPU implementation offloads FEM assembly and post-processing to CUDA kernels, using a one-thread-per-element mapping to exploit massive parallelism while avoiding inter-thread dependencies.

- **Kernel-based element assembly**:
  - assembly is implemented as a GPU kernel where each thread processes one element  
  - per element, each thread:
    1. loads nodal indices and coordinates  
    2. computes shape functions, Jacobians, and gradients  
    3. assembles the local stiffness matrix and load vector  
    4. writes results to global memory  
  - computations use explicit loops compatible with Numba CUDA  

- **Thread-to-element mapping**:
  - 1D grid launch, one thread per element  
  - extra threads exit early when the element index exceeds mesh size  
  - enables uniform work distribution without synchronization during element evaluation  

- **Local memory usage**:
  - per-thread temporary arrays are stored using `cuda.local.array`:
    - DOF indices, coordinates  
    - local stiffness matrix and load vector  
    - shape functions and derivatives  
  - thread-private memory avoids race conditions and synchronization overhead  

- **Force vector assembly (atomics)**:
  - shared nodes require thread-safe accumulation  
  - global force vector is assembled using `cuda.atomic.add` to ensure correctness  

- **Post-processing on GPU**:
  - derived fields are computed in a separate GPU kernel  
  - each thread evaluates gradients and stores element-wise averaged results  

- **Solver integration**:
  - the linear system is solved on the GPU using CuPy sparse Conjugate Gradient (CG) 
  - sparse matrices are converted to CuPy formats and the solution phase runs fully on GPU before copying results back to CPU memory

---

### 3.6.4. Optimization Techniques Applied

The Numba CUDA implementation applies GPU-focused optimizations to maximize throughput and reduce memory/control-flow inefficiencies.

- **Massive parallelism**:
  - GPU executes tens of thousands of threads concurrently  
  - enables element-level parallelism far beyond CPU core counts  

- **Block size tuning**:
  - kernel launch configuration is tuned for occupancy vs. register pressure  
  - 128 threads per block provides good performance for register-heavy FEM kernels  

- **Memory coalescing**:
  - memory access patterns are structured so consecutive threads access contiguous memory  
  - improves global memory bandwidth utilization  

- **Register and local memory management**:
  - small per-thread arrays are kept in registers when possible  
  - larger arrays may spill to local memory but remain thread-private and cached efficiently  

- **Warp divergence minimization**:
  - control flow minimizes conditional branches  
  - aside from bounds checks, threads follow identical execution paths  

---

### 3.6.5. Challenges and Limitations

The Numba CUDA implementation provides high element-level parallelism but introduces GPU-specific constraints related to debugging, limited language support, atomic overhead, and CPU-GPU data movement.

| Limitation | Impact |
|-----------|--------|
| Debugging complexity | GPU kernel debugging is difficult; Python debuggers cannot be used and failures may be silent or cause kernel crashes |
| Limited NumPy support | Only a restricted subset is supported inside kernels; unsupported operations must be rewritten with explicit loops/arithmetic |
| Atomic operation overhead | `atomicAdd` introduces serialization in force vector accumulation and may become a bottleneck for higher connectivity |
| Memory transfer overhead | Host-device transfers (PCIe) add overhead; for small meshes transfer cost can dominate runtime |
| Partial CPU-GPU workflow | Some steps remain on CPU (e.g., COO index generation, boundary conditions), reducing end-to-end GPU acceleration |

---

### 3.6.6. Performance Characteristics and Role

The Numba CUDA implementation accelerates element-level FEM computation on GPU, with end-to-end performance increasingly dominated by solver bandwidth and host-device transfers.

**Expected scaling**

| Stage | Scaling Behavior | Dominant Factor |
|------|------------------|-----------------|
| Element assembly | O(N_elements) | GPU throughput |
| Post-processing | O(N_elements) | GPU throughput |
| Linear system solution | O(iterations × nnz) | GPU memory bandwidth |
| Data transfer | O(N) | PCIe bandwidth |

**Profiling observations (large meshes)**

- GPU occupancy typically reaches ~50-75%
- assembly achieves substantial speedup relative to CPU JIT  
- sparse solver becomes memory-bandwidth bound
- end-to-end speedup improves as problem size increases  

**Role in the implementation suite**

- first GPU-based implementation in the study  
- validates GPU acceleration using Python-native kernels (Numba CUDA)  
- serves as reference for comparison with raw CUDA (CuPy RawKernel) approaches  

---

### 3.6.7. Summary

The Numba CUDA implementation enables GPU acceleration of FEM assembly and post-processing using Python syntax:

Key observations include:

- Thousands of GPU threads execute element computations concurrently  
- Python-based kernel development significantly reduces development effort  
- Performance approaches that of hand-written CUDA kernels  
- Atomic operations and memory transfers limit scalability for smaller problems  

This implementation represents a practical and accessible entry point for GPU acceleration, bridging the gap between CPU-based JIT execution and fully optimized raw CUDA implementations.

## 3.7. Implementation 6: GPU CuPy (RawKernel)

### 3.7.1. Overview

The GPU CuPy implementation represents the most performance-oriented approach, using CuPy's `RawKernel` to execute hand-written CUDA C kernels directly on the GPU. This provides maximum control over GPU execution while leveraging CuPy's ecosystem for sparse matrix operations and iterative solvers.

| Attribute | Description |
|-----------|-------------|
| Technology | CuPy RawKernel (CUDA C), CuPy sparse |
| Execution Model | GPU SIMT with native CUDA C kernels |
| Role | Maximum GPU performance, production-quality implementation |
| Dependencies | NumPy, SciPy, CuPy |

---

### 3.7.2. Technology Background

CuPy is a NumPy-compatible GPU array library that enables accelerated numerical computing using NVIDIA GPUs. It provides GPU-resident arrays, sparse matrix support, and iterative solvers running directly on the GPU.

**CuPy overview**

- Drop-in NumPy replacement (`import cupy as cp`) with a similar API  
- GPU arrays stored in GPU memory (VRAM)  
- Sparse matrices in CSR/CSC/COO formats on GPU  
- GPU iterative solvers (e.g., CG, GMRES)  
- RawKernel interface for custom CUDA C/C++ kernels  

**RawKernel execution model**

- `RawKernel` embeds CUDA C/C++ code directly in Python, enabling:
  - full CUDA feature set  
  - maximum performance (no Python overhead in the kernel)  
  - explicit control over memory, synchronization, and shared memory  
- Kernels are compiled once and cached for reuse.  

**Comparison with Numba CUDA**

| Aspect | Numba CUDA | CuPy RawKernel |
|--------|------------|----------------|
| Kernel language | Python | CUDA C/C++ |
| Performance | ~90-95% of peak | ~100% of peak |
| Shared memory | Basic support | Full control |
| Warp primitives | Limited | Full access |
| Learning curve | Lower | Higher |

**GPU memory model**

![GPU Memory Architeture](images/documents/tutorial2/gpu_memory.png)


---

### 3.7.3. Implementation Strategy

This implementation uses CuPy RawKernel to execute custom CUDA C kernels while keeping the full FEM pipeline GPU-resident, including assembly, sparse matrix construction, solving, and post-processing.

- **CUDA C kernel architecture (RawKernel)**:
  - two primary kernels are embedded as CUDA C string literals:
    - **Assembly kernel** (`quad8_assembly_kernel`)
      - one thread per element  
      - computes 8×8 stiffness matrix (64 values)  
      - writes values to global COO value array  
      - atomic accumulation into the global force vector  
    - **Post-processing kernel** (`quad8_postprocess_kernel`)
      - one thread per element  
      - evaluates velocity gradient at 4 Gauss points  
      - averages to centroid velocity  
      - writes velocity components and magnitude  

- **Kernel source structure (assembly)**:
  - thread index computed from `blockIdx`, `blockDim`, `threadIdx`  
  - thread-local arrays for element data and local matrices  
  - fixed quadrature/integration loops matching CPU formulation  
  - scatter step writes flattened 8×8 values  
  - force vector assembled via atomic updates  

- **GPU-accelerated COO index generation**:
  - COO row/column indices are generated on GPU using vectorized CuPy ops:
    - creates all \(N_{el} \times 64\) indices in parallel  
    - avoids CPU-side index generation and CPU-GPU synchronization  
  - CUDA kernel computes only the COO values  

- **Sparse matrix construction on GPU**:
  - build COO matrix with CuPy sparse  
  - convert COO → CSR on GPU (duplicates merged automatically)  
  - sparse matrix remains GPU-resident  

- **GPU sparse solver**:
  - system solved fully on GPU using CuPy sparse solvers:
    - diagonal equilibration on GPU  
    - Jacobi preconditioning via linear operator  
    - Conjugate Gradient (CG) fully on GPU  
    - de-equilibration on GPU

---

### 3.7.4. Optimization Techniques Applied

The CuPy RawKernel implementation applies CUDA C-level optimizations to maximize kernel efficiency and ensure solver robustness.

- **Inline CUDA C shape function derivatives**:
  - derivatives are computed directly inside the kernel using explicit CUDA C expressions  
  - avoids function call overhead and enables compiler optimization  

- **Explicit Jacobian and inverse computation**:
  - Jacobian, determinant, and inverse are computed inside the kernel using:
    - explicit loops over the 8 nodes  
    - fixed-size operations suitable for compiler unrolling  
    - direct 2×2 determinant and inverse evaluation  

- **Atomic force vector update**:
  - nodal force accumulation uses CUDA atomics (`atomicAdd`)  
  - ensures correctness when multiple elements contribute to shared nodes  

- **Solver fallback strategy**:
  - attempts CG first  
  - falls back to GMRES if CG fails  
  - improves robustness under numerically difficult cases  

---

### 3.7.5. Challenges and Limitations

The CuPy RawKernel approach delivers high performance but increases implementation complexity and introduces GPU-specific constraints.

| Limitation | Impact |
|-----------|--------|
| CUDA C complexity | Requires CUDA expertise (thread/memory hierarchy, occupancy, register pressure, divergence), increasing development cost |
| Debugging challenges | Harder than Python/Numba; limited tooling, silent crashes, and race conditions are difficult to diagnose |
| Kernel compilation overhead | First-run JIT compilation adds latency but is largely eliminated by CuPy kernel caching |
| GPU memory constraints | Large meshes may exceed VRAM due to sparse CSR storage and working buffers, limiting problem size |
| CuPy sparse solver limitations | Fewer/more limited solver and preconditioner options than SciPy; possible numerical differences mitigated via GMRES fallback |

**Kernel compilation cost (first use)**

| Kernel | Compilation Time |
|--------|------------------|
| Assembly | ~200-500 ms |
| Post-processing | ~100-200 ms |

**GPU memory usage example (per 100K nodes)**

| Component | Memory |
|-----------|--------|
| Coordinates (x, y) | ~1.6 MB |
| Connectivity | ~3.2 MB |
| Sparse matrix (CSR) | ~50-100 MB |
| Solution vectors | ~0.8 MB |
| Working memory | Variable |


---

### 3.7.6. Performance Characteristics and Role

The CuPy RawKernel implementation achieves the highest GPU performance by combining custom CUDA C kernels with a fully GPU-resident sparse solve pipeline.

**GPU utilization (typical)**

| Metric | Typical Value | Main Limiter |
|--------|---------------|--------------|
| Occupancy | 50–75% | Register pressure |
| Memory throughput | 70–85% peak | Coalescing |
| Compute utilization | 60–80% | Kernel efficiency |

**Performance breakdown (large problems)**

| Stage | Time Fraction | Notes |
|-------|---------------|-------|
| Mesh loading | <5% | I/O bound |
| Assembly kernel | 5–15% | Highly parallel |
| Matrix construction | 5–10% | CuPy sparse ops |
| Linear solve | 60–80% | Memory-bandwidth bound |
| Post-processing | 5–10% | Highly parallel |
| Data transfer | <5% | PCIe overhead |

**Scaling characteristics**

| Problem Size | GPU Advantage | Notes |
|--------------|---------------|------|
| <10K elements | Minimal | Transfer overhead dominates |
| 10K–100K | Significant (5–20×) | Good GPU utilization |
| 100K–1M | Maximum (20–100×) | Full GPU saturation |
| >1M | Memory limited | May require multi-GPU |

**Comparison with CPU approaches (high level)**

| Aspect | CPU Baseline | Numba CPU | GPU CuPy |
|--------|--------------|-----------|----------|
| Parallelism | 1 core | multi-core | 1000s of threads |
| Bandwidth | ~50–100 GB/s | ~50–100 GB/s | ~500–900 GB/s |
| Latency | Low | Low | Higher (PCIe) |
| Throughput | Moderate | Good | Excellent |

---

### 3.7.7. Summary

The GPU CuPy implementation with RawKernel represents the most performance-optimized endpoint of this implementation spectrum:

Key observations include:

- Native CUDA C kernels provide maximum GPU performance
- Full GPU-resident pipeline (assembly, solve, post-processing) minimizes PCIe transfers
- GPU-based COO index generation avoids CPU bottlenecks present in Numba CUDA
- Sparse solver dominates runtime once assembly is accelerated
- Development and debugging complexity is significantly higher than Numba CUDA

This implementation establishes the upper bound for single-GPU performance in this project and provides a production-quality reference design combining custom CUDA kernels with CuPy’s sparse linear algebra ecosystem.

---

# 4. Performance Evaluation

## 4.1 Motivation and Scope

This section presents a systematic benchmark study of the finite element solver developed in this work, with the objective of **quantifying performance gains across execution models**, from conventional CPU-based implementations to fully GPU-resident solvers.

Rather than restricting the analysis to a single machine, the benchmark was designed as a **cross-hardware evaluation**, where identical solver implementations were executed on multiple systems equipped with different NVIDIA GPUs. This approach enables a clear separation between:

- algorithmic effects (assembly strategy, solver configuration), and  
- hardware effects (CPU vs GPU, GPU architecture, memory bandwidth, VRAM capacity).

All implementations solve the *same mathematical problem* using the *same FEM formulation*, ensuring that observed differences arise exclusively from the execution model and underlying hardware.

---

## 4.2 Benchmark Objectives

The benchmark addresses the following key questions:

1. **CPU scaling limits**  
   How far can performance be improved on CPU using:
   - threading,
   - multiprocessing, and
   - JIT compilation with Numba,
   before memory bandwidth and Python overhead become dominant?

2. **GPU acceleration impact**  
   What is the performance gain when offloading:
   - element-level assembly,
   - sparse linear system solution, and
   - post-processing
   to the GPU using Numba CUDA and CuPy RawKernel?

3. **Cross-GPU scalability**  
   How does solver performance scale across GPUs with different compute capabilities, memory bandwidth, and VRAM capacity?

---

## 4.3 Solver Variants Under Test

All benchmark runs use the same mesh, boundary conditions, numerical parameters, and convergence criteria. Only the execution backend changes.

| Solver Variant | Execution Target | Description | Primary Role |
|---------------|------------------|-------------|--------------|
| **CPU Baseline** | CPU | Sequential NumPy/SciPy | Correctness reference |
| **CPU Threaded** | CPU | ThreadPool-based batching | Evaluate GIL-limited threading |
| **CPU Multiprocess** | CPU | Process-level parallelism | True CPU parallelism |
| **Numba JIT (CPU)** | CPU | `@njit` + `prange` | High-performance shared-memory CPU |
| **Numba CUDA** | GPU | Python CUDA kernels | GPU acceleration with Python kernels |
| **GPU CuPy (RawKernel)** | GPU | CUDA C kernels + CuPy sparse | Maximum single-GPU performance |

This progression reflects a deliberate transition from interpreter-driven execution to compiled and accelerator-based computation.

---

## 4.4 Testing Environment

At this stage of the project, the experimental evaluation is conducted using a selected set of servers (**ids: #1/#4/#5**) and problem sizes. In a subsequent phase, the study will be extended to include a larger pool of computational servers as well as significantly larger models, with increased numbers of nodes and elements, in order to further highlight performance, scalability, and hardware sensitivity. The current setup therefore represents an initial and controlled benchmarking baseline.

### Contributing Servers

Benchmark data aggregated from **5 servers**:

| # | Hostname | CPU | Cores | RAM | GPU | VRAM | Records |
|---|----------|-----|-------|-----|-----|------|---------|
| 1 | DESKTOP-3MCDHQ7 | AMD64 Family 25 Model 97 St... | 12 | - | NVIDIA GeForce RT... | 15.9 GB | 237 |
| 2 | DESKTOP-B968RT3 | AMD64 Family 25 Model 97 St... | 12 | - | NVIDIA GeForce RT... | 15.9 GB | 33 |
| 3 | KRATOS | Intel64 Family 6 Model 183 ... | 28 | - | NVIDIA GeForce RT... | 12.0 GB | 81 |
| 4 | MERCURY | 13th Gen Intel(R) Core(TM) ... | 20 | 94.3 GB | NVIDIA GeForce RT... | 24.0 GB | 275 |
| 5 | RICKYROG700 | Intel64 Family 6 Model 198 ... | 24 | - | NVIDIA GeForce RT... | 31.8 GB | 238 |

### Test Meshes

| Model | Size | Nodes | Elements | Matrix NNZ |
|-------|------|-------|----------|------------|
| Backward-Facing Step | XS | 287 | 82 | 3,873 |
| Backward-Facing Step | M | 195,362 | 64,713 | 3,042,302 |
| Backward-Facing Step | L | 766,088 | 254,551 | 11,973,636 |
| Backward-Facing Step | XL | 1,283,215 | 426,686 | 20,066,869 |
| Elbow 90° | XS | 411 | 111 | 5,063 |
| Elbow 90° | M | 161,984 | 53,344 | 2,503,138 |
| Elbow 90° | L | 623,153 | 206,435 | 9,712,725 |
| Elbow 90° | XL | 1,044,857 | 346,621 | 16,304,541 |
| S-Bend | XS | 387 | 222 | 4,109 |
| S-Bend | M | 196,078 | 64,787 | 3,048,794 |
| S-Bend | L | 765,441 | 254,034 | 11,952,725 |
| S-Bend | XL | 1,286,039 | 427,244 | 20,097,467 |
| T-Junction | XS | 393 | 102 | 5,357 |
| T-Junction | M | 196,420 | 64,987 | 3,057,464 |
| T-Junction | L | 768,898 | 255,333 | 12,012,244 |
| T-Junction | XL | 1,291,289 | 429,176 | 20,186,313 |
| Venturi | XS | 341 | 86 | 4,061 |
| Venturi | M | 194,325 | 64,334 | 3,023,503 |
| Venturi | L | 763,707 | 253,704 | 11,934,351 |
| Venturi | XL | 1,284,412 | 427,017 | 20,083,132 |
| Y-Shaped | XS | 201 | 52 | 2,571 |
| Y-Shaped | M | 195,853 | 48,607 | 2,336,363 |
| Y-Shaped | L | 772,069 | 192,308 | 9,242,129 |
| Y-Shaped | XL | 1,357,953 | 338,544 | 16,265,217 |

### Solver Configuration

| Parameter | Value |
|-----------|-------|
| Problem Type | 2D Potential Flow (Laplace) |
| Element Type | Quad-8 (8-node serendipity quadrilateral) |
| Linear Solver | CG |
| Tolerance | 1e-08 |
| Max Iterations | 15,000,000 |
| Preconditioner | Jacobi |

### Implementations Tested

| # | Implementation | File | Parallelism Strategy |
|---|----------------|------|----------------------|
| 1 | CPU Baseline | `quad8_cpu_v3.py` | Sequential Python loops |
| 2 | CPU Threaded | `quad8_cpu_threaded.py` | ThreadPoolExecutor (GIL-limited) |
| 3 | CPU Multiprocess | `quad8_cpu_multiprocess.py` | multiprocessing.Pool |
| 4 | Numba CPU | `quad8_numba.py` | @njit + prange |
| 5 | Numba CUDA | `quad8_numba_cuda.py` | @cuda.jit kernels |
| 6 | CuPy GPU | `quad8_gpu_v3.py` | CUDA C RawKernels |

In line with this scope, the benchmarks presented below are conducted on three representative systems and on meshes with a limited number of nodes and elements, ensuring consistent and reproducible measurements:


| System | GPU Model | VRAM | Benchmark Relevance |
|------|-----------|------|---------------------|
| RICKYROG700 | RTX 5090 | 31.8 GB | Upper performance ceiling |
| MERCURY | RTX 4090 | 24 GB | High-end reference GPU |
| DESKTOP-B968RT3 | RTX 5060 Ti | 15.9 GB | Mid-range GPU |

---

![DESCRIPTION](images/documents/tutorial2/assembly_vs_solve_breakdown_2x2_mesh_sizes.svg)

### 4.4.1 Assembly vs. Solve Time Breakdown Across Mesh Sizes

Figure XXX presents a detailed breakdown of total execution time into assembly and solve phases for all solver implementations, evaluated across four increasingly large mesh sizes. This decomposition is essential to understand not only which implementation is faster, but why performance changes with scale, revealing the underlying computational bottlenecks that dominate each regime.

For the smallest mesh (201 nodes), total runtimes are extremely short for all implementations, and performance is governed almost entirely by fixed overheads rather than sustained computation. The CPU baseline and lightweight threaded execution perform efficiently due to minimal setup costs, while multiprocessing exhibits a severe assembly penalty, clearly visible in the figure, caused by process spawning and inter-process communication overhead. GPU-based implementations (Numba CUDA and CuPy GPU) show relatively larger solve fractions despite low absolute runtimes, reflecting kernel launch latency and synchronization costs that cannot be amortized at this scale. These results confirm that accelerator-based execution is structurally inefficient for very small FEM problems, regardless of hardware capability.

At the intermediate mesh size (194,325 nodes), the performance profile enters a transition regime. Assembly time increases substantially for CPU-based implementations, especially for the baseline solver, while the solve phase becomes the dominant contributor for most execution models. Threaded and multiprocessing approaches reduce assembly time relative to the baseline, but this primarily exposes the sparse solver as the new bottleneck rather than eliminating it. Numba JIT significantly compresses assembly cost, making the solve phase overwhelmingly dominant. GPU-based solvers show a pronounced reduction in assembly time compared to CPU approaches; however, the solve phase remains substantial, indicating that performance is increasingly constrained by sparse linear algebra and memory access patterns rather than element-level computation.

For the large mesh (766,088 nodes), solver dominance becomes unequivocal. All CPU-based implementations spend the vast majority of their execution time in the iterative solver, with assembly contributing only a secondary fraction of the total cost—even when JIT compilation is employed. This reflects the inherently memory-bandwidth-bound nature of sparse matrix-vector operations on CPUs. In contrast, GPU implementations dramatically reduce assembly time to near-negligible levels and significantly lower overall solve time. The figure shows that GPU parallelism is highly effective at eliminating element-level bottlenecks; nevertheless, the solve phase remains the largest contributor even on the GPU.

This behavior is further reinforced for the largest mesh (1,357,953 nodes). Across all CPU execution models, runtime is almost entirely dominated by the solve phase, rendering additional assembly optimizations largely irrelevant. GPU-based solvers maintain minimal assembly costs and comparatively moderate solve times, but the solver still accounts for the majority of execution time. The convergence of assembly times between GPU and Numba CUDA at this scale indicates that performance is governed primarily by memory bandwidth and sparse access patterns, rather than kernel-level computational throughput.

This analysis highlights that FEM performance optimization is inherently scale-dependent. While CPU-level parallelism and JIT compilation provide meaningful gains at moderate sizes, they are insufficient to overcome the fundamental limitations of sparse linear algebra on CPUs. GPU acceleration effectively removes assembly as a bottleneck and substantially mitigates solver cost, making it the only viable strategy for large-scale problems. However, even on GPUs, further performance improvements must focus on solver algorithms, preconditioning strategies, and memory efficiency, rather than kernel-level optimizations alone.



![DESCRIPTION](images/documents/tutorial2/CPU_GPU_runtime_crossover_interpolated.svg)

*Figure xxx- Interpolated CPU-GPU runtime crossover as a function of problem size.*


### 4.4.2 CPU-GPU Runtime Crossover Analysis

Figure 4.5 presents the interpolated runtime crossover between CPU-based and GPU-based solver executions as a function of problem size. This analysis aims to identify the **break-even point** at which GPU acceleration becomes consistently advantageous over CPU execution, providing a quantitative criterion for hardware-aware solver selection.

At small problem sizes, the CPU implementation exhibits lower total runtime, which is primarily explained by its minimal startup overhead. GPU-based execution, while massively parallel, incurs fixed costs related to kernel launch, device synchronization, and data movement between host and device memory. In this regime, these overheads dominate total execution time, rendering GPU acceleration inefficient despite its superior theoretical throughput.

As the number of nodes increases, CPU runtime grows approximately linearly, reflecting the combined cost of element assembly and iterative sparse linear solves executed in a memory-bound environment. In contrast, the GPU runtime curve exhibits a much flatter slope. Once the problem size exceeds a critical threshold, the GPU is able to amortize its fixed overheads and exploit fine-grained parallelism across thousands of threads, leading to substantially better scalability.

The intersection point of the two curves defines the **CPU-GPU crossover region**, beyond which GPU execution consistently outperforms CPU execution. This crossover is not a single fixed value but rather a narrow interval, influenced by factors such as solver configuration, sparsity pattern, and memory access behavior. Importantly, this transition occurs well below the largest mesh sizes considered in this study, indicating that GPU acceleration is not merely beneficial for extreme-scale problems, but already advantageous at moderately large FEM models.

Beyond the crossover point, the divergence between CPU and GPU runtimes increases rapidly. This behavior confirms that CPU-based solvers become increasingly constrained by memory bandwidth and cache inefficiency, while GPU-based solvers sustain higher effective throughput due to wider memory interfaces and higher concurrency. The gap widens further as problem size grows, reinforcing the conclusion that CPUs do not scale favorably for large sparse FEM systems, even when augmented with threading or JIT compilation.

This crossover analysis provides a clear and actionable performance guideline: **CPU execution is preferable only for small-scale problems**, where overhead dominates, whereas **GPU execution becomes the superior choice once the problem size exceeds the crossover threshold**. This result complements the assembly-versus-solve breakdown by offering a global, hardware-agnostic perspective on performance scalability, and directly motivates the cross-GPU comparisons presented in the subsequent sections.



![DESCRIPTION](images/documents/tutorial2/geometry_small_multiples_runtime_scaling.svg)

**Figure xxx - Assembly and solve time breakdown for different solver strategies across multiple mesh sizes.**

### 4.4.3 Critical Analysis of Runtime Scaling and CPU-GPU Transition

The results presented in the previous figures reveal a clear and consistent transition in performance behaviour as problem size increases, highlighting the distinct computational regimes in which CPU-based and GPU-accelerated solvers operate.

For small-scale problems, CPU solvers—both sequential and parallel—exhibit competitive performance due to their low execution overhead and efficient handling of limited workloads. In this regime, the total runtime is dominated by fixed costs such as setup, memory allocation, and solver initialization, which reduces the relative benefit of parallel execution. Consequently, GPU-based solvers do not provide a measurable advantage for coarse meshes, as kernel launch overheads and data transfer costs outweigh the benefits of massive parallelism.

As the number of nodes increases, a progressive shift in computational dominance becomes evident. Assembly time grows approximately linearly with mesh size, while solver time increases more rapidly due to the expanding sparse linear system and its associated memory access patterns. CPU-based solvers, including multithreaded and Numba JIT implementations, begin to exhibit limited scalability in this regime. Although parallelism mitigates some of the computational burden, performance becomes increasingly constrained by memory bandwidth and cache efficiency rather than raw compute capability.

Beyond an intermediate problem size, a distinct CPU-GPU crossover point is observed. At this stage, GPU-based solvers consistently outperform all CPU variants, with total execution time scaling more favourably as mesh resolution increases. This behaviour is primarily driven by the solver phase, where the GPU’s high memory bandwidth and massive thread-level parallelism enable more efficient sparse matrix-vector operations. The assembly phase, while still relevant, becomes secondary in determining overall performance for large-scale simulations.

Importantly, the crossover point is not purely hardware-dependent but emerges from the interaction between problem size, algorithmic structure, and architectural characteristics. The results demonstrate that GPU acceleration becomes increasingly advantageous once the solver phase dominates runtime and parallel workload granularity is sufficient to amortize GPU overheads.

This analysis confirms that CPU-based approaches remain suitable for small and moderately sized problems, while GPU acceleration is essential for maintaining scalability in large-scale finite element simulations. The findings reinforce the importance of selecting solver strategies based on both problem size and computational architecture, rather than relying on a one-size-fits-all execution model.



![DESCRIPTION](images/documents/tutorial2/normalized_time_per_element_venturi.svg)

**Figure xxx - Runtime scaling of the FEM solver as a function of the number of nodes for different geometries.**

### 4.4.3 Runtime Scaling and CPU-GPU Crossover Analysis

Figure XXX shows the normalized runtime per element as a function of the number of elements for the Venturi geometry, using logarithmic scales on both axes. This representation isolates scaling efficiency from absolute runtime and provides a clearer view of how each execution model behaves asymptotically as problem size increases.

For small meshes (≈10³ elements), the normalized runtime per element is relatively high and scattered across implementations. In this regime, fixed overheads dominate execution, and GPU-based solvers (Numba CUDA and CuPy GPU) exhibit significantly worse efficiency per element than CPU-based approaches. This behavior reflects kernel launch latency, memory transfer overhead, and GPU context management costs, which cannot be amortized when the computational workload per element is small. Lightweight CPU approaches, particularly Numba JIT, achieve the lowest per-element cost in this range due to minimal overhead and efficient compiled execution.

As the number of elements increases toward the mid-scale regime (≈10⁵-10⁶ elements), a clear change in scaling behavior emerges. CPU-based implementations show an increasing runtime per element, indicating deteriorating efficiency as sparse solver costs and memory bandwidth limitations begin to dominate. In contrast, GPU-based solvers exhibit a decreasing runtime per element, demonstrating improved amortization of overheads and more effective utilization of parallel hardware resources. This region corresponds to the CPU-GPU crossover, where GPU execution transitions from being overhead-bound to throughput-efficient.

Beyond the crossover point (≈10⁶ elements), GPU implementations clearly dominate. Both CuPy GPU and Numba CUDA show the lowest and flattest curves, indicating near-optimal scaling where additional elements incur only marginal increases in per-element cost. This behavior highlights the advantage of massive thread-level parallelism and high memory bandwidth when handling large sparse systems. Among GPU approaches, CuPy consistently achieves slightly lower per-element runtimes than Numba CUDA, reflecting lower kernel abstraction overhead and more optimized execution paths.

CPU-based solvers, including multiprocessing and threading, display the opposite trend: their per-element runtime increases steadily with mesh size. This confirms that CPU execution becomes increasingly constrained by memory access patterns and sparse linear algebra operations, which do not scale favorably with core count alone. Multiprocessing shows particularly poor efficiency at small scales and only moderate improvement at larger sizes, underscoring the cost of inter-process communication.

Overall, the figure provides strong empirical evidence that GPU acceleration is essential for achieving scalable FEM performance at large problem sizes. While CPU-based solvers remain efficient and competitive for small meshes, their asymptotic behavior is fundamentally limited. GPU-based approaches, by contrast, demonstrate improving efficiency with scale and clearly superior asymptotic performance, making them the preferred execution model for high-resolution, production-scale finite element simulations.



![DESCRIPTION](images/documents/tutorial2/pareto_frontier_per_mesh_size_markers_angles.svg)

**Figure xxx - Pareto frontier of average total runtime versus average solver iterations for different mesh sizes and execution models.**  

### 4.4.4 Pareto-Based Performance Trade-off Analysis

Figure xxx presents a Pareto-based analysis of solver performance, explicitly relating average total runtime to average solver iteration count across four increasing mesh sizes. This representation provides a multidimensional view of efficiency, allowing runtime performance to be evaluated jointly with numerical effort, rather than in isolation.

For the smallest mesh (201 nodes), all implementations exhibit very similar iteration counts, confirming that convergence behavior is independent of the execution backend at this scale. Performance differences are therefore entirely driven by execution overhead. In this regime, the Pareto frontier is defined by CPU-based approaches, particularly the baseline and threaded CPU implementations, which achieve minimal runtime with no GPU-related initialization or data transfer costs. Multiprocessing is clearly Pareto-dominated, exhibiting both higher runtime and no numerical advantage. GPU and Numba CUDA solutions also lie off the Pareto frontier, as fixed GPU overheads outweigh any benefit from parallel execution for such small systems.

At approximately 200k nodes (195,853 nodes), a clear transition occurs. While iteration counts remain clustered across all solvers—indicating preserved numerical equivalence—the runtime dimension separates sharply by architecture. GPU-based solvers (GPU and Numba CUDA) move decisively toward the Pareto frontier, achieving substantially lower runtimes for iteration counts comparable to CPU-based methods. In contrast, baseline CPU, threaded, and Numba CPU implementations become Pareto-dominated due to rapidly increasing wall-clock time, despite similar convergence behavior. This confirms that the performance divergence is architectural rather than algorithmic.

For larger meshes (≈772k nodes), the Pareto structure becomes even more pronounced. GPU and Numba CUDA implementations clearly define the Pareto frontier, combining low runtime with iteration counts indistinguishable from CPU solvers. CPU and threaded implementations occupy the upper-right region of the plots, reflecting both higher runtime and no numerical benefit. Multiprocessing, while improving over baseline CPU in absolute time, remains Pareto-dominated due to its limited scalability and overhead costs. Numba CPU retains acceptable iteration efficiency but becomes increasingly dominated in runtime as sparse solver and memory bandwidth limitations saturate CPU resources.

At the largest mesh size (1,357,953 nodes), GPU dominance is unequivocal. GPU-based solvers achieve order-of-magnitude reductions in runtime while maintaining iteration counts consistent with all other implementations. The Pareto frontier is exclusively defined by GPU and Numba CUDA approaches, demonstrating that no CPU-based execution model offers a competitive trade-off at this scale. The vertical alignment of iteration counts across all solvers further reinforces that numerical behavior is invariant, and that the Pareto advantage arises solely from superior execution efficiency.

This Pareto analysis leads to three key conclusions:

1. For small-scale problems, CPU-based solvers are Pareto-optimal, as minimal overhead outweighs any benefit from accelerator hardware;
2. For medium-scale problems, the Pareto frontier begins to shift toward GPU-based execution, marking the onset of the CPU-GPU crossover;
3. For large-scale problems, GPU-based solvers fully dominate the Pareto frontier, delivering the best achievable balance between runtime and numerical effort.

This analysis reinforces the central finding of the performance study: GPU acceleration is not merely faster in absolute terms, but becomes structurally superior as problem size increases, while fully preserving numerical consistency across all execution models.


![DESCRIPTION](images/documents/tutorial2/performance_envelope_y_shaped.svg)

**Figure xxx - Performance envelope across execution models for the Y-shaped geometry.**  

### 4.4.5. Performance Envelope Analysis for the Y-Shaped Geometry

The performance envelope clearly reveals distinct computational regimes as the problem size increases. For small meshes, CPU-based solvers define the lower envelope, achieving the shortest runtimes due to minimal overhead and immediate execution. In this regime, GPU implementations are penalized by kernel launch latency, memory allocation, and host-device data transfer costs, which outweigh the benefits of massive parallelism.

As mesh complexity increases, a clear crossover point emerges where GPU-based solvers begin to outperform all CPU alternatives. Beyond this threshold, the envelope shifts decisively toward GPU execution, indicating superior scalability and throughput. The widening gap between GPU and CPU curves highlights the asymptotic advantage of GPU architectures for element-level parallel workloads characteristic of FEM assembly and post-processing.

An important observation is that Numba CUDA and CuPy-based implementations form the lower bound of the envelope for large meshes, confirming that once overheads are amortized, execution efficiency is primarily governed by available parallelism and memory bandwidth rather than interpreter or compilation strategy.

This figure demonstrates that solver optimality is strongly mesh-dependent. While CPU execution remains appropriate for small-scale problems, GPU acceleration defines the optimal performance envelope for medium to large meshes, justifying its use as the default strategy in high-resolution FEM simulations.


![DESCRIPTION](images/documents/tutorial2/venturi_iterations_vs_nodes_all_solvers.svg)

**Figure xxx - Number of Conjugate Gradient iterations as a function of mesh size for all solver implementations (Venturi geometry).** 

### 4.4.6 Solver Convergence Behaviour Across Mesh Sizes

This figure provides a crucial validation of the numerical consistency of the entire implementation suite. Across all solver backends and execution models, the number of CG iterations exhibits an almost identical growth trend as mesh size increases. This confirms that convergence behaviour is governed by the mathematical properties of the discretized system — namely mesh resolution, conditioning of the stiffness matrix, and boundary conditions — rather than by the underlying execution architecture.

For small meshes, the iteration count remains low and tightly clustered across all solvers, reflecting well-conditioned systems and rapid convergence. As the number of nodes increases, the iteration count grows steadily, which is expected for elliptic problems discretized with higher resolution. Importantly, this growth is uniform across CPU and GPU implementations, demonstrating that GPU acceleration does not alter the numerical trajectory of the solver.

The absence of divergence between CPU and GPU curves is particularly significant. It indicates that all implementations:
- Apply identical preconditioning strategies (Jacobi),
- Use consistent convergence tolerances,
- Preserve numerical precision within acceptable floating-point limits.

This result also reinforces the interpretation of performance gains observed in runtime benchmarks: speedups achieved by GPU-based solvers arise exclusively from faster execution of assembly, sparse linear algebra, and vector operations, not from reduced solver work or relaxed convergence criteria.

From a performance analysis standpoint, this figure isolates **runtime efficiency** as the sole differentiating factor between solvers. Since the iteration count is invariant with respect to execution model, any reduction in total runtime directly reflects architectural advantages such as increased parallelism, higher memory bandwidth, and reduced instruction overhead.

This convergence analysis confirms that:
- All solver implementations are numerically equivalent and directly comparable.
- GPU acceleration preserves solver robustness and stability.
- Performance improvements observed in later sections are genuine computational gains rather than numerical artefacts.

This result is fundamental for the credibility of the benchmarking study and validates the fairness of the cross-platform performance comparison.


![DESCRIPTION](images/documents/tutorial2/y_shaped_cpu_2x2_execution_models.svg)

**Figure xxx - Execution time comparison across solver implementations for the Y-shaped geometry (CPU-based and GPU-based models).**  

### 4.4.7 Comparative Execution Time Breakdown for the Y-Shaped Geometry

Figure XXX provides a consolidated comparison of execution time scaling for the Y-Shaped geometry across multiple execution models (CPU, threaded, multiprocessing, and Numba JIT CPU), explicitly accounting for different CPU architectures. This representation strengthens the robustness of the performance analysis by demonstrating that the observed trends are not tied to a single processor, but persist across heterogeneous hardware configurations.

For the smallest mesh size, execution times remain tightly clustered across all CPUs and execution models. Differences between Intel and AMD processors are minimal and largely masked by fixed overheads such as solver initialization, memory allocation, and Python runtime setup. In this regime, absolute performance is dominated by non-scalable costs rather than architectural efficiency, and all execution models behave similarly regardless of CPU family.

As mesh size increases, the scaling behavior becomes more clearly differentiated. The baseline CPU implementation exhibits near-linear growth in execution time across all processors, indicating that performance is primarily constrained by interpreter overhead and memory bandwidth rather than core count. While absolute runtimes vary slightly between CPUs, the slope of the curves remains remarkably consistent, confirming that the baseline implementation is architecture-agnostic but fundamentally limited in scalability.

The threaded execution model improves performance at small and medium scales but shows increasing divergence between CPUs as mesh size grows. This reflects sensitivity to core count, cache hierarchy, and NUMA effects. Nevertheless, the overall scaling trend remains similar to the baseline CPU, reinforcing that Python threading provides limited benefits for compute-bound FEM workloads due to the Global Interpreter Lock (GIL).

The multiprocessing approach displays the highest variability across CPUs. While it reduces execution time relative to baseline CPU for larger meshes, its scaling is less stable and more sensitive to hardware characteristics. This behavior is consistent with the overhead of process creation, inter-process communication, and memory duplication, which amplify architectural differences and reduce predictability.

The Numba JIT CPU implementation demonstrates the most consistent and favorable scaling among CPU-based approaches. Across all tested processors, its execution time grows more slowly with mesh size, and inter-CPU variability is significantly reduced. This confirms that JIT compilation effectively removes interpreter overhead and enables more efficient parallel execution, making performance primarily dependent on raw memory bandwidth and vectorized execution rather than Python runtime behavior.

A key insight from this figure is that, although absolute runtimes vary between CPUs, the relative ordering of execution models is preserved across architectures. Numba JIT consistently outperforms pure Python approaches, while baseline and threaded CPU executions remain the least scalable. This stability indicates that the conclusions drawn from earlier sections generalize across different hardware environments.

Overall, this analysis reinforces several important findings:

1. CPU-based solvers scale predictably but are ultimately limited by memory bandwidth and core count.
2. Threading and multiprocessing introduce variability without fundamentally changing scaling behavior.
3. Numba JIT provides a robust and portable performance improvement across CPU architectures.
4. Hardware differences affect absolute performance but do not alter the structural performance hierarchy.

This figure therefore strengthens the validity of the study’s conclusions by demonstrating that the observed performance patterns are architecturally robust, while also highlighting the intrinsic scalability limits of CPU-based FEM solvers when compared to GPU-accelerated approaches discussed in subsequent sections.

![DESCRIPTION](images/documents/tutorial2/y_shaped_gpu_side_by_side.svg)

**Figure xxx - Side-by-side execution time comparison of GPU-based solvers for the Y-shaped geometry.**  

### 4.4.8 GPU-Centric Performance Comparison for the Y-Shaped Geometry

This comparison isolates GPU execution behavior by removing CPU-based solvers from the analysis, allowing a focused evaluation of how different GPU programming approaches affect performance. For smaller mesh sizes, execution times for Numba CUDA and CuPy are relatively close, with neither implementation showing a decisive advantage. In this regime, kernel launch overheads, JIT compilation costs, and memory transfers dominate runtime, masking fine-grained kernel efficiency differences.

As mesh size increases, performance divergence becomes more apparent. The CuPy RawKernel implementation consistently achieves lower execution times compared to Numba CUDA for medium and large meshes. This behavior reflects the reduced abstraction overhead and finer control over memory access, kernel structure, and execution configuration afforded by native CUDA C kernels. In contrast, while Numba CUDA provides a more accessible development model, its Python-based kernel definition introduces additional overhead and less aggressive compiler optimization opportunities.

The scaling trends observed indicate that both GPU implementations benefit from increasing arithmetic intensity, but CuPy demonstrates superior asymptotic efficiency. This suggests that once fixed overheads are amortized, kernel-level optimization and memory coalescing become the dominant factors influencing performance. The widening gap at larger problem sizes highlights the cumulative impact of these low-level optimizations.

Importantly, both GPU approaches maintain similar solver iteration counts and numerical behavior, confirming that the observed performance differences are purely architectural and implementation-driven. This reinforces the interpretation that RawKernel-based execution represents the upper bound of achievable single-GPU performance within the scope of this project.

This figure demonstrates that:
- Numba CUDA offers a strong balance between performance and development productivity;
- CuPy RawKernel achieves the best absolute performance for large-scale problems;
- Kernel-level control becomes increasingly important as problem size grows.

This analysis justifies the inclusion of both approaches in the study and positions CuPy RawKernel as the reference implementation for maximum-performance GPU execution in the final benchmarking comparisons.

![DESCRIPTION](images/documents/tutorial2/y_shaped_runtime_speedup.svg)

**Figure xxx - Runtime speedup of GPU-based implementations relative to the CPU baseline for the Y-shaped geometry.**  

### 4.4.9 GPU Speedup Analysis for the Y-Shaped Geometry
 
Figure XXX provides a combined view of absolute runtime scaling and relative speedup versus the CPU baseline for the Y-Shaped geometry, offering a comprehensive perspective on how different execution models behave as the number of nodes increases.

From the runtime curves (top panel), a clear hierarchy emerges. For small meshes (≈200 nodes), all implementations exhibit very low absolute runtimes, and differences between execution models are marginal. In this regime, GPU-based solvers (CuPy GPU and Numba CUDA) show no meaningful advantage and, in absolute terms, remain comparable to CPU execution. This behavior reflects the dominance of fixed overheads — kernel launch latency, device synchronization, and host-device data transfers — which prevent GPUs from exploiting parallelism at such small problem sizes. Consequently, GPU acceleration is ineffective and provides little to no speedup.

As mesh size increases, runtime growth becomes strongly execution-model dependent. CPU, threaded, and Numba CPU implementations show steep scaling trends, with execution time increasing rapidly as the number of nodes grows. Multiprocessing reduces the slope relative to the CPU baseline but remains significantly slower than GPU-based approaches due to process management overhead and limited scalability. In contrast, GPU implementations exhibit substantially flatter scaling curves, indicating much better asymptotic behavior.

This transition is more clearly quantified in the speedup plot (bottom panel). For the smallest mesh, all speedups remain close to unity, confirming that GPU execution does not amortize its overhead. However, beyond the intermediate mesh size (≈200k nodes), a pronounced acceleration regime emerges. GPU-based solvers rapidly surpass all CPU-based implementations, with speedup increasing sharply as problem size grows.

The CuPy RawKernel implementation consistently achieves the highest speedup across all large meshes, reaching several tens of times faster than the CPU baseline at the largest scale. Numba CUDA follows the same qualitative trend but achieves systematically lower speedups. This gap reflects differences in kernel maturity and execution efficiency: CuPy benefits from native CUDA C kernels with tighter control over memory access patterns, better register allocation, and reduced abstraction overhead, which become increasingly important as arithmetic intensity grows.

An important observation is the sublinear growth of speedup at the largest mesh sizes. Although GPU acceleration remains substantial, the rate of improvement decreases, indicating a transition from compute-bound assembly phases to solver-dominated execution. At this stage, sparse linear algebra operations become memory-bandwidth-bound, even on the GPU, imposing a fundamental limit on achievable acceleration.

Overall, this figure demonstrates that:

1. GPU acceleration is highly scale-dependent and ineffective for small FEM problems;
2. A clear CPU-GPU crossover occurs as mesh size increases, after which GPUs dominate;
3. CuPy RawKernel consistently delivers the highest asymptotic speedup;
4. Sparse solver performance ultimately constrains end-to-end scalability.

These results provide strong empirical confirmation that GPU acceleration is essential for large-scale FEM simulations, while also highlighting that further performance gains at extreme scales must focus on solver algorithms and memory efficiency rather than kernel-level optimizations alone.



![DESCRIPTION](images/documents/tutorial2/y_shaped_total_time_breakdown_2x2.svg)

**Figure XXX - Detailed runtime breakdown of the total execution time for the Y-shaped geometry across CPU and GPU execution models.** 

### 4.4.10 Runtime Breakdown Across Execution Models for the Y-Shaped Geometry
 
This figure decomposes the total runtime into its main computational stages—mesh loading, system assembly, boundary condition application, linear system solution, and post-processing—for the different execution models considered in this study. The comparison is performed for a representative mesh size, enabling direct inspection of how each execution model redistributes computational cost across the FEM pipeline.

The runtime breakdown highlights fundamental differences in how CPU- and GPU-based implementations allocate computational effort across the FEM workflow. In the CPU baseline and threaded variants, system assembly represents a dominant fraction of the total runtime. This reflects the interpreter-bound nature of element-level loops and sparse matrix insertion, where Python overhead and memory indirection significantly limit performance.

In contrast, the Numba JIT CPU implementation exhibits a markedly different profile. Assembly time is substantially reduced and no longer dominates the execution, confirming the effectiveness of JIT compilation and parallel execution in eliminating Python-level overhead. As a result, the linear solver becomes the primary runtime contributor, indicating a shift from compute-bound to memory-bandwidth-bound behavior.

The GPU-based implementations further accentuate this transition. For both Numba CUDA and CuPy RawKernel, the assembly and post-processing stages account for only a small fraction of total runtime. These stages scale efficiently on the GPU due to massive parallelism and high arithmetic throughput. The sparse linear solve clearly dominates the execution time, accounting for the majority of runtime. This dominance reflects the intrinsic memory-bound nature of sparse matrix-vector products, even on high-bandwidth GPU architectures.

A notable distinction between the two GPU implementations lies in the relative cost of non-solver stages. The CuPy RawKernel implementation exhibits slightly lower assembly and post-processing fractions compared to Numba CUDA, consistent with the use of native CUDA C kernels and reduced abstraction overhead. Boundary condition application and mesh loading remain negligible across all execution models, confirming that they do not materially influence performance at scale.

Overall, this breakdown provides critical insight into performance bottlenecks:
- CPU-based implementations are limited primarily by assembly overhead.
- JIT compilation shifts the bottleneck toward the solver.
- GPU acceleration virtually eliminates assembly cost, exposing the solver as the dominant constraint.
- Further performance gains on GPU would require more advanced sparse solvers or preconditioning strategies rather than additional kernel-level optimization.

This figure therefore complements the speedup analysis by explaining *why* acceleration saturates and *where* future optimization efforts should be focused within the FEM pipeline.


### 4.5 RTX 5090 Performance

Key results from performance benchmarks comparing FEM solver implementations.

**Backward-Facing Step (XS)** (287 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 30ms ± 2ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 4ms | 1.7x | 3 |
| CPU Multiprocess | 996ms ± 48ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 5.0x | 3 |
| Numba CUDA | 49ms ± 6ms | 0.6x | 3 |
| CuPy GPU | 58ms ± 0ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-0" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.7,0.0,5.0,0.6,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**Backward-Facing Step (M)** (195,362 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 41.59s ± 0.12s | 1.0x | 3 |
| CPU Threaded | 31.23s ± 0.18s | 1.3x | 3 |
| CPU Multiprocess | 23.66s ± 0.22s | 1.8x | 3 |
| Numba CPU | 23.41s ± 0.17s | 1.8x | 3 |
| Numba CUDA | 2.56s ± 0.10s | 16.2x | 3 |
| CuPy GPU | 1.45s ± 0.01s | 28.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-1" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.3,1.8,1.8,16.2,28.6]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (XS)** (411 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 42ms ± 5ms | 1.0x | 3 |
| CPU Threaded | 21ms ± 4ms | 2.0x | 3 |
| CPU Multiprocess | 911ms ± 28ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 1ms | 8.0x | 3 |
| Numba CUDA | 53ms ± 7ms | 0.8x | 3 |
| CuPy GPU | 82ms ± 4ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-2" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.0,0.0,8.0,0.8,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (M)** (161,984 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 36.82s ± 0.16s | 1.0x | 3 |
| CPU Threaded | 21.98s ± 0.16s | 1.7x | 3 |
| CPU Multiprocess | 16.65s ± 0.48s | 2.2x | 3 |
| Numba CPU | 21.35s ± 0.18s | 1.7x | 3 |
| Numba CUDA | 2.12s ± 0.07s | 17.3x | 3 |
| CuPy GPU | 1.25s ± 0.02s | 29.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-3" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.7,2.2,1.7,17.3,29.4]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (XS)** (387 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 27ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 15ms ± 3ms | 1.8x | 3 |
| CPU Multiprocess | 860ms ± 42ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 1ms | 4.2x | 3 |
| Numba CUDA | 54ms ± 5ms | 0.5x | 3 |
| CuPy GPU | 82ms ± 3ms | 0.3x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-4" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.8,0.0,4.2,0.5,0.3]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (M)** (196,078 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 42.00s ± 0.14s | 1.0x | 3 |
| CPU Threaded | 34.94s ± 0.62s | 1.2x | 3 |
| CPU Multiprocess | 26.73s ± 0.47s | 1.6x | 3 |
| Numba CPU | 27.79s ± 0.19s | 1.5x | 3 |
| Numba CUDA | 2.80s ± 0.14s | 15.0x | 3 |
| CuPy GPU | 1.77s ± 0.12s | 23.7x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-5" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.2,1.6,1.5,15.0,23.7]}],"yAxisName":"Speedup (x)"}'></div>


**T-Junction (XS)** (393 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 34ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 17ms ± 1ms | 2.0x | 3 |
| CPU Multiprocess | 890ms ± 23ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 5.7x | 3 |
| Numba CUDA | 70ms ± 4ms | 0.5x | 3 |
| CuPy GPU | 71ms ± 5ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-6" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.0,0.0,5.7,0.5,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**T-Junction (M)** (196,420 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 1m 39.2s ± 105.6s | 1.0x | 3 |
| CPU Threaded | 31.54s ± 0.61s | 3.1x | 3 |
| CPU Multiprocess | 25.29s ± 0.04s | 3.9x | 3 |
| Numba CPU | 24.14s ± 0.35s | 4.1x | 3 |
| Numba CUDA | 2.67s ± 0.10s | 37.2x | 3 |
| CuPy GPU | 1.64s ± 0.05s | 60.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-7" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[3.1,3.9,4.1,37.2,60.5]}],"yAxisName":"Speedup (x)"}'></div>

**Venturi (XS)** (341 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 30ms ± 2ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 4ms | 1.6x | 3 |
| CPU Multiprocess | 1.05s ± 0.02s | 0.0x | 3 |
| Numba CPU | <0.01s ± 1ms | 6.2x | 3 |
| Numba CUDA | 52ms ± 5ms | 0.6x | 3 |
| CuPy GPU | 76ms ± 4ms | 0.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-8" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.6,0.0,6.2,0.6,0.4]}],"yAxisName":"Speedup (x)"}'></div>

**Venturi (M)** (194,325 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 41.96s ± 0.16s | 1.0x | 3 |
| CPU Threaded | 31.34s ± 0.42s | 1.3x | 3 |
| CPU Multiprocess | 24.63s ± 1.60s | 1.7x | 3 |
| Numba CPU | 23.02s ± 0.36s | 1.8x | 3 |
| Numba CUDA | 2.48s ± 0.03s | 16.9x | 3 |
| CuPy GPU | 1.66s ± 0.03s | 25.3x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-9" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.3,1.7,1.8,16.9,25.3]}],"yAxisName":"Speedup (x)"}'></div>


**Y-Shaped (XS)** (201 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 19ms ± 2ms | 1.0x | 3 |
| CPU Threaded | 11ms ± 1ms | 1.8x | 3 |
| CPU Multiprocess | 1.00s ± 0.01s | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 3.5x | 3 |
| Numba CUDA | 40ms ± 2ms | 0.5x | 3 |
| CuPy GPU | 59ms ± 4ms | 0.3x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-10" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.8,0.0,3.5,0.5,0.3]}],"yAxisName":"Speedup (x)"}'></div>

**Y-Shaped (M)** (195,853 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 34.38s ± 1.80s | 1.0x | 3 |
| CPU Threaded | 26.06s ± 0.34s | 1.3x | 3 |
| CPU Multiprocess | 22.53s ± 0.22s | 1.5x | 3 |
| Numba CPU | 19.15s ± 0.03s | 1.8x | 3 |
| Numba CUDA | 2.36s ± 0.05s | 14.6x | 3 |
| CuPy GPU | 1.54s ± 0.01s | 22.3x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-11" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.3,1.5,1.8,14.6,22.3]}],"yAxisName":"Speedup (x)"}'></div>


### 4.5.1 Critical Analysis RTX 5090

#### 4.5.1.1 Small-Scale Problems (XS meshes)

Across all geometries with **XS meshes (≈200-400 nodes)**, the results show a **clear and consistent pattern**:

- **CPU-based implementations dominate** in absolute performance.
- **Numba JIT CPU** delivers the best performance overall, with speedups ranging from **3.5× to 8×** relative to the baseline.
- **GPU implementations (Numba CUDA and CuPy GPU)** are systematically *slower* than the CPU baseline, with speedups in the range **0.3×-0.8×**, i.e., actual slowdowns.

This behavior is expected and technically justified. For small meshes, the FEM workload is dominated by:
- Kernel launch overhead,
- CPU-GPU memory transfers,
- GPU context synchronization.

These fixed costs outweigh the benefits of massive parallelism. Even on a high-end accelerator such as the RTX 5090, GPU execution cannot amortize these overheads when the number of elements is insufficient. The results therefore confirm that **GPU acceleration is not advantageous for small FEM problems**, regardless of GPU capability.

Additionally, **CPU multiprocessing performs extremely poorly** in this regime, often exceeding 800-1000 ms, due to process startup and IPC overhead dominating the actual computation.


#### 4.5.1.2 Medium-Scale Problems (M meshes)

For **medium-scale meshes (≈160k-200k nodes)**, the performance profile changes radically and consistently across all geometries.

Key observations:

- **GPU acceleration becomes dominant**, with large and stable speedups.
- **Numba CUDA** achieves speedups between **14× and 37×**.
- **CuPy GPU (RawKernel)** consistently outperforms Numba CUDA, reaching speedups between **22× and 60×**.
- **CPU-based approaches saturate**, rarely exceeding **~2× speedup**, even with multiprocessing or JIT compilation.

This regime marks the **CPU-GPU crossover point**, where the arithmetic intensity and parallel workload are sufficient to fully utilize the GPU. The RTX 5090’s high memory bandwidth and massive thread parallelism allow element-level assembly and post-processing to become negligible relative to the solver.

Notably:
- The **largest speedups occur for the most complex geometries** (e.g., T-Junction and Venturi), indicating that GPU efficiency improves with increasing sparsity complexity and solver workload.
- **CuPy RawKernel consistently outperforms Numba CUDA** by a factor of ~1.4-1.7×, reflecting lower abstraction overhead and more optimized kernel execution.

#### 4.5.1.3 Solver-Dominated Regime and GPU Efficiency

For medium meshes, the performance gains plateau around **20×-60×**, despite the extreme computational power of the RTX 5090. This reflects a fundamental algorithmic limit:

- Once assembly and post-processing are accelerated, the **sparse linear solver dominates runtime**.
- Sparse Conjugate Gradient is inherently **memory-bandwidth bound**, even on GPUs.
- Further gains would require more advanced preconditioners or solver algorithms, not faster kernels.

This observation aligns with the runtime breakdown analysis and confirms that the RTX 5090 is **not compute-limited**, but rather constrained by sparse memory access patterns intrinsic to FEM solvers.

#### 4.5.1.4 Comparative Assessment of Execution Models

From a practical standpoint, the benchmark results support the following conclusions:

| Regime | Best Execution Model | Rationale |
|------|----------------------|-----------|
| XS meshes | Numba JIT CPU | Minimal overhead, compiled execution |
| M meshes | CuPy GPU (RawKernel) | Maximum throughput, full GPU residency |
| CPU-only environments | Numba JIT CPU | Best balance of speed and memory efficiency |
| GPU prototyping | Numba CUDA | Easier development, acceptable performance |
| Production GPU | CuPy RawKernel | Highest and most consistent speedups |

The RTX 5090 demonstrates **excellent scalability** once the problem size justifies GPU usage. However, it also highlights that **hardware capability alone is insufficient**: algorithmic structure, execution model, and problem scale are decisive.

The benchmark results confirm that the RTX 5090 is exceptionally well suited for **large-scale FEM simulations**, delivering order-of-magnitude speedups over CPU execution when properly utilized. At the same time, the data reinforces several critical best practices:

- GPU acceleration should be **selectively applied**, not used indiscriminately.
- Small and interactive FEM problems are better served by optimized CPU execution.
- For large-scale production workloads, **RawKernel-based GPU implementations provide the highest return on investment**.
- The solver, not the kernel, becomes the ultimate bottleneck at scale.

The RTX 5090 establishes a clear upper bound for single-GPU FEM performance in this study, validating the architectural choices made in the GPU implementations while providing quantitative evidence of when and why GPU acceleration is most effective.

### Bottleneck Evolution

As optimizations progress, the computational bottleneck shifts:

#### Backward-Facing Step (XS) - 287 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (64%) | Post-Proc (18%) |
| CPU Threaded | Assembly (46%) | Post-Proc (22%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (50%) |
| Numba CPU | Solve (46%) | BC (13%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (69%) | BC (25%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-0" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":63.7},{"name":"Solve","value":7.2},{"name":"Apply BC","value":0.7},{"name":"Post-Process","value":17.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-1" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.4},{"name":"Solve","value":68.7},{"name":"Apply BC","value":24.7},{"name":"Post-Process","value":0.7}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-2" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[63.7,46.0,49.7,12.5,6.0,2.4]},{"name":"Solve","data":[7.2,15.8,0.3,45.9,86.7,68.7]},{"name":"Apply BC","data":[0.7,3.3,0.1,12.9,1.3,24.7]},{"name":"Post-Process","data":[17.8,22.3,49.7,1.3,2.4,0.7]}],"yAxisName":"Percentage (%)"}'></div>

#### Backward-Facing Step (M) - 195,362 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (53%) | Assembly (36%) |
| CPU Threaded | Solve (72%) | Assembly (17%) |
| CPU Multiprocess | Solve (88%) | Assembly (5%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (51%) | Assembly (25%) |
| CuPy GPU | Solve (90%) | BC (10%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-3" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":36.5},{"name":"Solve","value":53.3},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":10.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-4" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":89.9},{"name":"Apply BC","value":9.6},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-5" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[36.5,17.2,5.4,0.6,25.1,0.1]},{"name":"Solve","data":[53.3,72.5,87.7,97.0,51.3,89.9]},{"name":"Apply BC","data":[0.2,1.9,2.0,2.4,22.2,9.6]},{"name":"Post-Process","data":[10.0,8.5,4.9,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (XS) - 411 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (70%) | Post-Proc (20%) |
| CPU Threaded | Assembly (50%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (45%) | BC (17%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (65%) | BC (30%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-6" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":69.6},{"name":"Solve","value":5.8},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":20.3}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-7" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.4},{"name":"Solve","value":65.0},{"name":"Apply BC","value":29.8},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-8" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[69.6,50.4,50.5,12.9,5.8,2.4]},{"name":"Solve","data":[5.8,10.9,0.3,44.7,87.3,65.0]},{"name":"Apply BC","data":[1.1,4.6,0.1,16.7,1.6,29.8]},{"name":"Post-Process","data":[20.3,23.7,49.0,1.5,2.2,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (M) - 161,984 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (56%) | Assembly (35%) |
| CPU Threaded | Solve (68%) | Assembly (20%) |
| CPU Multiprocess | Solve (84%) | Assembly (7%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (50%) | Assembly (25%) |
| CuPy GPU | Solve (85%) | BC (15%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-9" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":34.8},{"name":"Solve","value":55.5},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":9.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-10" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":84.8},{"name":"Apply BC","value":14.7},{"name":"Post-Process","value":0.1}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-11" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[34.8,20.3,6.9,0.5,25.3,0.1]},{"name":"Solve","data":[55.5,67.8,84.5,97.2,50.2,84.8]},{"name":"Apply BC","data":[0.3,2.1,2.4,2.2,23.0,14.7]},{"name":"Post-Process","data":[9.4,9.7,6.1,0.0,0.1,0.1]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (XS) - 387 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (65%) | Post-Proc (19%) |
| CPU Threaded | Assembly (46%) | Post-Proc (26%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (44%) | BC (17%) |
| Numba CUDA | Solve (84%) | Assembly (8%) |
| CuPy GPU | Solve (69%) | BC (24%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-12" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":65.2},{"name":"Solve","value":8.8},{"name":"Apply BC","value":1.4},{"name":"Post-Process","value":19.2}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-13" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.9},{"name":"Solve","value":69.0},{"name":"Apply BC","value":24.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-14" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[65.2,45.8,50.5,10.8,7.7,1.9]},{"name":"Solve","data":[8.8,14.5,0.3,43.7,84.2,69.0]},{"name":"Apply BC","data":[1.4,4.9,0.1,16.7,2.0,24.4]},{"name":"Post-Process","data":[19.2,26.2,49.0,1.1,2.0,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (M) - 196,078 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (59%) | Assembly (32%) |
| CPU Threaded | Solve (74%) | Assembly (17%) |
| CPU Multiprocess | Solve (89%) | Assembly (5%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (55%) | Assembly (24%) |
| CuPy GPU | Solve (89%) | BC (10%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-15" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":31.9},{"name":"Solve","value":59.0},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":8.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-16" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":89.2},{"name":"Apply BC","value":10.4},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-17" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[31.9,16.6,4.9,0.6,24.1,0.1]},{"name":"Solve","data":[59.0,74.1,88.8,97.2,54.9,89.2]},{"name":"Apply BC","data":[0.2,1.5,1.7,2.2,19.7,10.4]},{"name":"Post-Process","data":[8.8,7.8,4.6,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### T-Junction (XS) - 393 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (67%) | Post-Proc (20%) |
| CPU Threaded | Assembly (52%) | Post-Proc (22%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (47%) | BC (16%) |
| Numba CUDA | Solve (89%) | Assembly (5%) |
| CuPy GPU | Solve (65%) | BC (30%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-20" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":67.1},{"name":"Solve","value":8.3},{"name":"Apply BC","value":0.9},{"name":"Post-Process","value":20.1}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-21" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.9},{"name":"Solve","value":64.8},{"name":"Apply BC","value":29.7},{"name":"Post-Process","value":0.4}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-22" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[67.1,52.4,50.2,13.1,5.2,1.9]},{"name":"Solve","data":[8.3,14.2,0.4,46.8,88.6,64.8]},{"name":"Apply BC","data":[0.9,4.2,0.1,15.5,1.5,29.7]},{"name":"Post-Process","data":[20.1,21.8,49.2,1.2,2.1,0.4]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (M) - 196,420 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (83%) | Assembly (14%) |
| CPU Threaded | Solve (75%) | Assembly (16%) |
| CPU Multiprocess | Solve (89%) | Assembly (5%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (55%) | Assembly (24%) |
| CuPy GPU | Solve (88%) | BC (11%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-23" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":13.6},{"name":"Solve","value":82.6},{"name":"Apply BC","value":0.1},{"name":"Post-Process","value":3.7}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-24" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":88.3},{"name":"Apply BC","value":11.3},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-25" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[13.6,15.7,5.0,0.6,24.1,0.1]},{"name":"Solve","data":[82.6,75.1,88.5,97.1,54.7,88.3]},{"name":"Apply BC","data":[0.1,1.7,1.8,2.3,20.1,11.3]},{"name":"Post-Process","data":[3.7,7.5,4.6,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### Venturi (XS) - 341 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (65%) | Post-Proc (19%) |
| CPU Threaded | Assembly (47%) | Post-Proc (22%) |
| CPU Multiprocess | Post-Proc (50%) | Assembly (50%) |
| Numba CPU | Solve (45%) | BC (16%) |
| Numba CUDA | Solve (85%) | Assembly (7%) |
| CuPy GPU | Solve (60%) | BC (34%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-27" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":65.2},{"name":"Solve","value":7.6},{"name":"Apply BC","value":1.2},{"name":"Post-Process","value":18.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-28" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.3},{"name":"Solve","value":59.8},{"name":"Apply BC","value":34.3},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-29" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[65.2,47.1,49.5,12.3,6.7,2.3]},{"name":"Solve","data":[7.6,14.7,0.3,44.9,85.4,59.8]},{"name":"Apply BC","data":[1.2,4.7,0.1,16.4,1.8,34.3]},{"name":"Post-Process","data":[18.8,22.5,49.9,1.7,2.3,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (M) - 194,325 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (55%) | Assembly (35%) |
| CPU Threaded | Solve (72%) | Assembly (17%) |
| CPU Multiprocess | Solve (87%) | Assembly (6%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (52%) | Assembly (25%) |
| CuPy GPU | Solve (81%) | BC (19%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-30" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":35.3},{"name":"Solve","value":54.7},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":9.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-31" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":80.8},{"name":"Apply BC","value":18.8},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-32" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[35.3,17.4,6.0,0.5,25.0,0.1]},{"name":"Solve","data":[54.7,72.3,86.6,97.1,52.2,80.8]},{"name":"Apply BC","data":[0.2,1.8,2.2,2.3,21.5,18.8]},{"name":"Post-Process","data":[9.8,8.4,5.2,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### Y-Shaped (XS) - 201 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (58%) | Post-Proc (18%) |
| CPU Threaded | Assembly (45%) | Solve (18%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (39%) | Assembly (13%) |
| Numba CUDA | Solve (80%) | Assembly (8%) |
| CuPy GPU | Solve (61%) | BC (32%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-35" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":58.0},{"name":"Solve","value":10.7},{"name":"Apply BC","value":1.3},{"name":"Post-Process","value":18.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-36" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.3},{"name":"Solve","value":61.3},{"name":"Apply BC","value":32.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-37" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[58.0,45.3,50.0,13.1,8.4,2.3]},{"name":"Solve","data":[10.7,17.8,0.3,38.9,80.4,61.3]},{"name":"Apply BC","data":[1.3,6.1,0.1,10.2,1.3,32.4]},{"name":"Post-Process","data":[18.0,17.2,49.5,1.6,2.5,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Y-Shaped (M) - 195,853 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (56%) | Assembly (34%) |
| CPU Threaded | Solve (74%) | Assembly (16%) |
| CPU Multiprocess | Solve (87%) | Assembly (6%) |
| Numba CPU | Solve (96%) | BC (3%) |
| Numba CUDA | Solve (51%) | BC (26%) |
| CuPy GPU | Solve (80%) | BC (18%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-38" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":34.2},{"name":"Solve","value":55.9},{"name":"Apply BC","value":0.4},{"name":"Post-Process","value":9.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-39" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":80.4},{"name":"Apply BC","value":18.0},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-40" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[34.2,16.2,5.7,0.7,21.3,0.1]},{"name":"Solve","data":[55.9,73.8,86.8,96.2,51.1,80.4]},{"name":"Apply BC","data":[0.4,2.2,2.3,3.0,25.9,18.0]},{"name":"Post-Process","data":[9.4,7.7,5.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


### Why Each Optimization Helps

| Transition | Reason |
|------------|--------|
| Baseline → Threaded | Limited by Python GIL; threads only help for I/O |
| Threaded → Multiprocess | Bypasses GIL via separate processes; IPC overhead limits gains |
| Multiprocess → Numba CPU | JIT compilation eliminates interpreter overhead; true parallel loops |
| Numba CPU → Numba CUDA | GPU parallelism: thousands of threads vs dozens of CPU cores |
| Numba CUDA → CuPy GPU | CUDA C kernels more optimized than Numba-generated PTX |

### RTX 5090 Bottleneck Analysis:
- 
The benchmark results clearly demonstrate that, on the RTX 5090, the computational bottleneck of the FEM pipeline shifts almost entirely from low-level numerical kernels to higher-level algorithmic components. Once GPU acceleration is introduced, the element assembly and post-processing stages become negligible in relative terms, often accounting for less than 2-3% of total runtime in medium-sized meshes. Instead, the **linear solver phase** emerges as the dominant cost, consistently representing **80-90% of the total execution time** in the CuPy implementation. This behavior highlights the fundamentally **memory-bound nature of sparse linear algebra**, particularly the repeated Sparse Matrix-Vector multiplications (SpMV) required by the Conjugate Gradient method. Although the RTX 5090 provides extremely high memory bandwidth, the irregular access patterns of sparse matrices prevent full utilization of the hardware’s theoretical throughput.  

In addition, the **application of boundary conditions** becomes a visible secondary bottleneck, especially for small-scale (XS) problems, where fixed overheads such as kernel launches, synchronization points, and conditional logic are not amortized by problem size. This explains why GPU-based approaches may underperform highly optimized CPU JIT implementations for small meshes, even on top-tier GPUs. From an optimization perspective, these results indicate that further performance gains on the RTX 5090 will not come from accelerating assembly kernels, but rather from **reducing solver iteration counts**, improving preconditioning strategies, and restructuring boundary condition handling to minimize synchronization and memory traffic. Overall, the RTX 5090 effectively exposes the **numerical and memory-efficiency limits** of the current solver design, marking a transition point where performance is constrained more by algorithmic choices than by raw computational power.


### 4.6 RTX 4090 Performance

Key results from performance benchmarks comparing FEM solver implementations.

**Backward-Facing Step (XS)** (287 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 25ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 17ms ± 2ms | 1.5x | 3 |
| CPU Multiprocess | 233ms ± 10ms | 0.1x | 3 |
| Numba CPU | <0.01s ± 0ms | 6.1x | 3 |
| Numba CUDA | 45ms ± 10ms | 0.6x | 3 |
| CuPy GPU | 51ms ± 4ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-0" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.5,0.1,6.1,0.6,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**Backward-Facing Step (M)** (195,362 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 46.68s ± 1.66s | 1.0x | 3 |
| CPU Threaded | 37.41s ± 0.38s | 1.2x | 3 |
| CPU Multiprocess | 26.60s ± 0.18s | 1.8x | 3 |
| Numba CPU | 29.17s ± 2.04s | 1.6x | 3 |
| Numba CUDA | 2.40s ± 0.03s | 19.4x | 3 |
| CuPy GPU | 1.27s ± 0.02s | 36.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-1" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.2,1.8,1.6,19.4,36.6]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (XS)** (411 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 38ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 1ms | 2.1x | 3 |
| CPU Multiprocess | 230ms ± 5ms | 0.2x | 3 |
| Numba CPU | <0.01s ± 2ms | 5.5x | 3 |
| Numba CUDA | 44ms ± 1ms | 0.9x | 3 |
| CuPy GPU | 53ms ± 2ms | 0.7x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-2" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.1,0.2,5.5,0.9,0.7]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (M)** (161,984 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 43.70s ± 0.62s | 1.0x | 3 |
| CPU Threaded | 27.97s ± 0.44s | 1.6x | 3 |
| CPU Multiprocess | 19.95s ± 1.64s | 2.2x | 3 |
| Numba CPU | 27.40s ± 1.48s | 1.6x | 3 |
| Numba CUDA | 1.97s ± 0.02s | 22.1x | 3 |
| CuPy GPU | 1.05s ± 0.05s | 41.7x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-3" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.6,2.2,1.6,22.1,41.7]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (XS)** (387 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 34ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 17ms ± 3ms | 2.1x | 3 |
| CPU Multiprocess | 219ms ± 6ms | 0.2x | 3 |
| Numba CPU | <0.01s ± 2ms | 5.8x | 3 |
| Numba CUDA | 63ms ± 6ms | 0.5x | 3 |
| CuPy GPU | 62ms ± 8ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-4" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.1,0.2,5.8,0.5,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (M)** (196,078 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 38.82s ± 0.42s | 1.0x | 3 |
| CPU Threaded | 42.24s ± 1.49s | 0.9x | 3 |
| CPU Multiprocess | 31.68s ± 1.12s | 1.2x | 3 |
| Numba CPU | 33.08s ± 0.33s | 1.2x | 3 |
| Numba CUDA | 2.44s ± 0.05s | 15.9x | 3 |
| CuPy GPU | 1.54s ± 0.05s | 25.2x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-5" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[0.9,1.2,1.2,15.9,25.2]}],"yAxisName":"Speedup (x)"}'></div>


**T-Junction (XS)** (393 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 35ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 2ms | 1.9x | 3 |
| CPU Multiprocess | 232ms ± 8ms | 0.1x | 3 |
| Numba CPU | <0.01s ± 1ms | 6.1x | 3 |
| Numba CUDA | 48ms ± 4ms | 0.7x | 3 |
| CuPy GPU | 64ms ± 8ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-6" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.9,0.1,6.1,0.7,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**T-Junction (M)** (196,420 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 48.55s ± 0.08s | 1.0x | 3 |
| CPU Threaded | 41.12s ± 0.61s | 1.2x | 3 |
| CPU Multiprocess | 29.56s ± 1.14s | 1.6x | 3 |
| Numba CPU | 28.00s ± 1.65s | 1.7x | 3 |
| Numba CUDA | 2.33s ± 0.03s | 20.8x | 3 |
| CuPy GPU | 1.43s ± 0.01s | 34.0x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-7" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.2,1.6,1.7,20.8,34.0]}],"yAxisName":"Speedup (x)"}'></div>

**Venturi (XS)** (341 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 30ms ± 2ms | 1.0x | 3 |
| CPU Threaded | 15ms ± 3ms | 2.0x | 3 |
| CPU Multiprocess | 203ms ± 11ms | 0.1x | 3 |
| Numba CPU | <0.01s ± 0ms | 6.6x | 3 |
| Numba CUDA | 60ms ± 9ms | 0.5x | 3 |
| CuPy GPU | 53ms ± 2ms | 0.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-8" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.0,0.1,6.6,0.5,0.6]}],"yAxisName":"Speedup (x)"}'></div>

**Venturi (M)** (194,325 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 32.22s ± 0.16s | 1.0x | 3 |
| CPU Threaded | 38.35s ± 0.25s | 0.8x | 3 |
| CPU Multiprocess | 27.15s ± 1.38s | 1.2x | 3 |
| Numba CPU | 29.96s ± 0.27s | 1.1x | 3 |
| Numba CUDA | 2.38s ± 0.08s | 13.5x | 3 |
| CuPy GPU | 1.40s ± 0.05s | 23.0x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-9" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[0.8,1.2,1.1,13.5,23.0]}],"yAxisName":"Speedup (x)"}'></div>


**Y-Shaped (XS)** (201 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 20ms ± 3ms | 1.0x | 6 |
| CPU Threaded | <0.01s ± 1ms | 2.1x | 3 |
| CPU Multiprocess | 169ms ± 5ms | 0.1x | 3 |
| Numba CPU | <0.01s ± 0ms | 5.3x | 3 |
| Numba CUDA | 53ms ± 13ms | 0.4x | 3 |
| CuPy GPU | 52ms ± 6ms | 0.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-10" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.1,0.1,5.3,0.4,0.4]}],"yAxisName":"Speedup (x)"}'></div>

**Y-Shaped (M)** (195,853 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 25.05s ± 0.49s | 1.0x | 3 |
| CPU Threaded | 31.45s ± 0.54s | 0.8x | 3 |
| CPU Multiprocess | 23.40s ± 0.34s | 1.1x | 3 |
| Numba CPU | 22.86s ± 0.32s | 1.1x | 3 |
| Numba CUDA | 2.20s ± 0.08s | 11.4x | 3 |
| CuPy GPU | 1.35s ± 0.01s | 18.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-11" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[0.8,1.1,1.1,11.4,18.6]}],"yAxisName":"Speedup (x)"}'></div>


### 4.6.1 Critical Analysis - RTX 4090 Performance

#### 4.6.1.1 Small-Scale Problems (XS meshes)

For **small-scale meshes (XS, ≈200-400 nodes)**, the performance behavior on the RTX 4090 closely mirrors that observed on the RTX 5090, with an even stronger emphasis on overhead dominance. Across all geometries, **CPU-based implementations clearly outperform GPU-based approaches**. In particular, **Numba JIT CPU** consistently delivers the best absolute performance, achieving speedups in the range of **5× to 6.6×** relative to the CPU baseline, while GPU executions remain below parity.

Both **Numba CUDA** and **CuPy GPU** exhibit speedups between **0.4× and 0.9×**, indicating systematic slowdowns. This outcome is technically expected: for XS meshes, the FEM workload is too small to amortize the fixed costs associated with GPU execution, including kernel launch latency, device synchronization, and internal memory management. On the RTX 4090, these overheads are comparable in magnitude to the total computation time, effectively neutralizing the benefits of GPU parallelism.

As in the RTX 5090 case, **CPU multiprocessing performs particularly poorly**, often exceeding hundreds of milliseconds, due to process creation and inter-process communication overhead. These results confirm that, regardless of GPU tier, **small FEM problems are fundamentally ill-suited for GPU acceleration**, and that optimized CPU JIT execution remains the most effective strategy in this regime.

#### 4.6.1.2 Medium-Scale Problems (M meshes)

For **medium-scale meshes (M, ≈160k-200k nodes)**, the RTX 4090 transitions into a regime where GPU acceleration becomes clearly beneficial. In this setting, both GPU implementations deliver substantial and stable speedups across all geometries. **Numba CUDA** achieves speedups between **11× and 22×**, while **CuPy GPU** consistently outperforms it, reaching speedups in the range of **18× to 42×**.

Although these gains are significant, they remain systematically lower than those observed on the RTX 5090. This indicates that, while the RTX 4090 is capable of efficiently exploiting large-scale parallelism, it reaches saturation earlier due to lower memory bandwidth and reduced peak throughput. CPU-based approaches, including Numba CPU and multiprocessing, largely plateau at **~1.5×-2× speedup**, confirming that the CPU becomes the limiting factor once problem size increases.

Importantly, the strongest GPU speedups are again observed for more complex geometries, such as **T-Junction** and **Elbow 90°**, where solver workload and sparsity structure increase arithmetic intensity. This reinforces the conclusion that GPU efficiency improves with problem complexity and solver dominance.

#### 4.6.1.3 Solver-Dominated Regime and Architectural Limits

In the medium-mesh regime, the RTX 4090 clearly enters a **solver-dominated execution profile**. Once assembly and post-processing are accelerated on the GPU, the **sparse linear solver accounts for the vast majority of runtime**, typically exceeding **80% of total execution time** in the CuPy GPU implementation. This behavior confirms that performance is constrained by **memory-bound Sparse Matrix-Vector multiplications**, rather than by floating-point throughput.

Compared to the RTX 5090, the RTX 4090 exhibits a stronger sensitivity to memory access patterns and cache efficiency. While the solver benefits from GPU parallelism, irregular sparsity patterns limit effective bandwidth utilization, preventing linear scaling with hardware capability. Additionally, the **application of boundary conditions** emerges as a noticeable secondary cost, particularly in cases where scattered memory writes and synchronization are required.

These observations indicate that further performance improvements on the RTX 4090 would require **algorithmic-level optimizations**, such as improved preconditioning, reduced iteration counts, or alternative solver strategies, rather than additional kernel-level tuning.

#### 4.6.1.4 Comparative Assessment of Execution Models

From a practical and architectural perspective, the RTX 4090 benchmarks support the following conclusions:

| Regime | Best Execution Model | Rationale |
|------|----------------------|-----------|
| XS meshes | Numba JIT CPU | Lowest overhead, efficient JIT execution |
| M meshes | CuPy GPU (RawKernel) | Best GPU throughput and solver integration |
| CPU-only systems | Numba JIT CPU | Optimal balance of performance and simplicity |
| GPU prototyping | Numba CUDA | Easier development, reasonable speedups |
| Production GPU | CuPy RawKernel | Most stable and highest GPU performance |

Overall, the RTX 4090 demonstrates **excellent scalability for medium-scale FEM problems**, delivering order-of-magnitude speedups over CPU execution when problem size justifies GPU use. However, the results also show that its performance ceiling is reached earlier than that of the RTX 5090, underscoring the importance of memory bandwidth and solver efficiency in large-scale FEM workloads.

The benchmarks confirm that the RTX 4090 is a strong and well-balanced GPU for FEM acceleration, but they also reinforce a key insight of this study: **once GPU acceleration is enabled, performance is dictated more by algorithmic structure and sparse linear algebra efficiency than by raw compute power alone**.



### Bottleneck Evolution

As optimizations progress, the computational bottleneck shifts:

#### Backward-Facing Step (XS) - 287 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (68%) | Post-Proc (20%) |
| CPU Threaded | Assembly (52%) | Post-Proc (21%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (48%) |
| Numba CPU | Solve (45%) | Assembly (14%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (72%) | BC (22%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-0" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.2},{"name":"Solve","value":7.2},{"name":"Apply BC","value":0.9},{"name":"Post-Process","value":19.7}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-1" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.0},{"name":"Solve","value":71.7},{"name":"Apply BC","value":22.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-2" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.2,51.6,49.8,14.2,6.3,2.0]},{"name":"Solve","data":[7.2,11.7,0.9,44.6,86.8,71.7]},{"name":"Apply BC","data":[0.9,3.7,0.3,12.2,1.2,22.4]},{"name":"Post-Process","data":[19.7,21.0,48.4,3.0,2.1,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Backward-Facing Step (M) - 195,362 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (59%) | Assembly (32%) |
| CPU Threaded | Solve (67%) | Assembly (19%) |
| CPU Multiprocess | Solve (92%) | Assembly (3%) |
| Numba CPU | Solve (98%) | BC (1%) |
| Numba CUDA | Solve (49%) | Assembly (30%) |
| CuPy GPU | Solve (89%) | BC (11%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-3" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":31.9},{"name":"Solve","value":58.9},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":9.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-4" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":88.7},{"name":"Apply BC","value":10.9},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-5" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[31.9,19.1,3.5,0.6,29.8,0.1]},{"name":"Solve","data":[58.9,67.3,92.1,98.0,49.1,88.7]},{"name":"Apply BC","data":[0.2,1.2,1.7,1.4,20.0,10.9]},{"name":"Post-Process","data":[9.0,12.5,2.7,0.0,0.2,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (XS) - 411 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (21%) |
| CPU Threaded | Assembly (53%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (47%) |
| Numba CPU | Solve (39%) | BC (16%) |
| Numba CUDA | Solve (86%) | Assembly (7%) |
| CuPy GPU | Solve (67%) | BC (27%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-6" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.7},{"name":"Solve","value":6.3},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":21.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-7" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.0},{"name":"Solve","value":66.9},{"name":"Apply BC","value":27.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-8" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.7,52.8,50.7,15.1,6.6,2.0]},{"name":"Solve","data":[6.3,11.5,0.9,39.3,86.0,66.9]},{"name":"Apply BC","data":[1.1,4.7,0.5,15.9,1.9,27.4]},{"name":"Post-Process","data":[21.0,23.6,47.4,3.0,2.4,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (M) - 161,984 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (63%) | Assembly (28%) |
| CPU Threaded | Solve (64%) | Assembly (21%) |
| CPU Multiprocess | Solve (92%) | Assembly (4%) |
| Numba CPU | Solve (98%) | BC (1%) |
| Numba CUDA | Solve (49%) | Assembly (31%) |
| CuPy GPU | Solve (83%) | BC (16%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-9" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":28.5},{"name":"Solve","value":63.4},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":7.9}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-10" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":83.1},{"name":"Apply BC","value":16.5},{"name":"Post-Process","value":0.1}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-11" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[28.5,21.0,3.8,0.5,30.9,0.1]},{"name":"Solve","data":[63.4,64.1,91.6,98.2,48.7,83.1]},{"name":"Apply BC","data":[0.2,1.2,1.9,1.3,19.2,16.5]},{"name":"Post-Process","data":[7.9,13.7,2.6,0.0,0.2,0.1]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (XS) - 387 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (68%) | Post-Proc (17%) |
| CPU Threaded | Assembly (55%) | Post-Proc (17%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (46%) |
| Numba CPU | Solve (39%) | BC (16%) |
| Numba CUDA | Solve (91%) | Assembly (4%) |
| CuPy GPU | Solve (66%) | BC (29%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-12" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":67.5},{"name":"Solve","value":7.2},{"name":"Apply BC","value":1.2},{"name":"Post-Process","value":17.1}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-13" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.6},{"name":"Solve","value":65.8},{"name":"Apply BC","value":29.0},{"name":"Post-Process","value":0.4}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-14" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[67.5,54.7,51.2,14.6,4.0,1.6]},{"name":"Solve","data":[7.2,12.5,1.2,38.9,91.2,65.8]},{"name":"Apply BC","data":[1.2,4.5,0.5,15.9,1.2,29.0]},{"name":"Post-Process","data":[17.1,17.2,46.0,2.0,1.6,0.4]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (M) - 196,078 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (47%) | Assembly (42%) |
| CPU Threaded | Solve (71%) | Assembly (17%) |
| CPU Multiprocess | Solve (94%) | Assembly (3%) |
| Numba CPU | Solve (98%) | BC (1%) |
| Numba CUDA | Solve (53%) | Assembly (28%) |
| CuPy GPU | Solve (89%) | BC (11%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-15" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":41.6},{"name":"Solve","value":46.8},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":11.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-16" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":88.6},{"name":"Apply BC","value":11.1},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-17" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[41.6,16.9,2.9,0.5,27.8,0.1]},{"name":"Solve","data":[46.8,70.8,93.5,98.2,52.9,88.6]},{"name":"Apply BC","data":[0.3,1.1,1.4,1.3,18.3,11.1]},{"name":"Post-Process","data":[11.4,11.2,2.2,0.0,0.2,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### T-Junction (XS) - 393 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (20%) |
| CPU Threaded | Assembly (52%) | Post-Proc (23%) |
| CPU Multiprocess | Assembly (52%) | Post-Proc (46%) |
| Numba CPU | Solve (41%) | Assembly (15%) |
| Numba CUDA | Solve (85%) | Assembly (8%) |
| CuPy GPU | Solve (73%) | BC (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-19" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.8},{"name":"Solve","value":6.9},{"name":"Apply BC","value":1.0},{"name":"Post-Process","value":19.7}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-20" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.4},{"name":"Solve","value":72.8},{"name":"Apply BC","value":23.1},{"name":"Post-Process","value":0.4}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-21" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.8,52.4,52.3,14.6,7.8,1.4]},{"name":"Solve","data":[6.9,12.2,1.0,40.5,84.6,72.8]},{"name":"Apply BC","data":[1.0,4.1,0.5,14.6,1.7,23.1]},{"name":"Post-Process","data":[19.7,22.8,45.5,2.1,2.4,0.4]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (M) - 196,420 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (60%) | Assembly (31%) |
| CPU Threaded | Solve (70%) | Assembly (18%) |
| CPU Multiprocess | Solve (93%) | Assembly (3%) |
| Numba CPU | Solve (98%) | BC (1%) |
| Numba CUDA | Solve (51%) | Assembly (29%) |
| CuPy GPU | Solve (87%) | BC (12%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-22" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":31.4},{"name":"Solve","value":59.8},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":8.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-23" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":87.4},{"name":"Apply BC","value":12.2},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-24" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[31.4,17.5,3.0,0.5,29.1,0.1]},{"name":"Solve","data":[59.8,69.8,93.0,98.0,51.2,87.4]},{"name":"Apply BC","data":[0.2,1.1,1.5,1.5,18.6,12.2]},{"name":"Post-Process","data":[8.6,11.6,2.5,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (XS) - 341 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (65%) | Post-Proc (17%) |
| CPU Threaded | Assembly (50%) | Post-Proc (23%) |
| CPU Multiprocess | Assembly (52%) | Post-Proc (46%) |
| Numba CPU | Solve (43%) | BC (16%) |
| Numba CUDA | Solve (90%) | Assembly (5%) |
| CuPy GPU | Solve (65%) | BC (30%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-25" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":65.1},{"name":"Solve","value":6.4},{"name":"Apply BC","value":1.3},{"name":"Post-Process","value":17.1}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-26" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.7},{"name":"Solve","value":64.7},{"name":"Apply BC","value":30.1},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-27" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[65.1,49.8,52.0,14.0,4.8,1.7]},{"name":"Solve","data":[6.4,12.3,1.0,43.3,89.5,64.7]},{"name":"Apply BC","data":[1.3,4.9,0.5,15.6,1.4,30.1]},{"name":"Post-Process","data":[17.1,22.5,45.9,3.0,2.0,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (M) - 194,325 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (45%) | Assembly (43%) |
| CPU Threaded | Solve (68%) | Assembly (19%) |
| CPU Multiprocess | Solve (92%) | Assembly (3%) |
| Numba CPU | Solve (98%) | BC (1%) |
| Numba CUDA | Solve (48%) | Assembly (29%) |
| CuPy GPU | Solve (81%) | BC (19%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-28" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":42.7},{"name":"Solve","value":45.2},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":11.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-29" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":81.0},{"name":"Apply BC","value":18.7},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-30" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[42.7,18.7,3.3,0.6,29.3,0.1]},{"name":"Solve","data":[45.2,67.7,92.5,98.0,48.3,81.0]},{"name":"Apply BC","data":[0.3,1.2,1.7,1.4,21.5,18.7]},{"name":"Post-Process","data":[11.8,12.4,2.5,0.0,0.2,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### Y-Shaped (XS) - 201 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (63%) | Post-Proc (17%) |
| CPU Threaded | Assembly (44%) | Post-Proc (22%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (48%) |
| Numba CPU | Solve (39%) | Assembly (14%) |
| Numba CUDA | Solve (89%) | Assembly (5%) |
| CuPy GPU | Solve (60%) | BC (30%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-33" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":62.8},{"name":"Solve","value":8.5},{"name":"Apply BC","value":1.3},{"name":"Post-Process","value":17.1}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-34" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":2.9},{"name":"Solve","value":60.4},{"name":"Apply BC","value":30.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-35" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[62.8,43.8,49.8,14.0,4.9,2.9]},{"name":"Solve","data":[8.5,16.4,1.0,39.4,88.6,60.4]},{"name":"Apply BC","data":[1.3,5.2,0.4,11.0,1.1,30.4]},{"name":"Post-Process","data":[17.1,21.9,48.0,3.3,2.2,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Y-Shaped (M) - 195,853 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (45%) | Solve (42%) |
| CPU Threaded | Solve (71%) | Assembly (17%) |
| CPU Multiprocess | Solve (92%) | Assembly (3%) |
| Numba CPU | Solve (97%) | BC (2%) |
| Numba CUDA | Solve (50%) | Assembly (26%) |
| CuPy GPU | Solve (79%) | BC (19%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-36" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":44.8},{"name":"Solve","value":42.4},{"name":"Apply BC","value":0.5},{"name":"Post-Process","value":12.3}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-37" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":79.4},{"name":"Apply BC","value":19.0},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-38" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[44.8,16.7,3.0,0.6,25.9,0.1]},{"name":"Solve","data":[42.4,70.9,92.2,97.3,50.3,79.4]},{"name":"Apply BC","data":[0.5,1.4,2.0,2.1,22.1,19.0]},{"name":"Post-Process","data":[12.3,11.0,2.7,0.0,0.2,0.0]}],"yAxisName":"Percentage (%)"}'></div>



### Why Each Optimization Helps

| Transition | Reason |
|------------|--------|
| Baseline → Threaded | Limited by Python GIL; threads only help for I/O |
| Threaded → Multiprocess | Bypasses GIL via separate processes; IPC overhead limits gains |
| Multiprocess → Numba CPU | JIT compilation eliminates interpreter overhead; true parallel loops |
| Numba CPU → Numba CUDA | GPU parallelism: thousands of threads vs dozens of CPU cores |
| Numba CUDA → CuPy GPU | CUDA C kernels more optimized than Numba-generated PTX |

### 4.6.2 Bottleneck Evolution - RTX 4090

The profiling results show a **clear, repeatable migration of the dominant bottleneck** as the execution model evolves from pure Python CPU to fully GPU-resident kernels. In **XS meshes**, the **CPU baseline is overwhelmingly assembly-bound** (≈63-69% across geometries), with **post-processing** consistently appearing as the secondary cost (≈17-21%). This indicates that, at small scales, runtime is dominated by Python-level loop overhead and per-element bookkeeping rather than linear algebra. Threading reduces assembly share (typically to ≈44-55%) but does not change the structural picture: **the workload remains dominated by serial Python-side orchestration**, and multiprocessing exaggerates post-processing/IPC effects (≈45-50%), confirming that fixed overheads eclipse useful work.

Once JIT compilation is introduced (**Numba CPU**), the bottleneck **shifts from assembly to the solver**: in XS cases, “Solve” typically becomes the primary component (≈39-45%), while assembly collapses to ≈14-15%. This is the expected signature of removing interpreter overhead—assembly becomes “cheap enough” that the sparse solve emerges as the limiting step. On the GPU, the shift becomes even more pronounced: for **Numba CUDA**, **Solve dominates massively** (≈85-91% in XS; ≈48-53% in M), and assembly shrinks to single digits in XS but remains non-negligible in M (≈26-31%), suggesting that kernel efficiency and memory traffic begin to matter once the workload scales.

The most important insight appears in the **CuPy GPU profiles**, where assembly becomes almost negligible (≈0.1-2.9%), and the execution becomes **solver-dominated in every medium-scale case** (≈79-89% Solve). However, unlike Numba CUDA, CuPy exposes a consistent **secondary bottleneck in boundary condition application** (≈11-19% in M; ≈22-30% in XS). This indicates that once the main kernels are highly optimized, **BC handling becomes a memory-bound, synchronization-heavy stage** that resists acceleration—likely due to scatter writes, masking, or sparse index updates that reduce coalescing and increase overhead. Overall, the RTX 4090 profiling confirms that performance improvements are not limited by “more GPU” but by **algorithmic structure**: after assembly is optimized away, the **sparse solver and BC application define the hard ceiling**, and further gains would require solver-level advances (e.g., better preconditioning, fewer iterations, or alternative sparse methods), not just faster kernels.


### 4.7 RTX 5060 Ti  Performance

Key results from performance benchmarks comparing FEM solver implementations.

**Backward-Facing Step (XS)** (287 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 35ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 17ms ± 0ms | 2.1x | 3 |
| CPU Multiprocess | 796ms ± 24ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 5.4x | 3 |
| Numba CUDA | 53ms ± 0ms | 0.7x | 3 |
| CuPy GPU | 70ms ± 9ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-0" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.1,0.0,5.4,0.7,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**Backward-Facing Step (M)** (195,362 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 39.07s ± 0.20s | 1.0x | 3 |
| CPU Threaded | 25.93s ± 0.65s | 1.5x | 3 |
| CPU Multiprocess | 17.22s ± 0.02s | 2.3x | 3 |
| Numba CPU | 16.88s ± 2.95s | 2.3x | 3 |
| Numba CUDA | 3.19s ± 0.10s | 12.2x | 3 |
| CuPy GPU | 2.22s ± 0.06s | 17.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-1" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Backward-Facing Step (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.5,2.3,2.3,12.2,17.6]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (XS)** (411 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 44ms ± 0ms | 1.0x | 3 |
| CPU Threaded | 36ms ± 2ms | 1.2x | 3 |
| CPU Multiprocess | 800ms ± 20ms | 0.1x | 3 |
| Numba CPU | <0.01s ± 0ms | 5.6x | 3 |
| Numba CUDA | 59ms ± 3ms | 0.7x | 3 |
| CuPy GPU | 94ms ± 8ms | 0.5x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-2" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.2,0.1,5.6,0.7,0.5]}],"yAxisName":"Speedup (x)"}'></div>

**Elbow 90° (M)** (161,984 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 33.09s ± 0.59s | 1.0x | 3 |
| CPU Threaded | 19.19s ± 0.07s | 1.7x | 3 |
| CPU Multiprocess | 12.42s ± 0.04s | 2.7x | 3 |
| Numba CPU | 14.30s ± 0.05s | 2.3x | 3 |
| Numba CUDA | 2.98s ± 0.61s | 11.1x | 3 |
| CuPy GPU | 1.90s ± 0.06s | 17.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-3" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Elbow 90\u00b0 (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.7,2.7,2.3,11.1,17.4]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (XS)** (387 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 34ms ± 0ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 0ms | 1.9x | 3 |
| CPU Multiprocess | 789ms ± 23ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 4.2x | 3 |
| Numba CUDA | 63ms ± 4ms | 0.5x | 3 |
| CuPy GPU | 91ms ± 7ms | 0.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-4" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.9,0.0,4.2,0.5,0.4]}],"yAxisName":"Speedup (x)"}'></div>

**S-Bend (M)** (196,078 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 40.28s ± 0.10s | 1.0x | 3 |
| CPU Threaded | 26.05s ± 0.07s | 1.5x | 3 |
| CPU Multiprocess | 21.00s ± 0.71s | 1.9x | 3 |
| Numba CPU | 18.06s ± 0.23s | 2.2x | 3 |
| Numba CUDA | 3.47s ± 0.09s | 11.6x | 3 |
| CuPy GPU | 2.52s ± 0.10s | 16.0x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-5" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - S-Bend (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.5,1.9,2.2,11.6,16.0]}],"yAxisName":"Speedup (x)"}'></div>


**T-Junction (XS)** (393 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 44ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 22ms ± 0ms | 2.0x | 3 |
| CPU Multiprocess | 828ms ± 66ms | 0.1x | 3 |
| Numba CPU | 13ms ± 4ms | 3.4x | 3 |
| Numba CUDA | 65ms ± 1ms | 0.7x | 3 |
| CuPy GPU | 101ms ± 17ms | 0.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-6" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[2.0,0.1,3.4,0.7,0.4]}],"yAxisName":"Speedup (x)"}'></div>

**T-Junction (M)** (196,420 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 39.19s ± 0.47s | 1.0x | 3 |
| CPU Threaded | 25.37s ± 0.43s | 1.5x | 3 |
| CPU Multiprocess | 18.31s ± 0.15s | 2.1x | 3 |
| Numba CPU | 18.77s ± 5.17s | 2.1x | 3 |
| Numba CUDA | 3.35s ± 0.03s | 11.7x | 3 |
| CuPy GPU | 2.43s ± 0.24s | 16.2x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-7" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - T-Junction (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.5,2.1,2.1,11.7,16.2]}],"yAxisName":"Speedup (x)"}'></div>


**Venturi (XS)** (341 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 34ms ± 0ms | 1.0x | 3 |
| CPU Threaded | 18ms ± 0ms | 1.9x | 3 |
| CPU Multiprocess | 889ms ± 17ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 4.9x | 3 |
| Numba CUDA | 53ms ± 1ms | 0.6x | 3 |
| CuPy GPU | 92ms ± 16ms | 0.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-8" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.9,0.0,4.9,0.6,0.4]}],"yAxisName":"Speedup (x)"}'></div>

**Venturi (M)** (194,325 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 38.42s ± 0.09s | 1.0x | 3 |
| CPU Threaded | 23.74s ± 0.41s | 1.6x | 3 |
| CPU Multiprocess | 17.85s ± 0.55s | 2.2x | 3 |
| Numba CPU | 15.80s ± 0.42s | 2.4x | 3 |
| Numba CUDA | 3.20s ± 0.05s | 12.0x | 3 |
| CuPy GPU | 2.50s ± 0.11s | 15.4x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-9" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Venturi (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.6,2.2,2.4,12.0,15.4]}],"yAxisName":"Speedup (x)"}'></div>


**Y-Shaped (XS)** (201 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 23ms ± 1ms | 1.0x | 3 |
| CPU Threaded | 13ms ± 1ms | 1.8x | 3 |
| CPU Multiprocess | 889ms ± 9ms | 0.0x | 3 |
| Numba CPU | <0.01s ± 0ms | 3.8x | 3 |
| Numba CUDA | 44ms ± 1ms | 0.5x | 3 |
| CuPy GPU | 65ms ± 7ms | 0.3x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-10" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (XS)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.8,0.0,3.8,0.5,0.3]}],"yAxisName":"Speedup (x)"}'></div>

**Y-Shaped (M)** (195,853 nodes)

| Implementation | Total Time | Speedup vs Baseline | N |
|----------------|------------|---------------------|---|
| CPU Baseline | 31.21s ± 0.74s | 1.0x | 3 |
| CPU Threaded | 20.27s ± 0.04s | 1.5x | 3 |
| CPU Multiprocess | 16.05s ± 0.79s | 1.9x | 3 |
| Numba CPU | 13.41s ± 0.09s | 2.3x | 3 |
| Numba CUDA | 2.89s ± 0.04s | 10.8x | 3 |
| CuPy GPU | 2.30s ± 0.05s | 13.6x | 3 |

<div class="echart-container bar-chart" id="exec-speedup-11" style="height:250px" data-chart='{"type":"bar","title":"Speedup vs CPU Baseline - Y-Shaped (M)","categories":["CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Speedup","data":[1.5,1.9,2.3,10.8,13.6]}],"yAxisName":"Speedup (x)"}'></div>

#### 4.7.1 Critical Analysis - NVIDIA RTX 5060 Ti

##### 4.7.1.1 Small-Scale Problems (XS meshes)

Across all geometries with XS meshes (≈200-400 nodes), the RTX 5060 Ti exhibits behavior that is fully consistent with overhead-dominated execution regimes. GPU-based implementations (Numba CUDA and CuPy GPU) consistently underperform relative to CPU-based approaches, with speedups ranging between 0.3× and 0.7×, indicating actual slowdowns.

This outcome is not a limitation of the implementation but a direct consequence of the problem scale. At this size, FEM workloads are dominated by:

* Kernel launch latency,
* GPU context initialization,
* Host-device memory transfers,
* JIT compilation overhead (Numba CUDA).

These fixed costs cannot be amortized when the number of elements is small. Even lightweight GPU kernels incur latency that exceeds the total compute time required by optimized CPU execution. As a result, Numba JIT CPU clearly dominates, delivering speedups between 3.4× and 5.6×, while threaded CPU execution provides modest but consistent gains.

CPU multiprocessing performs particularly poorly, often exceeding 700-900 ms, confirming that process creation and inter-process communication overhead completely dominate at small scales. Overall, these results reinforce that GPU acceleration on the RTX 5060 Ti is fundamentally unsuitable for small FEM problems, regardless of implementation quality.

##### 4.7.1.2 Medium-Scale Problems (M meshes)

For medium-scale meshes (≈160k-200k nodes), the performance landscape changes decisively and consistently across all geometries. This regime marks the CPU-GPU crossover point for the RTX 5060 Ti.

Key observations include:

* GPU acceleration becomes clearly beneficial, with speedups ranging from ~10× to ~18×;
* CuPy GPU consistently outperforms Numba CUDA, achieving the highest speedups across all geometries;
* CPU-based approaches saturate, rarely exceeding ~2.5× speedup, even with multiprocessing or JIT compilation.

The RTX 5060 Ti, while not a high-end accelerator, is able to fully exploit GPU parallelism once the FEM workload becomes sufficiently large. Assembly and post-processing costs are effectively suppressed, and the solver phase dominates runtime. Compared to high-end GPUs (e.g., RTX 4090/5090), the achieved speedups are lower, but still represent an order-of-magnitude improvement over CPU execution.

Notably, the relative performance ranking of execution models is stable:

* CuPy GPU > Numba CUDA > Numba CPU ≈ CPU Multiprocess > Threaded > CPU baseline.

This consistency confirms that performance differences are driven by architectural efficiency rather than numerical behavior or solver convergence.

##### 4.7.1.3 Solver-Dominated Regime and Architectural Limits

Despite the substantial gains observed at medium scale, the RTX 5060 Ti shows earlier saturation of speedup growth compared to higher-end GPUs. Peak speedups plateau around ~15×-18×, reflecting architectural constraints:

* Lower memory bandwidth relative to flagship GPUs,
* Fewer SMs and reduced parallel occupancy,
* Limited ability to hide sparse memory access latency.

Once assembly is fully accelerated, the sparse iterative solver becomes the dominant bottleneck. Sparse matrix-vector products are inherently memory-bound, and on the RTX 5060 Ti this limitation becomes apparent sooner. Consequently, further kernel-level optimizations yield diminishing returns, and additional speedup would require:

* More advanced preconditioning,
* Solver algorithm changes,
* Multi-GPU or hybrid CPU-GPU strategies.

This behavior confirms that the RTX 5060 Ti is not compute-limited, but rather constrained by memory bandwidth and sparse access patterns, which are intrinsic to FEM solvers.

##### 4.7.1.4 Practical Assessment and Positioning of the RTX 5060 Ti

From a practical and methodological perspective, the RTX 5060 Ti occupies a well-defined and coherent role within the performance spectrum explored in this study:

| Regime                           | Best Execution Model | Rationale                                                 |
| -------------------------------- | -------------------- | --------------------------------------------------------- |
| XS meshes                        | Numba JIT CPU        | Minimal overhead, compiled execution                      |
| M meshes                         | CuPy GPU             | Best balance of throughput and efficiency                 |
| CPU-only systems                 | Numba JIT CPU        | Strong performance without GPU                            |
| GPU prototyping                  | Numba CUDA           | Easier development, acceptable speed                      |
| Production workloads (mid-scale) | CuPy GPU             | Maximum achievable acceleration on this class of hardware |

While the RTX 5060 Ti does not reach the extreme speedups observed on flagship GPUs, it consistently delivers order-of-magnitude acceleration for realistic FEM workloads at medium scale. This makes it a highly attractive option for:

* Cost-sensitive environments,
* Workstations without high-end GPUs,
* Educational, research, and prototyping contexts,
* Moderate-resolution production simulations.

The RTX 5060 Ti confirms the central conclusions of this study:
GPU acceleration is scale-dependent, architecture-dependent, and solver-limited. When applied appropriately, even a mid-range GPU provides substantial benefits. However, indiscriminate GPU usage—particularly for small problems—remains inefficient. The results validate the RTX 5060 Ti as a capable and well-balanced accelerator for medium-scale FEM workloads, while clearly delineating its limits relative to higher-end architectures.

### Bottleneck Evolution

As optimizations progress, the computational bottleneck shifts:

#### Backward-Facing Step (XS) - 287 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (68%) | Post-Proc (20%) |
| CPU Threaded | Assembly (51%) | Post-Proc (25%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (55%) | BC (16%) |
| Numba CUDA | Solve (88%) | Assembly (6%) |
| CuPy GPU | Solve (72%) | BC (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-0" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.1},{"name":"Solve","value":7.8},{"name":"Apply BC","value":0.8},{"name":"Post-Process","value":20.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-1" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.8},{"name":"Solve","value":72.1},{"name":"Apply BC","value":23.2},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-2" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.1,50.9,50.3,10.5,6.0,1.8]},{"name":"Solve","data":[7.8,13.5,0.4,55.0,88.2,72.1]},{"name":"Apply BC","data":[0.8,4.1,0.1,15.6,1.4,23.2]},{"name":"Post-Process","data":[20.0,24.6,48.9,0.8,1.8,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Backward-Facing Step (M) - 195,362 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (50%) | Solve (36%) |
| CPU Threaded | Solve (56%) | Assembly (28%) |
| CPU Multiprocess | Solve (78%) | Assembly (12%) |
| Numba CPU | Solve (94%) | BC (4%) |
| Numba CUDA | Solve (50%) | Assembly (27%) |
| CuPy GPU | Solve (88%) | BC (12%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-3" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":49.7},{"name":"Solve","value":36.4},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":13.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-4" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.2},{"name":"Solve","value":88.1},{"name":"Apply BC","value":11.6},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-5" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[49.7,28.0,12.1,1.4,26.8,0.2]},{"name":"Solve","data":[36.4,56.2,77.7,94.4,49.8,88.1]},{"name":"Apply BC","data":[0.3,2.8,3.7,4.2,22.2,11.6]},{"name":"Post-Process","data":[13.6,13.1,6.5,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (XS) - 411 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (71%) | Post-Proc (20%) |
| CPU Threaded | Assembly (59%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (53%) | BC (22%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (74%) | BC (22%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-6" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":70.5},{"name":"Solve","value":6.2},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":19.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-7" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.5},{"name":"Solve","value":74.3},{"name":"Apply BC","value":21.9},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-8" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[70.5,59.3,50.0,10.4,6.4,1.5]},{"name":"Solve","data":[6.2,9.1,0.5,52.6,87.4,74.3]},{"name":"Apply BC","data":[1.1,4.1,0.2,21.7,1.9,21.9]},{"name":"Post-Process","data":[19.6,24.1,49.2,0.8,1.8,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (M) - 161,984 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (47%) | Solve (40%) |
| CPU Threaded | Solve (51%) | Assembly (31%) |
| CPU Multiprocess | Solve (73%) | Assembly (15%) |
| Numba CPU | Solve (94%) | BC (4%) |
| Numba CUDA | Solve (48%) | Assembly (28%) |
| CuPy GPU | Solve (81%) | BC (19%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-9" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":47.4},{"name":"Solve","value":39.7},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-10" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.2},{"name":"Solve","value":81.1},{"name":"Apply BC","value":18.5},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-11" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[47.4,30.8,14.8,1.4,28.0,0.2]},{"name":"Solve","data":[39.7,51.0,72.7,94.4,47.8,81.1]},{"name":"Apply BC","data":[0.3,3.1,4.3,4.1,23.1,18.5]},{"name":"Post-Process","data":[12.6,15.1,8.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (XS) - 387 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (19%) |
| CPU Threaded | Assembly (50%) | Post-Proc (20%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (52%) | BC (21%) |
| Numba CUDA | Solve (89%) | Assembly (5%) |
| CuPy GPU | Solve (72%) | BC (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-12" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.8},{"name":"Solve","value":8.0},{"name":"Apply BC","value":1.4},{"name":"Post-Process","value":18.9}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-13" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.8},{"name":"Solve","value":72.4},{"name":"Apply BC","value":23.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-14" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.8,50.4,50.2,9.6,5.1,1.8]},{"name":"Solve","data":[8.0,16.8,0.5,52.4,88.5,72.4]},{"name":"Apply BC","data":[1.4,6.8,0.2,20.7,2.0,23.4]},{"name":"Post-Process","data":[18.9,20.1,48.9,1.1,2.1,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (M) - 196,078 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (46%) | Solve (41%) |
| CPU Threaded | Solve (58%) | Assembly (26%) |
| CPU Multiprocess | Solve (80%) | Assembly (11%) |
| Numba CPU | Solve (95%) | BC (4%) |
| Numba CUDA | Solve (52%) | Assembly (26%) |
| CuPy GPU | Solve (87%) | BC (12%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-15" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":46.1},{"name":"Solve","value":41.2},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-16" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":87.3},{"name":"Apply BC","value":12.4},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-17" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[46.1,26.4,10.6,1.3,25.8,0.1]},{"name":"Solve","data":[41.2,58.4,80.4,94.7,52.3,87.3]},{"name":"Apply BC","data":[0.3,2.2,3.6,4.0,20.9,12.4]},{"name":"Post-Process","data":[12.4,12.9,5.4,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (XS) - 393 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (71%) | Post-Proc (19%) |
| CPU Threaded | Assembly (53%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (48%) |
| Numba CPU | Solve (59%) | BC (17%) |
| Numba CUDA | Solve (88%) | Assembly (6%) |
| CuPy GPU | Solve (74%) | BC (22%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-20" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":70.9},{"name":"Solve","value":6.9},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":19.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-21" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.5},{"name":"Solve","value":74.2},{"name":"Apply BC","value":22.1},{"name":"Post-Process","value":0.4}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-22" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[70.9,53.0,51.1,7.3,5.7,1.5]},{"name":"Solve","data":[6.9,13.4,0.4,59.1,88.4,74.2]},{"name":"Apply BC","data":[1.1,4.4,0.1,16.5,1.8,22.1]},{"name":"Post-Process","data":[19.0,23.9,48.2,1.2,1.7,0.4]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (M) - 196,420 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (49%) | Solve (37%) |
| CPU Threaded | Solve (57%) | Assembly (28%) |
| CPU Multiprocess | Solve (79%) | Assembly (11%) |
| Numba CPU | Solve (95%) | BC (4%) |
| Numba CUDA | Solve (51%) | Assembly (27%) |
| CuPy GPU | Solve (87%) | BC (13%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-23" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":49.4},{"name":"Solve","value":37.2},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":13.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-24" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":87.0},{"name":"Apply BC","value":12.7},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-25" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[49.4,27.7,11.3,1.3,26.8,0.1]},{"name":"Solve","data":[37.2,56.9,79.2,94.9,51.5,87.0]},{"name":"Apply BC","data":[0.3,2.2,3.3,3.8,20.6,12.7]},{"name":"Post-Process","data":[13.0,13.1,6.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (XS) - 341 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (19%) |
| CPU Threaded | Assembly (51%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (49%) | BC (21%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (67%) | BC (29%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-27" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":69.0},{"name":"Solve","value":7.1},{"name":"Apply BC","value":1.4},{"name":"Post-Process","value":19.5}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-28" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.6},{"name":"Solve","value":66.7},{"name":"Apply BC","value":29.3},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-29" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[69.0,50.6,50.0,10.0,5.9,1.6]},{"name":"Solve","data":[7.1,13.3,0.4,49.3,87.1,66.7]},{"name":"Apply BC","data":[1.4,5.7,0.2,20.5,2.3,29.3]},{"name":"Post-Process","data":[19.5,24.3,49.3,0.9,1.9,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (M) - 194,325 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (48%) | Solve (39%) |
| CPU Threaded | Solve (55%) | Assembly (29%) |
| CPU Multiprocess | Solve (78%) | Assembly (12%) |
| Numba CPU | Solve (94%) | BC (5%) |
| Numba CUDA | Solve (49%) | Assembly (28%) |
| CuPy GPU | Solve (79%) | BC (21%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-30" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":47.9},{"name":"Solve","value":39.0},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-31" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":78.6},{"name":"Apply BC","value":21.1},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-32" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[47.9,28.7,11.9,1.5,27.9,0.1]},{"name":"Solve","data":[39.0,55.2,78.0,93.7,49.4,78.6]},{"name":"Apply BC","data":[0.3,2.5,3.8,4.8,21.5,21.1]},{"name":"Post-Process","data":[12.8,13.7,6.3,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


#### Y-Shaped (XS) - 201 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (63%) | Post-Proc (20%) |
| CPU Threaded | Assembly (44%) | Post-Proc (26%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (49%) |
| Numba CPU | Solve (52%) | BC (15%) |
| Numba CUDA | Solve (86%) | Assembly (6%) |
| CuPy GPU | Solve (61%) | BC (34%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-35" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":63.4},{"name":"Solve","value":9.8},{"name":"Apply BC","value":1.2},{"name":"Post-Process","value":20.3}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-36" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.9},{"name":"Solve","value":60.5},{"name":"Apply BC","value":34.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-37" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[63.4,43.9,50.6,9.7,6.2,1.9]},{"name":"Solve","data":[9.8,15.9,0.4,51.7,86.5,60.5]},{"name":"Apply BC","data":[1.2,5.5,0.1,14.5,1.6,34.4]},{"name":"Post-Process","data":[20.3,25.6,48.7,1.1,2.2,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Y-Shaped (M) - 195,853 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (46%) | Solve (41%) |
| CPU Threaded | Solve (59%) | Assembly (26%) |
| CPU Multiprocess | Solve (79%) | Assembly (10%) |
| Numba CPU | Solve (93%) | BC (6%) |
| Numba CUDA | Solve (48%) | BC (26%) |
| CuPy GPU | Solve (79%) | BC (20%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-38" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":46.4},{"name":"Solve","value":40.5},{"name":"Apply BC","value":0.6},{"name":"Post-Process","value":12.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-39" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":79.4},{"name":"Apply BC","value":19.5},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-40" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[46.4,25.5,10.3,1.5,24.3,0.1]},{"name":"Solve","data":[40.5,58.9,79.3,92.8,47.7,79.4]},{"name":"Apply BC","data":[0.6,2.9,4.2,5.6,26.4,19.5]},{"name":"Post-Process","data":[12.4,12.6,6.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>


### Why Each Optimization Helps

| Transition | Reason |
|------------|--------|
| Baseline → Threaded | Limited by Python GIL; threads only help for I/O |
| Threaded → Multiprocess | Bypasses GIL via separate processes; IPC overhead limits gains |
| Multiprocess → Numba CPU | JIT compilation eliminates interpreter overhead; true parallel loops |
| Numba CPU → Numba CUDA | GPU parallelism: thousands of threads vs dozens of CPU cores |
| Numba CUDA → CuPy GPU | CUDA C kernels more optimized than Numba-generated PTX |

#### 4060Ti Bottleneck critical analysis

On the RTX 5060 Ti, the bottleneck evolution follows a clear and internally consistent pattern that mirrors the architectural constraints of a mid-range GPU. For XS meshes (≈200-400 nodes), execution is overwhelmingly dominated by fixed overheads rather than computation: CPU implementations spend most of their time in assembly and post-processing, while GPU-based solvers appear solve-dominated mainly due to kernel launch latency, device synchronization, and host-device transfer costs. In this regime, GPU execution is intrinsically inefficient and cannot amortize its overhead, and even multiprocessing on the CPU performs poorly due to IPC and process startup costs. As mesh size increases to the M regime (≈160k-200k nodes), the performance bottleneck shifts decisively toward the sparse linear solve across all implementations. 

CPU baselines remain partially assembly-bound, but threaded and multiprocessing variants reduce assembly time and expose the solver as the dominant cost. Numba CPU effectively removes interpreter and assembly overhead, yet becomes almost entirely solve-bound, highlighting the fundamental memory-bandwidth limitations of sparse linear algebra on CPUs. GPU implementations, by contrast, virtually eliminate assembly cost and achieve substantially lower total runtimes; however, their performance is ultimately constrained by the same solver-dominated behavior, with a non-negligible contribution from boundary condition application and synchronization. Overall, the RTX 5060 Ti demonstrates that GPU acceleration is only advantageous beyond a clear problem-size threshold, and that once assembly is removed from the critical path, further performance gains depend primarily on improving solver efficiency and memory access patterns rather than kernel-level optimizations alone.

## 4.8 Cross-Platform Comparative Analysis

This section consolidates the benchmark results presented in Sections 4.5-4.7 into a unified comparative analysis.  
Rather than reiterating individual measurements, the focus here is on **interpreting performance trends**, **explaining architectural effects**, and **extracting general conclusions** regarding execution models and GPU classes.

---

### 4.8.1 CPU vs GPU: Where the Paradigm Shifts

Across all geometries and medium-to-large meshes, a clear and consistent transition point emerges:

- **Small meshes (XS)**  
  GPU execution is systematically slower than optimized CPU variants due to:
  - kernel launch overhead,
  - PCIe latency,
  - underutilization of GPU parallelism.

- **Medium meshes (M)**  
  GPU acceleration becomes dominant, with speedups ranging from:
  - **~11× (RTX 5060 Ti)**  
  - **~20-40× (RTX 4090)**  
  - **~30-60× (RTX 5090)**  

This confirms that GPU acceleration is not universally beneficial, but **highly problem-size dependent**.

---

### 4.8.2 CPU Scaling Limits

The benchmark reveals well-defined limits for CPU-based optimization strategies.

| CPU Strategy | Observed Benefit | Limiting Factor |
|-------------|------------------|-----------------|
| Threading | 1.2× - 2.1× | Python GIL |
| Multiprocessing | 1.5× - 2.7× | IPC overhead |
| Numba JIT | 2× - 6× | Memory bandwidth |

Even with aggressive JIT compilation, **CPU performance saturates early**.  
For medium meshes, the solver becomes:

- **memory-bound**, and  
- dominated by **sparse matrix-vector products**.

This explains why Numba CPU converges to similar performance as multiprocessing for large problems.

---

### 4.8.3 GPU Acceleration: Numba CUDA vs CuPy RawKernel

A consistent hierarchy is observed across all GPUs:

| GPU Execution Model | Characteristics | Performance |
|--------------------|------------------|-------------|
| Numba CUDA | Python-defined kernels, easier development | High |
| CuPy RawKernel | Native CUDA C, full control | Highest |

Key observations:

- **CuPy GPU consistently outperforms Numba CUDA** for medium meshes.
- Gains range from **1.3× to 1.8×** over Numba CUDA.
- The advantage increases with:
  - mesh size,
  - solver dominance,
  - memory bandwidth pressure.

This confirms that **kernel maturity and low-level control matter** once GPU execution becomes solver-bound.

---

### 4.8.4 Cross-GPU Performance Scaling

A core objective of this benchmark was to separate **software scaling** from **hardware scaling**.

#### Aggregate Speedup (Medium Meshes)

| GPU | Typical Speedup vs CPU Baseline |
|----|--------------------------------|
| RTX 5060 Ti | 11× - 18× |
| RTX 4090 | 20× - 42× |
| RTX 5090 | 25× - 60× |

However, performance does **not** scale linearly with theoretical FLOPs.

#### Interpretation

- The FEM solver is **memory-bandwidth dominated**, not compute-bound.
- Higher-end GPUs benefit from:
  - larger L2 cache,
  - higher memory throughput,
  - better latency hiding.
- The RTX 5090 advantage is strongest for:
  - CG-heavy cases,
  - large sparse matrices,
  - solver-dominated workloads.

This confirms that **architectural balance**, not raw FLOPs, drives FEM performance.

---

### 4.8.5 Bottleneck Evolution Across Platforms

A central insight from the benchmark is the **systematic migration of bottlenecks**:

| Execution Stage | CPU Baseline | Numba CPU | GPU (CuPy) |
|----------------|-------------|-----------|------------|
| Assembly | Dominant | Minor | Negligible |
| Solve | Secondary | Dominant | Overwhelming |
| Apply BC | Minor | Minor | Non-negligible |
| Post-processing | Visible | Minimal | Negligible |

Key implications:

- GPU acceleration **eliminates assembly as a bottleneck**.
- The **linear solver dominates runtime** in all optimized variants.
- On GPU, boundary condition application becomes visible due to:
  - atomic operations,
  - irregular memory access,
  - limited arithmetic intensity.

This validates the design decision to prioritize GPU-resident solvers.

---

### 4.8.6 Efficiency vs Absolute Performance

While the RTX 5090 delivers the highest absolute performance, efficiency considerations are relevant:

| GPU | Relative Performance | Cost / Power Consideration |
|----|----------------------|----------------------------|
| RTX 5060 Ti | Moderate | High efficiency per cost |
| RTX 4090 | Very high | Balanced performance |
| RTX 5090 | Extreme | Diminishing returns |

For production environments, this suggests:

- **Mid-range GPUs** are sufficient for moderate FEM workloads.
- **High-end GPUs** are justified for:
  - very large meshes,
  - repeated simulations,
  - solver-dominated pipelines.

---

### 4.8.7 Robustness and Numerical Consistency

Crucially, acceleration does **not** alter numerical behavior:

- Identical CG iteration counts across platforms.
- Consistent residual norms at convergence.
- No divergence or fallback behavior observed.

This confirms that performance gains are achieved **without sacrificing numerical correctness**.

---

### 4.8.8 Consolidated Summary

| Aspect | Key Conclusion |
|------|----------------|
| CPU optimization | Quickly saturates |
| GPU benefit | Strongly size-dependent |
| Best execution model | CuPy RawKernel |
| Dominant bottleneck | Sparse solver |
| Best scaling factor | Memory bandwidth |
| Best overall GPU | RTX 5090 |
| Best cost-efficiency | RTX 5060 Ti |

---

### 4.8.9 Final Insight

The benchmark demonstrates that **GPU acceleration fundamentally changes the performance landscape of FEM solvers**, but only when:

- the problem size is sufficiently large,
- data remains resident on the GPU,
- solver execution dominates the pipeline.

Beyond this point, performance becomes a function of **memory architecture rather than algorithmic complexity**, placing modern GPUs at a decisive advantage over CPUs for large-scale finite element simulations.


## Conclusions

### Key Findings

> Conclusions based on mean times across 3 servers.

#### Backward-Facing Step (XS) - 287 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.5x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.8x speedup.

3. **JIT Compilation:** Numba CPU delivers 5.4x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 71% of total time.

#### Backward-Facing Step (M) - 195,362 nodes

1. **Maximum Speedup:** CuPy GPU achieves 25.1x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.3x speedup.

3. **JIT Compilation:** Numba CPU delivers 1.8x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 89% of total time.

#### Elbow 90° (XS) - 411 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.5x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.6x speedup.

3. **JIT Compilation:** Numba CPU delivers 6.1x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 69% of total time.

#### Elbow 90° (M) - 161,984 nodes

1. **Maximum Speedup:** CuPy GPU achieves 26.2x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.6x speedup.

3. **JIT Compilation:** Numba CPU delivers 1.7x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 83% of total time.

#### S-Bend (XS) - 387 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.4x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.9x speedup.

3. **JIT Compilation:** Numba CPU delivers 4.7x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 69% of total time.

#### S-Bend (M) - 196,078 nodes

1. **Maximum Speedup:** CuPy GPU achieves 20.7x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.2x speedup.

3. **JIT Compilation:** Numba CPU delivers 1.5x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 88% of total time.


#### T-Junction (XS) - 393 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.5x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 2.0x speedup.

3. **JIT Compilation:** Numba CPU delivers 4.6x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 71% of total time.

#### T-Junction (M) - 196,420 nodes

1. **Maximum Speedup:** CuPy GPU achieves 30.4x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.7x speedup.

3. **JIT Compilation:** Numba CPU delivers 2.4x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 87% of total time.

#### Venturi (XS) - 341 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.4x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.9x speedup.

3. **JIT Compilation:** Numba CPU delivers 6.0x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 64% of total time.

#### Venturi (M) - 194,325 nodes

1. **Maximum Speedup:** CuPy GPU achieves 20.6x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.2x speedup.

3. **JIT Compilation:** Numba CPU delivers 1.7x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 80% of total time.

#### Y-Shaped (XS) - 201 nodes

1. **Maximum Speedup:** CuPy GPU achieves 0.4x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.9x speedup.

3. **JIT Compilation:** Numba CPU delivers 4.1x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 61% of total time.

#### Y-Shaped (M) - 195,853 nodes

1. **Maximum Speedup:** CuPy GPU achieves 17.3x speedup over CPU Baseline.

2. **Threading Effect:** CPU Threaded shows 1.2x speedup.

3. **JIT Compilation:** Numba CPU delivers 1.6x speedup by eliminating interpreter overhead.

4. **GPU Bottleneck:** On GPU, the iterative solver consumes 80% of total time.


### Recommendations

| Use Case | Recommended Implementation |
|----------|---------------------------|
| Development/debugging | CPU Baseline or Numba CPU |
| Production (no GPU) | Numba CPU |
| Production (with GPU) | CuPy GPU |
| Small meshes (<10K nodes) | Numba CPU (GPU overhead not worthwhile) |
| Large meshes (>100K nodes) | CuPy GPU |