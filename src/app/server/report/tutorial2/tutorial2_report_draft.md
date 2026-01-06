# 1. General Overview - Finite Element Method

The Finite Element Method (FEM) is a numerical technique widely used to approximate solutions of partial differential equations arising in engineering and scientific problems. Its main strength lies in its ability to handle complex geometries, heterogeneous materials, and general boundary conditions, which are often intractable using analytical approaches.

The fundamental idea of FEM is to replace a continuous problem by a discrete one. The physical domain is subdivided into a finite number of smaller regions, called elements, over which the unknown field is approximated using interpolation functions. By assembling the contributions of all elements, the original continuous problem is transformed into a system of algebraic equations that can be solved numerically.

IMAGE1

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

IMAGE2

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

IMAGE 4


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

!!!Nota: Acrescentar nota sobre Transição do MATLAB para o Python; Pre-Implementation


This section presents multiple implementations of the same FEM problem using different execution models on CPU and GPU. All implementations share an identical numerical formulation, discretization, boundary conditions, and solver configuration; observed differences arise exclusively from the execution strategy and computational backend.

The implementations cover sequential CPU execution, shared-memory and process-based CPU parallelism, just-in-time compiled CPU execution using Numba, and GPU-based execution using Numba CUDA and CuPy with custom raw kernels. Together, these approaches span execution models ranging from interpreter-driven execution to compiled and accelerator-based computation.

Numerical equivalence is preserved across all implementations, enabling direct and fair comparison of execution behavior, performance, and scalability under consistent numerical conditions.

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

#### 3.2.2.1. NumPy and SciPy Ecosystem
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

# Implementation 2: CPU Threaded

## 1. Overview

The CPU Threaded implementation extends the CPU baseline by introducing parallelism through Python’s `concurrent.futures.ThreadPoolExecutor`. The objective is to evaluate whether multi-threading can accelerate FEM assembly and post-processing despite the presence of Python’s Global Interpreter Lock (GIL).

Unlike the baseline, which executes all element-level operations sequentially, this implementation partitions the mesh into batches processed concurrently by multiple threads. The approach relies on the fact that NumPy releases the GIL during computational kernels, allowing partial overlap of execution across threads.

| Attribute | Description |
|-----------|-------------|
| Technology | Python ThreadPoolExecutor (`concurrent.futures`) |
| Execution Model | Multi-threaded with GIL constraints |
| Role | Evaluate benefits and limits of threading on CPU |
| Dependencies | NumPy, SciPy, concurrent.futures (stdlib) |

---

## 2. Technology Background

### 2.1 Python Threading and the Global Interpreter Lock

Python’s Global Interpreter Lock (GIL) enforces serialized execution of Python bytecode, preventing true parallel execution of CPU-bound workloads across threads. This simplifies memory management but significantly constrains scalability for numerical applications implemented at the Python level.

However, many NumPy operations release the GIL during execution, including:

- Vectorized array arithmetic  
- Dense linear algebra routines (BLAS/LAPACK)  
- Element-wise mathematical kernels  

This behavior enables limited concurrency when the computation is structured to maximize time spent inside GIL-released NumPy kernels, while minimizing Python-level control flow.

### 2.2 ThreadPoolExecutor Execution Model

The `ThreadPoolExecutor` abstraction provides a pool of reusable worker threads and a future-based execution model.

Key characteristics include:

- Persistent worker threads, reducing creation overhead  
- Asynchronous task submission via `Future` objects  
- Automatic synchronization and cleanup through context management  
- Dynamic scheduling that enables basic load balancing  

This abstraction simplifies parallel orchestration while preserving shared-memory access to NumPy arrays.

### 2.3 Implications for FEM Workloads

Relative to the CPU baseline, the expected impact of threading on FEM operations is mixed:

| Operation | GIL Released | Expected Benefit |
|----------|--------------|------------------|
| Python loop iteration | No | None |
| Sparse matrix indexing | No | None |
| NumPy dense kernels | Yes | Moderate |
| Element-wise NumPy ops | Yes | Moderate |

The overall benefit therefore depends on increasing the ratio of GIL-free numerical computation relative to GIL-held Python coordination.

---

## 3. Implementation Strategy

### 3.1 Batch-Based Parallelization

To amortize threading overhead and reduce GIL contention, elements are grouped into fixed-size batches. Each batch is processed by a single thread, enabling coarse-grained parallelism:

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



Each thread operates independently on a contiguous range of elements, computing local stiffness contributions and storing results in thread-local buffers.

### 3.2 Element Batch Processing

Each batch computes stiffness matrices and load contributions for a subset of elements and stores results in pre-allocated arrays using COO (Coordinate) format.

Key steps include:

1. Pre-allocation of output arrays for rows, columns, and values  
2. Sequential processing of elements within the batch  
3. Computation of local stiffness matrices using NumPy operations  
4. Storage of local contributions in thread-local COO arrays  

This design avoids shared writes during assembly and minimizes synchronization.

### 3.3 Parallel Assembly Orchestration

The main assembly routine dispatches batches to worker threads using a thread pool. Results are collected asynchronously, allowing faster threads to return without blocking on slower batches. After all threads complete, individual COO arrays are concatenated and converted to CSR format.

### 3.4 COO-Based Global Assembly

Unlike the baseline implementation, which performs incremental insertion into a LIL matrix, this implementation assembles the global stiffness matrix using COO format:

| Aspect | Baseline (LIL) | Threaded (COO) |
|------|----------------|----------------|
| Thread safety | Not thread-safe | Naturally thread-safe |
| Insertion pattern | Incremental | Batched |
| Duplicate handling | Explicit | Automatic on CSR conversion |
| Parallel suitability | Poor | High |

The final `COO → CSR` conversion automatically merges duplicate entries arising from shared nodes between elements.

### 3.5 Post-Processing Parallelization

Derived field computation (velocity and magnitude) follows the same batch-based threading strategy. Each thread processes a disjoint subset of elements and writes results into non-overlapping regions of the output arrays, avoiding data races.

### 3.6 Linear System Solution

The linear solver is identical to the CPU baseline. SciPy’s Conjugate Gradient solver is used with the same preconditioning and convergence criteria. No Python-level threading is applied to the solver phase, as SciPy internally manages optimized numerical kernels and threading via BLAS libraries.

---

## 4. Optimization Techniques Applied

### 4.1 Batch Size Selection

Batch size is a critical tuning parameter controlling the balance between coordination overhead and load balance. Empirical testing indicates that batch sizes between 500 and 2000 elements provide the best trade-off for typical problem sizes.

### 4.2 Pre-allocation of Thread-Local Buffers

Each batch allocates fixed-size arrays once per thread invocation, avoiding repeated dynamic memory allocation within inner loops. This reduces overhead and improves cache locality.

### 4.3 Inlined Element Computation

Element stiffness computation is implemented directly within the batch function to minimize function call overhead and maximize time spent in GIL-released NumPy kernels.

### 4.4 Shared Read-Only Data

Mesh coordinates, connectivity, and quadrature data are shared across threads as read-only NumPy arrays. This avoids memory duplication while maintaining thread safety.

---

## 5. Challenges and Limitations

### 5.1 GIL Contention

Despite NumPy releasing the GIL during numerical kernels, a substantial fraction of execution time remains GIL-bound due to Python loops, indexing, and sparse data manipulation. This fundamentally limits scalability.

### 5.2 Memory Bandwidth Saturation

All threads share the same memory subsystem, leading to contention and diminishing returns beyond a modest number of threads.

### 5.3 Thread Management Overhead

Task submission, scheduling, and result aggregation introduce non-negligible overhead, which dominates execution time for small problem sizes.

### 5.4 Limited Solver Parallelism

The solver phase remains effectively sequential at the Python level. While underlying BLAS libraries may use threads, overall solver performance is memory-bound and does not benefit significantly from additional Python threading.

---

## 6. Performance Characteristics and Role

### 6.1 Expected Scaling Behavior

Thread-level parallelism yields sub-linear speedup governed by Amdahl’s Law. Only portions of the assembly and post-processing phases benefit from concurrent execution.

### 6.2 Practical Speedup Regime

Empirical behavior typically shows:

- Modest gains with 2–4 threads  
- Diminishing returns beyond 4–8 threads  
- Potential slowdowns when contention outweighs parallel benefits  

### 6.3 Role in the Implementation Suite

This implementation serves as an intermediate reference between the sequential CPU baseline and more aggressive parallelization strategies. It highlights the structural limitations imposed by the GIL and motivates approaches that bypass it entirely.

---

## 7. Summary

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

![Multithreading](multithreading.png)



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

# Implementation 4: Numba JIT CPU

## 1. Overview

The Numba JIT CPU implementation leverages Just-In-Time compilation to translate Python code into optimized native machine code at runtime. By compiling element-level FEM kernels and enabling parallel execution through Numba’s `prange` construct, this approach achieves true multi-threaded execution while preserving shared-memory semantics.

This implementation combines the low memory overhead of shared-memory execution with performance characteristics close to compiled languages, eliminating Python interpreter overhead from the dominant FEM assembly and post-processing phases.

| Attribute | Description |
|-----------|-------------|
| Technology | Numba JIT compiler with LLVM backend |
| Execution Model | JIT-compiled, multi-threaded shared memory |
| Role | High-performance CPU parallel execution |
| Dependencies | NumPy, SciPy, Numba |

---

## 2. Technology Background

### 2.1 Just-In-Time Compilation

Numba is a Just-In-Time compiler that translates Python functions into optimized machine code using the LLVM compiler infrastructure. The compilation pipeline transforms Python source code through intermediate representations into native CPU instructions.

Key advantages relative to interpreted execution include:

- Elimination of Python interpreter overhead  
- Native execution speed comparable to C/Fortran  
- Automatic loop unrolling and inlining  
- SIMD vectorization via LLVM  
- Execution without Global Interpreter Lock (GIL) constraints  

### 2.2 The `@njit` Compilation Model

The implementation uses Numba’s `@njit` decorator to enforce *nopython* mode. In this mode:

- All Python bytecode is bypassed  
- Type inference is resolved at compile time  
- Unsupported Python and NumPy features are disallowed  

Compilation caching is enabled to persist generated machine code across executions, amortizing compilation cost.

### 2.3 Parallel Execution with `prange`

Parallelism is achieved using Numba’s `prange`, which distributes loop iterations across CPU threads. Unlike Python threading:

- Execution occurs without GIL constraints  
- Threads operate in shared memory  
- Work distribution follows an OpenMP-style model  
- Near-linear speedup is achievable for independent iterations  

### 2.4 Relevance for FEM

For FEM workloads, JIT compilation directly targets the dominant computational bottlenecks:

- Element stiffness matrix computation  
- Element-level assembly loops  
- Derived field computation  

Sparse matrix construction and iterative solvers remain in SciPy, preserving numerical equivalence with previous implementations.

---

## 3. Implementation Strategy

### 3.1 Function-Level JIT Compilation

All computational kernels are implemented as Numba-compiled functions. Element stiffness computation, boundary condition evaluation, and post-processing kernels are explicitly written using loop-based formulations compatible with Numba’s *nopython* mode.

This ensures that the entire element-level computation executes in compiled code without interpreter intervention.

### 3.2 Parallel Element Assembly

Global assembly is implemented through a parallel loop over elements. Each iteration:

1. Gathers nodal coordinates for a single element  
2. Computes the local stiffness matrix and load vector  
3. Writes local contributions into pre-allocated COO arrays  

Parallelism is applied at the element level using `prange`, ensuring that each element is processed independently and concurrently.

### 3.3 Explicit Loop-Based Kernels

Unlike vectorized NumPy formulations, all numerical operations are expressed as explicit loops. This allows LLVM to:

- Fully unroll small fixed-size loops  
- Inline function calls  
- Eliminate temporary array allocations  
- Apply SIMD vectorization to inner loops  

This explicit structure is critical for achieving high performance in JIT-compiled FEM kernels.

### 3.4 Parallel Post-Processing

Derived field computation (velocity and magnitude) follows the same compiled parallel pattern. Each element gathers local solution values, evaluates gradients at integration points, and stores results in element-wise output arrays.

### 3.5 Solver Integration

Sparse matrix construction and the Conjugate Gradient solver are executed outside Numba using SciPy. The JIT boundary is placed at the array level: Numba generates COO-format data, and SciPy handles sparse matrix assembly and solution.

---

## 4. Optimization Techniques Applied

### 4.1 Interpreter Elimination

The dominant optimization is the complete removal of Python interpreter overhead from element-level computation. All inner loops execute as native machine code.

### 4.2 Loop Unrolling and Inlining

Small, fixed-size loops are fully unrolled by LLVM. Nested function calls within JIT-compiled code are typically inlined, eliminating function call overhead.

### 4.3 SIMD Vectorization

LLVM applies SIMD vectorization to inner arithmetic loops where data layout permits, enabling multiple operations per CPU cycle.

### 4.4 Memory Access Optimization

COO data is written sequentially in element-major order, improving cache locality and reducing memory write overhead.

### 4.5 Shared-Memory Parallelism

Parallel execution uses shared memory without data duplication, preserving memory efficiency relative to multiprocessing-based approaches.

---

## 5. Challenges and Limitations

### 5.1 Compilation Overhead

Initial execution incurs JIT compilation cost on the order of hundreds of milliseconds. This overhead is amortized for large problems and eliminated on subsequent runs via caching.

### 5.2 Limited NumPy and SciPy Support

Only a subset of NumPy functionality is supported in *nopython* mode. Unsupported operations must be rewritten using explicit loops, increasing implementation complexity.

### 5.3 Debugging Complexity

Debugging JIT-compiled code is more difficult than debugging pure Python code. Stack traces may be less informative, and interactive debuggers cannot step into compiled regions.

### 5.4 Memory Allocation in Parallel Regions

Allocating arrays inside parallel loops increases overhead. Performance is improved by minimizing allocations within parallel regions.

### 5.5 Solver Dominance at Scale

As assembly and post-processing accelerate, the sparse solver increasingly dominates total runtime and limits further speedup.

---

## 6. Performance Characteristics and Role

### 6.1 Expected Scaling

| Stage | Scaling Behavior | Dominant Factor |
|------|------------------|-----------------|
| Assembly | O(N_elements) | Compiled arithmetic |
| Post-processing | O(N_elements) | Compiled arithmetic |
| Linear system solution | O(iterations × nnz) | Sparse memory bandwidth |
| Boundary conditions | O(N_boundary) | Minor relative cost |

### 6.2 Profiling Observations

For large meshes:

- Assembly and post-processing times are reduced by one to two orders of magnitude  
- Solver time becomes the dominant runtime component  
- Parallel efficiency remains high until memory bandwidth saturation  

### 6.3 Role in the Implementation Suite

The Numba JIT CPU implementation represents the highest-performing CPU-based solution in this study. It establishes the upper bound for shared-memory CPU execution and serves as the reference point for evaluating GPU-based implementations.

---

## 7. Summary

The Numba JIT CPU implementation eliminates Python interpreter overhead and enables true shared-memory parallelism for FEM assembly and post-processing.

Key observations include:

- Explicit loop-based kernels outperform vectorized NumPy formulations  
- True parallel execution is achieved without GIL constraints  
- Memory efficiency is preserved relative to multiprocessing  
- Sparse solver performance ultimately limits end-to-end speedup  

This implementation provides the most efficient CPU-based execution model in the study and forms a natural transition toward GPU-based acceleration.

# Implementation 5: Numba CUDA

## 1. Overview

The Numba CUDA implementation extends the FEM solver to GPU execution using Numba’s `@cuda.jit` decorator, enabling the definition of GPU kernels using Python syntax. This approach provides access to massive GPU parallelism while avoiding direct CUDA C/C++ development, offering a balance between development productivity and performance.

Element-level FEM computations are offloaded to the GPU using a one-thread-per-element mapping, while sparse linear system solution is performed on the GPU using CuPy’s sparse solvers.

| Attribute | Description |
|-----------|-------------|
| Technology | Numba CUDA (`@cuda.jit`) |
| Execution Model | GPU SIMT execution |
| Role | GPU acceleration with Python-native kernels |
| Dependencies | NumPy, SciPy, Numba, CuPy |

---

## 2. Technology Background

### 2.1 Numba CUDA Programming Model

Numba extends its JIT compilation framework to NVIDIA GPUs through the `@cuda.jit` decorator. Python functions annotated as CUDA kernels are compiled to PTX (Parallel Thread Execution) code and executed on the GPU.

GPU execution follows the CUDA SIMT model, where thousands of lightweight threads execute the same kernel concurrently.

### 2.2 CUDA Execution Hierarchy

GPU kernels are launched using a hierarchical structure:

- **Grid**: All threads launched by a kernel invocation  
- **Block**: A group of threads that can cooperate via shared memory  
- **Thread**: The smallest execution unit  
- **Warp**: A group of 32 threads executing in lockstep  

Threads are indexed using `cuda.grid(1)`, enabling a direct mapping between thread index and FEM element index.

### 2.3 GPU Memory Hierarchy

GPU memory is organized in multiple tiers:

- **Registers**: Fastest, thread-private storage  
- **Local memory**: Thread-private, may spill to device memory  
- **Shared memory**: Fast, block-shared memory  
- **Global memory**: Large but high-latency device memory  

The implementation primarily uses registers and local memory for element-level arrays and global memory for mesh data and assembled results.

### 2.4 Relevance for FEM

GPU execution is particularly well suited for FEM workloads with large numbers of independent elements. Element-level stiffness matrix computation exhibits high arithmetic intensity and minimal inter-thread dependency, making it ideal for SIMT execution.

---

## 3. Implementation Strategy

### 3.1 Kernel-Based Element Assembly

Element assembly is implemented as a GPU kernel where each thread processes a single element. For each element, the thread:

1. Loads nodal indices and coordinates  
2. Computes shape functions, Jacobians, and gradients  
3. Assembles the local stiffness matrix and load vector  
4. Writes results to global memory  

All computations are performed using explicit loops compatible with Numba CUDA.

### 3.2 Thread-to-Element Mapping

A one-thread-per-element strategy is used:

- Each GPU thread computes one element  
- Threads are launched in 1D grids  
- Excess threads exit early if the element index exceeds the mesh size  

This mapping ensures uniform work distribution and avoids inter-thread synchronization during element computation.

### 3.3 Local Memory Usage

Per-thread temporary arrays are allocated using `cuda.local.array`, including:

- Element DOF indices  
- Nodal coordinates  
- Local stiffness matrix and load vector  
- Shape functions and derivatives  

These arrays are private to each thread, eliminating race conditions and synchronization overhead.

### 3.4 Force Vector Assembly with Atomics

Because multiple elements share nodes, assembly of the global force vector requires atomic operations. Thread-safe accumulation is performed using `cuda.atomic.add` to ensure correctness.

### 3.5 Post-Processing on GPU

Derived field computation (velocity and magnitude) is implemented as a separate GPU kernel. Each thread processes one element, evaluates gradients at integration points, and stores averaged results.

### 3.6 Solver Integration

The linear system is solved on the GPU using CuPy’s sparse Conjugate Gradient solver. Sparse matrices are converted to CuPy formats, allowing the entire solution phase to execute on the GPU before transferring results back to CPU memory.

---

## 4. Optimization Techniques Applied

### 4.1 Massive Parallelism

The GPU executes tens of thousands of threads concurrently, enabling element-level parallelism far beyond CPU core counts.

### 4.2 Block Size Tuning

Kernel launch configuration is tuned to balance occupancy and register pressure. A block size of 128 threads provides good performance for the register-heavy FEM kernels.

### 4.3 Memory Coalescing

Global memory accesses are structured so that consecutive threads write to contiguous memory regions, improving memory coalescing and bandwidth utilization.

### 4.4 Register and Local Memory Management

Small per-thread arrays are kept in registers where possible. Larger arrays may spill to local memory, but remain private and efficiently cached.

### 4.5 Warp Divergence Minimization

Kernel control flow is designed to minimize conditional branches. Aside from bounds checking at kernel entry, all threads follow identical execution paths.

---

## 5. Challenges and Limitations

### 5.1 Debugging Complexity

Debugging GPU kernels is significantly more difficult than CPU code. Python debuggers cannot be used inside kernels, and runtime errors may lead to silent failures or kernel crashes.

### 5.2 Limited NumPy Support

Only a restricted subset of NumPy functionality is available in CUDA kernels. Unsupported operations must be reimplemented using explicit arithmetic and loops.

### 5.3 Atomic Operation Overhead

Atomic updates to the global force vector introduce serialization and reduce scalability. While acceptable for this problem, atomics may become a bottleneck for higher connectivity or more complex FEM formulations.

### 5.4 Memory Transfer Overhead

Data must be explicitly transferred between host and device memory. For small problem sizes, PCIe transfer overhead can dominate execution time.

### 5.5 Partial CPU-GPU Workflow

Some tasks, such as COO index generation and boundary condition application, remain on the CPU due to complexity or limited performance benefit on GPU.

---

## 6. Performance Characteristics and Role

### 6.1 Expected Scaling

| Stage | Scaling Behavior | Dominant Factor |
|------|------------------|-----------------|
| Element assembly | O(N_elements) | GPU throughput |
| Post-processing | O(N_elements) | GPU throughput |
| Linear system solution | O(iterations × nnz) | GPU memory bandwidth |
| Data transfer | O(N) | PCIe bandwidth |

### 6.2 Profiling Observations

For large meshes:

- GPU occupancy reaches 50–75%  
- Assembly speedup is substantial compared to CPU JIT  
- Sparse solver performance becomes memory-bound  
- End-to-end speedup improves with increasing problem size  

### 6.3 Role in the Implementation Suite

The Numba CUDA implementation is the first fully GPU-based approach in the study. It demonstrates the feasibility of GPU acceleration using Python-native kernels and establishes a reference point for evaluating raw CUDA kernel implementations.

---

## 7. Summary

The Numba CUDA implementation enables GPU acceleration of FEM assembly and post-processing using Python syntax:

Key observations include:

- Thousands of GPU threads execute element computations concurrently  
- Python-based kernel development significantly reduces development effort  
- Performance approaches that of hand-written CUDA kernels  
- Atomic operations and memory transfers limit scalability for smaller problems  

This implementation represents a practical and accessible entry point for GPU acceleration, bridging the gap between CPU-based JIT execution and fully optimized raw CUDA implementations.

# Implementation 6: GPU CuPy (RawKernel)

## 1. Overview

The GPU CuPy implementation (`quad8_gpu_v3.py`) represents the most performance-oriented approach, using CuPy's `RawKernel` to execute hand-written CUDA C kernels directly on the GPU. This provides maximum control over GPU execution while leveraging CuPy's ecosystem for sparse matrix operations and iterative solvers.

| Attribute | Description |
|-----------|-------------|
| Technology | CuPy RawKernel (CUDA C), CuPy sparse |
| Execution Model | GPU SIMT with native CUDA C kernels |
| Role | Maximum GPU performance, production-quality implementation |
| Source File | `quad8_gpu_v3.py` |
| Dependencies | NumPy, SciPy, CuPy |

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

CuPy's `RawKernel` allows embedding CUDA C code directly in Python. This provides:

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

The assembly kernel follows a CUDA C structure with:

- Thread index derived from `blockIdx`, `blockDim`, and `threadIdx`
- Thread-local arrays for element data and local matrices
- Fixed integration loops matching the CPU formulation
- Scatter step writing flattened element stiffness matrix values
- Atomic force vector accumulation

### 3.3 GPU-Accelerated COO Index Generation

Unlike the Numba CUDA version which generates COO indices on CPU, this implementation uses vectorized CuPy operations on GPU:

- Generates all \(N_{el} \times 64\) row indices in parallel
- Generates all \(N_{el} \times 64\) column indices in parallel
- Avoids CPU-GPU synchronization for index generation
- Uses CuPy’s optimized array operations for index construction

The kernel computes only the COO values; index arrays are generated on GPU.

### 3.4 Sparse Matrix Construction

After kernel execution, COO data is converted to CSR:

- COO sparse matrix is created on GPU using CuPy sparse
- COO → CSR conversion merges duplicates automatically
- Entire sparse matrix remains GPU-resident

### 3.5 GPU Sparse Solver

The linear system is solved entirely on GPU using CuPy’s sparse solvers:

- Diagonal equilibration is performed on GPU
- Jacobi preconditioning is implemented on GPU using a linear operator abstraction
- Conjugate Gradient (CG) runs fully on GPU
- Solution is de-equilibrated on GPU

All solver operations (SpMV, vector updates, dot products) run without CPU round-trips.

### 3.6 GPU Post-Processing

Velocity computation is performed using a dedicated RawKernel:

- One thread per element
- Gradient evaluated at 4 Gauss points
- Element-averaged velocity and magnitude stored in GPU arrays
- Pressure computed from Bernoulli using vectorized GPU operations

---

## 4. Optimization Techniques Applied

### 4.1 CUDA C Shape Function Derivatives

Shape function derivatives are computed inline inside the kernel using explicit CUDA C expressions, avoiding function call overhead and enabling compiler optimization.

### 4.2 Jacobian and Inverse in CUDA C

The Jacobian matrix, determinant, and inverse are computed explicitly inside the kernel:

- Explicit loops over the 8 nodes
- Fixed-size operations suitable for unrolling
- Determinant and inverse computed directly from 2×2 Jacobian

### 4.3 Atomic Force Vector Update

Force vector accumulation is performed using CUDA atomics (`atomicAdd`) to ensure correctness when multiple elements share nodes.

### 4.4 Solver Fallback Strategy

The implementation includes a solver fallback mechanism:

- Attempts CG solve first
- Falls back to GMRES if CG fails
- Improves robustness under cases of numerical difficulty

---

## 5. Challenges and Limitations

### 5.1 CUDA C Complexity

RawKernel development requires CUDA expertise, including:

- Thread hierarchy reasoning
- Memory hierarchy and access optimization
- Occupancy, register pressure, and divergence control

This increases development cost relative to Python-kernel approaches.

### 5.2 Debugging Challenges

Debugging RawKernel code is harder than Python or Numba CUDA:

- Python debugger is not applicable inside kernels
- `printf` debugging is limited and expensive
- Memory errors can crash kernels without clear error traces
- Race conditions are difficult to diagnose

### 5.3 Kernel Compilation Overhead

RawKernel code is JIT-compiled on first use:

| Kernel | Compilation Time |
|--------|------------------|
| Assembly | ~200-500 ms |
| Post-processing | ~100-200 ms |

CuPy caches compiled kernels so subsequent executions have near-zero compilation cost.

### 5.4 GPU Memory Constraints

Large problems can be limited by GPU VRAM:

| Component | Memory per 100K nodes |
|-----------|------------------------|
| Coordinates (x, y) | ~1.6 MB |
| Connectivity | ~3.2 MB |
| Sparse matrix (CSR) | ~50-100 MB |
| Solution vectors | ~0.8 MB |
| Working memory | Variable |

For very large meshes, multi-GPU or out-of-core strategies may be required.

### 5.5 CuPy Sparse Solver Limitations

CuPy sparse solvers have limitations relative to SciPy:

- Fewer preconditioner options
- Some solver features less mature
- Occasional numerical differences

GMRES fallback addresses robustness concerns.

---

## 6. Performance Characteristics and Role

### 6.1 GPU Utilization Analysis

| Metric | Typical Value | Optimization Target |
|--------|---------------|---------------------|
| Occupancy | 50-75% | Register pressure |
| Memory throughput | 70-85% peak | Coalescing |
| Compute utilization | 60-80% | Algorithm efficiency |

### 6.2 Performance Breakdown

Expected time distribution for large problems:

| Stage | Time Fraction | Notes |
|-------|---------------|-------|
| Mesh loading | <5% | I/O bound |
| Assembly kernel | 5-15% | Highly parallel |
| Matrix construction | 5-10% | CuPy sparse ops |
| Linear solve | 60-80% | Memory bandwidth bound |
| Post-processing | 5-10% | Highly parallel |
| Data transfer | <5% | PCIe overhead |

### 6.3 Scaling Characteristics

| Problem Size | GPU Advantage | Notes |
|--------------|---------------|------|
| <10K elements | Minimal | Transfer overhead dominates |
| 10K-100K | Significant (5-20×) | Good GPU utilization |
| 100K-1M | Maximum (20-100×) | Full GPU saturation |
| >1M | Memory limited | May require multi-GPU |

### 6.4 Comparison with CPU Implementations

| Aspect | CPU Baseline | Numba CPU | GPU CuPy |
|--------|--------------|-----------|----------|
| Assembly | O(N) sequential | O(N/P) parallel | O(N/T) massively parallel |
| Threads/cores | 1 | 4-32 | 1000s |
| Memory bandwidth | 50-100 GB/s | 50-100 GB/s | 500-900 GB/s |
| Latency | Low | Low | Higher (PCIe) |
| Throughput | Moderate | Good | Excellent |

---

## 7. Summary

The GPU CuPy implementation with RawKernel represents the most performance-optimized endpoint of this implementation spectrum:

Key observations include:

- Native CUDA C kernels provide maximum GPU performance
- Full GPU-resident pipeline (assembly, solve, post-processing) minimizes PCIe transfers
- GPU-based COO index generation avoids CPU bottlenecks present in Numba CUDA
- Sparse solver dominates runtime once assembly is accelerated
- Development and debugging complexity is significantly higher than Numba CUDA

This implementation establishes the upper bound for single-GPU performance in this project and provides a production-quality reference design combining custom CUDA kernels with CuPy’s sparse linear algebra ecosystem.
