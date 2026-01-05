# Common Foundation

## 1. Overview

All six solver implementations developed in this project share a common mathematical formulation, software architecture, and interface contract. This deliberate design ensures that performance comparisons are meaningful and scientifically valid: observed differences in execution time reflect only the parallelization strategy and execution model employed, not variations in the underlying algorithm, numerical method, or problem formulation.

The common foundation encompasses four principal areas:

- **Mathematical formulation**: Quad-8 serendipity elements solving the 2D Laplace equation for velocity potential
- **Numerical methods**: Gauss-Legendre quadrature for integration, Jacobi-preconditioned Conjugate Gradient for linear system solution
- **Software architecture**: Unified solver API, progress callback system, and standardized result format
- **Shared computational modules**: Shape functions, integration rules, element matrices, and boundary condition handlers

This section documents these shared components in detail, establishing the baseline against which implementation-specific optimizations and parallelization strategies are evaluated.

---

## 2. Mathematical Formulation

### 2.1 Governing Equation

The solver addresses two-dimensional steady-state potential problems governed by Laplace's equation:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

where $u(x, y)$ represents the velocity potential field over a bounded domain $\Omega \subset \mathbb{R}^2$ with boundary $\Gamma = \partial\Omega$.

This elliptic partial differential equation arises in numerous physical contexts:

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

### 2.2 Weak Formulation

Applying the Galerkin weighted residual method with test functions $v \in H^1_0(\Omega)$, the weak form becomes:

$$\int_\Omega \nabla u \cdot \nabla v \, d\Omega = \int_\Gamma v \, g \, d\Gamma$$

where $g$ represents the Neumann (flux) boundary data. This formulation reduces continuity requirements on $u$ from $C^2$ to $H^1$, enabling piecewise polynomial approximation over finite elements.

### 2.3 Finite Element Discretization

The domain $\Omega$ is partitioned into $N_{el}$ non-overlapping quadrilateral elements $\Omega_e$:

$$\Omega = \bigcup_{e=1}^{N_{el}} \Omega_e, \quad \Omega_i \cap \Omega_j = \emptyset \text{ for } i \neq j$$

Within each element, the field variable is approximated using shape functions:

$$u^{(e)}(\mathbf{x}) = \sum_{i=1}^{8} N_i(\mathbf{x}) \, u_i^{(e)}$$

where $N_i$ are the element shape functions and $u_i^{(e)}$ are the nodal values.

Substituting into the weak form and applying element-by-element assembly yields the global linear system:

$$\mathbf{K} \mathbf{u} = \mathbf{f}$$

where:
- $\mathbf{K} \in \mathbb{R}^{N_{dof} \times N_{dof}}$ is the global stiffness matrix (sparse, symmetric, positive-definite)
- $\mathbf{u} \in \mathbb{R}^{N_{dof}}$ is the vector of nodal unknowns
- $\mathbf{f} \in \mathbb{R}^{N_{dof}}$ is the global load vector

---

## 3. Element Formulation: 8-Node Serendipity Quadrilateral

### 3.1 Element Topology

All implementations use the Quad-8 isoparametric serendipity element, which provides quadratic interpolation along element edges without interior nodes. Each element comprises 8 nodes: 4 corner nodes and 4 mid-edge nodes.

```
        η
        ↑
    4───7───3
    │       │
    8   ·   6  → ξ
    │       │
    1───5───2
```

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

### 3.2 Shape Functions

The shape functions in parametric coordinates $(\xi, \eta) \in [-1, 1]^2$ are defined as follows.

**Corner nodes** (nodes 1-4):

$$N_1 = \frac{1}{4}(1 - \xi)(1 - \eta)(-\xi - \eta - 1)$$

$$N_2 = \frac{1}{4}(1 + \xi)(1 - \eta)(\xi - \eta - 1)$$

$$N_3 = \frac{1}{4}(1 + \xi)(1 + \eta)(\xi + \eta - 1)$$

$$N_4 = \frac{1}{4}(1 - \xi)(1 + \eta)(-\xi + \eta - 1)$$

**Mid-edge nodes** (nodes 5-8):

$$N_5 = \frac{1}{2}(1 - \xi^2)(1 - \eta)$$

$$N_6 = \frac{1}{2}(1 + \xi)(1 - \eta^2)$$

$$N_7 = \frac{1}{2}(1 - \xi^2)(1 + \eta)$$

$$N_8 = \frac{1}{2}(1 - \xi)(1 - \eta^2)$$

These functions satisfy the Kronecker delta property: $N_i(\mathbf{x}_j) = \delta_{ij}$, ensuring that nodal values are interpolated exactly.

### 3.3 Shape Function Derivatives

The derivatives with respect to parametric coordinates are computed analytically. For the corner nodes:

$$\frac{\partial N_1}{\partial \xi} = \frac{(2\xi + \eta)(1 - \eta)}{4}, \quad \frac{\partial N_1}{\partial \eta} = \frac{(2\eta + \xi)(1 - \xi)}{4}$$

$$\frac{\partial N_2}{\partial \xi} = \frac{(2\xi - \eta)(1 - \eta)}{4}, \quad \frac{\partial N_2}{\partial \eta} = \frac{(2\eta - \xi)(1 + \xi)}{4}$$

$$\frac{\partial N_3}{\partial \xi} = \frac{(2\xi + \eta)(1 + \eta)}{4}, \quad \frac{\partial N_3}{\partial \eta} = \frac{(2\eta + \xi)(1 + \xi)}{4}$$

$$\frac{\partial N_4}{\partial \xi} = \frac{(2\xi - \eta)(1 + \eta)}{4}, \quad \frac{\partial N_4}{\partial \eta} = \frac{(2\eta - \xi)(1 - \xi)}{4}$$

For the mid-edge nodes:

$$\frac{\partial N_5}{\partial \xi} = \xi(\eta - 1), \quad \frac{\partial N_5}{\partial \eta} = \frac{\xi^2 - 1}{2}$$

$$\frac{\partial N_6}{\partial \xi} = \frac{1 - \eta^2}{2}, \quad \frac{\partial N_6}{\partial \eta} = -(1 + \xi)\eta$$

$$\frac{\partial N_7}{\partial \xi} = -\xi(1 + \eta), \quad \frac{\partial N_7}{\partial \eta} = \frac{1 - \xi^2}{2}$$

$$\frac{\partial N_8}{\partial \xi} = \frac{\eta^2 - 1}{2}, \quad \frac{\partial N_8}{\partial \eta} = (\xi - 1)\eta$$

### 3.4 Isoparametric Mapping and Jacobian

The isoparametric formulation uses the same shape functions to map both geometry and field variables:

$$x = \sum_{i=1}^{8} N_i(\xi, \eta) \, x_i, \quad y = \sum_{i=1}^{8} N_i(\xi, \eta) \, y_i$$

The Jacobian matrix of this transformation is:

$$\mathbf{J} = \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} \\ \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} \end{bmatrix} = \begin{bmatrix} \sum_i \frac{\partial N_i}{\partial \xi} x_i & \sum_i \frac{\partial N_i}{\partial \eta} x_i \\ \sum_i \frac{\partial N_i}{\partial \xi} y_i & \sum_i \frac{\partial N_i}{\partial \eta} y_i \end{bmatrix}$$

Or in matrix form: $\mathbf{J} = \mathbf{X}_e^T \, \mathbf{D}_\psi$, where $\mathbf{X}_e$ is the $(8 \times 2)$ matrix of nodal coordinates and $\mathbf{D}_\psi$ is the $(8 \times 2)$ matrix of parametric derivatives.

The determinant $|\mathbf{J}|$ represents the local area scaling factor and must be positive for valid element geometry.

### 3.5 Physical Derivatives (B-Matrix)

Derivatives with respect to physical coordinates are obtained through the chain rule:

$$\begin{bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{bmatrix} = \mathbf{J}^{-1} \begin{bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \end{bmatrix}$$

The $\mathbf{B}$ matrix, containing derivatives of all shape functions with respect to physical coordinates, is computed as:

$$\mathbf{B} = \mathbf{D}_\psi \, \mathbf{J}^{-1}$$

where $\mathbf{B} \in \mathbb{R}^{8 \times 2}$ with $B_{i,1} = \frac{\partial N_i}{\partial x}$ and $B_{i,2} = \frac{\partial N_i}{\partial y}$.

### 3.6 Rationale for Quad-8 Selection

The Quad-8 element was selected for this project based on several considerations relevant to high-performance computing evaluation:

| Characteristic | Benefit for Performance Study |
|----------------|------------------------------|
| Quadratic accuracy | Captures curved boundaries and field gradients accurately |
| 8 DOFs per element | Creates meaningful per-element computational workload |
| Dense 8×8 stiffness matrix | 64 floating-point values per element exposes parallelization opportunities |
| No interior nodes | Simpler connectivity than full biquadratic elements |
| Standard formulation | Well-documented, verifiable against reference implementations |

---

## 4. Numerical Integration

### 4.1 Gauss-Legendre Quadrature

Element integrals are evaluated using Gauss-Legendre quadrature, which provides optimal accuracy for polynomial integrands. For a function $f(\xi, \eta)$ over the reference domain $[-1, 1]^2$:

$$\int_{-1}^{1} \int_{-1}^{1} f(\xi, \eta) \, d\xi \, d\eta \approx \sum_{i=1}^{n_{ip}} w_i \, f(\xi_i, \eta_i)$$

where $(\xi_i, \eta_i)$ are the integration point coordinates and $w_i$ are the corresponding weights.

### 4.2 Integration Rules Used

Three quadrature rules are employed across the solver stages:

**3×3 Gauss-Legendre (9 points) — Element Stiffness Assembly**

The 9-point rule integrates polynomials up to degree 5 exactly. For Quad-8 elements with quadratic shape functions, the integrand $\mathbf{B}^T \mathbf{B} |\mathbf{J}|$ can reach degree 4-5 for mildly distorted elements, making the 9-point rule appropriate.

$$G = \sqrt{0.6} \approx 0.7746$$

| Point | $\xi$ | $\eta$ | Weight |
|-------|-------|--------|--------|
| 1 | $-G$ | $-G$ | $25/81$ |
| 2 | $0$ | $-G$ | $40/81$ |
| 3 | $+G$ | $-G$ | $25/81$ |
| 4 | $-G$ | $0$ | $40/81$ |
| 5 | $0$ | $0$ | $64/81$ |
| 6 | $+G$ | $0$ | $40/81$ |
| 7 | $-G$ | $+G$ | $25/81$ |
| 8 | $0$ | $+G$ | $40/81$ |
| 9 | $+G$ | $+G$ | $25/81$ |

**2×2 Gauss-Legendre (4 points) — Velocity Post-Processing**

A reduced 4-point rule is used for computing velocity gradients during post-processing, providing degree 3 accuracy with lower computational cost:

$$G = \sqrt{1/3} \approx 0.5774$$

| Point | $\xi$ | $\eta$ | Weight |
|-------|-------|--------|--------|
| 1 | $-G$ | $-G$ | $1$ |
| 2 | $+G$ | $-G$ | $1$ |
| 3 | $+G$ | $+G$ | $1$ |
| 4 | $-G$ | $+G$ | $1$ |

**3-point 1D Gauss-Legendre — Robin Boundary Conditions**

Edge integrals for Robin boundary conditions use 3-point 1D quadrature:

$$G = \sqrt{0.6} \approx 0.7746$$

| Point | $\xi$ | Weight |
|-------|-------|--------|
| 1 | $-G$ | $5/9$ |
| 2 | $0$ | $8/9$ |
| 3 | $+G$ | $5/9$ |

### 4.3 Element Stiffness Matrix Computation

The element stiffness matrix is computed as:

$$\mathbf{K}_e = \int_{\Omega_e} \mathbf{B}^T \mathbf{B} \, d\Omega = \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}(\xi, \eta)^T \mathbf{B}(\xi, \eta) \, |\mathbf{J}(\xi, \eta)| \, d\xi \, d\eta$$

Using 9-point quadrature:

$$\mathbf{K}_e = \sum_{p=1}^{9} w_p \, \mathbf{B}_p^T \mathbf{B}_p \, |\mathbf{J}_p|$$

where $\mathbf{B}_p = \mathbf{B}(\xi_p, \eta_p)$ and $|\mathbf{J}_p| = |\mathbf{J}(\xi_p, \eta_p)|$.

In component form:

$$K_{e,ij} = \sum_{p=1}^{9} w_p \, |\mathbf{J}_p| \left( B_{p,i1} B_{p,j1} + B_{p,i2} B_{p,j2} \right)$$

This formulation is identical across all implementations; only the execution model (sequential, threaded, JIT-compiled, or GPU-parallel) varies.

---

## 5. Boundary Conditions

### 5.1 Dirichlet Boundary Conditions

Dirichlet (essential) boundary conditions specify fixed potential values at designated boundary nodes:

$$u = \bar{u} \quad \text{on } \Gamma_D$$

These are implemented using row/column elimination: for each constrained degree of freedom $i$ with prescribed value $\bar{u}_i$:

1. Set $K_{ii} = 1$ and $K_{ij} = K_{ji} = 0$ for $j \neq i$
2. Set $f_i = \bar{u}_i$
3. Modify $f_j \leftarrow f_j - K_{ji} \bar{u}_i$ for all $j \neq i$ (to preserve symmetry)

In the project context, Dirichlet conditions are applied at outlet boundaries where the potential is fixed.

### 5.2 Robin Boundary Conditions

Robin (mixed) boundary conditions combine flux and potential contributions at inlet boundaries:

$$p \cdot u + \frac{\partial u}{\partial n} = \gamma \quad \text{on } \Gamma_R$$

where $p$ is a coefficient and $\gamma$ represents the prescribed combination of flux and potential.

The weak form contribution from Robin boundaries is:

$$\int_{\Gamma_R} p \, u \, v \, d\Gamma = \int_{\Gamma_R} \gamma \, v \, d\Gamma$$

This yields additional stiffness and load contributions:

$$\mathbf{H}_e = \int_{\Gamma_e} p \, \mathbf{N}^T \mathbf{N} \, d\Gamma, \quad \mathbf{P}_e = \int_{\Gamma_e} \gamma \, \mathbf{N} \, d\Gamma$$

For quadratic (3-node) edges, these integrals are evaluated using 3-point 1D Gauss-Legendre quadrature with quadratic edge shape functions:

$$N_1^{edge}(\xi) = \frac{\xi(\xi - 1)}{2}, \quad N_2^{edge}(\xi) = 1 - \xi^2, \quad N_3^{edge}(\xi) = \frac{\xi(\xi + 1)}{2}$$

The edge Jacobian accounts for the physical edge length:

$$|J_{edge}| = \sqrt{\left(\frac{dx}{d\xi}\right)^2 + \left(\frac{dy}{d\xi}\right)^2}$$

### 5.3 Boundary Detection

Boundary nodes are identified geometrically based on coordinate tolerance. The implementation detects:

- **Inlet boundary**: Left edge of domain (minimum $x$ coordinate)
- **Outlet boundary**: Right edge of domain (maximum $x$ coordinate)

A tolerance parameter (`bc_tolerance = 1e-9`) handles floating-point precision in coordinate comparisons.

---

## 6. Linear Solver Strategy

### 6.1 Conjugate Gradient Method

All implementations use the Conjugate Gradient (CG) method for solving the linear system $\mathbf{K}\mathbf{u} = \mathbf{f}$. CG is particularly suitable for this application because:

1. **Symmetric positive-definite system**: The stiffness matrix $\mathbf{K}$ from elliptic PDEs satisfies the SPD requirement
2. **Memory efficiency**: Only matrix-vector products required, no explicit factorization
3. **Predictable convergence**: Error reduction bounded by condition number
4. **Parallelization potential**: Core operations (SpMV, dot products, axpy) are data-parallel

The CG algorithm generates a sequence of iterates $\mathbf{u}^{(k)}$ that minimize the $\mathbf{K}$-norm of the error over a Krylov subspace of increasing dimension.

### 6.2 Jacobi Preconditioning

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

### 6.3 Solver Configuration

The following parameters are held constant across all implementations:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | Conjugate Gradient | Optimal for SPD systems |
| Preconditioner | Jacobi (diagonal) | Parallelizes uniformly |
| Relative tolerance | $10^{-8}$ | Engineering accuracy |
| Absolute tolerance | $0$ | Rely on relative criterion |
| Maximum iterations | 15,000 | Sufficient for test problems |
| Progress reporting | Every 50 iterations | Balance monitoring vs. overhead |

### 6.4 Convergence Monitoring

The solver monitors convergence using the relative residual norm:

$$\text{rel\_res} = \frac{\|\mathbf{r}^{(k)}\|_2}{\|\mathbf{b}\|_2} = \frac{\|\mathbf{f} - \mathbf{K}\mathbf{u}^{(k)}\|_2}{\|\mathbf{f}\|_2}$$

Convergence is declared when $\text{rel\_res} < 10^{-8}$ or the iteration count exceeds the maximum.

---

## 7. Post-Processing: Derived Fields

### 7.1 Velocity Field Computation

The velocity field is computed as the negative gradient of the potential:

$$\mathbf{v} = -\nabla u = -\begin{bmatrix} \frac{\partial u}{\partial x} \\ \frac{\partial u}{\partial y} \end{bmatrix}$$

For each element, the gradient is evaluated at 4 Gauss points and averaged:

$$\mathbf{v}_e = \frac{1}{4} \sum_{p=1}^{4} \left( -\mathbf{B}_p^T \mathbf{u}_e \right)$$

where $\mathbf{u}_e$ is the vector of nodal solution values for element $e$.

### 7.2 Velocity Magnitude

The velocity magnitude per element:

$$|\mathbf{v}|_e = \frac{1}{4} \sum_{p=1}^{4} \sqrt{v_{x,p}^2 + v_{y,p}^2}$$

### 7.3 Pressure Field

Pressure is computed from Bernoulli's equation for incompressible flow:

$$p = p_0 - \frac{1}{2} \rho |\mathbf{v}|^2$$

where:
- $p_0 = 101328.8$ Pa (reference pressure)
- $\rho = 0.6125$ kg/m³ (fluid density)

These constants are configurable parameters in the solver constructor.

---

## 8. Computational Pipeline

### 8.1 Pipeline Stages

Every solver implementation follows an identical six-stage pipeline, ensuring that timing comparisons are structurally equivalent:

```
┌─────────────────┐
│   1. LOAD MESH  │  Read HDF5 file → coordinates, connectivity
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. ASSEMBLE     │  Element loops → global K, f (PRIMARY TARGET)
│    SYSTEM       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. APPLY BCs    │  Robin inlet + Dirichlet outlet
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. SOLVE        │  Preconditioned CG iterations (SECONDARY TARGET)
│    SYSTEM       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. COMPUTE      │  Velocity, pressure fields (TERTIARY TARGET)
│    DERIVED      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. EXPORT       │  Save results to HDF5
│    RESULTS      │
└─────────────────┘
```

### 8.2 Parallelization Targets

| Stage | Computational Pattern | Parallelization Opportunity |
|-------|----------------------|----------------------------|
| Load Mesh | I/O bound | Limited (disk/memory bandwidth) |
| Assemble System | Element-independent loops | **High** (embarrassingly parallel) |
| Apply BCs | Sequential modifications | Low (small fraction of runtime) |
| Solve System | Sparse matrix-vector products | **Medium** (memory bandwidth limited) |
| Compute Derived | Element-independent loops | **High** (embarrassingly parallel) |
| Export Results | I/O bound | Limited (disk bandwidth) |

The assembly stage exhibits the highest parallelization potential because each element's stiffness matrix can be computed independently. The solve stage benefits from parallel SpMV but faces memory bandwidth constraints characteristic of sparse computations. Post-processing mirrors assembly in its parallel structure.

### 8.3 Timing Instrumentation

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

## 9. Software Architecture

### 9.1 Solver Interface Contract

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

### 9.2 SolverWrapper: Unified Factory

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

### 9.3 Progress Callback System

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

### 9.4 Result Format

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

## 10. Shared Computational Modules

### 10.1 Module Organization

Each implementation variant includes adapted versions of four core computational modules. While the mathematical operations are identical, each version is optimized for its execution model:

| Module | Purpose | CPU (NumPy) | Numba JIT | CuPy GPU | CUDA Kernel |
|--------|---------|-------------|-----------|----------|-------------|
| `shape_n_der8` | Shape functions, derivatives, Jacobian | `np.zeros`, `np.linalg` | `@njit`, explicit loops | `cp.zeros`, `cp.linalg` | Inlined in kernel |
| `genip2dq` | Gauss point coordinates and weights | `np.array` constants | `@njit`, return arrays | `cp.array` constants | Helper function |
| `elem_quad8` | Element stiffness matrix | `np.outer`, matrix ops | `@njit`, nested loops | `cp.outer`, matrix ops | Full kernel |
| `robin_quadr` | Robin BC edge integration | NumPy loops | `@njit` loops | CuPy loops | CPU fallback |

### 10.2 Implementation Adaptations

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

### 10.3 Mathematical Equivalence

Despite implementation differences, all versions compute mathematically identical results (within floating-point precision). This is verified by:

1. Comparing solution vectors across implementations
2. Checking that relative differences are within machine epsilon ($\approx 10^{-15}$)
3. Ensuring identical iteration counts for CG convergence

This equivalence is essential for valid performance comparisons: timing differences reflect execution model efficiency, not algorithmic variations.

---

## 11. Mesh Format and I/O

### 11.1 HDF5 Mesh Format

Meshes are stored in HDF5 format for efficient I/O operations:

```
mesh.h5
├── coordinates/
│   ├── x    (float64, shape: Nnodes)
│   └── y    (float64, shape: Nnodes)
└── connectivity/
    └── quad8 (int32, shape: Nelements × 8)
```

### 11.2 Format Advantages

HDF5 provides several advantages for this application:

| Feature | Benefit |
|---------|---------|
| Binary format | Faster I/O than text formats |
| Compression support | Reduced storage for large meshes |
| Memory mapping | Efficient access patterns |
| Platform independence | Cross-platform compatibility |
| Hierarchical structure | Organized data layout |
| Partial reads | Future extensibility for distributed computing |

### 11.3 Legacy Format Support

For compatibility, the solver also supports:

- **NPZ** (NumPy compressed archive): Fast binary format
- **Excel (.xlsx)**: Human-readable, useful for small test cases

All formats are converted to the internal NumPy array representation upon loading.

### 11.4 Mesh Loading Implementation

```python
def load_mesh(self) -> None:
    """Load mesh from HDF5 file."""
    import h5py
    
    with h5py.File(self.mesh_file, 'r') as f:
        self.x = f['coordinates/x'][:]
        self.y = f['coordinates/y'][:]
        self.quad8 = f['connectivity/quad8'][:]
    
    self.Nnds = len(self.x)
    self.Nels = len(self.quad8)
```

---

## 12. Summary

The common foundation described in this section ensures that all six solver implementations operate on identical mathematical and algorithmic ground. Key design decisions supporting fair performance comparison include:

1. **Identical FEM formulation**: Quad-8 elements, 9-point quadrature, Robin/Dirichlet BCs
2. **Uniform solver strategy**: Jacobi-preconditioned CG with fixed tolerance
3. **Consistent interfaces**: Same constructor signature, result format, callback system
4. **Equivalent computational modules**: Mathematically identical, adapted for each execution model
5. **Standardized timing**: Per-stage instrumentation with identical granularity

With this foundation established, the following sections examine how each implementation variant exploits parallelism within this common framework, and how the resulting performance characteristics differ across problem sizes and computational stages.

---
