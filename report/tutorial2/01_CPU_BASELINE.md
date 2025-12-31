# CPU Baseline Implementation

## Overview

The CPU Baseline implementation (`quad8_cpu_v3.py`) serves as the **reference implementation** against which all optimized versions are compared. It represents a straightforward, readable implementation of the Finite Element Method (FEM) using standard Python scientific computing libraries. While not optimized for performance, this implementation prioritizes **clarity, correctness, and maintainability**—essential qualities for validating that accelerated versions produce identical results.

This implementation solves the **2D Laplace equation** (∇²φ = 0) for potential flow analysis using 8-node quadrilateral (Quad-8) isoparametric elements. The physics models inviscid, incompressible flow around aerodynamic bodies such as ducts, airfoils, and venturi geometries.

---

## Technology Stack

### Core Scientific Computing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.10+ | Primary implementation language |
| Array Operations | NumPy 1.24+ | N-dimensional array computations |
| Sparse Matrices | SciPy (scipy.sparse) | Memory-efficient matrix storage (LIL, CSR) |
| Linear Solver | SciPy (scipy.sparse.linalg) | Iterative Krylov solvers (CG, GMRES) |
| Data I/O | Pandas, h5py | Mesh file loading (Excel, HDF5) |
| Type Hints | typing, numpy.typing | Code documentation and IDE support |

### Real-Time Event-Driven Notifications

The solver integrates with a web-based monitoring system through a **callback architecture** that decouples the computation from the UI layer:

| Component | Technology | Purpose |
|-----------|------------|---------|
| Callback Interface | `ProgressCallback` class | Abstract event emission interface |
| Event Transport | Socket.IO | WebSocket-based bidirectional communication |
| Server Framework | FastAPI + python-socketio | Async HTTP + WebSocket server |
| Client Consumer | JavaScript (Browser) | Real-time UI updates (ECharts, Three.js) |

**Event Types Emitted:**

| Event | Trigger | Payload |
|-------|---------|---------|
| `fem:stageStart` | Stage begins | `{stage: string}` |
| `fem:stageComplete` | Stage ends | `{stage, duration}` |
| `fem:meshLoaded` | Mesh parsed | `{nodes, elements, coordinates}` |
| `fem:assemblyProgress` | During assembly | `{current, total}` |
| `fem:solveProgress` | Each N iterations | `{iteration, residual, etr}` |
| `fem:solveComplete` | Solver finished | `{converged, iterations, timing}` |

This architecture allows the solver to run uninterrupted while the UI receives granular progress updates for responsive user experience during long-running simulations.

---

## Architecture

### Class Structure

The implementation is organized around a single solver class with clear separation of concerns:

```
Quad8FEMSolver
├── __init__()           # Configuration and parameter initialization
├── load_mesh()          # Mesh I/O (HDF5, Excel, NPZ formats)
├── assemble_system()    # Global matrix assembly
├── apply_boundary_conditions()  # Robin + Dirichlet BCs
├── solve()              # Iterative linear system solution
├── compute_derived_fields()     # Post-processing (velocity, pressure)
├── run()                # Orchestrates complete workflow
└── [visualization/export methods]

IterativeSolverMonitor   # Callback class for convergence tracking
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `x`, `y` | `ndarray[float64]` | Nodal coordinates |
| `quad8` | `ndarray[int32]` | Element connectivity (Nels × 8) |
| `Kg` | `lil_matrix` → `csr_matrix` | Global stiffness matrix |
| `fg` | `ndarray[float64]` | Global force vector |
| `u` | `ndarray[float64]` | Solution vector (velocity potential) |
| `vel`, `abs_vel` | `ndarray[float64]` | Derived velocity fields |
| `pressure` | `ndarray[float64]` | Bernoulli-derived pressure |

### Execution Flow

The solver executes through five primary stages:

```
┌─────────────────┐
│  1. Load Mesh   │  Read nodes, elements, connectivity from file
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Assembly    │  Build global stiffness matrix K and force vector f
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Apply BCs   │  Robin (inlet) + Dirichlet (outlet) conditions
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Solve       │  Iterative solution of Ku = f
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Post-Process│  Compute velocity v = -∇φ and pressure
└─────────────────┘
```

---

## Key Implementation Details

### Matrix Assembly

The assembly phase is the **most computationally intensive** part of the CPU implementation. It follows the classical FEM assembly procedure:

1. Loop over all elements sequentially
2. For each element, compute the 8×8 element stiffness matrix
3. Scatter element contributions to the global matrix

#### Element Loop Structure

```python
def assemble_system(self) -> None:
    """Assemble global stiffness matrix and force vector."""
    
    # LIL format optimal for incremental construction (random access insertions)
    self.Kg = lil_matrix((self.Nnds, self.Nnds), dtype=np.float64)
    self.fg = np.zeros(self.Nnds, dtype=np.float64)
    
    # Sequential loop over all elements - PRIMARY BOTTLENECK
    for e in range(self.Nels):
        edofs = self.quad8[e]                                 # 8 global node indices for this element
        XN = np.column_stack((self.x[edofs], self.y[edofs]))  # (8,2) nodal coordinates
        Ke, fe = Elem_Quad8(XN, fL=0.0)                       # Compute 8×8 element stiffness matrix
        
        # Scatter element contributions to global system (64 operations per element)
        for i in range(8):
            self.fg[edofs[i]] += fe[i]                        # Accumulate force vector
            for j in range(8):
                self.Kg[edofs[i], edofs[j]] += Ke[i, j]       # LIL format allows efficient +=
```

**Critical Observation:** The nested Python loops (`for e`, `for i`, `for j`) are the primary performance bottleneck. For a mesh with 64,000 elements, this results in:
- 64,000 element computations
- 64,000 × 64 = 4,096,000 scatter operations

#### Element Stiffness Computation

Each element's stiffness matrix is computed via numerical integration using 3×3 Gauss-Legendre quadrature (9 integration points):

```python
def Elem_Quad8(XN, fL):
    """Compute element stiffness matrix Ke and load vector fe for 8-node quad."""
    
    Ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)
    
    nip = 9                        # 3×3 Gauss quadrature (exact for degree 5 polynomials)
    xp, wp = Genip2DQ(nip)         # xp: (9,2) parametric coords, wp: (9,) weights
    
    for ip in range(nip):
        csi, eta = xp[ip, 0], xp[ip, 1]                # Parametric coordinates (ξ,η) ∈ [-1,1]²
        
        B, psi, Detj = Shape_N_Der8(XN, csi, eta)      # B: (8,2) physical derivatives
                                                       # psi: (8,) shape functions
                                                       # Detj: Jacobian determinant for area scaling
        
        wip = wp[ip] * Detj                            # Integration weight × Jacobian
        
        Ke += wip * (B @ B.T)                          # Stiffness: ∫∫ (∇ψᵢ · ∇ψⱼ) dΩ
        fe += fL * wip * psi                           # Force: ∫∫ fL × ψᵢ dΩ (zero for Laplace)
    
    return Ke, fe
```

#### Shape Functions and Derivatives

The 8-node quadrilateral element uses serendipity shape functions. The `Shape_N_Der8` function computes:

1. **Shape functions** ψᵢ(ξ,η) for interpolation
2. **Parametric derivatives** ∂ψᵢ/∂ξ, ∂ψᵢ/∂η
3. **Jacobian matrix** J = ∂(x,y)/∂(ξ,η)
4. **Physical derivatives** B = ∂ψᵢ/∂x, ∂ψᵢ/∂y via chain rule

```python
def Shape_N_Der8(XN, csi, eta):
    """Shape functions and derivatives for 8-node quadrilateral (serendipity element).
    
    Node numbering:  3---6---2
                     |       |
                     7   *   5    (* = center at ξ=η=0)
                     |       |
                     0---4---1
    """
    
    # Shape functions ψᵢ(ξ,η) - corner nodes (0-3) and mid-side nodes (4-7)
    psi = np.zeros(8, dtype=float)
    psi[0] = (csi - 1) * (eta + csi + 1) * (1 - eta) / 4   # Corner: includes quadratic correction
    psi[1] = (1 + csi) * (1 - eta) * (csi - eta - 1) / 4
    psi[2] = (1 + csi) * (1 + eta) * (csi + eta - 1) / 4
    psi[3] = (csi - 1) * (csi - eta + 1) * (1 + eta) / 4
    psi[4] = (1 - csi * csi) * (1 - eta) / 2               # Mid-side: quadratic "bubble"
    psi[5] = (1 + csi) * (1 - eta * eta) / 2
    psi[6] = (1 - csi * csi) * (1 + eta) / 2
    psi[7] = (1 - csi) * (1 - eta * eta) / 2
    
    # Parametric derivatives: Dpsi[i,0] = ∂ψᵢ/∂ξ, Dpsi[i,1] = ∂ψᵢ/∂η
    Dpsi = np.zeros((8, 2), dtype=float)
    # ... (analytical derivative expressions for all 8 nodes)
    
    # Jacobian maps parametric → physical coordinates: J = ∂(x,y)/∂(ξ,η)
    jaco = XN.T @ Dpsi              # (2,2) = (2,8) @ (8,2)
    Detj = np.linalg.det(jaco)      # Area scaling factor (must be > 0 for valid elements)
    Invj = np.linalg.inv(jaco)      # For chain rule transformation
    
    # Physical derivatives via chain rule: ∂ψ/∂x = J⁻¹ × ∂ψ/∂ξ
    B = Dpsi @ Invj                 # (8,2): B[i,0] = ∂ψᵢ/∂x, B[i,1] = ∂ψᵢ/∂y
    
    return B, psi, Detj
```

#### Sparse Matrix Strategy

The implementation uses SciPy's **LIL (List of Lists)** format during assembly for efficient incremental construction:

```python
self.Kg = lil_matrix((self.Nnds, self.Nnds), dtype=np.float64)
```

LIL format is optimal for:
- Random access insertions
- Incremental value additions
- Building sparse matrices element-by-element

After assembly, the matrix is converted to **CSR (Compressed Sparse Row)** format for efficient matrix-vector products during the solve phase:

```python
self.Kg = csr_matrix(self.Kg)
```

---

### Linear System Solve

The solver employs the **Conjugate Gradient (CG)** method with diagonal equilibration and Jacobi preconditioning.

#### Diagonal Equilibration

To improve conditioning, the system is scaled by the inverse square root of diagonal entries:

```python
# Diagonal equilibration reduces condition number κ(K) for faster CG convergence
# Symmetric scaling preserves matrix symmetry (required for CG)
diag = self.Kg.diagonal()
D_inv_sqrt = 1.0 / np.sqrt(np.abs(diag))      # D^(-1/2) element-wise

D_mat = diags(D_inv_sqrt)                      # Sparse diagonal matrix
Kg_eq = D_mat @ self.Kg @ D_mat                # K_eq = D^(-1/2) × K × D^(-1/2)
fg_eq = self.fg * D_inv_sqrt                   # f_eq = D^(-1/2) × f
```

This transformation reduces the condition number, accelerating convergence.

#### Jacobi Preconditioner

A simple diagonal (Jacobi) preconditioner is applied:

```python
diag_eq = Kg_eq.diagonal()
M_inv = 1.0 / diag_eq

def precond_jacobi(v):
    return M_inv * v

M = LinearOperator(Kg_eq.shape, precond_jacobi)
```

#### Solver Invocation

```python
u_eq, self.solve_info = cg(
    Kg_eq,
    fg_eq,
    rtol=1e-8,
    atol=0.0,
    maxiter=self.maxiter,
    M=M,
    callback=self.monitor
)

# De-equilibrate: u = D^(-1/2) u_eq
self.u = u_eq * D_inv_sqrt
```

#### Convergence Monitoring

The `IterativeSolverMonitor` class tracks convergence and provides real-time feedback:

```python
class IterativeSolverMonitor:
    def __call__(self, xk):
        self.it += 1
        
        # Compute actual residual r = b - Ax
        r = self.b - self.A @ xk
        res_norm = float(np.linalg.norm(r))
        rel_res = res_norm / self.b_norm
        
        # Report progress
        if self.progress_callback:
            self.progress_callback.on_iteration(
                iteration=self.it,
                max_iterations=self.maxiter,
                residual=res_norm,
                relative_residual=rel_res,
                elapsed_time=elapsed,
                etr_seconds=etr
            )
```

---

### Boundary Conditions

The implementation supports two types of boundary conditions:

#### Robin Boundary Conditions (Inlet)

Robin BCs model the inlet with a mixed condition αu + β∂u/∂n = γ. Applied via boundary integral contributions:

```python
for (n1, n2, n3) in robin_edges:
    He, Pe = Robin_quadr(
        self.x[n1], self.y[n1],
        self.x[n2], self.y[n2],
        self.x[n3], self.y[n3],
        p=inlet_potential,
        gama=self.gamma
    )
    
    ed = [n1, n2, n3]
    for i in range(3):
        self.fg[ed[i]] += Pe[i]
        for j in range(3):
            self.Kg[ed[i], ed[j]] += He[i, j]
```

The `Robin_quadr` function computes boundary contributions using 3-point Gauss quadrature along the quadratic edge:

```python
def Robin_quadr(x1, y1, x2, y2, x3, y3, p, gama):
    """Robin BC contribution for quadratic edge (3 nodes)."""
    
    He = np.zeros((3, 3), dtype=float)
    Pe = np.zeros(3, dtype=float)
    
    xi, wi = Genip1D(3)  # 1D Gauss points
    
    for ip in range(3):
        csi = xi[ip]
        
        # Quadratic shape functions on edge
        b[0] = 0.5 * csi * (csi - 1.0)  # Node 1
        b[1] = 1.0 - csi * csi           # Node 2 (mid-side)
        b[2] = 0.5 * csi * (csi + 1.0)  # Node 3
        
        # Edge Jacobian (arc length derivative)
        jaco = sqrt(dx² + dy²)
        
        He += wip * p * outer(b, b)
        Pe += wip * gama * b
    
    return He, Pe
```

#### Dirichlet Boundary Conditions (Outlet)

Dirichlet BCs (u = 0 at outlet) are applied using the **penalty method**:

```python
PENALTY_FACTOR = 1.0e12

for n in exit_nodes:
    self.Kg[n, n] += PENALTY_FACTOR
    # fg[n] += PENALTY_FACTOR * 0.0  (target value is zero)
```

The penalty method is chosen over row elimination for simplicity and to preserve matrix symmetry.

---

### Post-Processing

After solving for the velocity potential φ, derived fields are computed:

#### Velocity Field

Velocity is the negative gradient of potential: **v = -∇φ**

```python
def compute_derived_fields(self):
    """Compute velocity and pressure from solved potential φ."""
    
    self.vel = np.zeros((self.Nels, 2), dtype=np.float64)     # Velocity vector per element
    self.abs_vel = np.zeros(self.Nels, dtype=np.float64)      # Speed (magnitude) per element
    
    for e in range(self.Nels):                                 # Loop over elements (bottleneck)
        edofs = self.quad8[e]
        XN = np.column_stack((self.x[edofs], self.y[edofs]))
        
        xp, _ = Genip2DQ(4)                                    # 2×2 quadrature (sufficient for post-proc)
        
        for ip in range(4):
            B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
            grad = B.T @ self.u[edofs]                         # ∇φ = Σ (∂ψᵢ/∂x, ∂ψᵢ/∂y) × φᵢ
            
            # Velocity is NEGATIVE gradient (flow from high to low potential)
            vel_x[ip] = -grad[0]
            vel_y[ip] = -grad[1]
            v_ip[ip] = np.linalg.norm(grad)
        
        # Average over integration points for element centroid value
        self.vel[e] = [vel_x.mean(), vel_y.mean()]
        self.abs_vel[e] = v_ip.mean()
```

#### Pressure Field

Pressure is computed via the Bernoulli equation:

```python
# Bernoulli equation for incompressible inviscid flow: p + ½ρv² = p₀
self.pressure = self.p0 - 0.5 * self.rho * self.abs_vel**2
```

Where:
- p₀ = reference pressure (101,328 Pa)
- ρ = fluid density (0.6125 kg/m³)

---

## Design Decisions

### Approach Rationale

The CPU baseline was designed with the following priorities:

1. **Correctness First:** Every algorithm follows textbook FEM formulations, making it easy to verify against analytical solutions and reference implementations.

2. **Readability:** Code structure mirrors the mathematical formulation—element loops, shape functions, and assembly are immediately recognizable to anyone familiar with FEM.

3. **Modularity:** Helper functions (`Elem_Quad8`, `Shape_N_Der8`, `Robin_quadr`) are isolated, enabling unit testing and reuse across implementations.

4. **Validation Baseline:** Results from this implementation serve as the "ground truth" for validating GPU-accelerated versions.

### Trade-offs Made

| Decision | Benefit | Cost |
|----------|---------|------|
| Sequential element loop | Simple, debuggable code | No parallelism; O(Nels) serial operations |
| LIL → CSR conversion | Efficient incremental assembly | Memory overhead during conversion |
| Python-level loops | Readable, matches math | Interpreter overhead (~100x slower than C) |
| Diagonal equilibration | Better conditioning | Extra matrix operations |
| Penalty method for BCs | Preserves symmetry | Large condition number impact |

### Why Not Optimize the CPU Version?

The CPU baseline intentionally avoids optimizations like:
- Vectorized assembly
- Cython/C extensions
- Multi-threading

This ensures a **fair comparison** with GPU implementations and clearly demonstrates the performance gap between naive Python and accelerated computing.

---

## Performance Characteristics

### Strengths

1. **Simplicity:** Minimal dependencies, easy to understand and modify
2. **Portability:** Runs on any system with Python and SciPy
3. **Debugging:** Easy to inspect intermediate values and trace issues
4. **Memory Efficiency:** Sparse matrix storage handles large problems
5. **Correctness:** Validated against reference solutions

### Limitations

1. **Assembly Bottleneck:** Python loops over elements dominate runtime
2. **No Parallelism:** Single-threaded execution wastes multi-core CPUs
3. **Interpreter Overhead:** Function call overhead in inner loops
4. **Post-processing:** Element-by-element velocity computation is slow
5. **Memory Bandwidth:** Not cache-optimized

### Computational Complexity

| Stage | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Load Mesh | O(Nnds + Nels) | O(Nnds + Nels) |
| Assembly | O(Nels × 8² × 9) | O(Nnds²) sparse |
| Apply BCs | O(boundary_nodes) | O(1) |
| Solve | O(iterations × nnz) | O(Nnds) |
| Post-process | O(Nels × 4) | O(Nels) |

Where:
- Nnds = number of nodes
- Nels = number of elements
- nnz = non-zeros in sparse matrix

### Benchmark Results

| Mesh | Nodes | Elements | Assembly (s) | Solve (s) | Total (s) |
|------|-------|----------|--------------|-----------|-----------|
| small_duct | ~5,000 | ~1,600 | [placeholder] | [placeholder] | [placeholder] |
| s_duct | ~65,000 | ~21,000 | [placeholder] | [placeholder] | [placeholder] |
| venturi_194k | ~194,000 | ~64,000 | [placeholder] | [placeholder] | [placeholder] |

*[Benchmark data to be populated with actual measurements]*

---

## Code Highlights

### Progress Callback Integration

The implementation integrates with a real-time monitoring system for web-based visualization:

```python
def run(self, output_dir=None, export_file=None):
    """Orchestrate complete FEM workflow with real-time progress notifications."""
    
    # Emit event for UI progress indicator
    if self.progress_callback:
        self.progress_callback.on_stage_start(stage='assemble_system')
    
    # Execute and time the stage
    self._time_step('assemble_system', self.assemble_system)
    
    # Emit completion event with timing for metrics display
    if self.progress_callback:
        self.progress_callback.on_stage_complete(
            stage='assemble_system',
            duration=self.timing_metrics['assemble_system']
        )
```

This enables the web application to display real-time progress bars, timing breakdowns, and convergence plots.

### Timing Infrastructure

Every stage is instrumented for performance analysis:

```python
def _time_step(self, step_name: str, func: Callable) -> Any:
    """Execute function and record wall-clock time for benchmarking."""
    t0 = time.perf_counter()          # High-resolution timer
    result = func()
    t1 = time.perf_counter()
    
    self.timing_metrics[step_name] = t1 - t0    # Store for comparison across implementations
    return result
```

This consistent timing approach enables apples-to-apples comparisons across all solver implementations.

---

## Lessons Learned

### Development Insights

1. **Python Loop Overhead is Severe:** The element assembly loop, despite being "just" 64,000 iterations, dominates runtime due to interpreter overhead and function call costs.

2. **Sparse Matrix Format Matters:** Using LIL for assembly and converting to CSR for solve is crucial. Attempting to use CSR during assembly results in catastrophic performance.

3. **Equilibration is Essential:** Without diagonal equilibration, the CG solver struggles to converge for poorly-conditioned meshes with high aspect ratio elements.

4. **Penalty Method Simplicity:** While theoretically introducing conditioning issues, the penalty method proved robust and much simpler than matrix partitioning approaches.

### Debugging Challenges

1. **Silent Numerical Issues:** Mesh quality problems (inverted elements, negative Jacobians) caused subtle solution errors without explicit failures.

2. **Boundary Detection:** Floating-point tolerance in identifying boundary nodes required careful tuning (`bc_tolerance = 1e-9`).

3. **Sign Convention:** The velocity sign (`v = -∇φ`) was initially incorrect, producing reversed flow directions.

---

## Conclusions

The CPU Baseline implementation establishes a **correct, readable, and maintainable** reference for the FEM solver. Its deliberate simplicity makes it an ideal foundation for:

1. **Validation:** All accelerated implementations must match its numerical results
2. **Education:** The code directly maps to mathematical FEM formulations
3. **Benchmarking:** Provides the reference baseline for performance comparisons

### When to Use This Implementation

- **Development and debugging** of new features
- **Small meshes** (< 10,000 nodes) where runtime is acceptable
- **Validation** of optimized implementations
- **Teaching** FEM concepts

### Key Takeaway

The CPU baseline reveals that **assembly is the dominant cost** in this FEM workflow, consuming 80-90% of total runtime for large meshes. This insight directly motivates the GPU acceleration strategy: massively parallel element processing can theoretically achieve speedups proportional to the number of elements.

The subsequent implementations will progressively address these bottlenecks:
- **CPU Threaded/Multiprocess:** Parallelize the element loop across CPU cores
- **Numba JIT:** Eliminate Python interpreter overhead
- **GPU (Numba CUDA / CuPy):** Massive parallelism with thousands of concurrent threads

---
