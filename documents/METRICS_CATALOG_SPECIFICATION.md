# FEMulator Pro - Metrics Catalog Specification

## Overview

This document describes all metrics available in the FEMulator Pro Metrics Catalog. Each metric is categorized, described, and evaluated for academic relevance. Implementation proposals include UI mockups and data requirements.

---

## Metric Categories

| Category | Icon | Purpose | Metric Count |
|----------|------|---------|--------------|
| Live Solver Monitor | ⚡ | Real-time solver progress visualization | 4 |
| Solution Quality | 📊 | Post-solve numerical analysis | 4 |
| Performance Breakdown | ⏱️ | Timing analysis per solver stage | 3 |
| Comparative Analysis | 📈 | Cross-run and cross-solver comparisons | 3 |
| Mesh Information | 🔷 | Model geometry characteristics | 2 |
| System Information | 💻 | Hardware and environment details | 3 |

**Total: 19 metrics**

---

## Relevance Classification

| Level | Description | Academic Value |
|-------|-------------|----------------|
| ⭐⭐⭐ **Critical** | Essential for demonstrating project goals | Must-have for presentation |
| ⭐⭐ **Important** | Adds significant value to analysis | Strongly recommended |
| ⭐ **Nice-to-have** | Supplementary information | Optional enhancement |

---

## Category 1: Live Solver Monitor

Real-time metrics that update during the iterative solve process. These demonstrate the application's responsiveness and provide insight into solver behavior.

### 1.1 Progress Ring

| Attribute | Value |
|-----------|-------|
| **ID** | `progress-ring` |
| **Type** | Real-time |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
A circular progress indicator showing the current iteration as a percentage of maximum iterations. Includes the estimated time remaining (ETR) calculated from the current solve rate.

**Academic Value:**
- Demonstrates real-time WebSocket communication
- Shows responsive UI updates during compute-intensive operations
- Provides user feedback during long-running solvers

**Data Requirements:**
```javascript
{
  iteration: number,      // Current iteration (e.g., 5000)
  max_iterations: number, // Maximum iterations (e.g., 15000)
  etr_seconds: number,    // Estimated time remaining
  elapsed_time: number    // Time since solve started
}
```

**UI Proposal:**
```
        ┌─────────┐
       ╱    65%    ╲
      │             │
      │  Iter 9800  │
      │   / 15000   │
       ╲           ╱
        └─────────┘
         ETR: 2m 47s
```

**Implementation Notes:**
- SVG-based circular progress with stroke-dasharray animation
- Update throttled to ~10fps to avoid performance impact
- Color transitions: blue (0-50%) → green (50-90%) → gold (90-100%)

---

### 1.2 Convergence Plot

| Attribute | Value |
|-----------|-------|
| **ID** | `convergence-plot` |
| **Type** | Real-time |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
A line chart showing the residual norm versus iteration number on a logarithmic scale. This is the quintessential visualization for iterative solver convergence behavior.

**Academic Value:**
- **Highly relevant** for FEM/numerical methods courses
- Demonstrates understanding of iterative solver convergence
- Shows the effect of preconditioning (Jacobi) on convergence rate
- Can reveal solver issues (stagnation, divergence, oscillation)

**Data Requirements:**
```javascript
{
  history: [
    { iteration: 50, residual: 7.065e-02 },
    { iteration: 100, residual: 7.146e-02 },
    // ... accumulated during solve
  ],
  tolerance: 1e-8  // Target tolerance line
}
```

**UI Proposal:**
```
  Residual
  1e+0 ┤●●●●
       │    ●●●
  1e-3 ┤      ●●●●
       │          ●●●●
  1e-6 ┤              ●●●●●
       │                   ●●●●●●●●
  1e-9 ┼─────────────────────────────── tol
       └────────────────────────────────
        0    2500   5000   7500   10000
                   Iteration
```

**Implementation Notes:**
- Canvas or SVG-based rendering
- Log scale Y-axis (1e+0 to 1e-12 typical range)
- Horizontal dashed line at target tolerance
- Accumulate residual history in client during solve
- Support for zoom/pan on completed plots

---

### 1.3 Stage Timeline

| Attribute | Value |
|-----------|-------|
| **ID** | `stage-timeline` |
| **Type** | Real-time |
| **Relevance** | ⭐⭐ Important |
| **Default** | Enabled |

**Description:**
A horizontal segmented bar showing the progression through solver stages: Load Mesh → Assemble System → Apply BC → Solve → Post-Process. The current stage is highlighted, completed stages show their duration.

**Academic Value:**
- Visualizes the FEM workflow stages
- Shows relative time spent in each phase
- Demonstrates understanding of FEM pipeline

**Data Requirements:**
```javascript
{
  stages: ['load_mesh', 'assemble_system', 'apply_bc', 'solve_system', 'compute_derived'],
  current_stage: 'solve_system',
  completed: {
    'load_mesh': 0.26,
    'assemble_system': 0.12,
    'apply_bc': 1.07
  }
}
```

**UI Proposal:**
```
  Load   Assemble   BC    ▶ Solve          Post
  ├────┼──────────┼─────┼════════════════┼─────┤
  0.3s    0.1s     1.1s      5.5s+         ...

  ████ Completed  ════ In Progress  ░░░░ Pending
```

**Implementation Notes:**
- Flexbox-based segmented bar
- Pulse animation on current stage
- Tooltips showing exact durations
- Proportional widths based on expected duration (or actual once complete)

---

### 1.4 Residual Display

| Attribute | Value |
|-----------|-------|
| **ID** | `residual-display` |
| **Type** | Real-time |
| **Relevance** | ⭐⭐ Important |
| **Default** | Enabled |

**Description:**
Numeric display of the current residual norm and relative residual in scientific notation. Updates with each progress event.

**Academic Value:**
- Shows actual convergence metrics
- Demonstrates understanding of stopping criteria
- Complements the convergence plot with exact values

**Data Requirements:**
```javascript
{
  residual: 1.101e-9,
  relative_residual: 1.511e-8,
  tolerance: 1e-8
}
```

**UI Proposal:**
```
  ┌─────────────────────────────┐
  │  Residual:  1.101e-09  ✓    │
  │  Relative:  1.511e-08       │
  │  Target:    1.000e-08       │
  └─────────────────────────────┘
```

**Implementation Notes:**
- Color coding: green when below tolerance, amber when close, red when far
- Checkmark indicator when converged
- Monospace font for alignment

---

## Category 2: Solution Quality

Metrics computed after the solve completes, analyzing the quality and characteristics of the solution.

### 2.1 Solution Range

| Attribute | Value |
|-----------|-------|
| **ID** | `solution-range` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Displays the minimum and maximum values of the computed potential field (u), along with mean and standard deviation. Optionally includes a mini histogram showing distribution.

**Academic Value:**
- Validates solution reasonableness (non-physical values indicate problems)
- Shows understanding of the physical meaning of the solution
- Useful for comparing across different meshes/conditions

**Data Requirements:**
```javascript
{
  u_range: [0.0, 11.825],
  u_mean: 5.755,
  u_std: 4.391
}
```

**UI Proposal:**
```
  ┌─ Solution (Velocity Potential) ─┐
  │                                 │
  │  Min:   0.000                   │
  │  Max:   11.825                  │
  │  Mean:  5.755                   │
  │  Std:   4.391                   │
  │                                 │
  │  ░░▒▒▓▓████▓▓▒▒░░  histogram   │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Mini sparkline histogram using SVG bars
- Tooltips explaining physical interpretation
- Highlight anomalous values (negative potential, etc.)

---

### 2.2 Velocity Field Stats

| Attribute | Value |
|-----------|-------|
| **ID** | `velocity-stats` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Statistics for the velocity field computed as v = -∇u. Includes minimum, maximum, and mean velocity magnitude. This is the primary physical quantity of interest for flow analysis.

**Academic Value:**
- **Core deliverable** - velocity field is the main output
- Validates physical reasonableness (inlet/outlet velocities)
- Demonstrates understanding of potential flow theory

**Data Requirements:**
```javascript
{
  vel_min: 0.0,
  vel_max: 42.15,
  vel_mean: 12.34,
  // Derived from abs_vel array
}
```

**UI Proposal:**
```
  ┌─ Velocity Field (m/s) ──────────┐
  │                                 │
  │  Min:   0.00   (stagnation)     │
  │  Max:   42.15  (throat)         │
  │  Mean:  12.34                   │
  │                                 │
  │  ████████████████░░░░░░░░░░░░░  │
  │  0              42.15           │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Color gradient bar matching the visualization colormap
- Annotations for physical interpretation
- Link to 3D visualization color scale

---

### 2.3 Pressure Distribution

| Attribute | Value |
|-----------|-------|
| **ID** | `pressure-stats` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐ Important |
| **Default** | Disabled |

**Description:**
Pressure statistics derived from Bernoulli's equation: p = p₀ - ½ρv². Shows the pressure drop across the domain and validates energy conservation.

**Academic Value:**
- Demonstrates understanding of Bernoulli principle
- Shows connection between potential flow and pressure
- Useful for engineering analysis (pressure drop calculations)

**Data Requirements:**
```javascript
{
  p0: 101328.8,        // Reference pressure (Pa)
  rho: 0.6125,         // Density (kg/m³)
  p_min: 98234.5,      // Minimum pressure
  p_max: 101328.8,     // Maximum pressure
  delta_p: 3094.3      // Pressure drop
}
```

**UI Proposal:**
```
  ┌─ Pressure (Bernoulli) ──────────┐
  │                                 │
  │  p₀:     101,329 Pa             │
  │  p_min:   98,235 Pa             │
  │  p_max:  101,329 Pa             │
  │  Δp:       3,094 Pa             │
  │                                 │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Format large numbers with thousand separators
- Show pressure in Pa or kPa based on magnitude
- Optional conversion to other units

---

### 2.4 Convergence Quality

| Attribute | Value |
|-----------|-------|
| **ID** | `convergence-quality` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Summary of convergence behavior: whether the solver converged, final iteration count versus maximum, final residual achieved, and convergence rate estimate.

**Academic Value:**
- **Essential** for validating solution quality
- Shows understanding of iterative solver behavior
- Important for comparing solver implementations

**Data Requirements:**
```javascript
{
  converged: true,
  iterations: 9871,
  max_iterations: 15000,
  final_residual: 1.608e-9,
  final_relative_residual: 1.438e-8,
  tolerance: 1e-8
}
```

**UI Proposal:**
```
  ┌─ Convergence Status ────────────┐
  │                                 │
  │  Status:    ✓ CONVERGED         │
  │  Iterations: 9,871 / 15,000     │
  │  Final ||r||: 1.61e-09          │
  │  Relative:    1.44e-08          │
  │  Target:      1.00e-08          │
  │                                 │
  │  ████████████████░░░░  65.8%    │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Large checkmark/X for convergence status
- Progress bar showing iterations used
- Color coding: green (converged), red (not converged)

---

## Category 3: Performance Breakdown

Timing metrics for analyzing computational performance across solver stages.

### 3.1 Timing Waterfall

| Attribute | Value |
|-----------|-------|
| **ID** | `timing-waterfall` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Visual horizontal bar chart showing time spent in each solver stage. Bars are proportional to duration, with percentage annotations. This is the primary performance visualization.

**Academic Value:**
- **Core deliverable** for GPU acceleration project
- Clearly shows where time is spent (solve dominates for CPU, assembly for GPU)
- Essential for comparing implementations
- Demonstrates understanding of performance bottlenecks

**Data Requirements:**
```javascript
{
  timing_metrics: {
    load_mesh: 0.26,
    assemble_system: 0.12,
    apply_bc: 1.07,
    solve_system: 5.53,
    compute_derived: 0.006,
    total_program_time: 6.99
  }
}
```

**UI Proposal:**
```
  ┌─ Timing Breakdown ──────────────────────────────┐
  │                                                 │
  │  load_mesh      ██ 0.26s                  3.7%  │
  │  assemble       █ 0.12s                   1.7%  │
  │  apply_bc       ██████████ 1.07s         15.3%  │
  │  solve          ████████████████████████ 5.53s  79.1%  │
  │  post_process   ░ 0.01s                   0.1%  │
  │                 ─────────────────────────────── │
  │  TOTAL          6.99s                           │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

**Implementation Notes:**
- Horizontal bars with proportional widths
- Color coding by stage type (I/O, compute, solver)
- Hover for exact values
- Bold emphasis on dominant stage

---

### 3.2 Timing Table

| Attribute | Value |
|-----------|-------|
| **ID** | `timing-table` |
| **Type** | Post-solve |
| **Relevance** | ⭐ Nice-to-have |
| **Default** | Disabled |

**Description:**
Tabular view of all timing metrics including sub-timings not shown in the waterfall (e.g., matrix conversion, GPU transfer times).

**Academic Value:**
- Provides detailed breakdown for analysis
- Useful for identifying optimization opportunities
- Supports data export for reports

**Data Requirements:**
Same as Timing Waterfall, plus any sub-timings available.

**UI Proposal:**
```
  ┌─ Detailed Timings ──────────────────────┐
  │                                         │
  │  Stage              Time     % Total    │
  │  ─────────────────────────────────────  │
  │  load_mesh          0.260s     3.72%    │
  │  assemble_system    0.120s     1.72%    │
  │    └─ gpu_transfer  0.015s     0.21%    │
  │  apply_bc           1.070s    15.31%    │
  │  solve_system       5.530s    79.11%    │
  │  compute_derived    0.006s     0.09%    │
  │  ─────────────────────────────────────  │
  │  TOTAL              6.986s   100.00%    │
  │                                         │
  └─────────────────────────────────────────┘
```

**Implementation Notes:**
- Monospace alignment for numbers
- Expandable rows for sub-timings
- Copy-to-clipboard functionality

---

### 3.3 Throughput Metrics

| Attribute | Value |
|-----------|-------|
| **ID** | `throughput` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐ Important |
| **Default** | Disabled |

**Description:**
Derived performance metrics: elements processed per second during assembly, iterations per second during solve, and overall throughput.

**Academic Value:**
- Enables fair comparison across different mesh sizes
- Shows efficiency independent of problem size
- Useful for scalability analysis

**Data Requirements:**
```javascript
{
  elements: 338544,
  nodes: 1357953,
  iterations: 9871,
  timing_metrics: { ... },
  // Computed:
  elements_per_second: 2821200,  // elements / assemble_time
  iterations_per_second: 1785,   // iterations / solve_time
  dofs_per_second: 245422        // nodes / total_time
}
```

**UI Proposal:**
```
  ┌─ Throughput ────────────────────┐
  │                                 │
  │  Assembly:  2.82M elements/s    │
  │  Solver:    1,785 iter/s        │
  │  Overall:   194k nodes/s        │
  │                                 │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Format large numbers with SI prefixes (k, M)
- Show comparison to reference values if available

---

## Category 4: Comparative Analysis

Metrics that compare the current run against historical data or other solver implementations.

### 4.1 Solver Comparison

| Attribute | Value |
|-----------|-------|
| **ID** | `solver-comparison` |
| **Type** | Comparative |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Bar chart comparing the current solver's performance against other implementations for the same model. Shows GPU, Numba CUDA, Numba, CPU Multiprocess, CPU Threaded, and CPU baseline.

**Academic Value:**
- **Primary deliverable** for GPU acceleration project
- Directly demonstrates the value of different parallelization strategies
- Essential for project presentation and report
- Shows understanding of performance trade-offs

**Data Requirements:**
```javascript
// From Benchmark API for same model
{
  current: { solver: 'gpu', total_time: 6.99 },
  comparisons: [
    { solver: 'numba_cuda', total_time: 13.79 },
    { solver: 'numba', total_time: 329.75 },
    { solver: 'cpu_multiprocess', total_time: 330.49 },
    { solver: 'cpu_threaded', total_time: 387.59 },
    { solver: 'cpu', total_time: 542.47 }
  ]
}
```

**UI Proposal:**
```
  ┌─ Solver Comparison (Y-Tube X-Large) ────────────┐
  │                                                 │
  │  GPU          ████ 6.99s                  1.0x  │
  │  Numba CUDA   ████████ 13.79s             2.0x  │
  │  Numba        ████████████████████ 329.8s 47.2x │
  │  Multiprocess ████████████████████ 330.5s 47.3x │
  │  Threaded     ██████████████████████ 387.6s 55x │
  │  CPU          ████████████████████████ 542.5s 78x│
  │                                                 │
  │  ▲ Current run                                  │
  └─────────────────────────────────────────────────┘
```

**Implementation Notes:**
- Fetch from `/api/benchmark` filtered by model
- Highlight current run
- Show speedup factor relative to slowest (or CPU baseline)
- Handle missing data gracefully

---

### 4.2 Historical Best

| Attribute | Value |
|-----------|-------|
| **ID** | `historical-best` |
| **Type** | Comparative |
| **Relevance** | ⭐ Nice-to-have |
| **Default** | Disabled |

**Description:**
Compares the current run against the best recorded time for the same model and solver combination. Shows whether this is a new record or how far from the best.

**Academic Value:**
- Tracks optimization progress over time
- Useful for regression detection
- Demonstrates reproducibility

**Data Requirements:**
```javascript
{
  current_time: 6.99,
  best_time: 6.85,
  best_date: '2024-12-26',
  is_new_record: false,
  difference_percent: 2.04
}
```

**UI Proposal:**
```
  ┌─ Historical Best ───────────────┐
  │                                 │
  │  Current:   6.99s               │
  │  Best:      6.85s (Dec 26)      │
  │  Diff:      +2.04%              │
  │                                 │
  │  ░░░░░░░░░░░░░░░░████           │
  │                   ▲ You are here │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Query benchmark API for best time
- Celebration animation for new records
- Show date of best run

---

### 4.3 Speedup Factors

| Attribute | Value |
|-----------|-------|
| **ID** | `speedup-factors` |
| **Type** | Comparative |
| **Relevance** | ⭐⭐⭐ Critical |
| **Default** | Enabled |

**Description:**
Shows the speedup factor of the current solver compared to the CPU baseline. This is the headline number for GPU acceleration projects.

**Academic Value:**
- **The key metric** for demonstrating project success
- "78x faster than CPU" is the headline result
- Essential for abstracts, presentations, and reports

**Data Requirements:**
```javascript
{
  current_solver: 'gpu',
  current_time: 6.99,
  cpu_baseline_time: 542.47,
  speedup: 77.6
}
```

**UI Proposal:**
```
  ┌───────────────────────────────────────┐
  │                                       │
  │            ████████████               │
  │               77.6×                   │
  │         faster than CPU               │
  │                                       │
  │    GPU: 6.99s  vs  CPU: 542.47s       │
  │                                       │
  └───────────────────────────────────────┘
```

**Implementation Notes:**
- Large, prominent display
- Fetch CPU baseline from benchmark API
- Handle case where CPU baseline doesn't exist
- Color coding: green (>10x), gold (>50x), rainbow (>100x)

---

## Category 5: Mesh Information

Metrics describing the computational mesh characteristics.

### 5.1 Mesh Statistics

| Attribute | Value |
|-----------|-------|
| **ID** | `mesh-stats` |
| **Type** | Post-solve |
| **Relevance** | ⭐⭐ Important |
| **Default** | Enabled |

**Description:**
Basic mesh statistics: node count, element count, element type (Quad-8), and complexity classification (Small/Medium/Large/X-Large).

**Academic Value:**
- Provides context for performance numbers
- Shows problem size for scalability analysis
- Demonstrates understanding of mesh characteristics

**Data Requirements:**
```javascript
{
  nodes: 1357953,
  elements: 338544,
  element_type: 'Quad-8',
  dofs: 1357953,  // For scalar problem, DOFs = nodes
  complexity: 'X-Large'  // Derived from element count
}
```

**UI Proposal:**
```
  ┌─ Mesh Statistics ───────────────┐
  │                                 │
  │  Nodes:      1,357,953          │
  │  Elements:   338,544            │
  │  Type:       Quad-8 (8-node)    │
  │  DOFs:       1,357,953          │
  │                                 │
  │  Complexity: ████ X-LARGE       │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Format numbers with thousand separators
- Color-coded complexity badge
- Tooltip explaining Quad-8 elements

---

### 5.2 Boundary Conditions

| Attribute | Value |
|-----------|-------|
| **ID** | `boundary-info` |
| **Type** | Post-solve |
| **Relevance** | ⭐ Nice-to-have |
| **Default** | Disabled |

**Description:**
Information about applied boundary conditions: Robin BC edges (inlet), Dirichlet nodes (outlet), and any special handling (unused nodes).

**Academic Value:**
- Shows understanding of BC application in FEM
- Documents the problem setup
- Useful for validating model configuration

**Data Requirements:**
```javascript
{
  robin_edges: 400,
  dirichlet_nodes: 402,
  unused_nodes: 338544,
  inlet_potential: 0.0,
  outlet_potential: 0.0  // Dirichlet value
}
```

**UI Proposal:**
```
  ┌─ Boundary Conditions ───────────┐
  │                                 │
  │  Inlet (Robin):                 │
  │    Edges: 400                   │
  │    γ = 2.5, p = 0               │
  │                                 │
  │  Outlet (Dirichlet):            │
  │    Nodes: 402                   │
  │    u = 0                        │
  │                                 │
  │  Penalty nodes: 338,544         │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Collapsible sections
- Mathematical notation for BC formulation

---

## Category 6: System Information

Hardware and software environment details.

### 6.1 Server Hardware

| Attribute | Value |
|-----------|-------|
| **ID** | `server-hardware` |
| **Type** | System |
| **Relevance** | ⭐⭐ Important |
| **Default** | Disabled |

**Description:**
Server hardware specifications: CPU model, core count, RAM, GPU model, GPU memory, CUDA version.

**Academic Value:**
- Essential for reproducibility
- Required for fair performance comparisons
- Documents the test environment

**Data Requirements:**
```javascript
{
  hostname: 'logus2k',
  cpu_model: 'AMD Ryzen 9 5950X 16-Core',
  cpu_cores: 20,
  ram_gb: 64,
  gpu_model: 'NVIDIA GeForce RTX 3090',
  gpu_memory_gb: 24,
  cuda_version: '12.1'
}
```

**UI Proposal:**
```
  ┌─ Server Hardware ───────────────┐
  │                                 │
  │  CPU:   AMD Ryzen 9 5950X       │
  │         20 cores                │
  │  RAM:   64 GB                   │
  │                                 │
  │  GPU:   NVIDIA RTX 3090         │
  │         24 GB VRAM              │
  │  CUDA:  12.1                    │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Fetch from `/api/benchmark/server-config`
- Icons for CPU/GPU
- Link to full specs if available

---

### 6.2 Client Hardware

| Attribute | Value |
|-----------|-------|
| **ID** | `client-hardware` |
| **Type** | System |
| **Relevance** | ⭐ Nice-to-have |
| **Default** | Disabled |

**Description:**
Browser and client system information: browser name/version, OS, WebGL GPU renderer, screen resolution.

**Academic Value:**
- Documents visualization capabilities
- Useful for troubleshooting rendering issues
- Shows Three.js WebGL requirements

**Data Requirements:**
```javascript
{
  browser: 'Chrome',
  browser_version: '120',
  os: 'Windows',
  os_version: '10/11',
  gpu_vendor: 'NVIDIA Corporation',
  gpu_renderer: 'NVIDIA GeForce RTX 4080',
  webgl_version: '2.0',
  screen: '2560x1440',
  pixel_ratio: 1.5
}
```

**UI Proposal:**
```
  ┌─ Client Environment ────────────┐
  │                                 │
  │  Browser:  Chrome 120           │
  │  OS:       Windows 10/11        │
  │                                 │
  │  WebGL:    2.0                  │
  │  GPU:      NVIDIA RTX 4080      │
  │  Screen:   2560×1440 @1.5x      │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Detect using `client-hardware.js`
- Show WebGL support status
- Warning if WebGL 2 not available

---

### 6.3 Solver Configuration

| Attribute | Value |
|-----------|-------|
| **ID** | `solver-info` |
| **Type** | System |
| **Relevance** | ⭐⭐ Important |
| **Default** | Disabled |

**Description:**
Solver parameters used for the current run: solver type, tolerance, maximum iterations, preconditioner type.

**Academic Value:**
- Documents solver configuration
- Required for reproducibility
- Shows understanding of solver parameters

**Data Requirements:**
```javascript
{
  solver_type: 'gpu',
  tolerance: 1e-8,
  max_iterations: 15000,
  preconditioner: 'Jacobi',
  equilibration: 'Diagonal'
}
```

**UI Proposal:**
```
  ┌─ Solver Configuration ──────────┐
  │                                 │
  │  Type:           GPU (CuPy)     │
  │  Tolerance:      1.0e-08        │
  │  Max Iterations: 15,000         │
  │  Preconditioner: Jacobi         │
  │  Equilibration:  Diagonal       │
  └─────────────────────────────────┘
```

**Implementation Notes:**
- Expandable for advanced parameters
- Link to documentation for each parameter

---

## Implementation Priority Matrix

### Phase 1: Critical Metrics (MVP)

| Metric | Category | Effort | Impact |
|--------|----------|--------|--------|
| Progress Ring | Live Monitor | Low | High |
| Convergence Plot | Live Monitor | Medium | High |
| Timing Waterfall | Performance | Medium | High |
| Speedup Factors | Comparative | Low | High |
| Convergence Quality | Solution | Low | High |

**Estimated effort:** 2-3 days

### Phase 2: Important Metrics

| Metric | Category | Effort | Impact |
|--------|----------|--------|--------|
| Stage Timeline | Live Monitor | Low | Medium |
| Residual Display | Live Monitor | Low | Medium |
| Solution Range | Solution | Low | Medium |
| Velocity Stats | Solution | Low | Medium |
| Solver Comparison | Comparative | Medium | High |
| Mesh Statistics | Mesh | Low | Medium |

**Estimated effort:** 2 days

### Phase 3: Nice-to-have Metrics

| Metric | Category | Effort | Impact |
|--------|----------|--------|--------|
| Pressure Stats | Solution | Low | Low |
| Timing Table | Performance | Low | Low |
| Throughput | Performance | Low | Medium |
| Historical Best | Comparative | Medium | Low |
| Boundary Info | Mesh | Low | Low |
| Server Hardware | System | Low | Medium |
| Client Hardware | System | Low | Low |
| Solver Config | System | Low | Medium |

**Estimated effort:** 1-2 days

---

## Data Flow Architecture

```
┌─────────────────┐     Socket.IO      ┌──────────────────┐
│   FEM Solver    │ ─────────────────▶ │  Metrics Store   │
│   (Server)      │   solve_progress   │  (Client)        │
└─────────────────┘   stage_complete   └────────┬─────────┘
                      solve_complete            │
                                                ▼
┌─────────────────┐                    ┌──────────────────┐
│ Benchmark API   │ ◀───── fetch ───── │  Metrics Panel   │
│ /api/benchmark  │                    │  (Runtime Tab)   │
└─────────────────┘                    └──────────────────┘
                                                │
                                                ▼
                                       ┌──────────────────┐
                                       │  Metric Widgets  │
                                       │  (Visualizations)│
                                       └──────────────────┘
```

---

## Summary

| Category | Metrics | Critical | Important | Nice-to-have |
|----------|---------|----------|-----------|--------------|
| Live Monitor | 4 | 2 | 2 | 0 |
| Solution Quality | 4 | 3 | 1 | 0 |
| Performance | 3 | 1 | 1 | 1 |
| Comparative | 3 | 2 | 0 | 1 |
| Mesh Info | 2 | 0 | 1 | 1 |
| System Info | 3 | 0 | 2 | 1 |
| **Total** | **19** | **8** | **7** | **4** |

**Recommendation:** Implement the 8 Critical metrics first for the project presentation, then add Important metrics as time permits.
