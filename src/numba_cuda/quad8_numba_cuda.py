"""
QUAD8 FEM Solver - Numba CUDA version.

Standalone solver class for 2D potential flow using Quad-8 elements.
Uses Numba @cuda.jit kernels as alternative to CuPy RawKernel.

Functionally equivalent to GPU solver with identical interface.

Note: Pylance shows false positive type errors for Numba CUDA APIs
due to incomplete type stubs. The code runs correctly at runtime.
"""
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false

import sys
import time
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Callable
from numpy.typing import NDArray

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsplg
from numba import cuda
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from scipy.sparse.linalg import cg, gmres, LinearOperator

from kernels_numba_cuda import (
    quad8_assembly_kernel,
    quad8_postprocess_kernel,
    get_gauss_points_9,
    get_gauss_points_4
)


# =============================================================================
# Progress Callback Monitor
# =============================================================================

# =============================================================================
# Progress Callback Monitor (GPU)
# =============================================================================

class GPUSolverMonitor:
    """Monitor for GPU CG iterations."""
    
    def __init__(self, A, b, every: int = 50, maxiter: int = 50000, 
                 verbose: bool = True, progress_callback=None):
        self.A = A
        self.b = b
        self.it: int = 0
        self.every: int = every
        self.maxiter: int = maxiter
        self.verbose = verbose
        self.start_time: float = time.perf_counter()
        self.b_norm = float(cp.linalg.norm(b))
        self.progress_callback = progress_callback
    
    def reset(self):
        """Reset for fallback solver."""
        self.it = 0
        self.start_time = time.perf_counter()
    
    def __call__(self, xk) -> None:
        self.it += 1
        
        if self.it % self.every != 0:
            return
        
        # Compute residual on GPU
        r = self.b - self.A @ xk
        res_norm = float(cp.linalg.norm(r))
        rel_res = res_norm / self.b_norm
        
        elapsed = time.perf_counter() - self.start_time
        time_per_it = elapsed / self.it
        remaining = self.maxiter - self.it
        etr = remaining * time_per_it
        
        def fmt(seconds: float) -> str:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m:02d}m:{s:02d}s"
        
        progress = 100.0 * self.it / self.maxiter
        
        if self.verbose:
            print(
                f"  Iter {self.it}/{self.maxiter} ({progress:.1f}%), "
                f"||r|| = {res_norm:.3e}, rel = {rel_res:.3e}, "
                f"ETR: {fmt(etr)}"
            )
        
        if self.progress_callback is not None:
            self.progress_callback.on_iteration(
                iteration=self.it,
                max_iterations=self.maxiter,
                residual=res_norm,
                relative_residual=rel_res,
                elapsed_time=elapsed,
                etr_seconds=etr
            )


class IterativeSolverMonitor:
    """Monitor for CG iterations with actual residual computation."""

    def __init__(self, A, b, every: int = 10, maxiter: int = 15000, progress_callback=None):
        self.A = A
        self.b = b
        self.it: int = 0
        self.every: int = every
        self.maxiter: int = maxiter
        self.start_time: float = time.perf_counter()
        self.b_norm = float(np.linalg.norm(b))
        self.progress_callback = progress_callback

    def __call__(self, xk: NDArray[np.float64]) -> None:
        self.it += 1
        
        # Send incremental solution updates
        if self.progress_callback is not None:
            if self.it == 1 or (self.it % 100 == 0):
                self.progress_callback.on_solution_increment(
                    iteration=self.it,
                    solution=xk
                )

        # Log progress at specified intervals
        if self.it % self.every != 0 and self.it != self.maxiter:
            return

        # Compute actual residual
        r = self.b - self.A @ xk
        res_norm = float(np.linalg.norm(r))
        rel_res = res_norm / self.b_norm

        elapsed = time.perf_counter() - self.start_time
        time_per_it = elapsed / self.it
        remaining = self.maxiter - self.it
        etr = remaining * time_per_it

        def fmt(seconds: float) -> str:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m:02d}m:{s:02d}s"

        progress = 100.0 * self.it / self.maxiter

        print(
            f"  Iter {self.it}/{self.maxiter} ({progress:.1f}%), "
            f"||r|| = {res_norm:.3e}, rel = {rel_res:.3e}, "
            f"ETR: {fmt(etr)}"
        )
        
        if self.progress_callback is not None:
            self.progress_callback.on_iteration(
                iteration=self.it,
                max_iterations=self.maxiter,
                residual=res_norm,
                relative_residual=rel_res,
                elapsed_time=elapsed,
                etr_seconds=etr
            )


# =============================================================================
# Main Solver Class
# =============================================================================

class Quad8FEMSolverNumbaCUDA:
    """
    QUAD8 FEM Solver - Numba CUDA version.
    
    Features:
    - Quad-8 (8-node quadrilateral) elements
    - Robin boundary conditions (inlet)
    - Dirichlet boundary conditions (outlet)
    - GPU assembly using Numba @cuda.jit kernels
    - CPU solver (SciPy CG) for linear system
    - GPU post-processing using Numba @cuda.jit
    """
    
    def __init__(
        self,
        mesh_file: Path | str,
        p0: float = 101328.8281,
        rho: float = 0.6125,
        gamma: float = 2.5,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 15000,
        bc_tolerance: float = 1e-9,
        cg_print_every: int = 50,
        assembly_print_every: int = 50000,
        implementation_name: str = "NumbaCUDA",
        verbose: bool = True,
        progress_callback=None
    ):
        """
        Initialize FEM solver.
        
        Args:
            mesh_file: Path to mesh file (.h5, .npz, .xlsx)
            p0: Reference pressure (Pa)
            rho: Fluid density (kg/m³)
            gamma: Specific heat ratio
            rtol: CG solver relative tolerance
            atol: CG solver absolute tolerance
            maxiter: Maximum CG iterations
            bc_tolerance: Tolerance for detecting boundary nodes
            cg_print_every: Print CG progress every N iterations
            assembly_print_every: Print assembly progress every N elements
            implementation_name: Identifier for output files
            verbose: Enable/disable progress printing
            progress_callback: Callback for progress events
        """
        # Physics parameters
        self.p0 = p0
        self.rho = rho
        self.gamma = gamma
        
        # Solver parameters
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.bc_tolerance = bc_tolerance
        self.cg_print_every = cg_print_every
        self.assembly_print_every = assembly_print_every
        
        # Configuration
        self.implementation_name = implementation_name
        self.verbose = verbose
        
        # Mesh data (CPU)
        self.mesh_file = Path(mesh_file)
        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64]
        self.quad8: NDArray[np.int32]
        self.Nnds: int
        self.Nels: int
        
        # System matrices (CPU)
        self.Kg: csr_matrix
        self.fg: NDArray[np.float64]
        
        # Solution
        self.u: NDArray[np.float64]
        self.vel: NDArray[np.float64]
        self.abs_vel: NDArray[np.float64]
        self.pressure: NDArray[np.float64]

        # Timing
        self.program_start_time: float = time.perf_counter()
        self.timing_metrics: Dict[str, float] = {}
        
        # Solver state
        self.monitor: GPUSolverMonitor
        self.solve_info: int
        self.converged: bool = False
        self.iterations: int = 0
        
        self.progress_callback = progress_callback

    def _time_step(self, step_name: str, func: Callable[..., Any]) -> Any:
        """Utility to time a function call and store the result."""
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        self.timing_metrics[step_name] = t1 - t0
        
        if self.verbose:
            print(f"  > Step '{step_name}' completed in {self.timing_metrics[step_name]:.4f} seconds.")
            
        return result

    # =========================================================================
    # Mesh Loading
    # =========================================================================
    
    def load_mesh(self) -> None:
        """Load mesh from file (.xlsx, .npz, .h5)."""
        if self.verbose:
            print(f"Loading mesh from {self.mesh_file.name}...")
        
        suffix = self.mesh_file.suffix.lower()
        
        if suffix == '.xlsx':
            import pandas as pd
            coord = pd.read_excel(self.mesh_file, sheet_name="coord")
            conec = pd.read_excel(self.mesh_file, sheet_name="conec")
            self.x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
            self.y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
            self.quad8 = conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1
            
        elif suffix == '.npz':
            data = np.load(self.mesh_file)
            self.x = data['x']
            self.y = data['y']
            self.quad8 = data['quad8']
            
        elif suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(self.mesh_file, 'r') as f:
                self.x = np.array(f['x'], dtype=np.float64)
                self.y = np.array(f['y'], dtype=np.float64)
                self.quad8 = np.array(f['quad8'], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported mesh format: {suffix}")
        
        self.Nnds = self.x.size
        self.Nels = self.quad8.shape[0]
        
        if self.verbose:
            print(f"  Loaded: {self.Nnds} nodes, {self.Nels} Quad-8 elements")

   
    # =========================================================================
    # Assembly (Numba CUDA)
    # =========================================================================

    def assemble_system(self) -> None:
        """Assemble global stiffness matrix using Numba CUDA kernel."""
        if self.verbose:
            print("Assembling global system (Numba CUDA)...")

        # === DIAGNOSTIC ===
        e = 0
        edofs = self.quad8[e]
        print(f"  Element 0 DOFs: {edofs}")
        print(f"  Element 0 coords:")
        for i in range(8):
            print(f"    Node {edofs[i]}: ({self.x[edofs[i]]:.6f}, {self.y[edofs[i]]:.6f})")
        print(f"  x range: [{self.x.min():.6f}, {self.x.max():.6f}]")
        print(f"  y range: [{self.y.min():.6f}, {self.y.max():.6f}]")
        # === END DIAGNOSTIC ===

        # Transfer data to GPU
        d_x = cuda.to_device(self.x)
        d_y = cuda.to_device(self.y)
        d_quad8 = cuda.to_device(self.quad8)
        
        # Get integration points
        xp, wp = get_gauss_points_9()
        d_xp = cuda.to_device(xp)
        d_wp = cuda.to_device(wp)
        
        # Allocate output arrays on GPU
        d_vals = cuda.device_array(self.Nels * 64, dtype=np.float64)
        
        # Initialize fg to zero
        d_fg_host = np.zeros(self.Nnds, dtype=np.float64)
        d_fg = cuda.to_device(d_fg_host)
        
        # Launch kernel
        threads_per_block = 128
        blocks = (self.Nels + threads_per_block - 1) // threads_per_block
        
        if self.verbose:
            print(f"  Launching kernel: {blocks} blocks x {threads_per_block} threads")
        
        quad8_assembly_kernel[blocks, threads_per_block](
            d_x, d_y, d_quad8, d_xp, d_wp, d_vals, d_fg
        )
        
        # Synchronize
        cuda.synchronize()

        # Copy results back to CPU
        vals = d_vals.copy_to_host()
        self.fg = d_fg.copy_to_host()
        
        print(f"  vals stats: min={vals.min():.3e}, max={vals.max():.3e}, sum={vals.sum():.3e}")
        
        # Build COO indices
        rows = np.zeros(self.Nels * 64, dtype=np.int32)
        cols = np.zeros(self.Nels * 64, dtype=np.int32)
        
        for e in range(self.Nels):
            edofs = self.quad8[e]
            base_idx = e * 64
            k = 0
            for i in range(8):
                for j in range(8):
                    rows[base_idx + k] = edofs[i]
                    cols[base_idx + k] = edofs[j]
                    k += 1
        
        # Build sparse matrix
        self.Kg = coo_matrix(
            (vals, (rows, cols)),
            shape=(self.Nnds, self.Nnds),
            dtype=np.float64
        ).tocsr()
        
        if self.verbose:
            print(f"  Assembled {self.Nels} elements")

    # =========================================================================
    # Boundary Conditions (CPU)
    # =========================================================================
    
    def apply_boundary_conditions(self) -> None:
        """Apply Robin (inlet) and Dirichlet (outlet) boundary conditions."""
        if self.verbose:
            print("Applying boundary conditions...")
        
        # Convert to LIL for efficient modification
        Kg_lil = lil_matrix(self.Kg)
        
        # Robin BC on inlet (minimum x)
        x_min = float(self.x.min())
        boundary_nodes = set(np.where(np.abs(self.x - x_min) < self.bc_tolerance)[0].tolist())
        
        robin_edges = []
        for e in range(self.Nels):
            n = self.quad8[e]
            edges = [
                (n[0], n[4], n[1]),
                (n[1], n[5], n[2]),
                (n[2], n[6], n[3]),
                (n[3], n[7], n[0]),
            ]
            for edge in edges:
                if all(k in boundary_nodes for k in edge):
                    robin_edges.append(edge)
        
        if self.verbose:
            print(f"  Applying {len(robin_edges)} Robin edges...")
        
        inlet_potential = 0.0
        
        for (n1, n2, n3) in robin_edges:
            He, Pe = self._robin_quadr(
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
                    Kg_lil[ed[i], ed[j]] += He[i, j]
        
        # Dirichlet BC on outlet (maximum x)
        exit_nodes = np.where(self.x == np.max(self.x))[0]
        
        if self.verbose:
            print(f"  Applying {len(exit_nodes)} Dirichlet nodes (u=0) via Penalty Method...")
        
        PENALTY_FACTOR = 1.0e12
        
        for n in exit_nodes:
            Kg_lil[n, n] += PENALTY_FACTOR
        
        # Fix unused nodes
        used_nodes_set = set(self.quad8.flatten().tolist())
        unused_count = 0
        for n in range(self.Nnds):
            if n not in used_nodes_set:
                Kg_lil[n, n] += PENALTY_FACTOR
                unused_count += 1
        
        if self.verbose and unused_count > 0:
            print(f"  Fixed {unused_count} unused nodes via Penalty Method")
        
        # Convert back to CSR
        self.Kg = Kg_lil.tocsr()
    
    def _robin_quadr(self, x1, y1, x2, y2, x3, y3, p, gama):
        """Robin boundary contribution (CPU implementation)."""
        He = np.zeros((3, 3), dtype=np.float64)
        Pe = np.zeros(3, dtype=np.float64)
        
        # 3-point 1D Gauss quadrature
        G = np.sqrt(0.6)
        xi = np.array([-G, 0.0, G])
        wi = np.array([5.0, 8.0, 5.0]) / 9.0
        
        for ip in range(3):
            csi = xi[ip]
            
            b = np.array([
                0.5 * csi * (csi - 1.0),
                1.0 - csi * csi,
                0.5 * csi * (csi + 1.0)
            ])
            
            db = np.array([csi - 0.5, -2.0 * csi, csi + 0.5])
            
            xx = db[0] * x1 + db[1] * x2 + db[2] * x3
            yy = db[0] * y1 + db[1] * y2 + db[2] * y3
            
            jaco = np.sqrt(xx * xx + yy * yy)
            
            wip = jaco * wi[ip]
            He += (wip * p) * np.outer(b, b)
            Pe += (wip * gama) * b
        
        return He, Pe

    # =========================================================================
    # Solver (GPU - CuPy)
    # =========================================================================
    
    def solve(self) -> NDArray[np.float64]:
        """Solve linear system using CuPy GPU-accelerated CG solver."""
        if self.verbose:
            print("Preparing GPU solver (CuPy)...")
        
        t0_convert = time.perf_counter()
        
        # Convert to CuPy sparse matrix on GPU
        Kg_gpu = cpsparse.csr_matrix(self.Kg)
        fg_gpu = cp.asarray(self.fg)
        
        diag_values = Kg_gpu.diagonal()
        if self.verbose:
            print(f"  Kg Diagonal Min: {float(diag_values.min()):.3e}")
            print(f"  Kg Diagonal Max: {float(diag_values.max()):.3e}")
        
        initial_residual_norm = float(cp.linalg.norm(fg_gpu))
        if self.verbose:
            print(f"  Initial L2 residual norm (b): {initial_residual_norm:.3e}")
        
        self.timing_metrics['convert_to_gpu'] = time.perf_counter() - t0_convert
        
        # Solver setup
        MAXITER = self.maxiter
        TOL = 1e-8
        
        self.converged = False
        self.iterations = MAXITER
        
        # Diagonal equilibration
        if self.verbose:
            print("Applying diagonal equilibration...")
        
        diag = Kg_gpu.diagonal()
        diag_safe = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        D_inv_sqrt = 1.0 / cp.sqrt(cp.abs(diag_safe))
        
        # Equilibrated system
        Kg_eq = Kg_gpu.multiply(D_inv_sqrt[:, None]).multiply(D_inv_sqrt[None, :])
        fg_eq = fg_gpu * D_inv_sqrt
        
        # Jacobi preconditioner
        diag_eq = Kg_eq.diagonal()
        diag_eq_safe = cp.where(cp.abs(diag_eq) < 1e-14, 1.0, diag_eq)
        
        def jacobi_precond(x):
            return x / diag_eq_safe
        
        M = cpsplg.LinearOperator(
            shape=Kg_eq.shape,
            matvec=jacobi_precond,
            dtype=cp.float64
        )
        
        if self.verbose:
            print(f"Solving with GPU CG (tol={TOL:.1e}, maxiter={MAXITER})...")
        
        # GPU Solver Monitor
        self.monitor = GPUSolverMonitor(
            Kg_eq, fg_eq,
            every=self.cg_print_every,
            maxiter=MAXITER,
            verbose=self.verbose,
            progress_callback=self.progress_callback
        )
        
        t0_solve = time.perf_counter()
        
        try:
            u_eq, info = cpsplg.cg(
                Kg_eq,
                fg_eq,
                x0=cp.zeros_like(fg_eq),
                M=M,
                tol=TOL,
                maxiter=MAXITER,
                callback=self.monitor
            )
            solver_name = "CG"
            
        except Exception as e:
            if self.verbose:
                print(f"  CG failed: {e}")
                print("  Falling back to GMRES...")
            
            self.monitor.reset()
            
            try:
                u_eq, info = cpsplg.gmres(
                    Kg_eq,
                    fg_eq,
                    x0=cp.zeros_like(fg_eq),
                    M=M,
                    tol=TOL,
                    maxiter=MAXITER,
                    restart=50,
                    callback=self.monitor
                )
                solver_name = "GMRES"
                
            except Exception as e2:
                if self.verbose:
                    print(f"  GMRES also failed: {e2}")
                raise RuntimeError("Both CG and GMRES failed") from e2
        
        t1_solve = time.perf_counter()
        
        # Undo equilibration
        u_gpu = u_eq * D_inv_sqrt
        
        self.timing_metrics['solve_system'] = t1_solve - t0_solve
        self.iterations = self.monitor.it
        
        # Convergence check
        if info == 0:
            residual = cp.linalg.norm(Kg_gpu @ u_gpu - fg_gpu)
            rel_residual = residual / cp.linalg.norm(fg_gpu)
            
            if self.verbose:
                print(f"\n✓ {solver_name} converged in {self.iterations} iterations")
                print(f"  True residual norm:     {float(residual):.3e}")
                print(f"  True relative residual: {float(rel_residual):.3e}")
                print(f"  Solver time: {t1_solve - t0_solve:.4f}s")
            
            self.converged = True
        else:
            if self.verbose:
                print(f"\n✗ {solver_name} did not converge (info={info})")
            self.converged = False
        
        # Keep solution on GPU for post-processing, also store CPU copy
        self.u_gpu = u_gpu
        self.u = u_gpu.get()
        
        return self.u

    # =========================================================================
    # Post-Processing (Numba CUDA)
    # =========================================================================
    
    def compute_derived_fields(self) -> None:
        """Compute velocity and pressure fields using Numba CUDA kernel."""
        if self.verbose:
            print("Computing velocity field (Numba CUDA)...")
        
        # Transfer data to GPU
        d_u = cuda.to_device(self.u)
        d_x = cuda.to_device(self.x)
        d_y = cuda.to_device(self.y)
        d_quad8 = cuda.to_device(self.quad8)
        
        # Get integration points
        xp = get_gauss_points_4()
        d_xp = cuda.to_device(xp)
        
        # Allocate output arrays on GPU
        d_vel = cuda.device_array((self.Nels, 2), dtype=np.float64)
        d_abs_vel = cuda.device_array(self.Nels, dtype=np.float64)
        
        # Launch kernel
        threads_per_block = 128
        blocks = (self.Nels + threads_per_block - 1) // threads_per_block
        
        quad8_postprocess_kernel[blocks, threads_per_block](
            d_u, d_x, d_y, d_quad8, d_xp, d_vel, d_abs_vel
        )
        
        # Synchronize
        cuda.synchronize()
        
        # Copy results back to CPU
        self.vel = d_vel.copy_to_host()
        self.abs_vel = d_abs_vel.copy_to_host()
        
        # Compute pressure from Bernoulli equation
        self.pressure = self.p0 - self.rho * self.abs_vel**2

    # =========================================================================
    # Main Workflow
    # =========================================================================
    
    def run(
        self,
        output_dir: Optional[Path | str] = None,
        export_file: Optional[Path | str] = None
    ) -> Dict[str, Any]:
        """Run complete FEM simulation workflow."""
        
        total_workflow_start = time.perf_counter()
        self.timing_metrics = {}
        
        # Stage 1: Load Mesh
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='load_mesh')
        
        self._time_step('load_mesh', self.load_mesh)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='load_mesh',
                duration=self.timing_metrics['load_mesh']
            )
            self.progress_callback.on_mesh_loaded(
                nodes=self.Nnds,
                elements=self.Nels,
                coordinates={'x': self.x.tolist(), 'y': self.y.tolist()},
                connectivity=self.quad8.tolist()
            )

        used_nodes = set(self.quad8.flatten())
        print(f"  Used nodes: {len(used_nodes)} out of {self.Nnds}")
        print(f"  Node index range in quad8: [{self.quad8.min()}, {self.quad8.max()}]")            
        
        # Stage 2: Assembly
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='assemble_system')
        
        self._time_step('assemble_system', self.assemble_system)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='assemble_system',
                duration=self.timing_metrics['assemble_system']
            )

        # Check for zero diagonals in assembled matrix
        diag = self.Kg.diagonal()
        zero_diag_count = np.sum(diag == 0.0)
        print(f"  Zero diagonals after assembly: {zero_diag_count}")
        print(f"  Diagonal range: [{diag.min():.3e}, {diag.max():.3e}]")            
        
        # Stage 3: Boundary Conditions
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='apply_bc')
        
        self._time_step('apply_bc', self.apply_boundary_conditions)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='apply_bc',
                duration=self.timing_metrics['apply_bc']
            )

        # Stage 4: Solve
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='solve_system')
        
        self._time_step('solve_system', self.solve)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='solve_system',
                duration=self.timing_metrics['solve_system']
            )
        
        # Stage 5: Post-Processing
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='compute_derived')
        
        self._time_step('compute_derived', self.compute_derived_fields)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='compute_derived',
                duration=self.timing_metrics['compute_derived']
            )
        
        # Final timing
        total_workflow_end = time.perf_counter()
        self.timing_metrics['total_workflow'] = total_workflow_end - total_workflow_start
        self.timing_metrics['total_program_time'] = total_workflow_end - self.program_start_time
        
        # Results dictionary
        results: Dict[str, Any] = {
            'u': self.u,
            'vel': self.vel,
            'abs_vel': self.abs_vel,
            'pressure': self.pressure,
            'converged': self.converged,
            'iterations': self.iterations,
            'timing_metrics': self.timing_metrics,
            'solution_stats': {
                'u_range': [float(self.u.min()), float(self.u.max())],
                'u_mean': float(self.u.mean()),
                'u_std': float(self.u.std())
            },
            'mesh_info': {
                'nodes': self.Nnds,
                'elements': self.Nels
            }
        }
        
        if self.verbose:
            print("\n✓ Numba CUDA simulation complete")
            print("\nStep-by-Step Timings (seconds):")
            for k, v in self.timing_metrics.items():
                print(f"  {k:<20}: {v:.4f}")
            print(f"\nResults summary:")
            print(f"  Converged: {results['converged']}")
            print(f"  Iterations: {results['iterations']}")
            print(f"  u range: [{results['solution_stats']['u_range'][0]:.6e}, "
                  f"{results['solution_stats']['u_range'][1]:.6e}]")
        
        return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Numba CUDA FEM Solver")
    print("=" * 50)
    
    # Check CUDA availability
    if not cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)
    
    print(f"CUDA Device: {cuda.get_current_device().name}")
    
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent.parent
    
    solver = Quad8FEMSolverNumbaCUDA(
        mesh_file=PROJECT_ROOT / "src/app/client/mesh/s_duct.h5",
        implementation_name="NumbaCUDA",
        maxiter=15000,
        cg_print_every=50
    )
    
    results = solver.run()
    
    print(f"\nFinal Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Total Time: {results['timing_metrics']['total_program_time']:.4f}s")
