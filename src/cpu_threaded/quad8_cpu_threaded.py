"""
QUAD8 FEM Solver - CPU Threaded version using ThreadPoolExecutor.

Standalone solver class for 2D potential flow using Quad-8 elements.
Uses concurrent.futures.ThreadPoolExecutor for parallel assembly and post-processing.

Demonstrates explicit Python threading without JIT compilation.
Note: Due to Python's GIL, threading benefits are limited to NumPy operations
that release the GIL internally.

Functionally equivalent to CPU solver with identical interface.
"""

import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from numpy.typing import NDArray

from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator


# =============================================================================
# Element Computation Functions (Pure NumPy)
# =============================================================================

def gauss_points_9():
    """Return 3x3 Gauss-Legendre points and weights."""
    G = np.sqrt(0.6)
    xp = np.array([
        [-G, -G], [0.0, -G], [G, -G],
        [-G, 0.0], [0.0, 0.0], [G, 0.0],
        [-G, G], [0.0, G], [G, G]
    ], dtype=np.float64)
    
    wp = np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float64) / 81.0
    
    return xp, wp


def gauss_points_4():
    """Return 2x2 Gauss-Legendre points."""
    G = np.sqrt(1.0 / 3.0)
    xp = np.array([
        [-G, -G], [G, -G], [G, G], [-G, G]
    ], dtype=np.float64)
    
    return xp


def compute_element_stiffness(XN: np.ndarray, xp: np.ndarray, wp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element stiffness matrix for a single Quad-8 element.
    
    Parameters
    ----------
    XN : (8, 2) ndarray
        Nodal coordinates
    xp : (9, 2) ndarray
        Integration points
    wp : (9,) ndarray
        Integration weights
    
    Returns
    -------
    Ke : (8, 8) ndarray
        Element stiffness matrix
    fe : (8,) ndarray
        Element force vector
    """
    Ke = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)
    
    for ip in range(9):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        w = wp[ip]
        
        # Shape function derivatives (parametric)
        Dpsi = np.zeros((8, 2), dtype=np.float64)
        
        Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
        Dpsi[1, 0] = (2 * csi - eta) * (1 - eta) / 4
        Dpsi[2, 0] = (2 * csi + eta) * (1 + eta) / 4
        Dpsi[3, 0] = (2 * csi - eta) * (1 + eta) / 4
        Dpsi[4, 0] = csi * (eta - 1)
        Dpsi[5, 0] = (1 - eta * eta) / 2
        Dpsi[6, 0] = -csi * (1 + eta)
        Dpsi[7, 0] = (eta * eta - 1) / 2
        
        Dpsi[0, 1] = (2 * eta + csi) * (1 - csi) / 4
        Dpsi[1, 1] = (2 * eta - csi) * (1 + csi) / 4
        Dpsi[2, 1] = (2 * eta + csi) * (1 + csi) / 4
        Dpsi[3, 1] = (2 * eta - csi) * (1 - csi) / 4
        Dpsi[4, 1] = (csi * csi - 1) / 2
        Dpsi[5, 1] = -(1 + csi) * eta
        Dpsi[6, 1] = (1 - csi * csi) / 2
        Dpsi[7, 1] = (csi - 1) * eta
        
        # Jacobian matrix
        jaco = XN.T @ Dpsi  # (2, 2)
        Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
        
        if Detj <= 1.0e-12:
            continue
        
        # Inverse Jacobian
        inv_det = 1.0 / Detj
        Invj = np.array([
            [jaco[1, 1] * inv_det, -jaco[0, 1] * inv_det],
            [-jaco[1, 0] * inv_det, jaco[0, 0] * inv_det]
        ], dtype=np.float64)
        
        # B matrix (derivatives w.r.t. physical coordinates)
        B = Dpsi @ Invj  # (8, 2)
        
        # Accumulate Ke
        wip = w * Detj
        Ke += wip * (B @ B.T)
    
    return Ke, fe


def compute_element_velocity(XN: np.ndarray, u_e: np.ndarray, xp: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute velocity at element centroid.
    
    Parameters
    ----------
    XN : (8, 2) ndarray
        Nodal coordinates
    u_e : (8,) ndarray
        Element solution values
    xp : (4, 2) ndarray
        Integration points
    
    Returns
    -------
    vel_x, vel_y, abs_vel : float
        Velocity components and magnitude
    """
    vel_x_sum = 0.0
    vel_y_sum = 0.0
    v_mag_sum = 0.0
    
    for ip in range(4):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        
        # Shape function derivatives (parametric)
        Dpsi = np.zeros((8, 2), dtype=np.float64)
        
        Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
        Dpsi[1, 0] = (2 * csi - eta) * (1 - eta) / 4
        Dpsi[2, 0] = (2 * csi + eta) * (1 + eta) / 4
        Dpsi[3, 0] = (2 * csi - eta) * (1 + eta) / 4
        Dpsi[4, 0] = csi * (eta - 1)
        Dpsi[5, 0] = (1 - eta * eta) / 2
        Dpsi[6, 0] = -csi * (1 + eta)
        Dpsi[7, 0] = (eta * eta - 1) / 2
        
        Dpsi[0, 1] = (2 * eta + csi) * (1 - csi) / 4
        Dpsi[1, 1] = (2 * eta - csi) * (1 + csi) / 4
        Dpsi[2, 1] = (2 * eta + csi) * (1 + csi) / 4
        Dpsi[3, 1] = (2 * eta - csi) * (1 - csi) / 4
        Dpsi[4, 1] = (csi * csi - 1) / 2
        Dpsi[5, 1] = -(1 + csi) * eta
        Dpsi[6, 1] = (1 - csi * csi) / 2
        Dpsi[7, 1] = (csi - 1) * eta
        
        # Jacobian matrix
        jaco = XN.T @ Dpsi
        Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
        inv_det = 1.0 / Detj
        
        Invj = np.array([
            [jaco[1, 1] * inv_det, -jaco[0, 1] * inv_det],
            [-jaco[1, 0] * inv_det, jaco[0, 0] * inv_det]
        ], dtype=np.float64)
        
        # B matrix
        B = Dpsi @ Invj
        
        # Gradient
        grad = B.T @ u_e  # (2,)
        
        # Velocity is negative gradient
        vel_x_sum += -grad[0]
        vel_y_sum += -grad[1]
        v_mag_sum += np.sqrt(grad[0]**2 + grad[1]**2)
    
    return vel_x_sum / 4.0, vel_y_sum / 4.0, v_mag_sum / 4.0


# =============================================================================
# Batch Processing Functions for Threading
# =============================================================================

def process_element_batch_assembly(args):
    """
    Process a batch of elements for assembly.
    
    Parameters
    ----------
    args : tuple
        (start_idx, end_idx, x, y, quad8, xp, wp)
    
    Returns
    -------
    rows, cols, vals, fe_batch : arrays
        COO data and force vectors for this batch
    """
    start_idx, end_idx, x, y, quad8, xp, wp = args
    
    batch_size = end_idx - start_idx
    
    # Pre-allocate for this batch
    rows = np.zeros(batch_size * 64, dtype=np.int32)
    cols = np.zeros(batch_size * 64, dtype=np.int32)
    vals = np.zeros(batch_size * 64, dtype=np.float64)
    fe_batch = np.zeros((batch_size, 8), dtype=np.float64)
    
    for local_e, e in enumerate(range(start_idx, end_idx)):
        edofs = quad8[e]
        
        # Build element coordinates
        XN = np.column_stack([x[edofs], y[edofs]])
        
        # Compute element matrices
        Ke, fe = compute_element_stiffness(XN, xp, wp)
        
        # Store force vector
        fe_batch[local_e] = fe
        
        # Store COO entries
        base_idx = local_e * 64
        k = 0
        for i in range(8):
            for j in range(8):
                rows[base_idx + k] = edofs[i]
                cols[base_idx + k] = edofs[j]
                vals[base_idx + k] = Ke[i, j]
                k += 1
    
    return start_idx, rows, cols, vals, fe_batch


def process_element_batch_velocity(args):
    """
    Process a batch of elements for velocity computation.
    
    Parameters
    ----------
    args : tuple
        (start_idx, end_idx, x, y, quad8, u, xp)
    
    Returns
    -------
    start_idx, vel_batch, abs_vel_batch : arrays
        Velocity data for this batch
    """
    start_idx, end_idx, x, y, quad8, u, xp = args
    
    batch_size = end_idx - start_idx
    
    vel_batch = np.zeros((batch_size, 2), dtype=np.float64)
    abs_vel_batch = np.zeros(batch_size, dtype=np.float64)
    
    for local_e, e in enumerate(range(start_idx, end_idx)):
        edofs = quad8[e]
        
        XN = np.column_stack([x[edofs], y[edofs]])
        u_e = u[edofs]
        
        vx, vy, vmag = compute_element_velocity(XN, u_e, xp)
        
        vel_batch[local_e, 0] = vx
        vel_batch[local_e, 1] = vy
        abs_vel_batch[local_e] = vmag
    
    return start_idx, vel_batch, abs_vel_batch


# =============================================================================
# Progress Callback Monitor
# =============================================================================

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

class Quad8FEMSolverThreaded:
    """
    QUAD8 FEM Solver - CPU Threaded version.
    
    Features:
    - Quad-8 (8-node quadrilateral) elements
    - Robin boundary conditions (inlet)
    - Dirichlet boundary conditions (outlet)
    - ThreadPoolExecutor for parallel assembly and post-processing
    - SciPy CG solver for linear system
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
        num_workers: int = None,
        batch_size: int = 1000,
        implementation_name: str = "CPUThreaded",
        verbose: bool = True,
        progress_callback=None
    ):
        """
        Initialize FEM solver.
        
        Args:
            mesh_file: Path to mesh file (.h5, .npz, .xlsx)
            p0: Reference pressure (Pa)
            rho: Fluid density (kg/m^3)
            gamma: Specific heat ratio
            rtol: CG solver relative tolerance
            atol: CG solver absolute tolerance
            maxiter: Maximum CG iterations
            bc_tolerance: Tolerance for detecting boundary nodes
            cg_print_every: Print CG progress every N iterations
            num_workers: Number of threads (default: CPU count)
            batch_size: Elements per batch for threading
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
        
        # Threading parameters
        self.num_workers = num_workers or os.cpu_count()
        self.batch_size = batch_size
        
        # Configuration
        self.implementation_name = implementation_name
        self.verbose = verbose
        
        # Mesh data
        self.mesh_file = Path(mesh_file)
        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64]
        self.quad8: NDArray[np.int32]
        self.Nnds: int
        self.Nels: int
        
        # System matrices
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
        self.monitor: IterativeSolverMonitor
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
    # Assembly (ThreadPoolExecutor)
    # =========================================================================
    
    def assemble_system(self) -> None:
        """Assemble global stiffness matrix using ThreadPoolExecutor."""
        if self.verbose:
            print(f"Assembling global system (ThreadPoolExecutor, {self.num_workers} workers)...")
        
        # Get integration points
        xp, wp = gauss_points_9()
        
        # Create batches
        batches = []
        for start in range(0, self.Nels, self.batch_size):
            end = min(start + self.batch_size, self.Nels)
            batches.append((start, end, self.x, self.y, self.quad8, xp, wp))
        
        if self.verbose:
            print(f"  Processing {len(batches)} batches of up to {self.batch_size} elements...")
        
        # Process batches in parallel
        all_rows = []
        all_cols = []
        all_vals = []
        fe_all = np.zeros((self.Nels, 8), dtype=np.float64)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_element_batch_assembly, batch) for batch in batches]
            
            for future in as_completed(futures):
                start_idx, rows, cols, vals, fe_batch = future.result()
                all_rows.append(rows)
                all_cols.append(cols)
                all_vals.append(vals)
                
                # Store force vectors
                batch_size = fe_batch.shape[0]
                fe_all[start_idx:start_idx + batch_size] = fe_batch
        
        # Combine COO data
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_vals)
        
        # Build sparse matrix
        self.Kg = coo_matrix(
            (vals, (rows, cols)),
            shape=(self.Nnds, self.Nnds),
            dtype=np.float64
        ).tocsr()
        
        # Accumulate force vector
        self.fg = np.zeros(self.Nnds, dtype=np.float64)
        for e in range(self.Nels):
            edofs = self.quad8[e]
            for i in range(8):
                self.fg[edofs[i]] += fe_all[e, i]
        
        if self.verbose:
            print(f"  Assembled {self.Nels} elements")

    # =========================================================================
    # Boundary Conditions
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
        """Robin boundary contribution."""
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
    # Solver
    # =========================================================================
    
    def solve(self) -> NDArray[np.float64]:
        """Solve linear system using CG with diagonal equilibration."""
        solve_start_time = time.perf_counter()

        if self.verbose:
            print("Preparing solver...")
        
        diag_values = self.Kg.diagonal()
        if self.verbose:
            print(f"  Kg Diagonal Min: {diag_values.min():.3e}")
            print(f"  Kg Diagonal Max: {diag_values.max():.3e}")
        
        initial_residual_norm = np.linalg.norm(self.fg)
        if self.verbose:
            print(f"  Initial L2 residual norm (b): {initial_residual_norm:.3e}")
        
        # Diagonal equilibration
        if self.verbose:
            print("Applying diagonal equilibration...")
        
        diag = self.Kg.diagonal()
        D_inv_sqrt = 1.0 / np.sqrt(np.abs(diag))
        
        from scipy.sparse import diags
        D_mat = diags(D_inv_sqrt)
        Kg_eq = D_mat @ self.Kg @ D_mat
        fg_eq = self.fg * D_inv_sqrt
        
        # Jacobi preconditioner
        diag_eq = Kg_eq.diagonal()
        M_inv = 1.0 / diag_eq
        
        def precond_jacobi(v):
            return M_inv * v
        
        M = LinearOperator(Kg_eq.shape, precond_jacobi)
        
        if self.verbose:
            print(f"Solving with CG (tol=1e-8, maxiter={self.maxiter})...")
        
        TARGET_TOL = 1e-8
        
        self.monitor = IterativeSolverMonitor(
            Kg_eq, fg_eq,
            every=self.cg_print_every,
            maxiter=self.maxiter,
            progress_callback=self.progress_callback
        )
        
        u_eq, self.solve_info = cg(
            Kg_eq, fg_eq,
            rtol=TARGET_TOL,
            atol=0.0,
            maxiter=self.maxiter,
            M=M,
            callback=self.monitor
        )
        
        # De-equilibrate
        self.u = u_eq * D_inv_sqrt
        
        # True residual check
        r = self.fg - self.Kg @ self.u
        true_residual = np.linalg.norm(r)
        true_relative_residual = true_residual / initial_residual_norm

        # Store for results
        self.final_residual = float(true_residual)
        self.relative_residual = float(true_relative_residual)        

        solve_end_time = time.perf_counter()
        total_solve_time = solve_end_time - solve_start_time
        self.timing_metrics['solve_system'] = total_solve_time

        self.converged = (self.solve_info == 0)
        self.iterations = self.monitor.it

        if self.verbose:
            if self.converged:
                print(f"✓ CG converged in {self.iterations} iterations")
            else:
                print(f"✗ CG did not converge (max iterations reached)")
            
            print(f"  True residual norm:     {true_residual:.3e}")
            print(f"  True relative residual: {true_relative_residual:.3e}")
            print(f"  Total solver wall time: {total_solve_time:.4f} seconds")
        
        return self.u

    # =========================================================================
    # Post-Processing (ThreadPoolExecutor)
    # =========================================================================
    
    def compute_derived_fields(self) -> None:
        """Compute velocity and pressure fields using ThreadPoolExecutor."""
        if self.verbose:
            print(f"Computing velocity field (ThreadPoolExecutor, {self.num_workers} workers)...")
        
        # Get integration points
        xp = gauss_points_4()
        
        # Create batches
        batches = []
        for start in range(0, self.Nels, self.batch_size):
            end = min(start + self.batch_size, self.Nels)
            batches.append((start, end, self.x, self.y, self.quad8, self.u, xp))
        
        # Process batches in parallel
        self.vel = np.zeros((self.Nels, 2), dtype=np.float64)
        self.abs_vel = np.zeros(self.Nels, dtype=np.float64)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_element_batch_velocity, batch) for batch in batches]
            
            for future in as_completed(futures):
                start_idx, vel_batch, abs_vel_batch = future.result()
                batch_size = vel_batch.shape[0]
                self.vel[start_idx:start_idx + batch_size] = vel_batch
                self.abs_vel[start_idx:start_idx + batch_size] = abs_vel_batch
        
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
        
        # Stage 2: Assembly
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='assemble_system')
        
        self._time_step('assemble_system', self.assemble_system)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='assemble_system',
                duration=self.timing_metrics['assemble_system']
            )
        
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
                'u_std': float(self.u.std()),
                'final_residual': self.final_residual,
                'relative_residual': self.relative_residual,
            },
            'mesh_info': {
                'nodes': self.Nnds,
                'elements': self.Nels,
                'matrix_nnz': self.Kg.nnz,
                'element_type': 'quad8',
                'nodes_per_element': 8,
            },
            'solver_config': {
                'linear_solver': 'cg',
                'tolerance': 1e-8,
                'max_iterations': self.maxiter,
                'preconditioner': 'jacobi',
            },
        }
        
        if self.verbose:
            print(f"\n✓ CPU Threaded simulation complete ({self.num_workers} workers)")
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
    print("CPU Threaded FEM Solver (ThreadPoolExecutor)")
    print("=" * 50)
    
    import os
    print(f"CPU Count: {os.cpu_count()}")
    
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent.parent
    
    solver = Quad8FEMSolverThreaded(
        mesh_file=PROJECT_ROOT / "data/input/exported_mesh_v6.h5",
        implementation_name="CPUThreaded",
        maxiter=15000,
        cg_print_every=50,
        num_workers=None,  # Use all CPUs
        batch_size=1000
    )
    
    results = solver.run()
    
    print(f"\nFinal Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Total Time: {results['timing_metrics']['total_program_time']:.4f}s")
