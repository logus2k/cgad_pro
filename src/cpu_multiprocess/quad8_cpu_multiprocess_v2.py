"""
QUAD8 FEM Solver - CPU Multiprocess version using multiprocessing.Pool.

Standalone solver class for 2D potential flow using Quad-8 elements.
Uses multiprocessing.Pool for parallel assembly and post-processing.

Demonstrates explicit Python multiprocessing - true parallelism bypassing GIL.

Functionally equivalent to CPU solver with identical interface.

Note: Uses Pool initializer pattern to broadcast large arrays to workers once,
avoiding Windows IPC bottlenecks for large meshes.
"""

import sys
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from numpy.typing import NDArray

from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator


# =============================================================================
# Worker State for Pool Initializer Pattern
# =============================================================================
# This dict holds shared data in each worker process, avoiding repeated 
# serialization of large arrays through IPC pipes.

_WORKER_STATE = {}


def _init_assembly_worker(x, y, quad8, xp, wp):
    """
    Initialize worker process with mesh data for assembly.
    Called once per worker when the Pool is created.
    """
    _WORKER_STATE['x'] = x
    _WORKER_STATE['y'] = y
    _WORKER_STATE['quad8'] = quad8
    _WORKER_STATE['xp'] = xp
    _WORKER_STATE['wp'] = wp


def _init_velocity_worker(x, y, quad8, u, xp):
    """
    Initialize worker process with mesh and solution data for velocity computation.
    Called once per worker when the Pool is created.
    """
    _WORKER_STATE['x'] = x
    _WORKER_STATE['y'] = y
    _WORKER_STATE['quad8'] = quad8
    _WORKER_STATE['u'] = u
    _WORKER_STATE['xp'] = xp


# =============================================================================
# Element Computation Functions (Pure NumPy) - Must be at module level for pickle
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
# Batch Processing Functions for Multiprocessing (must be at module level)
# =============================================================================

def _process_assembly_batch(args):
    """
    Process a batch of elements for assembly.
    Uses worker state initialized via Pool initializer.
    
    Parameters
    ----------
    args : tuple
        (start_idx, end_idx)
    
    Returns
    -------
    start_idx, rows, cols, vals, fe_batch : arrays
        COO data and force vectors for this batch
    """
    start_idx, end_idx = args
    
    # Get data from worker state (initialized once at pool creation)
    x = _WORKER_STATE['x']
    y = _WORKER_STATE['y']
    quad8 = _WORKER_STATE['quad8']
    xp = _WORKER_STATE['xp']
    wp = _WORKER_STATE['wp']
    
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


def _process_velocity_batch(args):
    """
    Process a batch of elements for velocity computation.
    Uses worker state initialized via Pool initializer.
    
    Parameters
    ----------
    args : tuple
        (start_idx, end_idx)
    
    Returns
    -------
    start_idx, vel_batch, abs_vel_batch : arrays
        Velocity data for this batch
    """
    start_idx, end_idx = args
    
    # Get data from worker state (initialized once at pool creation)
    x = _WORKER_STATE['x']
    y = _WORKER_STATE['y']
    quad8 = _WORKER_STATE['quad8']
    u = _WORKER_STATE['u']
    xp = _WORKER_STATE['xp']
    
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
# Legacy Batch Functions (backward compatibility)
# =============================================================================
# These are kept for any external code that may import them directly.
# They use the old signature with all data passed in args.

def process_element_batch_assembly(args):
    """
    Legacy: Process a batch of elements for assembly.
    Kept for backward compatibility - prefer using Pool initializer pattern.
    
    Parameters
    ----------
    args : tuple
        (start_idx, end_idx, x, y, quad8, xp, wp)
    
    Returns
    -------
    start_idx, rows, cols, vals, fe_batch : arrays
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
    Legacy: Process a batch of elements for velocity computation.
    Kept for backward compatibility - prefer using Pool initializer pattern.
    
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
            return f"{m:02d}:{s:02d}"

        print(
            f"    CG iter={self.it:>5} | "
            f"res={res_norm:.3e} | "
            f"rel={rel_res:.3e} | "
            f"elapsed={fmt(elapsed)} | "
            f"ETR≈{fmt(etr)}"
        )
        
        # Notify progress callback
        if self.progress_callback:
            self.progress_callback.on_solver_iteration(
                iteration=self.it,
                residual_norm=res_norm,
                relative_residual=rel_res,
                elapsed_time=elapsed,
                estimated_time_remaining=etr
            )


# =============================================================================
# Main Solver Class
# =============================================================================

class Quad8FEMSolverMultiprocess:
    """
    QUAD8 FEM Solver - CPU Multiprocess version.
    
    Features:
    - Quad-8 (8-node quadrilateral) elements
    - Robin boundary conditions (inlet)
    - Dirichlet boundary conditions (outlet)
    - multiprocessing.Pool for parallel assembly and post-processing
    - SciPy CG solver for linear system
    
    Note: Uses Pool initializer pattern to broadcast large arrays to workers
    once, avoiding Windows IPC bottlenecks for large meshes.
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
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
        implementation_name: str = "CPUMultiprocess",
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
            num_workers: Number of processes (default: CPU count)
            batch_size: Elements per batch for multiprocessing
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
        
        # Multiprocessing parameters
        self.num_workers = num_workers or cpu_count()
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
        
        # System matrices (csr_matrix or csr_array depending on scipy version)
        self.Kg: Any = None
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
    # Assembly (multiprocessing.Pool with initializer)
    # =========================================================================
    
    def assemble_system(self) -> None:
        """Assemble global stiffness matrix using multiprocessing.Pool."""
        if self.verbose:
            print(f"Assembling global system (multiprocessing.Pool, {self.num_workers} workers)...")
        
        # Get integration points
        xp, wp = gauss_points_9()
        
        # Create batch index pairs (only indices, not data!)
        batches = []
        for start in range(0, self.Nels, self.batch_size):
            end = min(start + self.batch_size, self.Nels)
            batches.append((start, end))
        
        if self.verbose:
            print(f"  Processing {len(batches)} batches of up to {self.batch_size} elements...")
        
        # Process batches in parallel using Pool with initializer
        # Data is broadcast once to workers via initializer, not per-batch
        all_rows = []
        all_cols = []
        all_vals = []
        fe_all = np.zeros((self.Nels, 8), dtype=np.float64)
        
        with Pool(
            processes=self.num_workers,
            initializer=_init_assembly_worker,
            initargs=(self.x, self.y, self.quad8, xp, wp)
        ) as pool:
            results = pool.map(_process_assembly_batch, batches)
        
        # Collect results
        for start_idx, rows, cols, vals, fe_batch in results:
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
                if edge[0] in boundary_nodes and edge[1] in boundary_nodes and edge[2] in boundary_nodes:
                    robin_edges.append(edge)
        
        if self.verbose:
            print(f"  Robin BC: {len(robin_edges)} edges on inlet (gamma={self.gamma})")
        
        # Add Robin BC contribution
        for edge in robin_edges:
            edge_x = self.x[list(edge)]
            edge_y = self.y[list(edge)]
            
            Le = np.sqrt((edge_x[2] - edge_x[0])**2 + (edge_y[2] - edge_y[0])**2)
            
            Me_1d = Le / 30 * np.array([
                [4, 2, -1],
                [2, 16, 2],
                [-1, 2, 4]
            ], dtype=np.float64)
            
            for i_local, i_global in enumerate(edge):
                for j_local, j_global in enumerate(edge):
                    Kg_lil[i_global, j_global] += self.gamma * Me_1d[i_local, j_local]
        
        # Dirichlet BC on outlet (maximum x)
        x_max = float(self.x.max())
        outlet_nodes = np.where(np.abs(self.x - x_max) < self.bc_tolerance)[0]
        
        if self.verbose:
            print(f"  Dirichlet BC: {len(outlet_nodes)} nodes on outlet (u=0)")
        
        # Apply Dirichlet by zeroing rows/cols
        for nd in outlet_nodes:
            Kg_lil[nd, :] = 0
            Kg_lil[:, nd] = 0
            Kg_lil[nd, nd] = 1.0
            self.fg[nd] = 0.0
        
        # Convert back to CSR
        self.Kg = Kg_lil.tocsr()

    # =========================================================================
    # Solver (CG with Jacobi preconditioner)
    # =========================================================================
    
    def solve(self) -> NDArray[np.float64]:
        """Solve the linear system using conjugate gradient with monitoring."""
        if self.verbose:
            print("Solving linear system (CG with Jacobi preconditioner)...")
        
        solve_start_time = time.perf_counter()

        # Diagonal equilibration
        diag = np.array(self.Kg.diagonal(), dtype=np.float64)
        diag[diag == 0] = 1.0
        D_inv_sqrt = 1.0 / np.sqrt(np.abs(diag))
        
        # Scale system
        D_sp = csr_matrix((D_inv_sqrt, (np.arange(self.Nnds), np.arange(self.Nnds))), 
                          shape=(self.Nnds, self.Nnds))
        A_eq = D_sp @ self.Kg @ D_sp
        b_eq = D_inv_sqrt * self.fg
        
        # Initial guess from force vector (scaled)
        x0 = D_inv_sqrt * self.fg
        x0 = x0 / (np.linalg.norm(x0) + 1e-16)
        
        # Jacobi preconditioner
        diag_eq = np.array(A_eq.diagonal(), dtype=np.float64)
        diag_eq[diag_eq == 0] = 1.0
        M_inv = 1.0 / diag_eq
        
        def jacobi_precond(x):
            return M_inv * x
        
        M = LinearOperator(shape=(self.Nnds, self.Nnds), matvec=jacobi_precond)  # type: ignore[call-arg]

        # Initial residual
        initial_residual_norm = np.linalg.norm(b_eq - A_eq @ x0)

        # Setup monitor
        self.monitor = IterativeSolverMonitor(
            A=A_eq,
            b=b_eq,
            every=self.cg_print_every,
            maxiter=self.maxiter,
            progress_callback=self.progress_callback
        )
        
        # Solve
        u_eq, self.solve_info = cg(
            A_eq, b_eq,
            x0=x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=M,
            callback=self.monitor
        )
        
        # De-equilibrate
        self.u = u_eq * D_inv_sqrt
        
        # True residual check
        r = self.fg - self.Kg @ self.u
        true_residual = np.linalg.norm(r)
        if initial_residual_norm > 1e-16:
            true_relative_residual = true_residual / initial_residual_norm
        else:
            true_relative_residual = 0.0 if true_residual < 1e-16 else float('inf')

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
    # Post-Processing (multiprocessing.Pool with initializer)
    # =========================================================================
    
    def compute_derived_fields(self) -> None:
        """Compute velocity and pressure fields using multiprocessing.Pool."""
        if self.verbose:
            print(f"Computing velocity field (multiprocessing.Pool, {self.num_workers} workers)...")
        
        # Get integration points
        xp = gauss_points_4()
        
        # Create batch index pairs (only indices, not data!)
        batches = []
        for start in range(0, self.Nels, self.batch_size):
            end = min(start + self.batch_size, self.Nels)
            batches.append((start, end))
        
        # Process batches in parallel using Pool with initializer
        self.vel = np.zeros((self.Nels, 2), dtype=np.float64)
        self.abs_vel = np.zeros(self.Nels, dtype=np.float64)
        
        with Pool(
            processes=self.num_workers,
            initializer=_init_velocity_worker,
            initargs=(self.x, self.y, self.quad8, self.u, xp)
        ) as pool:
            results = pool.map(_process_velocity_batch, batches)
        
        # Collect results
        for start_idx, vel_batch, abs_vel_batch in results:
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
            print(f"\n✓ CPU Multiprocess simulation complete ({self.num_workers} workers)")
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
    print("CPU Multiprocess FEM Solver (multiprocessing.Pool)")
    print("=" * 50)
    
    print(f"CPU Count: {cpu_count()}")
    
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent.parent
    
    solver = Quad8FEMSolverMultiprocess(
        mesh_file=PROJECT_ROOT / "data/input/exported_mesh_v6.h5",
        implementation_name="CPUMultiprocess",
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
