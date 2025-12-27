"""
QUAD8 FEM Solver - Numba JIT optimized version.

Standalone solver class for 2D potential flow using Quad-8 elements.
Uses Numba @njit for element-level computations and parallel assembly.

Functionally equivalent to CPU solver with identical interface.
"""

import sys
import time
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Callable
from numpy.typing import NDArray

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg, LinearOperator

from numba import njit, prange


# =============================================================================
# Parallel Assembly Kernel (Numba)
# =============================================================================

@njit(cache=True)
def robin_quadr_numba(x1, y1, x2, y2, x3, y3, p, gama):
    """
    Robin boundary contribution for a quadratic edge (3 nodes).
    Numba-optimized version.
    """
    He = np.zeros((3, 3), dtype=np.float64)
    Pe = np.zeros(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    
    # 3-point 1D Gauss quadrature
    G = np.sqrt(0.6)
    xi = np.array([-G, 0.0, G], dtype=np.float64)
    wi = np.array([5.0, 8.0, 5.0], dtype=np.float64) / 9.0
    
    for ip in range(3):
        csi = xi[ip]
        
        # Shape functions (quadratic edge)
        b[0] = 0.5 * csi * (csi - 1.0)
        b[1] = 1.0 - csi * csi
        b[2] = 0.5 * csi * (csi + 1.0)
        
        # Derivatives of shape functions
        db0 = csi - 0.5
        db1 = -2.0 * csi
        db2 = csi + 0.5
        
        # Derivatives of x and y along edge
        xx = db0 * x1 + db1 * x2 + db2 * x3
        yy = db0 * y1 + db1 * y2 + db2 * y3
        
        # Jacobian (edge length scaling)
        jaco = np.sqrt(xx * xx + yy * yy)
        
        # Weighted contributions
        wip = jaco * wi[ip]
        wipp = wip * p
        wipg = wip * gama
        
        # Assemble
        for i in range(3):
            Pe[i] += wipg * b[i]
            for j in range(3):
                He[i, j] += wipp * b[i] * b[j]
    
    return He, Pe


@njit(cache=True)
def compute_element_matrix(XN):
    """
    Compute element stiffness matrix for a single Quad-8 element.
    Inlined for Numba compatibility in parallel loops.
    """
    Ke = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)
    
    # 3x3 Gauss points
    G = np.sqrt(0.6)
    xp = np.array([
        [-G, -G], [0.0, -G], [G, -G],
        [-G, 0.0], [0.0, 0.0], [G, 0.0],
        [-G, G], [0.0, G], [G, G]
    ], dtype=np.float64)
    
    wp = np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float64) / 81.0
    
    for ip in range(9):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        
        # Shape function derivatives
        Dpsi = np.zeros((8, 2), dtype=np.float64)
        Dpsi[0, 0] = (2*csi + eta) * (1 - eta) / 4
        Dpsi[1, 0] = (2*csi - eta) * (1 - eta) / 4
        Dpsi[2, 0] = (2*csi + eta) * (1 + eta) / 4
        Dpsi[3, 0] = (2*csi - eta) * (1 + eta) / 4
        Dpsi[4, 0] = csi * (eta - 1)
        Dpsi[5, 0] = (1 - eta*eta) / 2
        Dpsi[6, 0] = -csi * (1 + eta)
        Dpsi[7, 0] = (eta*eta - 1) / 2
        
        Dpsi[0, 1] = (2*eta + csi) * (1 - csi) / 4
        Dpsi[1, 1] = (2*eta - csi) * (1 + csi) / 4
        Dpsi[2, 1] = (2*eta + csi) * (1 + csi) / 4
        Dpsi[3, 1] = (2*eta - csi) * (1 - csi) / 4
        Dpsi[4, 1] = (csi*csi - 1) / 2
        Dpsi[5, 1] = -(1 + csi) * eta
        Dpsi[6, 1] = (1 - csi*csi) / 2
        Dpsi[7, 1] = (csi - 1) * eta
        
        # Jacobian
        jaco = np.zeros((2, 2), dtype=np.float64)
        for i in range(8):
            jaco[0, 0] += XN[i, 0] * Dpsi[i, 0]
            jaco[0, 1] += XN[i, 0] * Dpsi[i, 1]
            jaco[1, 0] += XN[i, 1] * Dpsi[i, 0]
            jaco[1, 1] += XN[i, 1] * Dpsi[i, 1]
        
        Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
        inv_det = 1.0 / Detj
        
        Invj_00 = jaco[1, 1] * inv_det
        Invj_01 = -jaco[0, 1] * inv_det
        Invj_10 = -jaco[1, 0] * inv_det
        Invj_11 = jaco[0, 0] * inv_det
        
        # B matrix
        B = np.zeros((8, 2), dtype=np.float64)
        for i in range(8):
            B[i, 0] = Dpsi[i, 0] * Invj_00 + Dpsi[i, 1] * Invj_10
            B[i, 1] = Dpsi[i, 0] * Invj_01 + Dpsi[i, 1] * Invj_11
        
        wip = wp[ip] * Detj
        
        # Ke += wip * B @ B.T
        for i in range(8):
            for j in range(8):
                Ke[i, j] += wip * (B[i, 0] * B[j, 0] + B[i, 1] * B[j, 1])
    
    return Ke, fe


@njit(parallel=True, cache=True)
def assemble_all_elements(x, y, quad8, Nels):
    """
    Assemble all element stiffness matrices in parallel.
    
    Returns COO format data for sparse matrix construction.
    """
    # Pre-allocate COO arrays (64 entries per element: 8x8)
    rows = np.zeros(Nels * 64, dtype=np.int32)
    cols = np.zeros(Nels * 64, dtype=np.int32)
    vals = np.zeros(Nels * 64, dtype=np.float64)
    
    # Element force vectors
    fe_all = np.zeros((Nels, 8), dtype=np.float64)
    
    # Parallel loop over elements
    for e in prange(Nels):
        # Get element DOFs
        edof0 = quad8[e, 0]
        edof1 = quad8[e, 1]
        edof2 = quad8[e, 2]
        edof3 = quad8[e, 3]
        edof4 = quad8[e, 4]
        edof5 = quad8[e, 5]
        edof6 = quad8[e, 6]
        edof7 = quad8[e, 7]
        
        # Build element coordinates
        XN = np.zeros((8, 2), dtype=np.float64)
        XN[0, 0] = x[edof0]; XN[0, 1] = y[edof0]
        XN[1, 0] = x[edof1]; XN[1, 1] = y[edof1]
        XN[2, 0] = x[edof2]; XN[2, 1] = y[edof2]
        XN[3, 0] = x[edof3]; XN[3, 1] = y[edof3]
        XN[4, 0] = x[edof4]; XN[4, 1] = y[edof4]
        XN[5, 0] = x[edof5]; XN[5, 1] = y[edof5]
        XN[6, 0] = x[edof6]; XN[6, 1] = y[edof6]
        XN[7, 0] = x[edof7]; XN[7, 1] = y[edof7]
        
        # Compute element matrices
        Ke, fe = compute_element_matrix(XN)
        
        # Store force vector
        for i in range(8):
            fe_all[e, i] = fe[i]
        
        # Store COO entries
        base_idx = e * 64
        edofs = np.array([edof0, edof1, edof2, edof3, edof4, edof5, edof6, edof7], dtype=np.int32)
        
        k = 0
        for i in range(8):
            for j in range(8):
                rows[base_idx + k] = edofs[i]
                cols[base_idx + k] = edofs[j]
                vals[base_idx + k] = Ke[i, j]
                k += 1
    
    return rows, cols, vals, fe_all


@njit(parallel=True, cache=True)
def compute_derived_fields_numba(x, y, quad8, u, Nels):
    """
    Compute velocity fields in parallel.
    """
    vel = np.zeros((Nels, 2), dtype=np.float64)
    abs_vel = np.zeros(Nels, dtype=np.float64)
    
    # 2x2 Gauss points for post-processing
    G = np.sqrt(1.0 / 3.0)
    
    for e in prange(Nels):
        # Get element DOFs
        edof0 = quad8[e, 0]
        edof1 = quad8[e, 1]
        edof2 = quad8[e, 2]
        edof3 = quad8[e, 3]
        edof4 = quad8[e, 4]
        edof5 = quad8[e, 5]
        edof6 = quad8[e, 6]
        edof7 = quad8[e, 7]
        
        # Build element coordinates and solution
        XN = np.zeros((8, 2), dtype=np.float64)
        u_e = np.zeros(8, dtype=np.float64)
        
        XN[0, 0] = x[edof0]; XN[0, 1] = y[edof0]; u_e[0] = u[edof0]
        XN[1, 0] = x[edof1]; XN[1, 1] = y[edof1]; u_e[1] = u[edof1]
        XN[2, 0] = x[edof2]; XN[2, 1] = y[edof2]; u_e[2] = u[edof2]
        XN[3, 0] = x[edof3]; XN[3, 1] = y[edof3]; u_e[3] = u[edof3]
        XN[4, 0] = x[edof4]; XN[4, 1] = y[edof4]; u_e[4] = u[edof4]
        XN[5, 0] = x[edof5]; XN[5, 1] = y[edof5]; u_e[5] = u[edof5]
        XN[6, 0] = x[edof6]; XN[6, 1] = y[edof6]; u_e[6] = u[edof6]
        XN[7, 0] = x[edof7]; XN[7, 1] = y[edof7]; u_e[7] = u[edof7]
        
        vel_x_sum = 0.0
        vel_y_sum = 0.0
        v_mag_sum = 0.0
        
        # 4 integration points
        xp_list = [(-G, -G), (G, -G), (G, G), (-G, G)]
        
        for ip in range(4):
            if ip == 0:
                csi, eta = -G, -G
            elif ip == 1:
                csi, eta = G, -G
            elif ip == 2:
                csi, eta = G, G
            else:
                csi, eta = -G, G
            
            # Shape function derivatives
            Dpsi = np.zeros((8, 2), dtype=np.float64)
            Dpsi[0, 0] = (2*csi + eta) * (1 - eta) / 4
            Dpsi[1, 0] = (2*csi - eta) * (1 - eta) / 4
            Dpsi[2, 0] = (2*csi + eta) * (1 + eta) / 4
            Dpsi[3, 0] = (2*csi - eta) * (1 + eta) / 4
            Dpsi[4, 0] = csi * (eta - 1)
            Dpsi[5, 0] = (1 - eta*eta) / 2
            Dpsi[6, 0] = -csi * (1 + eta)
            Dpsi[7, 0] = (eta*eta - 1) / 2
            
            Dpsi[0, 1] = (2*eta + csi) * (1 - csi) / 4
            Dpsi[1, 1] = (2*eta - csi) * (1 + csi) / 4
            Dpsi[2, 1] = (2*eta + csi) * (1 + csi) / 4
            Dpsi[3, 1] = (2*eta - csi) * (1 - csi) / 4
            Dpsi[4, 1] = (csi*csi - 1) / 2
            Dpsi[5, 1] = -(1 + csi) * eta
            Dpsi[6, 1] = (1 - csi*csi) / 2
            Dpsi[7, 1] = (csi - 1) * eta
            
            # Jacobian
            jaco = np.zeros((2, 2), dtype=np.float64)
            for i in range(8):
                jaco[0, 0] += XN[i, 0] * Dpsi[i, 0]
                jaco[0, 1] += XN[i, 0] * Dpsi[i, 1]
                jaco[1, 0] += XN[i, 1] * Dpsi[i, 0]
                jaco[1, 1] += XN[i, 1] * Dpsi[i, 1]
            
            Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
            inv_det = 1.0 / Detj
            
            Invj_00 = jaco[1, 1] * inv_det
            Invj_01 = -jaco[0, 1] * inv_det
            Invj_10 = -jaco[1, 0] * inv_det
            Invj_11 = jaco[0, 0] * inv_det
            
            # B matrix and gradient
            grad_x = 0.0
            grad_y = 0.0
            for i in range(8):
                B_i0 = Dpsi[i, 0] * Invj_00 + Dpsi[i, 1] * Invj_10
                B_i1 = Dpsi[i, 0] * Invj_01 + Dpsi[i, 1] * Invj_11
                grad_x += B_i0 * u_e[i]
                grad_y += B_i1 * u_e[i]
            
            # Velocity is negative gradient
            vel_x_sum += -grad_x
            vel_y_sum += -grad_y
            v_mag_sum += np.sqrt(grad_x * grad_x + grad_y * grad_y)
        
        vel[e, 0] = vel_x_sum / 4.0
        vel[e, 1] = vel_y_sum / 4.0
        abs_vel[e] = v_mag_sum / 4.0
    
    return vel, abs_vel


# =============================================================================
# Progress Callback Monitor (same as CPU)
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

class Quad8FEMSolverNumba:
    """
    QUAD8 FEM Solver - Numba JIT optimized.
    
    Features:
    - Quad-8 (8-node quadrilateral) elements
    - Robin boundary conditions (inlet)
    - Dirichlet boundary conditions (outlet)
    - Parallel assembly using Numba prange
    - Jacobi-preconditioned CG solver
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
        implementation_name: str = "Numba",
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
    # Assembly (Numba Parallel)
    # =========================================================================
    
    def assemble_system(self) -> None:
        """Assemble global stiffness matrix using parallel Numba kernel."""
        if self.verbose:
            print("Assembling global system (Numba parallel)...")
        
        # Run parallel assembly
        rows, cols, vals, fe_all = assemble_all_elements(
            self.x, self.y, self.quad8, self.Nels
        )
        
        # Build sparse matrix from COO data
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
        from scipy.sparse import lil_matrix
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
            He, Pe = robin_quadr_numba(
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

        solve_end_time = time.perf_counter()
        total_solve_time = solve_end_time - solve_start_time
        self.timing_metrics['solve_system'] = total_solve_time

        self.converged = (self.solve_info == 0)

        if self.verbose:
            if self.converged:
                print(f"✓ CG converged in {self.monitor.it} iterations")
            else:
                print(f"✗ CG did not converge (max iterations reached)")
            
            print(f"  True residual norm:     {true_residual:.3e}")
            print(f"  True relative residual: {true_relative_residual:.3e}")
            print(f"  Total solver wall time: {total_solve_time:.4f} seconds")
        
        return self.u

    # =========================================================================
    # Post-Processing (Numba Parallel)
    # =========================================================================
    
    def compute_derived_fields(self) -> None:
        """Compute velocity and pressure fields using parallel Numba kernel."""
        if self.verbose:
            print("Computing velocity field (Numba parallel)...")
        
        self.vel, self.abs_vel = compute_derived_fields_numba(
            self.x, self.y, self.quad8, self.u, self.Nels
        )
        
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
            'iterations': self.monitor.it,
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
            print("\n✓ Numba simulation complete")
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
    # Warm up Numba JIT compilation
    print("Warming up Numba JIT compilation...")
    
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent.parent
    
    solver = Quad8FEMSolverNumba(
        mesh_file=PROJECT_ROOT / "data/input/exported_mesh_v6.h5",
        implementation_name="Numba",
        maxiter=15000,
        cg_print_every=50
    )
    
    results = solver.run()
    
    print(f"\nFinal Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Total Time: {results['timing_metrics']['total_program_time']:.4f}s")
