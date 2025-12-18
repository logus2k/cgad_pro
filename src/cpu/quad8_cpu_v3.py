"""
QUAD8 FEM Solver - CPU baseline (NumPy + SciPy)

Encapsulated solver class for 2D potential flow using Quad-8 elements.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Callable
from numpy.typing import NDArray

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator, gmres

# Project paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHARED_DIR = HERE.parent / "shared"

# Add shared to path
sys.path.insert(0, str(SHARED_DIR))

# Import shared utilities
from visualization_utils_fast import generate_all_visualizations
from export_utils_v2 import export_results

# Import CPU-specific functions
from elem_quad8_cpu import Elem_Quad8
from robin_quadr_cpu import Robin_quadr
from genip2dq_cpu import Genip2DQ
from shape_n_der8_cpu import Shape_N_Der8
from progress_callback import ProgressCallback


class IterativeSolverMonitor:
    """Monitor for CG iterations with actual residual computation."""

    def __init__(self, A, b, every: int = 10, maxiter: int = 15000, progress_callback=None):
        self.A = A  # System matrix
        self.b = b  # Right-hand side
        self.it: int = 0
        self.every: int = every
        self.maxiter: int = maxiter
        self.start_time: float = time.perf_counter()
        self.b_norm = float(np.linalg.norm(b))
        self.progress_callback = progress_callback

    def __call__(self, xk: NDArray[np.float64]) -> None:
        """
        Called by SciPy CG callback.

        Parameters
        ----------
        xk : ndarray
            Current solution estimate (equilibrated)
        """
        self.it += 1
        
        # Always check for incremental solution updates (independent of logging)
        if self.progress_callback is not None:
            # Send at iteration 1, 100, 200, 300, etc.
            if self.it == 1 or (self.it % 100 == 0):
                print(f"[DEBUG] Sending solution increment at iteration {self.it}")  # ← Debug log
                self.progress_callback.on_solution_increment(
                    iteration=self.it,
                    solution=xk
                )

        # Log progress at specified intervals
        if self.it % self.every != 0 and self.it != self.maxiter:
            return

        # Compute actual residual: r = b - A*x
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
        
        # Invoke callback for metrics
        if self.progress_callback is not None:
            self.progress_callback.on_iteration(
                iteration=self.it,
                max_iterations=self.maxiter,
                residual=res_norm,
                relative_residual=rel_res,
                elapsed_time=elapsed,
                etr_seconds=etr
            )  

            
class Quad8FEMSolver:
    """
    QUAD8 FEM Solver for 2D potential flow.
    
    Features:
    - Quad-8 (8-node quadrilateral) elements
    - Robin boundary conditions (inlet)
    - Dirichlet boundary conditions (outlet)
    - Jacobi-preconditioned CG solver
    - Automatic handling of unused nodes
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
        implementation_name: str = "CPU",
        verbose: bool = True,
        progress_callback=None
    ):
        """
        Initialize FEM solver.
        
        Args:
            mesh_file: Path to Excel mesh file with 'coord' and 'conec' sheets
            p0: Reference pressure (Pa)
            rho: Fluid density (kg/m³)
            gamma: Specific heat ratio
            rtol: CG solver relative tolerance
            atol: CG solver absolute tolerance
            maxiter: Maximum CG iterations
            bc_tolerance: Tolerance for detecting boundary nodes
            cg_print_every: Print CG progress every N iterations
            assembly_print_every: Print assembly progress every N elements
            implementation_name: Identifier for output files (e.g., "CPU", "GPU")
            verbose: Enable/disable progress printing
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
        
        # Mesh data (will be initialized in load_mesh)
        self.mesh_file = Path(mesh_file)
        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64]
        self.quad8: NDArray[np.int32]
        self.Nnds: int
        self.Nels: int
        
        # System matrices (will be initialized in assemble_system)
        self.Kg: lil_matrix | csr_matrix
        self.fg: NDArray[np.float64]
        
        # Solution (will be initialized in solve and compute_derived_fields)
        self.u: NDArray[np.float64]
        self.vel: NDArray[np.float64]
        self.abs_vel: NDArray[np.float64]
        self.pressure: NDArray[np.float64]

        # Record Global Program Start Time
        self.program_start_time: float = time.perf_counter()
        
        # Solver diagnostics (will be initialized in solve)
        self.monitor: IterativeSolverMonitor
        self.solve_info: int
        self.converged: bool = False
        self.timing_metrics: Dict[str, float] = {} # Initialize timing dictionary
        
        self.progress_callback = progress_callback

    def _time_step(self, step_name: str, func: Callable[..., Any]) -> Any:
        """Utility to time a function call and store the result."""
        t0 = time.perf_counter()
        # Execute the function
        result = func()
        t1 = time.perf_counter()
        
        # Store the metric
        self.timing_metrics[step_name] = t1 - t0
        
        # Print if verbose
        if self.verbose:
            print(f"  > Step '{step_name}' completed in {self.timing_metrics[step_name]:.4f} seconds.")
            
        return result    
        
    def load_mesh(self) -> None:
        """
        Load mesh from file. Supports multiple formats:
        - .xlsx (Excel) - Original format
        - .npz (NumPy) - Fast binary format
        - .h5/.hdf5 (HDF5) - Fastest format
        """
        if self.verbose:
            print(f"Loading mesh from {self.mesh_file.name}...")
        
        suffix = self.mesh_file.suffix.lower()
        
        if suffix == '.xlsx':
            # Excel format (original)
            coord = pd.read_excel(self.mesh_file, sheet_name="coord")
            conec = pd.read_excel(self.mesh_file, sheet_name="conec")
            
            # Convert coordinates from mm to m
            self.x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
            self.y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
            
            # Convert connectivity to 0-indexed
            self.quad8 = conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1
            
        elif suffix == '.npz':
            # NumPy compressed format
            data = np.load(self.mesh_file)
            self.x = data['x']
            self.y = data['y']
            self.quad8 = data['quad8']
            
        elif suffix == '.h5' or suffix == '.hdf5':
            # HDF5 format (fastest)
            try:
                import h5py  # type: ignore
            except ImportError:
                raise ImportError(
                    "HDF5 support requires h5py. Install with: pip install h5py"
                )
            
            with h5py.File(self.mesh_file, 'r') as f:
                # Read into NumPy arrays
                self.x = np.array(f['x'], dtype=np.float64)
                self.y = np.array(f['y'], dtype=np.float64)
                self.quad8 = np.array(f['quad8'], dtype=np.int32)
        else:
            raise ValueError(
                f"Unsupported mesh format: {suffix}\n"
                f"Supported formats: .xlsx, .npz, .h5, .hdf5"
            )
        
        self.Nnds = self.x.size
        self.Nels = self.quad8.shape[0]
        
        if self.verbose:
            print(f"  Loaded: {self.Nnds} nodes, {self.Nels} Quad-8 elements")
    
    def assemble_system(self) -> None:
        """Assemble global stiffness matrix and force vector."""
        if self.verbose:
            print("Assembling global system...")
        
        self.Kg = lil_matrix((self.Nnds, self.Nnds), dtype=np.float64)
        self.fg = np.zeros(self.Nnds, dtype=np.float64)
        
        for e in range(self.Nels):
            edofs = self.quad8[e]
            XN = np.column_stack((self.x[edofs], self.y[edofs]))
            Ke, fe = Elem_Quad8(XN, fL=0.0)
            
            for i in range(8):
                self.fg[edofs[i]] += fe[i]
                for j in range(8):
                    self.Kg[edofs[i], edofs[j]] += Ke[i, j]
            
            if self.verbose and (e + 1) % self.assembly_print_every == 0:
                print(f"  {e + 1}/{self.Nels} elements assembled")
    
    def apply_boundary_conditions(self) -> None:
        """Apply Robin (inlet) and Dirichlet (outlet) boundary conditions."""
        
        if self.verbose:
            print("Applying boundary conditions...")
        
        # Robin BC on minimum-x boundary (inlet) assembly (remains the same as before)
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
        
        # Inlet potential remains 0.0 (as validated by the direct solver)
        inlet_potential = 0.0 
        
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
        
        
        # Dirichlet BC on maximum-x boundary (outlet)
        exit_nodes = np.where(self.x == np.max(self.x))[0]
        
        if self.verbose:
            print(f"  Applying {len(exit_nodes)} Dirichlet nodes (u=0) via Penalty Method...")
        
        PENALTY_FACTOR = 1.0e12
        
        # Use the Penalty Method for fixing the potential
        for n in exit_nodes:
            # Set diagonal entry to Kg[n,n] + PENALTY_FACTOR
            self.Kg[n, n] += PENALTY_FACTOR
            # fg[n] += PENALTY_FACTOR * target_u (target_u = 0.0)
            
        # Fix unused nodes (orphaned nodes not in any element)
        used_nodes_set = set(self.quad8.flatten().tolist())
        unused_count = 0
        for n in range(self.Nnds):
            if n not in used_nodes_set:
                self.Kg[n, n] += PENALTY_FACTOR
                # fg[n] += PENALTY_FACTOR * 0.0
                unused_count += 1
        
        if self.verbose and unused_count > 0:
            print(f"  Fixed {unused_count} unused nodes via Penalty Method")


    def solve(self) -> NDArray[np.float64]:
        """Solve linear system using CG with diagonal equilibration."""
        
        solve_start_time = time.perf_counter()

        if self.verbose:
            print("Converting to CSR format...")
        
        self.Kg = csr_matrix(self.Kg)
        
        diag_values = self.Kg.diagonal()
        if self.verbose:
            print(f"  Kg Diagonal Min: {diag_values.min():.3e}")
            print(f"  Kg Diagonal Max: {diag_values.max():.3e}")
        
        initial_residual_norm = np.linalg.norm(self.fg)
        if self.verbose:
            print(f"  Initial L2 residual norm (b): {initial_residual_norm:.3e}")
        
        # Apply diagonal equilibration for better conditioning
        if self.verbose:
            print("Applying diagonal equilibration...")
        
        diag = self.Kg.diagonal()
        D_inv_sqrt = 1.0 / np.sqrt(np.abs(diag))
        
        # Create equilibrated system: D^(-1/2) * K * D^(-1/2) * u_eq = D^(-1/2) * f
        from scipy.sparse import diags
        D_mat = diags(D_inv_sqrt)
        Kg_eq = D_mat @ self.Kg @ D_mat
        fg_eq = self.fg * D_inv_sqrt
        
        # Use Jacobi preconditioner on equilibrated system
        diag_eq = Kg_eq.diagonal()
        M_inv = 1.0 / diag_eq
        
        def precond_jacobi(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return M_inv * v
        
        M = LinearOperator(Kg_eq.shape, precond_jacobi)
        
        if self.verbose:
            print(f"Solving with CG (tol={1e-8:.1e}, maxiter={self.maxiter})...")
        
        # Target tolerance
        TARGET_TOL = 1e-8
        
        # Initialize the monitor with A and b for residual computation
        self.monitor = IterativeSolverMonitor(
            Kg_eq,
            fg_eq,
            every=self.cg_print_every,
            maxiter=self.maxiter,
            progress_callback=self.progress_callback
        )
        
        # Solve with CG (more robust for large symmetric systems)
        u_eq, self.solve_info = cg(
            Kg_eq,
            fg_eq,
            rtol=TARGET_TOL,
            atol=0.0,
            maxiter=self.maxiter,
            M=M,
            callback=self.monitor
        )
        
        # De-equilibrate solution: u = D^(-1/2) * u_eq
        self.u = u_eq * D_inv_sqrt
        
        # --- TRUE residual check ---
        r = self.fg - self.Kg @ self.u
        true_residual = np.linalg.norm(r)
        true_relative_residual = true_residual / initial_residual_norm

        # STOP TIMER and store metric
        solve_end_time = time.perf_counter()
        total_solve_time = solve_end_time - solve_start_time
        self.timing_metrics['solve_system'] = total_solve_time

        # CG convergence is defined by info == 0
        self.converged = (self.solve_info == 0)

        if self.verbose:
            if self.converged:
                print(f"✓ CG converged in {self.monitor.it} iterations")
            else:
                print(f"✗ CG did not converge (max iterations reached)")
            
            print(f"  True residual norm:     {true_residual:.3e}")
            print(f"  True relative residual: {true_relative_residual:.3e}")
            print(f"  Target tol:             {TARGET_TOL:.1e}")
            print(f"  Total solver wall time: {total_solve_time:.4f} seconds")
        
        return self.u
    
    def compute_derived_fields(self) -> None:
        """Compute velocity and pressure fields from potential."""
        if self.verbose:
            print("Computing velocity field...")
        
        self.abs_vel = np.zeros(self.Nels, dtype=np.float64)
        self.vel = np.zeros((self.Nels, 2), dtype=np.float64)
        
        for e in range(self.Nels):
            edofs = self.quad8[e]
            XN = np.column_stack((self.x[edofs], self.y[edofs]))
            
            xp, _ = Genip2DQ(4)
            v_ip = np.zeros(4, dtype=np.float64)
            
            for ip in range(4):
                B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
                grad = B.T @ self.u[edofs]
                self.vel[e, 0] = grad[0]
                self.vel[e, 1] = grad[1]
                v_ip[ip] = np.linalg.norm(grad)
            
            self.abs_vel[e] = v_ip.mean()
        
        self.pressure = self.p0 - self.rho * self.abs_vel**2
    
    def print_statistics(self) -> None:
        """Print solution statistics."""
        if not self.verbose:
            return
        
        print(f"\nSolution statistics:")
        print(f"  u range: [{self.u.min():.6e}, {self.u.max():.6e}]")
        print(f"  u mean:  {self.u.mean():.6e}")
        print(f"  u std:   {self.u.std():.6e}")
    
    def visualize(self, output_dir: Optional[Path | str]) -> Dict[str, Path]:
        """Generates and saves visualizations."""
        """
        Generate all visualization plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of output file paths
        """

        if output_dir is None:
            output_dir = PROJECT_ROOT / "data/output/figures"        
        
        if self.verbose:
            print(f"Generating visualizations...")
        
        output_files = generate_all_visualizations(
            self.x, self.y, self.quad8,
            self.u,
            output_dir,
            implementation_name=self.implementation_name
        )
        
        if self.verbose:
            print(f"  Saved to {output_dir}")
        
        return output_files
    
    def export(self, output_file: Optional[Path | str]):
        """
        Export results to Excel.
        
        Args:
            output_file: Path to save Excel file
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = PROJECT_ROOT / f"data/output/Results_quad8_{self.implementation_name}.xlsx"
        
        if self.verbose:
            print(f"Exporting results...")
        
        export_results(
            output_file,
            self.x, self.y, self.quad8,
            self.u, self.vel, self.abs_vel, self.pressure,
            implementation_name=self.implementation_name,
            formats=['hdf5']   # ['hdf5', 'npz', 'csv']
        )
        
        if self.verbose:
            print(f"  Saved to {output_file}")
        
        return output_file

    def run(
        self,
        output_dir: Optional[Path | str] = None,
        export_file: Optional[Path | str] = None
    ) -> Dict[str, Any]:
        """Run complete FEM simulation workflow with progress callbacks."""
        
        # --- Start Timer for the Core Workflow ---
        total_workflow_start = time.perf_counter()
        self.timing_metrics = {}  # Ensure metrics are cleared/re-initialized
        
        # --- STAGE 1: Load Mesh ---
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='load_mesh')
        
        self._time_step('load_mesh', self.load_mesh)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='load_mesh',
                duration=self.timing_metrics['load_mesh']
            )
            # Emit mesh metadata
            self.progress_callback.on_mesh_loaded(
                nodes=self.Nnds,
                elements=self.Nels,
                coordinates={'x': self.x.tolist(), 'y': self.y.tolist()},
                connectivity=self.quad8.tolist()
            )
        
        # --- STAGE 2: Assembly ---
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='assemble_system')
        
        self._time_step('assemble_system', self.assemble_system)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='assemble_system',
                duration=self.timing_metrics['assemble_system']
            )
        
        # --- STAGE 3: Apply Boundary Conditions ---
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='apply_bc')
        
        self._time_step('apply_bc', self.apply_boundary_conditions)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='apply_bc',
                duration=self.timing_metrics['apply_bc']
            )
        
        # --- STAGE 4: Solve Linear System ---
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='solve_system')
        
        self._time_step('solve_system', self.solve)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='solve_system',
                duration=self.timing_metrics['solve_system']
            )
        
        # --- STAGE 5: Post-Processing ---
        if self.progress_callback:
            self.progress_callback.on_stage_start(stage='compute_derived')
        
        self._time_step('compute_derived', self.compute_derived_fields)
        
        if self.progress_callback:
            self.progress_callback.on_stage_complete(
                stage='compute_derived',
                duration=self.timing_metrics['compute_derived']
            )
        
        # Print statistics (console diagnostics)
        self._time_step('print_stats', self.print_statistics)
        
        # --- STAGE 6: Visualization (Optional) ---
        if output_dir is not None:
            if self.progress_callback:
                self.progress_callback.on_stage_start(stage='visualize')
            
            self._time_step('visualize', lambda: self.visualize(output_dir))
            
            if self.progress_callback:
                self.progress_callback.on_stage_complete(
                    stage='visualize',
                    duration=self.timing_metrics['visualize']
                )
        
        # --- STAGE 7: Export (Optional) ---
        if export_file is not None:
            if self.progress_callback:
                self.progress_callback.on_stage_start(stage='export')
            
            self._time_step('export', lambda: self.export(export_file))
            
            if self.progress_callback:
                self.progress_callback.on_stage_complete(
                    stage='export',
                    duration=self.timing_metrics['export']
                )
        
        # --- Final Total Workflow Time ---
        total_workflow_end = time.perf_counter()
        self.timing_metrics['total_workflow'] = total_workflow_end - total_workflow_start
        
        # --- Calculate and Store Total Program Time ---
        total_program_time = total_workflow_end - self.program_start_time
        self.timing_metrics['total_program_time'] = total_program_time
        
        # --- Prepare Results Dictionary ---
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
        
        # --- Emit Final Completion Event ---
        if self.progress_callback:
            self.progress_callback.on_solve_complete(
                converged=self.converged,
                iterations=self.monitor.it,
                timing_metrics=self.timing_metrics,
                solution_stats=results['solution_stats'],
                mesh_info=results['mesh_info']
            )
        
        # --- Console Output (if verbose) ---
        if self.verbose:
            print("\n✓ Simulation complete")
            
            # Print the detailed breakdown
            print("\nStep-by-Step Timings (seconds):")
            for k, v in self.timing_metrics.items():
                print(f"  {k:<20}: {v:.4f}")
            
            # Print the final total time
            print(f"  Total Program Wall Time: {total_program_time:.4f} seconds")
            
            # Print results summary
            print(f"\nResults summary:")
            print(f"  Converged: {results['converged']}")
            print(f"  Iterations: {results['iterations']}")
            print(f"  u range: [{results['solution_stats']['u_range'][0]:.6e}, "
                f"{results['solution_stats']['u_range'][1]:.6e}]")
        
        return results

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":
    # Create solver instance
    solver = Quad8FEMSolver(
        mesh_file=PROJECT_ROOT / "data/input/exported_mesh_v6.h5",
        implementation_name="CPU",
        maxiter=15000,
        cg_print_every=5   # 5 or 10 for frequent updates
    )
    
    # Run simulation (uses default output paths)
    results = solver.run()
    
    print(f"\nResults summary:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")

    total_program_time = results['timing_metrics'].get('total_program_time', 0.0)
    print(f"  Total Program Wall Time: {total_program_time:.4f} seconds")
