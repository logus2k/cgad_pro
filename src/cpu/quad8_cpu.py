"""
QUAD8 FEM Solver - CPU baseline (NumPy + SciPy)

Encapsulated solver class for 2D potential flow using Quad-8 elements.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from numpy.typing import NDArray

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator

# Project paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHARED_DIR = HERE.parent / "shared"

# Add shared to path
sys.path.insert(0, str(SHARED_DIR))

# Import shared utilities
from visualization_utils import generate_all_visualizations
from export_utils import export_results_to_excel

# Import CPU-specific functions
from elem_quad8_cpu import Elem_Quad8
from robin_quadr_cpu import Robin_quadr
from genip2dq_cpu import Genip2DQ
from shape_n_der8_cpu import Shape_N_Der8


class CGMonitor:
    """Callback monitor for CG solver iterations."""
    
    def __init__(self, every: int = 10):
        self.it: int = 0
        self.every: int = every
        self.residuals: list[float] = []

    def __call__(self, rk: NDArray[np.float64]) -> None:
        self.it += 1
        residual = float(np.linalg.norm(rk))
        self.residuals.append(residual)
        
        if self.it % self.every == 0:
            print(f"  CG iteration {self.it}, residual = {residual:.3e}")


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
        maxiter: int = 5000,
        bc_tolerance: float = 1e-9,  # ← Add this
        cg_print_every: int = 50,
        assembly_print_every: int = 50000,
        implementation_name: str = "CPU",
        verbose: bool = True
    ):
        """
        Initialize FEM solver.
        
        Args:
            mesh_file: Path to Excel mesh file with 'coord' and 'conec' sheets
            p0: Reference pressure (Pa)
            rho: Fluid density (kg/m³)
            gamma: Specific heat ratio
            rtol: CG solver relative tolerance
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
        
        # Solver diagnostics (will be initialized in solve)
        self.monitor: CGMonitor
        self.solve_info: int
        self.converged: bool = False
        
    def load_mesh(self) -> None:
        """Load mesh from Excel file."""
        if self.verbose:
            print(f"Loading mesh from {self.mesh_file.name}...")
        
        coord = pd.read_excel(self.mesh_file, sheet_name="coord")
        conec = pd.read_excel(self.mesh_file, sheet_name="conec")
        
        # Convert coordinates from mm to m
        self.x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
        self.y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
        
        # Convert connectivity to 0-indexed
        self.quad8 = conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1
        
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
        
        # Robin BC on minimum-x boundary (inlet)
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
        
        for (n1, n2, n3) in robin_edges:
            He, Pe = Robin_quadr(
                self.x[n1], self.y[n1],
                self.x[n2], self.y[n2],
                self.x[n3], self.y[n3],
                p=0.0,
                gama=self.gamma
            )
            
            ed = [n1, n2, n3]
            for i in range(3):
                self.fg[ed[i]] += Pe[i]
                for j in range(3):
                    self.Kg[ed[i], ed[j]] += He[i, j]
        
        # Dirichlet BC on maximum-x boundary (outlet)
        x_max = float(self.x.max())
        exit_nodes = np.where(np.abs(self.x - x_max) < self.bc_tolerance)[0]
        
        if self.verbose:
            print(f"  Applying {len(exit_nodes)} Dirichlet nodes...")
        
        for n in exit_nodes:
            self.Kg[n, :] = 0.0
            self.Kg[:, n] = 0.0
            self.Kg[n, n] = 1.0
            self.fg[n] = 0.0
        
        # Fix unused nodes (orphaned nodes not in any element)
        used_nodes_set = set(self.quad8.flatten().tolist())
        unused_count = 0
        for n in range(self.Nnds):
            if n not in used_nodes_set:
                self.Kg[n, :] = 0.0
                self.Kg[:, n] = 0.0
                self.Kg[n, n] = 1.0
                self.fg[n] = 0.0
                unused_count += 1
        
        if self.verbose and unused_count > 0:
            print(f"  Fixed {unused_count} unused nodes")
    
    def solve(self) -> NDArray[np.float64]:
        """Solve linear system using preconditioned CG."""
        if self.verbose:
            print("Converting to CSR format...")
        
        self.Kg = csr_matrix(self.Kg)
        
        # Jacobi preconditioner
        diag = self.Kg.diagonal().astype(np.float64, copy=True)
        diag[diag == 0.0] = 1.0
        M_inv = np.reciprocal(diag)
        
        def precond(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return M_inv * v
        
        M = LinearOperator(self.Kg.shape, precond)
        
        if self.verbose:
            print("Solving linear system (PCG)...")
        
        self.monitor = CGMonitor(every=self.cg_print_every)
        
        self.u, self.solve_info = cg(
            self.Kg,
            self.fg,
            rtol=self.rtol,
            maxiter=self.maxiter,
            M=M,
            callback=self.monitor
        )
        
        self.converged = (self.solve_info == 0)
        
        if self.verbose:
            if self.solve_info == 0:
                print(f"✓ CG converged in {self.monitor.it} iterations")
            elif self.solve_info > 0:
                print(f"✗ CG did not converge (max iterations reached)")
            else:
                print(f"✗ CG error (info={self.solve_info})")
        
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
    
    def visualize(self, output_dir: Path | str) -> Dict[str, Path]:
        """
        Generate all visualization plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of output file paths
        """
        output_dir = Path(output_dir)
        
        if self.verbose:
            print(f"Generating visualizations...")
        
        output_files = generate_all_visualizations(
            self.x, self.y, self.quad8,
            self.u, self.abs_vel, self.pressure,
            output_dir,
            implementation_name=self.implementation_name
        )
        
        if self.verbose:
            print(f"  Saved to {output_dir}")
        
        return output_files
    
    def export(self, output_file: Path | str) -> Path:
        """
        Export results to Excel.
        
        Args:
            output_file: Path to save Excel file
            
        Returns:
            Path to saved file
        """
        output_file = Path(output_file)
        
        if self.verbose:
            print(f"Exporting results...")
        
        export_results_to_excel(
            output_file,
            self.x, self.y, self.quad8,
            self.u, self.vel, self.abs_vel, self.pressure,
            implementation_name=self.implementation_name
        )
        
        if self.verbose:
            print(f"  Saved to {output_file}")
        
        return output_file
    
    def run(
        self,
        output_dir: Optional[Path | str] = None,
        export_file: Optional[Path | str] = None
    ) -> Dict[str, Any]:
        """
        Run complete FEM simulation workflow.
        
        Args:
            output_dir: Directory for visualization outputs. 
                    Defaults to PROJECT_ROOT/data/output/figures
            export_file: Path for Excel export.
                        Defaults to PROJECT_ROOT/data/output/Results_quad8_{implementation}.xlsx
            
        Returns:
            Dictionary with solution data and file paths
        """
        # Execute workflow
        self.load_mesh()
        self.assemble_system()
        self.apply_boundary_conditions()
        self.solve()
        self.compute_derived_fields()
        self.print_statistics()
        
        results: Dict[str, Any] = {
            'u': self.u,
            'vel': self.vel,
            'abs_vel': self.abs_vel,
            'pressure': self.pressure,
            'converged': self.converged,
            'iterations': self.monitor.it,
            'residuals': self.monitor.residuals
        }
        
        # Set defaults if not provided
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data/output/figures"
        
        if export_file is None:
            export_file = PROJECT_ROOT / f"data/output/Results_quad8_{self.implementation_name}.xlsx"
        
        # Generate outputs
        results['visualization_files'] = self.visualize(output_dir)
        results['export_file'] = self.export(export_file)
        
        if self.verbose:
            print("\n✓ Simulation complete")
        
        return results


# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":
    # Create solver instance
    solver = Quad8FEMSolver(
        mesh_file=PROJECT_ROOT / "data/input/converted_mesh_v3.xlsx",
        implementation_name="CPU"
    )
    
    # Run simulation (uses default output paths)
    results = solver.run()
    
    print(f"\nResults summary:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Final residual: {results['residuals'][-1]:.3e}")
