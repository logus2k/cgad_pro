"""
QUAD8 FEM Solver - GPU-accelerated version using CuPy (OPTIMIZED: RawKernel)

Encapsulated solver class for 2D potential flow using Quad-8 elements.

- Uses CuPy's RawKernel for Mass Assembly and Mass Post-Processing (major performance gain).
- Uses SciPy's GMRES solver on CPU for matrix solution stability comparison (Intentional).
"""

import numpy as np
import sys
import time
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, Callable, List
from numpy.typing import NDArray

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsplg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.sparse.linalg import spilu # Added spilu back for ILU preconditioner

# --- Project paths (Adjust as needed) ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHARED_DIR = HERE.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

# Import shared utilities
from visualization_utils import generate_all_visualizations
from export_utils import export_results_to_excel # Added export utility
from robin_quadr_gpu import Robin_quadr 

# =========================================================================
# 1. RAW KERNEL SOURCE: Mass Assembly (Ke and fe for ALL elements)
# =========================================================================
QUAD8_KERNEL_SOURCE = r"""
extern "C" __global__
void quad8_assembly_kernel(
    const double* x_in,
    const double* y_in,
    const int* quad8_in,
    const double* xp_in, // 9 IP points
    const double* wp_in, // 9 IP weights
    const int Nels,
    const int Nnds,
    // Output arrays
    double* vals_out, // 64 vals per element
    double* fg_out)   // Global force vector (atomically updated)
{
    // Element index from the thread ID
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Nels) return;

    const int N_NDS_PER_EL = 8;
    const int N_IP = 9;
    const int N_NZ_PER_EL = 64;
    const double fL = 0.0; 

    // --- 1. Get Nodal Coordinates (XN) ---
    double XN[N_NDS_PER_EL][2];
    int edofs[N_NDS_PER_EL];
    
    // Global connectivity access
    for (int i = 0; i < N_NDS_PER_EL; ++i) {
        edofs[i] = quad8_in[e * N_NDS_PER_EL + i];
        XN[i][0] = x_in[edofs[i]];
        XN[i][1] = y_in[edofs[i]];
    }

    // --- 2. Initialize Ke (8x8) and fe (8x1) ---
    double Ke[N_NDS_PER_EL][N_NDS_PER_EL] = {{0.0}};
    double fe[N_NDS_PER_EL] = {0.0};
    
    // --- 3. Integration Loop (NIP=9) ---
    for (int ip = 0; ip < N_IP; ++ip) {
        
        // Get csi, eta for this integration point
        double csi = xp_in[ip * 2 + 0];
        double eta = xp_in[ip * 2 + 1];
        double wp_ip = wp_in[ip];

        // --- 4. Shape_N_Der8 Logic (B matrix) ---
        double Dpsi[N_NDS_PER_EL][2];
        double psi[N_NDS_PER_EL];

        // Shape functions (psi)
        psi[0] = (csi-1)*(eta+csi+1)*(1-eta)/4;
        psi[1] = (1+csi)*(1-eta)*(csi-eta-1)/4;
        psi[2] = (1+csi)*(1+eta)*(csi+eta-1)/4;
        psi[3] = (csi-1)*(csi-eta+1)*(1+eta)/4;
        psi[4] = (1-csi*csi)*(1-eta)/2;
        psi[5] = (1+csi)*(1-eta*eta)/2;
        psi[6] = (1-csi*csi)*(1+eta)/2;
        psi[7] = (1-csi)*(1-eta*eta)/2;

        // Derivatives wrt (csi, eta) (Dpsi)
        Dpsi[0][0] = (2*csi+eta)*(1-eta)/4; Dpsi[0][1] = (2*eta+csi)*(1-csi)/4;
        Dpsi[1][0] = (2*csi-eta)*(1-eta)/4; Dpsi[1][1] = (2*eta-csi)*(1+csi)/4;
        Dpsi[2][0] = (2*csi+eta)*(1+eta)/4; Dpsi[2][1] = (2*eta+csi)*(1+csi)/4;
        Dpsi[3][0] = (2*csi-eta)*(1+eta)/4; Dpsi[3][1] = (2*eta-csi)*(1-csi)/4;
        Dpsi[4][0] = csi*(eta-1);          Dpsi[4][1] = (csi*csi-1)/2;
        Dpsi[5][0] = (1-eta*eta)/2;        Dpsi[5][1] = -(1+csi)*eta;
        Dpsi[6][0] = -csi*(1+eta);         Dpsi[6][1] = (1-csi*csi)/2;
        Dpsi[7][0] = (eta*eta-1)/2;        Dpsi[7][1] = (csi-1)*eta;
        
        // Jacobian (jaco)
        double J[2][2] = {{0.0}};
        for(int i=0; i<8; ++i) {
            J[0][0] += XN[i][0] * Dpsi[i][0]; J[0][1] += XN[i][0] * Dpsi[i][1];
            J[1][0] += XN[i][1] * Dpsi[i][0]; J[1][1] += XN[i][1] * Dpsi[i][1];
        }
        
        // Determinant and Inverse of Jacobian
        double Detj = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        
        if (Detj <= 1.0e-12) { return; }
        
        double InvJ[2][2];
        InvJ[0][0] = J[1][1] / Detj; InvJ[0][1] = -J[0][1] / Detj;
        InvJ[1][0] = -J[1][0] / Detj; InvJ[1][1] = J[0][0] / Detj;

        // B matrix (B = Dpsi @ InvJ)
        double B[N_NDS_PER_EL][2];
        for(int i=0; i<8; ++i) {
            B[i][0] = Dpsi[i][0] * InvJ[0][0] + Dpsi[i][1] * InvJ[1][0];
            B[i][1] = Dpsi[i][0] * InvJ[0][1] + Dpsi[i][1] * InvJ[1][1];
        }

        // --- 5. Assembly Accumulation (Ke += wip * B @ B.T) ---
        double wip = wp_ip * Detj;
        
        for (int i = 0; i < N_NDS_PER_EL; ++i) {
            // fe += fL * wip * psi
            fe[i] += fL * wip * psi[i];

            for (int j = 0; j < N_NDS_PER_EL; ++j) {
                // (B @ B.T) contribution
                double K_ij = B[i][0] * B[j][0] + B[i][1] * B[j][1];
                Ke[i][j] += wip * K_ij;
            }
        }
    }
    
    // --- 6. Global Scatter (Write results to global memory) ---
    int start_idx = e * N_NZ_PER_EL;
    
    int k = 0;
    for (int i = 0; i < N_NDS_PER_EL; ++i) {
        for (int j = 0; j < N_NDS_PER_EL; ++j) {
            vals_out[start_idx + k] = Ke[i][j];
            k++;
        }
    }

    // Atomically update the global force vector fg_out
    for (int i = 0; i < N_NDS_PER_EL; ++i) {
        atomicAdd(&fg_out[edofs[i]], fe[i]);
    }
}
"""

# =========================================================================
# 2. RAW KERNEL SOURCE: Mass Post-Processing (Compute Derived Fields)
# =========================================================================
QUAD8_POSTPROCESS_KERNEL_SOURCE = r"""
extern "C" __global__
void quad8_postprocess_kernel(
    const double* u_in,
    const double* x_in,
    const double* y_in,
    const int* quad8_in,
    const double* xp_in, // 4 IP points
    const int Nels,
    const double P0,
    const double RHO,
    // Output arrays
    double* abs_vel_out,
    double* vel_out) // 2D array: [Nels * 2]
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Nels) return;

    const int N_NDS_PER_EL = 8;
    const int N_IP = 4; // NIP=4 for post-processing

    // --- 1. Get Nodal Coords (XN) and Solution (u_e) ---
    double XN[N_NDS_PER_EL][2];
    double u_e[N_NDS_PER_EL];
    int edofs[N_NDS_PER_EL];
    
    for (int i = 0; i < N_NDS_PER_EL; ++i) {
        edofs[i] = quad8_in[e * N_NDS_PER_EL + i];
        XN[i][0] = x_in[edofs[i]];
        XN[i][1] = y_in[edofs[i]];
        u_e[i] = u_in[edofs[i]];
    }

    double v_ip_sum = 0.0;
    
    // --- 2. Integration Loop (NIP=4) ---
    for (int ip = 0; ip < N_IP; ++ip) {
        double csi = xp_in[ip * 2 + 0];
        double eta = xp_in[ip * 2 + 1];

        // --- 3. B matrix calculation (repeated logic) ---
        double Dpsi[N_NDS_PER_EL][2];
        
        // Derivatives wrt (csi, eta) (Dpsi)
        Dpsi[0][0] = (2*csi+eta)*(1-eta)/4; Dpsi[0][1] = (2*eta+csi)*(1-csi)/4;
        Dpsi[1][0] = (2*csi-eta)*(1-eta)/4; Dpsi[1][1] = (2*eta-csi)*(1+csi)/4;
        Dpsi[2][0] = (2*csi+eta)*(1+eta)/4; Dpsi[2][1] = (2*eta+csi)*(1+csi)/4;
        Dpsi[3][0] = (2*csi-eta)*(1+eta)/4; Dpsi[3][1] = (2*eta-csi)*(1-csi)/4;
        Dpsi[4][0] = csi*(eta-1);          Dpsi[4][1] = (csi*csi-1)/2;
        Dpsi[5][0] = (1-eta*eta)/2;        Dpsi[5][1] = -(1+csi)*eta;
        Dpsi[6][0] = -csi*(1+eta);         Dpsi[6][1] = (1-csi*csi)/2;
        Dpsi[7][0] = (eta*eta-1)/2;        Dpsi[7][1] = (csi-1)*eta;
        
        // Jacobian (J)
        double J[2][2] = {{0.0}};
        for(int i=0; i<8; ++i) {
            J[0][0] += XN[i][0] * Dpsi[i][0]; J[0][1] += XN[i][0] * Dpsi[i][1];
            J[1][0] += XN[i][1] * Dpsi[i][0]; J[1][1] += XN[i][1] * Dpsi[i][1];
        }

        // Determinant and Inverse of Jacobian
        double Detj = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        double InvJ[2][2];
        InvJ[0][0] =  J[1][1] / Detj; InvJ[0][1] = -J[0][1] / Detj;
        InvJ[1][0] = -J[1][0] / Detj; InvJ[1][1] =  J[0][0] / Detj;

        // B matrix (B = Dpsi @ InvJ)
        double B[N_NDS_PER_EL][2];
        for(int i=0; i<8; ++i) {
            B[i][0] = Dpsi[i][0] * InvJ[0][0] + Dpsi[i][1] * InvJ[1][0]; // d(phi)/dx
            B[i][1] = Dpsi[i][0] * InvJ[0][1] + Dpsi[i][1] * InvJ[1][1]; // d(phi)/dy
        }
        
        // --- 4. Gradient Calculation (grad = B.T @ u_e) ---
        double grad[2] = {0.0};
        for (int i = 0; i < N_NDS_PER_EL; ++i) {
            grad[0] += B[i][0] * u_e[i]; // d(u)/dx
            grad[1] += B[i][1] * u_e[i]; // d(u)/dy
        }

        // --- 5. Velocity and Norm ---
        double vel_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
        v_ip_sum += vel_norm;

        // Write the velocity at the first IP (approximation of element velocity)
        if (ip == 0) {
            vel_out[e * 2 + 0] = grad[0];
            vel_out[e * 2 + 1] = grad[1];
        }
    }
    
    // --- 6. Output (Average Velocity) ---
    // Average velocity magnitude (at NIP=4)
    double avg_vel = v_ip_sum / N_IP;
    abs_vel_out[e] = avg_vel;
}
"""

class IterativeSolverMonitor:
	"""Callback monitor for GMRES solver iterations (NumPy on CPU)."""
	
	def __init__(self, every: int = 50, maxiter: int = 5000):
		self.it = 0
		self.every = every
		self.maxiter = maxiter
		self.start_time = time.perf_counter()

	def __call__(self, rk: NDArray[np.float64]) -> None:
		self.it += 1

		if self.it % self.every != 0 and self.it != self.maxiter:
			return

		res_norm = float(np.linalg.norm(rk))
		elapsed = time.perf_counter() - self.start_time
		time_per_it = elapsed / self.it
		remaining = self.maxiter - self.it
		etr = remaining * time_per_it

		def fmt(sec):
			m = int(sec // 60)
			s = int(sec % 60)
			return f"{m:02d}m:{s:02d}s"

		progress = 100.0 * self.it / self.maxiter

		print(
			f"  GMRES iter {self.it}/{self.maxiter} ({progress:.1f}%), "
			f"residual = {res_norm:.3e}, "
			f"ETR: {fmt(etr)}"
		)


class Quad8FEMSolverGPU:
	"""
	QUAD8 FEM Solver — GPU-Optimized (RawKernel Assembly/Post-Process, CPU Solve)
	"""

	def __init__(
		self,
		mesh_file: Path | str, # Added type hint for mesh_file
		p0: float = 101328.8281,
		rho: float = 0.6125,
		gamma: float = 2.5,
		rtol: float = 1e-6,
		atol: float = 0.0,
		maxiter: int = 100,
		cg_print_every: int = 50,
		bc_tolerance: float = 1e-9,
		implementation_name: str = "GPU",
		verbose: bool = True,
	):
		# --- FIXES 1, 2, 3, 5: Assign mesh_file to self ---
		self.mesh_file = Path(mesh_file) 
		# --------------------------------------------------

		# Physics parameters
		self.p0 = p0
		self.rho = rho
		self.gamma = gamma

		# Solver parameters
		self.tol: float = 0.9 
		self.rtol = rtol
		self.atol = atol
		self.maxiter = maxiter
		self.cg_print_every = cg_print_every

		self.bc_tolerance = bc_tolerance
		self.implementation_name = implementation_name
		self.verbose = verbose

		self.program_start_time = time.perf_counter()
		self.timing_metrics: Dict[str, float] = {}
		
		# Solver diagnostics
		self.converged: bool = False
		self.iterations: int = 0
		self.residuals: List[float] = [] # FIX 4: Explicitly type hint as List[float]


	# ---------------
	# Timing utility 
	# ---------------
	def _time_step(self, name: str, fn: Callable) -> Any:
		t0 = time.perf_counter()
		out = fn()
		self.timing_metrics[name] = time.perf_counter() - t0
		if self.verbose:
			print(f"  > Step '{name}' completed in {self.timing_metrics[name]:.4f}s")
		return out

	# -------------
	# Mesh loading
	# -------------
	def load_mesh(self):
		if self.verbose:
			# FIX 5 is also resolved here by the __init__ assignment
			print(f"Loading mesh from {self.mesh_file.name}...") 

		coord = pd.read_excel(self.mesh_file, sheet_name="coord")
		conec = pd.read_excel(self.mesh_file, sheet_name="conec")

		self.x_cpu = coord["X"].to_numpy(dtype=np.float64) / 1000.0
		self.y_cpu = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
		self.quad8 = conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1 

		self.Nnds = self.x_cpu.size
		self.Nels = self.quad8.shape[0]

		# Transfer nodal data to GPU
		self.x = cp.asarray(self.x_cpu)
		self.y = cp.asarray(self.y_cpu)

		if self.verbose:
			print(f"  Loaded: {self.Nnds} nodes, {self.Nels} Quad-8 elements")

	# ---------
	# Assembly 
	# ---------
	def _genip2dq_9_gpu(self):
		"""Helper function to generate NIP=9 Gauss points (CuPy arrays)."""
		G = cp.sqrt(0.6)
		xp = cp.array([
			[-1.0, -1.0], [ 0.0, -1.0], [ 1.0, -1.0],
			[-1.0,  0.0], [ 0.0,  0.0], [ 1.0,  0.0],
			[-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0]
		], dtype=cp.float64) * G
		wp = cp.array(
			[25, 40, 25, 40, 64, 40, 25, 40, 25],
			dtype=cp.float64
		) / 81.0
		return xp, wp

	def assemble_system(self):
		if self.verbose:
			print("Assembling global system (Mass Assembly RawKernel)...")

		self.fg = cp.zeros(self.Nnds, dtype=cp.float64)
		quad8_cp = cp.asarray(self.quad8, dtype=cp.int32)
		
		# 1. Pre-allocate COO buffers
		N_NZ_per_el = 64
		total_nnz = self.Nels * N_NZ_per_el
		
		self._rows = cp.empty(total_nnz, dtype=cp.int32)
		self._cols = cp.empty(total_nnz, dtype=cp.int32)
		self._vals = cp.empty(total_nnz, dtype=cp.float64) 

		# 2. Vectorized Global Index Generation (ON GPU)
		local_i, local_j = cp.mgrid[0:8, 0:8]
		local_rows = cp.tile(local_i.ravel(), self.Nels)
		local_cols = cp.tile(local_j.ravel(), self.Nels)
		el_indices = cp.arange(self.Nels).repeat(N_NZ_per_el)
		
		self._rows = quad8_cp[el_indices, local_rows]
		self._cols = quad8_cp[el_indices, local_cols]
		
		# 3. Setup Integration Points
		xp, wp = self._genip2dq_9_gpu()

		# 4. Kernel Launch (Mass Assembly)
		kernel = cp.RawKernel(QUAD8_KERNEL_SOURCE, 'quad8_assembly_kernel')
		threads_per_block = 128
		blocks = (self.Nels + threads_per_block - 1) // threads_per_block

		kernel((blocks,), (threads_per_block,), (
			self.x, self.y, quad8_cp, 
			xp, wp, 
			self.Nels, self.Nnds,
			self._vals, self.fg
		))
		
		cp.cuda.Stream.null.synchronize()

	# --------------------
	# Boundary conditions 
	# --------------------

	def apply_boundary_conditions(self):
		if self.verbose:
			print("Applying boundary conditions...")

		x_min = self.x_cpu.min()
		x_max = self.x_cpu.max()

		# Dirichlet BC on maximum-x boundary (outlet)
		exit_nodes_cpu = np.where(self.x_cpu == x_max)[0]
		exit_nodes_gpu = cp.asarray(exit_nodes_cpu, dtype=cp.int32)

		# Robin BC on minimum-x boundary (inlet)
		boundary_nodes_cpu = set(
			np.where(np.abs(self.x_cpu - x_min) < self.bc_tolerance)[0].tolist()
		)

		# Find the edges forming the Robin boundary (Inlet)
		robin_edges = []
		for e in range(self.Nels):
			n = self.quad8[e]
			edges = [
				(n[0], n[4], n[1]), (n[1], n[5], n[2]),
				(n[2], n[6], n[3]), (n[3], n[7], n[0]), 
			]
			for edge in edges:
				if all(k in boundary_nodes_cpu for k in edge):
					# Use tuple(sorted) to ensure unique edges are found
					if tuple(sorted(edge)) not in [tuple(sorted(e)) for e in robin_edges]:
						robin_edges.append(edge)
						
		if self.verbose:
			print(f"  Applying {len(robin_edges)} Robin edges (Inlet)...")

		# --- CORRECTION: ALIGN ROBIN BC VALUE WITH CPU (inlet_potential=0.0) ---
		inlet_potential = 0.0 

		bc_rows = []
		bc_cols = []
		bc_vals = []

		for (n1, n2, n3) in robin_edges:
			He, Pe = Robin_quadr(
				self.x[n1], self.y[n1], self.x[n2], self.y[n2],
				self.x[n3], self.y[n3], p=inlet_potential, gama=self.gamma
			)
			
			ed = [n1, n2, n3]
			
			for i in range(3):
				# CORRECTION: Remove the scaling factor (x 10.0)
				self.fg[ed[i]] += Pe[i] 
				
				for j in range(3):
					bc_rows.append(ed[i])
					bc_cols.append(ed[j])
					bc_vals.append(cp.asnumpy(He[i, j]).item())

		# Merge bulk COO with Robin BC COO
		if bc_rows:
			bc_rows_cp = cp.asarray(bc_rows, dtype=cp.int32)
			bc_cols_cp = cp.asarray(bc_cols, dtype=cp.int32)
			bc_vals_cp = cp.asarray(bc_vals, dtype=cp.float64)

			self._rows = cp.concatenate([self._rows, bc_rows_cp])
			self._cols = cp.concatenate([self._cols, bc_cols_cp])
			self._vals = cp.concatenate([self._vals, bc_vals_cp])

		# --- FIX 2: DIRICHLET ELIMINATION (Outlet: COO filtering and High Penalty) ---
		PENALTY_FACTOR = 1.0e12 # ALIGNED with CPU penalty strength

		if self.verbose:
			print(f"  Applying {exit_nodes_cpu.size} Dirichlet nodes (u=0) via COO Identity/Penalty...")

		# 1. COO Filtering: Remove existing contributions to Dirichlet rows/cols
		mask = ~(
			cp.isin(self._rows, exit_nodes_gpu) |
			cp.isin(self._cols, exit_nodes_gpu)
		)

		rows_cp = self._rows[mask]
		cols_cp = self._cols[mask]
		vals_cp = self._vals[mask]

		# 2. Add Identity/Penalty rows for Dirichlet nodes
		rows_cp = cp.concatenate([rows_cp, exit_nodes_gpu])
		cols_cp = cp.concatenate([cols_cp, exit_nodes_gpu])
		# FIX: Apply the high penalty factor
		vals_cp = cp.concatenate([vals_cp, cp.ones(exit_nodes_gpu.size) * PENALTY_FACTOR])

		# Set RHS to 0.0 (target u=0.0)
		self.fg[exit_nodes_gpu] = 0.0

		# --- FIX 3: Fix unused nodes (orphaned nodes not in any element) ---
		used_nodes_set = set(self.quad8.flatten().tolist())
		unused_nodes_cpu = np.array([
			n for n in range(self.Nnds) if n not in used_nodes_set
		], dtype=np.int32)

		unused_count = unused_nodes_cpu.size

		if unused_count > 0:
			unused_nodes_gpu = cp.asarray(unused_nodes_cpu)
			
			# Append rows/cols for unused nodes
			rows_cp = cp.concatenate([rows_cp, unused_nodes_gpu])
			cols_cp = cp.concatenate([cols_cp, unused_nodes_gpu])
			
			# Append diagonal penalty values
			vals_cp = cp.concatenate([vals_cp, cp.ones(unused_nodes_gpu.size) * PENALTY_FACTOR]) 
			
			# Set RHS to 0.0 for unused nodes
			self.fg[unused_nodes_gpu] = 0.0

		if self.verbose: # The printout you saw was likely this one
			print(f"  Fixed {unused_count} unused nodes via Penalty Method")


		# Final sparse matrix
		self.Kg = cpsparse.coo_matrix(
			(vals_cp, (rows_cp, cols_cp)),
			shape=(self.Nnds, self.Nnds)
		).tocsr()






	# ------
	# Solve (CPU GMRES for stability/comparison)
	# ------
	def solve(self):
		if self.verbose:
			print("Converting system to CPU CSR for GMRES...")

		# GPU → CPU transfer
		K_cpu = csr_matrix(self.Kg.get())
		f_cpu = self.fg.get()

		initial_residual_norm = np.linalg.norm(f_cpu)

		if self.verbose:
			print(f"  Initial L2 residual norm (b): {initial_residual_norm:.3e}")

		# -----------------------------
		# Preconditioner (ILU/Jacobi)
		# -----------------------------
		M = None
		precond_name = "Jacobi"

		try:
			ilu = spilu(K_cpu.tocsc(), drop_tol=1e-6, fill_factor=50)
			def precond_ilu(v): return ilu.solve(v)
			M = LinearOperator(K_cpu.shape, precond_ilu)
			precond_name = "ILU"
		except Exception:
			if self.verbose:
				print("  ILU failed, falling back to Jacobi...")
			diag = K_cpu.diagonal().astype(np.float64, copy=True)
			diag[np.abs(diag) < 1e-12] = 1.0
			M_inv = np.reciprocal(diag)
			def precond_jacobi(v): return M_inv * v
			M = LinearOperator(K_cpu.shape, precond_jacobi)

		if self.verbose:
			print(f"Solving linear system (PCGMRES with {precond_name})...")

		# -----------------------------
		# GMRES solve
		# -----------------------------
		self.monitor = IterativeSolverMonitor(every=50, maxiter=self.maxiter)

		u_cpu, self.solve_info = gmres(
			K_cpu,
			f_cpu,
			rtol=self.rtol,
			atol=self.atol,
			maxiter=self.maxiter,
			M=M,
			callback=self.monitor,
			callback_type="legacy"
		)

		self.converged = (self.solve_info == 0)
		self.iterations = self.monitor.it
		
		final_res = np.linalg.norm(f_cpu - K_cpu @ u_cpu)
		
		# --- FIX 4: Explicitly cast NumPy float to standard Python float ---
		self.residuals = [float(initial_residual_norm), float(final_res)]
		# ------------------------------------------------------------------
		
		# Nullspace fix and transfer back to GPU
		u_cpu -= u_cpu.mean()
		self.u = cp.asarray(u_cpu)

		if self.verbose:
			if self.converged:
				print(f"✓ GMRES converged in {self.iterations} iterations")
			else:
				print(f"✗ GMRES did not converge (info={self.solve_info})")

	# ----------------
	# Post-processing 
	# ----------------
	def _genip2dq_4_gpu(self):
		"""Helper function to generate NIP=4 Gauss points (CuPy arrays)."""
		G = cp.sqrt(1.0 / 3.0)
		xp = G * cp.array([
			[-1.0, -1.0], [ 1.0, -1.0], [ 1.0,  1.0], [-1.0, 1.0]
		], dtype=cp.float64)
		return xp

	def compute_derived_fields(self):
		if self.verbose:
			print("Computing velocity field (Mass Post-Processing RawKernel)...")
		
		# 1. Prepare arrays
		quad8_cp = cp.asarray(self.quad8, dtype=cp.int32)
		self.abs_vel = cp.zeros(self.Nels, dtype=cp.float64)
		self.vel = cp.zeros((self.Nels, 2), dtype=cp.float64)
		
		# 2. Setup Integration Points
		xp_4 = self._genip2dq_4_gpu()
		
		# 3. Launch Kernel
		kernel = cp.RawKernel(QUAD8_POSTPROCESS_KERNEL_SOURCE, 'quad8_postprocess_kernel')
		threads_per_block = 128
		blocks = (self.Nels + threads_per_block - 1) // threads_per_block

		kernel((blocks,), (threads_per_block,), (
			self.u, self.x, self.y, quad8_cp, 
			xp_4,
			self.Nels,
			self.p0, self.rho,
			self.abs_vel, self.vel 
		))

		# 4. Final pressure calculation (Vectorized on GPU)
		self.pressure = self.p0 - self.rho * self.abs_vel**2
		
		cp.cuda.Stream.null.synchronize()


	# ----
	# Run 
	# ----
	def run(self):
		total_start = time.perf_counter()
		self.timing_metrics = {}

		self._time_step("load_mesh", self.load_mesh)
		self._time_step("assemble_system", self.assemble_system)
		self._time_step("apply_bc", self.apply_boundary_conditions)
		self._time_step("solve_system", self.solve)
		self._time_step("compute_derived", self.compute_derived_fields)

		self.timing_metrics["total_workflow"] = time.perf_counter() - total_start
		self.timing_metrics["total_program_time"] = (
			time.perf_counter() - self.program_start_time
		)

		if self.verbose:
			print("\n✓ GPU simulation complete")

		# ----------------
		# Visualization and Export (using CPU data)
		# ----------------
		output_dir = self.mesh_file.parent.parent / "output/figures" # Fix 5 resolved here
		export_path = self.mesh_file.parent.parent / f"output/Results_quad8_{self.implementation_name}.xlsx"
		
		if self.verbose:
			print("Generating visualizations...")

		output_files = generate_all_visualizations(
			self.x_cpu,
			self.y_cpu,
			self.quad8,
			cp.asnumpy(self.u),
			output_dir,
			implementation_name=self.implementation_name
		)

		export_results_to_excel(
			export_path,
			self.x_cpu, self.y_cpu, self.quad8,
			cp.asnumpy(self.u), 
			cp.asnumpy(self.vel), 
			cp.asnumpy(self.abs_vel), 
			cp.asnumpy(self.pressure),
			implementation_name=self.implementation_name
		)


		if self.verbose:
			for k, v in output_files.items():
				print(f"  Saved {k}: {v.name}")

		timing = self.timing_metrics
		if self.verbose:
			print("\nStep-by-Step Timings (seconds):")
			for k, v in timing.items():
				print(f"  {k:<20}: {v:.4f}")
			print(f"  Total Program Wall Time: {timing['total_program_time']:.4f} seconds")

		return {
			"u": cp.asnumpy(self.u),
			"vel": cp.asnumpy(self.vel),
			"abs_vel": cp.asnumpy(self.abs_vel),
			"pressure": cp.asnumpy(self.pressure),
			"converged": self.converged,
			"iterations": self.iterations,
			"timing_metrics": self.timing_metrics,
		}

if __name__ == "__main__":
	
	# Determine PROJECT_ROOT based on the execution context
	PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

	solver = Quad8FEMSolverGPU(
		mesh_file=PROJECT_ROOT / "data/input/converted_mesh_v5.xlsx",
		implementation_name="GPU-Optimized",
		maxiter=100,
		verbose=True
	)

	results = solver.run()

	print("\nResults summary:")
	print(f"  Converged: {results['converged']}")
	print(f"  Iterations: {results['iterations']}")
	print(f"  u range: [{results['u'].min():.6e}, {results['u'].max():.6e}]")

	timing = results["timing_metrics"]
	print("\nTiming breakdown (seconds):")
	for k, v in timing.items():
		print(f"  {k:<20}: {v:.4f}")
