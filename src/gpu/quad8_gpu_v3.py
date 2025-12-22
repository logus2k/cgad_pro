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
from scipy.sparse.linalg import LinearOperator, gmres, spilu, aslinearoperator
from scipy.sparse.linalg import spilu # Added spilu back for ILU preconditioner

# --- Project paths (Adjust as needed) ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHARED_DIR = HERE.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

# Import shared utilities
from visualization_utils_gpu import generate_all_visualizations
from export_utils_v2 import export_results
from robin_quadr_gpu import Robin_quadr 
from genip2dq_gpu import Genip2DQ
from shape_n_der8_gpu import Shape_N_Der8
from progress_callback import ProgressCallback


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
    double vel_x_sum = 0.0;  // ✅ ADD: Accumulate vx
    double vel_y_sum = 0.0;  // ✅ ADD: Accumulate vy
    
    // --- 2. Integration Loop (NIP=4) ---
    for (int ip = 0; ip < N_IP; ++ip) {
        double csi = xp_in[ip * 2 + 0];
        double eta = xp_in[ip * 2 + 1];

        // --- 3. B matrix calculation ---
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

        // --- 5. Accumulate velocity components ---
        // Velocity is negative gradient: v = -grad(u)
        vel_x_sum += -grad[0];  // ✅ FIX: Accumulate vx
        vel_y_sum += -grad[1];  // ✅ FIX: Accumulate vy
        
        // Velocity magnitude at this Gauss point
        double vel_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
        v_ip_sum += vel_norm;
    }
    
    // --- 6. Output (Average over all 4 Gauss points) ---
    vel_out[e * 2 + 0] = vel_x_sum / N_IP;  // ✅ FIX: Average vx
    vel_out[e * 2 + 1] = vel_y_sum / N_IP;  // ✅ FIX: Average vy
    abs_vel_out[e] = v_ip_sum / N_IP;        // Already correct
}
"""

# =========================================================================
# GPU SOLVER MONITOR (Progress Callback for CuPy solvers)
# =========================================================================
class GPUSolverMonitor:
	"""Monitor for GPU iterative solvers with actual residual computation"""
	def __init__(self, A, b, every: int = 50, maxiter: int = 50000, verbose: bool = True, progress_callback=None):
		self.A = A  # System matrix (equilibrated)
		self.b = b  # Right-hand side (equilibrated)
		self.every = every
		self.maxiter = maxiter
		self.verbose = verbose
		self.it = 0
		self.t_start = time.perf_counter()
		self.b_norm = float(cp.linalg.norm(b).get())
		self.progress_callback = progress_callback 
	
	def __call__(self, xk):
		"""Callback function called by CuPy solvers"""
		self.it += 1
		
		# ✅ ALWAYS check for incremental updates (independent of logging)
		if self.progress_callback is not None:
			if self.it == 1 or (self.it % 100 == 0):
				print(f"[DEBUG GPU] Sending solution increment at iteration {self.it}")
				# Convert CuPy array to CPU for transmission
				solution_cpu = xk.get() if hasattr(xk, 'get') else xk
				self.progress_callback.on_solution_increment(
					iteration=self.it,
					solution=solution_cpu
				)
		
		# Log progress at specified intervals
		if self.it % self.every == 0 or self.it == 1:
			# Compute actual residual: r = b - A*x
			r = self.b - self.A @ xk
			res_norm = float(cp.linalg.norm(r).get())
			rel_res = res_norm / self.b_norm if self.b_norm > 0 else 0.0
			
			elapsed = time.perf_counter() - self.t_start
			pct = 100.0 * self.it / self.maxiter
			
			# ETR calculation
			etr_sec = 0.0
			if self.it > 0:
				iters_left = self.maxiter - self.it
				time_per_iter = elapsed / self.it
				etr_sec = iters_left * time_per_iter
				etr_min = int(etr_sec // 60)
				etr_sec_rem = int(etr_sec % 60)
				etr_str = f"{etr_min:02d}m:{etr_sec_rem:02d}s"
			else:
				etr_str = "--:--"
			
			if self.verbose:
				print(f"  Iter {self.it}/{self.maxiter} ({pct:.1f}%), ||r|| = {res_norm:.3e}, rel = {rel_res:.3e}, ETR: {etr_str}")
			
			# Invoke callback for metrics
			if self.progress_callback is not None:
				self.progress_callback.on_iteration(
					iteration=self.it,
					max_iterations=self.maxiter,
					residual=res_norm,
					relative_residual=rel_res,
					elapsed_time=elapsed,
					etr_seconds=etr_sec
				)	
	
	def reset(self):
		"""Reset monitor for a new solve attempt"""
		self.it = 0
		self.t_start = time.perf_counter()


class IterativeSolverMonitor:
	"""Callback monitor for GMRES solver iterations (NumPy on CPU)."""
	
	def __init__(self, every: int = 50, maxiter: int = 50000):
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
		maxiter: int = 50000,
		cg_print_every: int = 50,
		bc_tolerance: float = 1e-9,
		implementation_name: str = "GPU",
		verbose: bool = True,
		progress_callback=None,
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
		
		# Mesh data (initialized in load_mesh)
		self.Nnds: int = 0
		self.Nels: int = 0
		
		# Solver diagnostics
		self.converged: bool = False
		self.iterations: int = 0
		self.residuals: List[float] = [] # FIX 4: Explicitly type hint as List[float]

		self.progress_callback = progress_callback

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
		"""
		Load mesh from file. Supports multiple formats:
		- .xlsx (Excel) - Slow but human-readable (14s for 196K nodes)
		- .npz (NumPy) - Fast binary format (0.3s for 196K nodes)
		- .h5 (HDF5) - Fastest, industry standard (0.2s for 196K nodes)
		
		Format is auto-detected from file extension.
		"""
		if self.verbose:
			print(f"Loading mesh from {self.mesh_file.name}...")
		
		suffix = self.mesh_file.suffix.lower()
		
		if suffix == '.xlsx':
			# Excel format (slow but original)
			coord = pd.read_excel(self.mesh_file, sheet_name="coord")
			conec = pd.read_excel(self.mesh_file, sheet_name="conec")
			
			self.x_cpu = coord["X"].to_numpy(dtype=np.float64) / 1000.0
			self.y_cpu = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
			self.quad8 = (conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1).copy()
			
		elif suffix == '.npz':
			# NumPy compressed format (fast)
			data = np.load(self.mesh_file)
			self.x_cpu = data['x']
			self.y_cpu = data['y']
			self.quad8 = data['quad8'].copy()
			
		elif suffix == '.h5' or suffix == '.hdf5':
			# HDF5 format (fastest)
			try:
				import h5py  # type: ignore
			except ImportError:
				raise ImportError(
					"HDF5 support requires h5py. Install with: pip install h5py"
				)
			
			with h5py.File(self.mesh_file, 'r') as f:
				# Type annotations to help Pylance understand h5py datasets
				x_dataset = f['x']  # type: ignore
				y_dataset = f['y']  # type: ignore
				quad8_dataset = f['quad8']  # type: ignore
				
				# Read into NumPy arrays
				self.x_cpu = np.array(x_dataset, dtype=np.float64)
				self.y_cpu = np.array(y_dataset, dtype=np.float64)
				self.quad8 = np.array(quad8_dataset, dtype=np.int32)
		else:
			raise ValueError(
				f"Unsupported mesh format: {suffix}\n"
				f"Supported formats: .xlsx, .npz, .h5, .hdf5\n"
				f"Use convert_mesh.py to convert Excel to binary format."
			)

		self.Nnds = self.x_cpu.size
		self.Nels = self.quad8.shape[0]

		# Transfer to GPU
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
		
		# CRITICAL FIX: Define the high penalty factor (REQUIRED for convergence)
		PENALTY_FACTOR = 1.0e12 

		# Dirichlet BC on maximum-x boundary (outlet)
		exit_nodes = np.where(self.x_cpu == x_max)[0]
		exit_nodes_gpu = cp.asarray(exit_nodes, dtype=cp.int32)
		
		# Robin BC on minimum-x boundary (inlet)
		boundary_nodes = set(
			np.where(np.abs(self.x_cpu - x_min) < self.bc_tolerance)[0].tolist()
		)

		# ---------------------------------
		# Robin BCs (temporary CPU buffers)
		# ---------------------------------
		bc_rows = []
		bc_cols = []
		bc_vals = []
        
		if self.verbose:
			print(f"  Applying {exit_nodes_gpu.size} Dirichlet nodes (u=0) and Robin edges (Inlet)...")


		for e in range(self.Nels):
			n = self.quad8[e]
			edges = [
				(n[0], n[4], n[1]),
				(n[1], n[5], n[2]),
				(n[2], n[6], n[3]),
				(n[3], n[7], n[0]),
			]
			for (n1, n2, n3) in edges:
				if n1 in boundary_nodes and n2 in boundary_nodes and n3 in boundary_nodes:
					He, Pe = Robin_quadr(
						self.x[n1], self.y[n1],
						self.x[n2], self.y[n2],
						self.x[n3], self.y[n3],
						p=0.0,
						gama=self.gamma
					)
					for i, ni in enumerate((n1, n2, n3)):
						self.fg[ni] += Pe[i]
						for j, nj in enumerate((n1, n2, n3)):
							bc_rows.append(ni)
							bc_cols.append(nj)
							bc_vals.append(He[i, j])

		# ---------------------------------
		# Merge bulk COO with BC COO
		# ---------------------------------
		if bc_rows:
			bc_rows_cp = cp.asarray(bc_rows, dtype=cp.int32)
			bc_cols_cp = cp.asarray(bc_cols, dtype=cp.int32)
			bc_vals_cp = cp.asarray(bc_vals, dtype=cp.float64)

			self._rows = cp.concatenate([self._rows, bc_rows_cp])
			self._cols = cp.concatenate([self._cols, bc_cols_cp])
			self._vals = cp.concatenate([self._vals, bc_vals_cp])

		# ---------------------------------
		# Dirichlet elimination (Outlet)
		# ---------------------------------
		mask = ~(
			cp.isin(self._rows, exit_nodes_gpu) |
			cp.isin(self._cols, exit_nodes_gpu)
		)

		rows_cp = self._rows[mask]
		cols_cp = self._cols[mask]
		vals_cp = self._vals[mask]

		# Identity rows for Dirichlet nodes
		rows_cp = cp.concatenate([rows_cp, exit_nodes_gpu])
		cols_cp = cp.concatenate([cols_cp, exit_nodes_gpu])
        
        # CRITICAL FIX: Apply the high penalty factor (1.0e12)
		vals_cp = cp.concatenate([vals_cp, cp.ones(exit_nodes_gpu.size) * PENALTY_FACTOR])

		self.fg[exit_nodes_gpu] = 0.0
        
		# ---------------------------------
		# Fix unused nodes (nodes not in any element)
		# ---------------------------------
		# Build temporary matrix to find which nodes are used
		temp_coo = cpsparse.coo_matrix(
			(vals_cp, (rows_cp, cols_cp)),
			shape=(self.Nnds, self.Nnds)
		)
		temp_csr = temp_coo.tocsr()
		
		# Find nodes with zero diagonal (unused nodes)
		diag_temp = temp_csr.diagonal()
		unused_mask = cp.abs(diag_temp) < 1e-14
		unused_nodes = cp.where(unused_mask)[0]
		
		if unused_nodes.size > 0:
			if self.verbose:
				print(f"  Fixing {unused_nodes.size} unused nodes via penalty method...")
			
			# Add penalty diagonal entries for unused nodes
			rows_cp = cp.concatenate([rows_cp, unused_nodes])
			cols_cp = cp.concatenate([cols_cp, unused_nodes])
			vals_cp = cp.concatenate([vals_cp, cp.ones(unused_nodes.size) * PENALTY_FACTOR])
			
			# Set RHS to zero for unused nodes
			self.fg[unused_nodes] = 0.0

		# ---------------------------------
		# Final sparse matrix
		# ---------------------------------
		self.Kg = cpsparse.coo_matrix(
			(vals_cp, (rows_cp, cols_cp)),
			shape=(self.Nnds, self.Nnds)
		).tocsr()


	def _print_system_diagnostics(self):
		"""Print comprehensive diagnostics about the system matrix and RHS"""
		if not self.verbose:
			return
		
		print("\n" + "="*70)
		print("SYSTEM DIAGNOSTICS")
		print("="*70)
		
		# Matrix properties
		diag = self.Kg.diagonal()
		nnz = self.Kg.nnz
		
		diag_min = cp.abs(diag).min().get()
		diag_max = cp.abs(diag).max().get()
		diag_mean = cp.abs(diag).mean().get()
		
		# Count zero/near-zero diagonals
		zero_diag_count = cp.sum(cp.abs(diag) < 1e-14).get()
		
		print(f"Matrix Properties:")
		print(f"  Shape: {self.Kg.shape}")
		print(f"  NNZ: {nnz} ({100*nnz/(self.Nnds**2):.4f}% dense)")
		print(f"  Diagonal min: {diag_min:.3e}")
		print(f"  Diagonal max: {diag_max:.3e}")
		print(f"  Diagonal mean: {diag_mean:.3e}")
		print(f"  Condition estimate (diag): {diag_max/diag_min:.3e}")
		print(f"  Zero/near-zero diagonals: {zero_diag_count}")
		
		# RHS properties
		fg_norm = cp.linalg.norm(self.fg).get()
		fg_min = cp.abs(self.fg).min().get()
		fg_max = cp.abs(self.fg).max().get()
		fg_nonzero = cp.count_nonzero(self.fg).get()
		
		print(f"\nRHS Properties:")
		print(f"  L2 norm: {fg_norm:.3e}")
		print(f"  Min (abs): {fg_min:.3e}")
		print(f"  Max (abs): {fg_max:.3e}")
		print(f"  Non-zero entries: {fg_nonzero}/{self.Nnds}")
		
		# Symmetry check (for sparse matrices, check data arrays)
		Kg_T = self.Kg.T.tocsr()
		diff = self.Kg - Kg_T
		sym_diff = cp.abs(diff.data).sum().get() if diff.nnz > 0 else 0.0
		
		print(f"\nSymmetry:")
		print(f"  ||Kg - Kg^T||_1: {sym_diff:.3e}")
		
		if sym_diff < 1e-10:
			print(f"  ✓ Matrix is symmetric")
		else:
			print(f"  ✗ Matrix is NOT symmetric (may affect solver choice)")
		
		print("="*70 + "\n")


	# -------------
	# Solver
	# -------------
	def solve(self):
		"""
		GPU-accelerated iterative solver with diagnostics.
		Tries CG first (for symmetric systems), falls back to GMRES if needed.
		"""
		if self.verbose:
			print("Preparing GPU solver with diagnostics...")
		
		t0_convert = time.perf_counter()
		
		# --- System Diagnostics ---
		self._print_system_diagnostics()
		
		self.timing_metrics["convert_to_cpu"] = time.perf_counter() - t0_convert

		# --- Solver Setup ---
		MAXITER = 50000 
		TOL = 1e-8
		
		self.converged = False
		self.iterations = MAXITER
		
		# --- Diagonal Equilibration (improves conditioning) ---
		if self.verbose:
			print("Applying diagonal equilibration...")
		
		diag = self.Kg.diagonal()
		diag_safe = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
		D_inv_sqrt = 1.0 / cp.sqrt(cp.abs(diag_safe))
		
		# Equilibrated system: D^(-1/2) * Kg * D^(-1/2) * (D^(1/2) * u) = D^(-1/2) * fg
		Kg_eq = self.Kg.multiply(D_inv_sqrt[:, None]).multiply(D_inv_sqrt[None, :])
		fg_eq = self.fg * D_inv_sqrt
		
		# --- GPU Jacobi Preconditioner ---
		diag_eq = Kg_eq.diagonal()
		diag_eq_safe = cp.where(cp.abs(diag_eq) < 1e-14, 1.0, diag_eq)
		
		def jacobi_precond(x):
			return x / diag_eq_safe
		
		M = cpsplg.LinearOperator(
			shape=Kg_eq.shape,
			matvec=jacobi_precond,
			dtype=cp.float64
		)
		
		# --- Try CG first (optimal for symmetric positive definite) ---
		if self.verbose:
			print(f"Solving with GPU CG (tol={TOL:.1e}, maxiter={MAXITER})...")
		
		monitor = GPUSolverMonitor(Kg_eq, fg_eq, every=50, maxiter=MAXITER, verbose=self.verbose, progress_callback=self.progress_callback)
		
		t0_solve = time.perf_counter()
		
		try:
			u_eq, info = cpsplg.cg(
				Kg_eq,
				fg_eq,
				x0=cp.zeros_like(fg_eq),
				M=M,
				tol=TOL,
				maxiter=MAXITER,
				callback=monitor
			)
			
			solver_name = "CG"
			
		except Exception as e:
			if self.verbose:
				print(f"  CG failed: {e}")
				print("  Falling back to GMRES...")
			
			monitor.reset()
			
			try:
				u_eq, info = cpsplg.gmres(
					Kg_eq,
					fg_eq,
					x0=cp.zeros_like(fg_eq),
					M=M,
					tol=TOL,
					maxiter=MAXITER,
					restart=50,
					callback=monitor
				)
				solver_name = "GMRES"
				
			except Exception as e2:
				if self.verbose:
					print(f"  GMRES also failed: {e2}")
				raise RuntimeError("Both CG and GMRES failed") from e2
		
		t1_solve = time.perf_counter()
		
		# Undo equilibration
		self.u = u_eq * D_inv_sqrt
		
		self.timing_metrics["solve_system"] = t1_solve - t0_solve
		self.iterations = monitor.it
		
		# --- Convergence Check ---
		if info == 0:
			# Verify true residual
			residual = cp.linalg.norm(self.Kg @ self.u - self.fg)
			rel_residual = residual / cp.linalg.norm(self.fg)
			
			if self.verbose:
				print(f"\n✓ {solver_name} converged in {self.iterations} iterations")
				print(f"  True residual norm:     {residual.get():.3e}")
				print(f"  True relative residual: {rel_residual.get():.3e}")
				print(f"  Solver time: {t1_solve - t0_solve:.4f}s")
			
			self.converged = True
		else:
			if self.verbose:
				print(f"\n✗ {solver_name} did not converge (info={info})")
			self.converged = False

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

	# ----------------
	# Post-processing 
	# ----------------
	def compute_derived_fields(self):
		"""
		Compute velocity and pressure fields using GPU RawKernel.
		Replaces the slow Python loop with mass-parallel GPU computation.
		"""
		if self.verbose:
			print("Computing derived fields...")

		# Allocate output arrays on GPU
		self.vel = cp.zeros((self.Nels, 2), dtype=cp.float64)
		self.abs_vel = cp.zeros(self.Nels, dtype=cp.float64)

		# Generate 4-point Gauss integration points for post-processing
		xp = self._genip2dq_4_gpu()
		
		# Prepare connectivity array
		quad8_cp = cp.asarray(self.quad8, dtype=cp.int32)

		# Launch GPU kernel for velocity computation
		kernel = cp.RawKernel(QUAD8_POSTPROCESS_KERNEL_SOURCE, 'quad8_postprocess_kernel')
		threads_per_block = 128
		blocks = (self.Nels + threads_per_block - 1) // threads_per_block

		kernel((blocks,), (threads_per_block,), (
			self.u,           # Solution vector
			self.x, self.y,   # Nodal coordinates
			quad8_cp,         # Connectivity
			xp,               # Integration points
			self.Nels,
			self.p0,          # Reference pressure
			self.rho,         # Density
			self.abs_vel,     # Output: velocity magnitude
			self.vel          # Output: velocity vector
		))
		
		cp.cuda.Stream.null.synchronize()

		# Compute pressure from Bernoulli equation
		self.pressure = self.p0 - self.rho * self.abs_vel**2


		# ----
		# Run 
		# ----
	def run(
		self,
		output_dir: Optional[Path | str] = None,
		export_file: Optional[Path | str] = None
	) -> Dict[str, Any]:
		"""Run complete FEM simulation workflow with progress callbacks."""
		
		# --- Start Timer for the Core Workflow ---
		total_workflow_start = time.perf_counter()
		self.timing_metrics = {}
		
		# --- STAGE 1: Load Mesh ---
		if self.progress_callback:
			self.progress_callback.on_stage_start(stage='load_mesh')
		
		self._time_step('load_mesh', self.load_mesh)
		
		if self.progress_callback:
			self.progress_callback.on_stage_complete(
				stage='load_mesh',
				duration=self.timing_metrics['load_mesh']
			)
			# Emit mesh metadata (convert CuPy arrays to CPU for serialization)
			self.progress_callback.on_mesh_loaded(
				nodes=self.Nnds,
				elements=self.Nels,
				coordinates={'x': self.x.get().tolist(), 'y': self.y.get().tolist()} if self.Nnds < 50000 else None,
				connectivity=self.quad8.tolist() if self.Nels < 10000 else None
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
		
		if self.verbose:
			print(f"\nSolution statistics:")
			print(f"  u range: [{self.u.min():.6e}, {self.u.max():.6e}]")
			print(f"  u mean:  {self.u.mean():.6e}")
			print(f"  u std:   {self.u.std():.6e}")
		
		# --- STAGE 6: Visualization (Optional) ---
		if output_dir is not None:
			if self.progress_callback:
				self.progress_callback.on_stage_start(stage='visualize')
			
			# Optional visualization code...
			
			if self.progress_callback:
				self.progress_callback.on_stage_complete(
					stage='visualize',
					duration=self.timing_metrics.get('visualize', 0.0)
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
		
		# Add derived fields if computed
		if hasattr(self, 'vel'):
			results['vel'] = self.vel
		if hasattr(self, 'abs_vel'):
			results['abs_vel'] = self.abs_vel
		if hasattr(self, 'pressure'):
			results['pressure'] = self.pressure
		
		# --- Emit Final Completion Event ---
		if self.progress_callback:
			self.progress_callback.on_solve_complete(
				converged=self.converged,
				iterations=self.iterations,
				timing_metrics=self.timing_metrics,
				solution_stats=results['solution_stats'],
				mesh_info=results['mesh_info']
			)
		
		# --- Console Output (if verbose) ---
		if self.verbose:
			print("\n✓ GPU simulation complete")
			
			print("\nStep-by-Step Timings (seconds):")
			for k, v in self.timing_metrics.items():
				print(f"  {k:<20}: {v:.4f}")
			
			print(f"  Total Program Wall Time: {total_program_time:.4f} seconds")
			
			print(f"\nResults summary:")
			print(f"  Converged: {results['converged']}")
			print(f"  Iterations: {results['iterations']}")
			print(f"  u range: [{results['solution_stats']['u_range'][0]:.6e}, "
				f"{results['solution_stats']['u_range'][1]:.6e}]")
		
		return results

if __name__ == "__main__":
	
	# Determine PROJECT_ROOT based on the execution context
	PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

	solver = Quad8FEMSolverGPU(
		mesh_file=PROJECT_ROOT / "src/app/client/mesh/h5/tube1_1_7m.h5",
		implementation_name="GPU",
		maxiter=50000,
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
