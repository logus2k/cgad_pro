"""
QUAD8 FEM — GPU baseline solver (CuPy)

Aligned with CPU solver structure:
- Same workflow stages
- Same BC logic
- GPU assembly, CPU solve (intentional)
- Ready for later GPU optimization
"""

import sys
from pathlib import Path
import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from elem_quad8_gpu import Elem_Quad8
from genip2dq_gpu import Genip2DQ
from shape_n_der8_gpu import Shape_N_Der8
from robin_quadr_gpu import Robin_quadr

# Project paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHARED_DIR = HERE.parent / "shared"

# Add shared to path
sys.path.insert(0, str(SHARED_DIR))

from visualization_utils import generate_all_visualizations


class IterativeSolverMonitor:
	"""
	Callback monitor for CG/GMRES iterations.

	NOTE:
	- This is IDENTICAL in behavior to the CPU version
	- Used only for reporting, never for convergence
	"""

	def __init__(self, every: int = 50, maxiter: int = 5000):
		self.it = 0
		self.every = every
		self.maxiter = maxiter
		self.start_time = time.perf_counter()

	def __call__(self, rk):
		"""
		GMRES legacy callback.

		Parameters
		----------
		rk : ndarray
			Preconditioned residual (M⁻¹ r). Diagnostic only.
		"""
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
			f"||M⁻¹ r|| = {res_norm:.3e}, "
			f"ETR: {fmt(etr)}"
		)


class Quad8FEMSolverGPU:
	"""
	QUAD8 FEM Solver — GPU baseline (CuPy assembly, CPU solve)

	This class is structurally aligned with the CPU solver:
	- Same lifecycle
	- Same method names
	- Same timing metrics
	- Same run() contract
	"""

	def __init__(
		self,
		mesh_file,
		p0=101328.8281,
		rho=0.6125,
		gamma=2.5,
		rtol=1e-10,
		atol=1e-16,
		maxiter=5000,
		cg_print_every=50,
		bc_tolerance=1e-9,
		implementation_name="GPU",
		verbose=True,
	):
		self.mesh_file = Path(mesh_file)
		self.p0 = p0
		self.rho = rho
		self.gamma = gamma

		self.rtol = rtol
		self.atol = atol
		self.maxiter = maxiter
		self.cg_print_every = cg_print_every

		self.bc_tolerance = bc_tolerance
		self.implementation_name = implementation_name
		self.verbose = verbose

		self.program_start_time = time.perf_counter()
		self.timing_metrics = {}

	# ---------------
	# Timing utility 
	# ---------------
	def _time_step(self, name, fn):
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
			print(f"Loading mesh from {self.mesh_file.name}...")

		coord = pd.read_excel(self.mesh_file, sheet_name="coord")
		conec = pd.read_excel(self.mesh_file, sheet_name="conec")

		self.x_cpu = coord["X"].to_numpy(dtype=np.float64) / 1000.0
		self.y_cpu = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
		self.quad8 = conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1

		self.Nnds = self.x_cpu.size
		self.Nels = self.quad8.shape[0]

		self.x = cp.asarray(self.x_cpu)
		self.y = cp.asarray(self.y_cpu)

		if self.verbose:
			print(f"  Loaded: {self.Nnds} nodes, {self.Nels} Quad-8 elements")

	# ---------
	# Assembly 
	# ---------
	def assemble_system(self):
		if self.verbose:
			print("Assembling global system (GPU, preallocated COO)...")

		self.fg = cp.zeros(self.Nnds, dtype=cp.float64)

		nnz_per_elem = 64
		nnz_total = self.Nels * nnz_per_elem

		rows = cp.empty(nnz_total, dtype=cp.int32)
		cols = cp.empty(nnz_total, dtype=cp.int32)
		vals = cp.empty(nnz_total, dtype=cp.float64)

		ptr = 0

		for e in range(self.Nels):
			edofs = self.quad8[e]
			XN = cp.column_stack((self.x[edofs], self.y[edofs]))

			Ke, fe = Elem_Quad8(XN, fL=0.0)
			self.fg[edofs] += fe

			for i in range(8):
				ri = edofs[i]
				for j in range(8):
					rows[ptr] = ri
					cols[ptr] = edofs[j]
					vals[ptr] = Ke[i, j]
					ptr += 1

		# Store GPU-side COO buffers
		self._rows = rows
		self._cols = cols
		self._vals = vals

	# --------------------
	# Boundary conditions 
	# --------------------
	def apply_boundary_conditions(self):
		if self.verbose:
			print("Applying boundary conditions...")

		x_min = self.x_cpu.min()
		x_max = self.x_cpu.max()

		# exit_nodes = np.where(np.abs(self.x_cpu - x_max) < self.bc_tolerance)[0]
		exit_nodes = np.where(self.x_cpu == x_max)[0]
		boundary_nodes = set(
			np.where(np.abs(self.x_cpu - x_min) < self.bc_tolerance)[0].tolist()
		)

		# ---------------------------------
		# Robin BCs (temporary CPU buffers)
		# ---------------------------------
		bc_rows = []
		bc_cols = []
		bc_vals = []

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
		# Dirichlet elimination
		# ---------------------------------
		exit_nodes_gpu = cp.asarray(exit_nodes, dtype=cp.int32)

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
		vals_cp = cp.concatenate([vals_cp, cp.ones(exit_nodes_gpu.size)])

		self.fg[exit_nodes_gpu] = 0.0

		# ---------------------------------
		# Final sparse matrix
		# ---------------------------------
		self.Kg = cpsparse.coo_matrix(
			(vals_cp, (rows_cp, cols_cp)),
			shape=(self.Nnds, self.Nnds)
		).tocsr()



	# ------
	# Solve
	# ------
	def solve(self):
		if self.verbose:
			print("Converting system to CPU CSR for GMRES...")

		# GPU → CPU
		K_cpu = csr_matrix(self.Kg.get())
		f_cpu = self.fg.get()

		diag_values = K_cpu.diagonal()
		if self.verbose:
			print(f"  Kg Diagonal Min: {diag_values.min():.3e}")
			print(f"  Kg Diagonal Max: {diag_values.max():.3e}")

		initial_residual_norm = np.linalg.norm(f_cpu)
		if self.verbose:
			print(f"  Initial L2 residual norm (b): {initial_residual_norm:.3e}")

		# -----------------------------
		# Preconditioner (SAME as CPU)
		# -----------------------------
		M = None
		precond_name = "Jacobi"

		try:
			from scipy.sparse.linalg import spilu

			if self.verbose:
				print("Building ILU preconditioner...")

			ilu = spilu(K_cpu.tocsc(), drop_tol=1e-6, fill_factor=50)

			def precond_ilu(v):
				return ilu.solve(v)

			M = LinearOperator(K_cpu.shape, precond_ilu)
			precond_name = "ILU"

		except Exception as e:
			if self.verbose:
				print(f"  ILU failed ({e}), falling back to Jacobi...")

		if M is None:
			if self.verbose:
				print("Building Jacobi preconditioner...")

			diag = K_cpu.diagonal().astype(np.float64, copy=True)
			diag[np.abs(diag) < 1e-12] = 1.0
			M_inv = np.reciprocal(diag)

			def precond_jacobi(v):
				return M_inv * v

			M = LinearOperator(K_cpu.shape, precond_jacobi)

		if self.verbose:
			print(f"Solving linear system (PCGMRES with {precond_name})...")

		# -----------------------------
		# GMRES parameters (SAME)
		# -----------------------------
		TARGET_RTOL = 1e-4
		TARGET_ATOL = 1e-16
		MAXITER = self.maxiter

		# Monitor (reporting only)
		self.monitor = IterativeSolverMonitor(
			every=50,
			maxiter=MAXITER
		)

		# -----------------------------
		# GMRES solve
		# -----------------------------
		u_cpu, self.solve_info = gmres(
			K_cpu,
			f_cpu,
			rtol=TARGET_RTOL,
			atol=TARGET_ATOL,
			maxiter=MAXITER,
			M=M,
			callback=self.monitor,
			callback_type="legacy"
		)

		# Convergence logic (IDENTICAL to CPU)
		self.converged = (self.solve_info == 0)

		# -----------------------------
		# Nullspace fix (presentation only)
		# -----------------------------
		u_cpu -= u_cpu.mean()

		if self.verbose:
			if self.converged:
				print(f"✓ GMRES converged in {self.monitor.it} iterations")
			elif self.solve_info > 0:
				print("✗ GMRES did not converge (max iterations reached)")
			else:
				print(f"✗ GMRES error (info={self.solve_info})")

		# CPU → GPU
		self.u = cp.asarray(u_cpu)




	# ----------------
	# Post-processing 
	# ----------------
	def compute_derived_fields(self):
		if self.verbose:
			print("Computing derived fields...")

		self.vel = cp.zeros((self.Nels, 2))
		self.abs_vel = cp.zeros(self.Nels)

		for e in range(self.Nels):
			edofs = self.quad8[e]
			XN = cp.column_stack((self.x[edofs], self.y[edofs]))

			xp, _ = Genip2DQ(4)
			v_ip = cp.zeros(4)

			for ip in range(4):
				B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
				grad = B.T @ self.u[edofs]
				self.vel[e] = grad
				v_ip[ip] = cp.linalg.norm(grad)

			self.abs_vel[e] = cp.mean(v_ip)

		self.pressure = self.p0 - self.rho * self.abs_vel**2

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
		# Visualization
		# ----------------
		if self.verbose:
			print("Generating visualizations...")

		output_dir = self.mesh_file.parent.parent / "output/figures"

		output_files = generate_all_visualizations(
			self.x_cpu,
			self.y_cpu,
			self.quad8,
			cp.asnumpy(self.u),
			output_dir,
			implementation_name=self.implementation_name
		)

		if self.verbose:
			for k, v in output_files.items():
				print(f"  Saved {k}: {v.name}")

		if self.verbose:
			for k, v in output_files.items():
				print(f"  Saved {k}: {v.name}")			

		return {
			"u": self.u,
			"vel": self.vel,
			"abs_vel": self.abs_vel,
			"pressure": self.pressure,
			"converged": self.converged,
			"iterations": None,  # direct solve
			"timing_metrics": self.timing_metrics,
		}

if __name__ == "__main__":
	from pathlib import Path

	PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

	solver = Quad8FEMSolverGPU(
		mesh_file=PROJECT_ROOT / "data/input/converted_mesh_v5.xlsx",
		implementation_name="GPU",
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
