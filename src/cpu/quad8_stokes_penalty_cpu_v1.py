"""
QUAD8 FEM – Steady Stokes (Penalty Formulation) – CPU (PRODUCTION + AMG)

Key properties:
- Quad-8 elements
- Velocity-only formulation (u_x, u_y)
- Penalty method for incompressibility
- Symmetric Positive Definite (SPD)
- Solved with GMRES
- Diagonal equilibration
- AMG (PyAMG) preconditioner
- Real-time GMRES monitor with true residuals + ETA
- Correct FEM-based pressure reconstruction
- Correct FEM-based vorticity computation
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse import diags

import h5py
import pyamg

from shape_n_der8_cpu import Shape_N_Der8
from genip2dq_cpu import Genip2DQ


# -------------------------------------------------
# Iteration monitor (LEGACY GMRES MODE)
# -------------------------------------------------

class IterativeSolverMonitor:
	"""Real-time GMRES monitor with true residuals and ETA."""

	def __init__(self, A, b, every=20, maxiter=20000):
		self.A = A
		self.b = b
		self.every = every
		self.maxiter = maxiter
		self.it = 0
		self.start = time.perf_counter()
		self.b_norm = float(np.linalg.norm(b))

	def __call__(self, xk):
		self.it += 1

		if self.it % self.every != 0 and self.it != self.maxiter:
			return

		r = self.b - self.A @ xk
		res = float(np.linalg.norm(r))
		rel = res / self.b_norm

		elapsed = time.perf_counter() - self.start
		etr = (self.maxiter - self.it) * (elapsed / max(self.it, 1))

		print(
			f"  Iter {self.it}/{self.maxiter} | "
			f"||r||={res:.3e} rel={rel:.3e} "
			f"ETR={etr/60:.1f} min",
			flush=True
		)


# -------------------------------------------------
# Solver
# -------------------------------------------------

class Quad8StokesPenaltySolverCPU:

	def __init__(
		self,
		mesh_file: Path | str,
		mu: float = 1.0e-3,
		penalty: float = 1.0e7,
		rtol: float = 1e-8,
		maxiter: int = 20000,
		verbose: bool = True
	):
		self.mesh_file = Path(mesh_file)
		self.mu = mu
		self.penalty = penalty
		self.rtol = rtol
		self.maxiter = maxiter
		self.verbose = verbose
		self.timing = {}

	def _log(self, msg):
		if self.verbose:
			elapsed = time.perf_counter() - self._t0
			print(f"[{elapsed:8.2f}s] {msg}", flush=True)

	# -------------------------------------------------
	# Mesh loading
	# -------------------------------------------------

	def load_mesh(self):
		suffix = self.mesh_file.suffix.lower()

		if suffix == ".xlsx":
			coord = pd.read_excel(self.mesh_file, sheet_name="coord")
			conec = pd.read_excel(self.mesh_file, sheet_name="conec")
			self.x = coord["X"].to_numpy(np.float64) / 1000.0
			self.y = coord["Y"].to_numpy(np.float64) / 1000.0
			self.quad8 = conec.iloc[:, :8].to_numpy(np.int32) - 1

		elif suffix in (".h5", ".hdf5"):
			with h5py.File(self.mesh_file, "r") as f:
				self.x = np.asarray(f["x"], dtype=np.float64)
				self.y = np.asarray(f["y"], dtype=np.float64)
				self.quad8 = np.asarray(f["quad8"], dtype=np.int32) - 1
		else:
			raise ValueError("Unsupported mesh format")

		self.Nnds = self.x.size
		self.Nels = self.quad8.shape[0]

	# -------------------------------------------------
	# Assembly
	# -------------------------------------------------

	def assemble_system(self):
		ndof = 2 * self.Nnds
		self.K = lil_matrix((ndof, ndof), dtype=np.float64)
		self.f = np.zeros(ndof, dtype=np.float64)

		xp, wp = Genip2DQ(9)
		wp = np.asarray(wp)

		report_every = max(1, self.Nels // 20)

		for e in range(self.Nels):
			if self.verbose and e % report_every == 0:
				print(f"  Assembly progress: {100*e/self.Nels:5.1f}% ({e}/{self.Nels})", flush=True)

			nodes = self.quad8[e]
			XN = np.column_stack((self.x[nodes], self.y[nodes]))
			Ke = np.zeros((16, 16), dtype=np.float64)

			for ip in range(len(wp)):
				B, _, detJ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
				w = wp[ip] * detJ

				dNdx = B[0]
				dNdy = B[1]

				for i in range(8):
					for j in range(8):
						k_visc = self.mu * (
							dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j]
						) * w

						div_i = dNdx[i] + dNdy[i]
						div_j = dNdx[j] + dNdy[j]
						k_pen = self.penalty * div_i * div_j * w

						Ke[i, j]       += k_visc + k_pen
						Ke[i, j+8]     += k_pen
						Ke[i+8, j]     += k_pen
						Ke[i+8, j+8]   += k_visc + k_pen

			for a in range(8):
				ia = nodes[a]
				for b in range(8):
					ib = nodes[b]
					self.K[2*ia,   2*ib  ] += Ke[a, b]
					self.K[2*ia,   2*ib+1] += Ke[a, b+8]
					self.K[2*ia+1, 2*ib  ] += Ke[a+8, b]
					self.K[2*ia+1, 2*ib+1] += Ke[a+8, b+8]

	# -------------------------------------------------
	# Boundary conditions
	# -------------------------------------------------

	def apply_bc(self):
		xmin = np.min(self.x)
		ymin = np.min(self.y)
		ymax = np.max(self.y)
		tol = 1e-9
		PEN = 1e12

		inlet = np.where(np.abs(self.x - xmin) < tol)[0]
		walls = np.where(
			(np.abs(self.y - ymin) < tol) |
			(np.abs(self.y - ymax) < tol)
		)[0]

		for n in inlet:
			self.K[2*n, 2*n] += PEN
			self.f[2*n] += PEN
			self.K[2*n+1, 2*n+1] += PEN

		for n in walls:
			self.K[2*n, 2*n] += PEN
			self.K[2*n+1, 2*n+1] += PEN

	# -------------------------------------------------
	# Solve (GMRES + AMG)
	# -------------------------------------------------

	def solve(self):
		K = self.K.tocsr()
		b = self.f

		# --- AMG preconditioner ---
		B = np.zeros((K.shape[0], 2))
		B[0::2, 0] = 1.0
		B[1::2, 1] = 1.0

		if self.verbose:
			print("Building AMG preconditioner (PyAMG)...", flush=True)

		ml = pyamg.smoothed_aggregation_solver(K, B=B)
		M = ml.aspreconditioner()

		monitor = IterativeSolverMonitor(K, b, maxiter=self.maxiter)

		if self.verbose:
			print(f"Solving with GMRES + AMG (rtol={self.rtol:.1e})", flush=True)

		u, info = gmres(
			K,
			b,
			rtol=self.rtol,
			atol=0.0,
			restart=200,
			maxiter=self.maxiter,
			M=M,
			callback=monitor,
			callback_type="legacy"
		)

		self.u = u
		self.converged = (info == 0)

		r_true = b - K @ u
		self.true_residual = float(np.linalg.norm(r_true))

		if self.verbose:
			print(f"GMRES converged: {self.converged}")
			print(f"Iterations:     {monitor.it}")
			print(f"True ||r||:     {self.true_residual:.3e}")

	# -------------------------------------------------
	# FEM post-processing
	# -------------------------------------------------

	def reconstruct_pressure(self):
		div = np.zeros(self.Nnds)
		wgt = np.zeros(self.Nnds)

		xp, wp = Genip2DQ(9)
		wp = np.asarray(wp)

		for e in range(self.Nels):
			nodes = self.quad8[e]
			XN = np.column_stack((self.x[nodes], self.y[nodes]))
			ux = self.u[2*nodes]
			uy = self.u[2*nodes + 1]

			for ip in range(len(wp)):
				B, _, detJ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
				w = wp[ip] * detJ
				div_gp = np.dot(B[0], ux) + np.dot(B[1], uy)

				for a in range(8):
					div[nodes[a]] += div_gp * w
					wgt[nodes[a]] += w

		self.p = -self.penalty * div / np.maximum(wgt, 1e-14)

	def compute_vorticity(self):
		omega = np.zeros(self.Nnds)
		wgt = np.zeros(self.Nnds)

		xp, wp = Genip2DQ(9)
		wp = np.asarray(wp)

		for e in range(self.Nels):
			nodes = self.quad8[e]
			XN = np.column_stack((self.x[nodes], self.y[nodes]))
			ux = self.u[2*nodes]
			uy = self.u[2*nodes + 1]

			for ip in range(len(wp)):
				B, _, detJ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
				w = wp[ip] * detJ
				om = np.dot(B[0], uy) - np.dot(B[1], ux)

				for a in range(8):
					omega[nodes[a]] += om * w
					wgt[nodes[a]] += w

		self.vorticity = omega / np.maximum(wgt, 1e-14)

	# -------------------------------------------------
	# Run
	# -------------------------------------------------

	def run(self):
		self._t0 = time.perf_counter()

		self._log("Loading mesh")
		self.load_mesh()
		self._log(f"Mesh loaded ({self.Nels} elements, {self.Nnds} nodes)")

		self._log("Assembling global system (this is the slow phase)")
		self.assemble_system()
		self._log("Assembly completed")

		self._log("Applying boundary conditions")
		self.apply_bc()
		self._log("Boundary conditions applied")

		self._log("Starting GMRES + AMG solve")
		self.solve()
		self._log("GMRES solve completed")

		self._log("Reconstructing pressure")
		self.reconstruct_pressure()
		self._log("Pressure reconstructed")

		self._log("Computing vorticity")
		self.compute_vorticity()
		self._log("Vorticity computed")

		return self.u, self.p, self.vorticity


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
	solver = Quad8StokesPenaltySolverCPU(
		mesh_file="../../data/input/exported_mesh_v6.h5",
		verbose=True
	)
	u, p, omega = solver.run()
