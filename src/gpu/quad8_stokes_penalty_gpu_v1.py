"""
QUAD8 FEM – Steady Stokes (Penalty Formulation) – GPU (PRODUCTION)

Features:
- Quad-8 FEM
- Velocity-only penalty Stokes (SPD)
- Conjugate Gradient (CG)
- RawKernel assembly
- Safe Jacobi preconditioner (no NaNs)
- Real-time stage logging
- Real-time CG residual + ETA monitoring
"""

from pathlib import Path
import time
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsplg
import h5py


# =================================================
# CUDA kernel (FULLY INLINED)
# =================================================

CUDA_SRC = r"""
extern "C" __global__
void quad8_stokes_penalty_assemble(
    const double* x,
    const double* y,
    const int*    quad8,
    int           Nels,
    double        mu,
    double        penalty,
    int*    rows,
    int*    cols,
    double* vals
){
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Nels) return;

    double Ke[16][16];
    #pragma unroll
    for (int i=0;i<16;i++)
        #pragma unroll
        for (int j=0;j<16;j++)
            Ke[i][j] = 0.0;

    int n[8];
    double X[8], Y[8];
    #pragma unroll
    for (int i=0;i<8;i++){
        n[i] = quad8[e*8 + i];
        X[i] = x[n[i]];
        Y[i] = y[n[i]];
    }

    const double gp[3] = { -0.774596669241483, 0.0, 0.774596669241483 };
    const double gw[3] = {  0.555555555555556, 0.888888888888889, 0.555555555555556 };

    for (int gx=0; gx<3; gx++)
    for (int gy=0; gy<3; gy++) {

        double xi = gp[gx];
        double eta = gp[gy];
        double wgt = gw[gx]*gw[gy];

        double dNdxi[8], dNdeta[8];

        dNdxi[0]=0.25*(1-eta)*(2*xi+eta);
        dNdxi[1]=0.25*(1-eta)*(2*xi-eta);
        dNdxi[2]=0.25*(1+eta)*(2*xi+eta);
        dNdxi[3]=0.25*(1+eta)*(2*xi-eta);
        dNdxi[4]=-xi*(1-eta);
        dNdxi[5]=0.5*(1-eta*eta);
        dNdxi[6]=-xi*(1+eta);
        dNdxi[7]=-0.5*(1-eta*eta);

        dNdeta[0]=0.25*(1-xi)*(xi+2*eta);
        dNdeta[1]=0.25*(1+xi)*(-xi+2*eta);
        dNdeta[2]=0.25*(1+xi)*(xi+2*eta);
        dNdeta[3]=0.25*(1-xi)*(-xi+2*eta);
        dNdeta[4]=-0.5*(1-xi*xi);
        dNdeta[5]=-eta*(1+xi);
        dNdeta[6]=0.5*(1-xi*xi);
        dNdeta[7]=-eta*(1-xi);

        double J11=0,J12=0,J21=0,J22=0;
        #pragma unroll
        for(int i=0;i<8;i++){
            J11+=dNdxi[i]*X[i]; J12+=dNdeta[i]*X[i];
            J21+=dNdxi[i]*Y[i]; J22+=dNdeta[i]*Y[i];
        }

        double detJ = J11*J22 - J12*J21;
        double invJ11 =  J22/detJ, invJ12 = -J12/detJ;
        double invJ21 = -J21/detJ, invJ22 =  J11/detJ;

        double dNdx[8], dNdy[8];
        #pragma unroll
        for(int i=0;i<8;i++){
            dNdx[i] = invJ11*dNdxi[i] + invJ12*dNdeta[i];
            dNdy[i] = invJ21*dNdxi[i] + invJ22*dNdeta[i];
        }

        double w = detJ * wgt;

        #pragma unroll
        for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<8;j++){
            double k_visc = mu*(dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j])*w;
            double div_i = dNdx[i] + dNdy[i];
            double div_j = dNdx[j] + dNdy[j];
            double k_pen = penalty * div_i * div_j * w;

            Ke[i][j]       += k_visc + k_pen;
            Ke[i][j+8]     += k_pen;
            Ke[i+8][j]     += k_pen;
            Ke[i+8][j+8]   += k_visc + k_pen;
        }
    }

    int base = e * 256;
    int k = 0;

    #pragma unroll
    for(int a=0;a<8;a++)
    #pragma unroll
    for(int b=0;b<8;b++){
        int ia=n[a], ib=n[b];
        int ux_i=2*ia, uy_i=2*ia+1;
        int ux_j=2*ib, uy_j=2*ib+1;

        rows[base+k]=ux_i; cols[base+k]=ux_j; vals[base+k++]=Ke[a][b];
        rows[base+k]=ux_i; cols[base+k]=uy_j; vals[base+k++]=Ke[a][b+8];
        rows[base+k]=uy_i; cols[base+k]=ux_j; vals[base+k++]=Ke[a+8][b];
        rows[base+k]=uy_i; cols[base+k]=uy_j; vals[base+k++]=Ke[a+8][b+8];
    }
}
"""


# =================================================
# CG monitor (Pylance-clean)
# =================================================

class CGMonitor:
	def __init__(self, A, b, maxiter):
		self.A = A
		self.b = b
		self.maxiter = maxiter
		self.it = 0
		self.start = time.perf_counter()
		self.bnorm = float(cp.linalg.norm(b).get())

	def __call__(self, xk):
		if self.it % 20 == 0 and self.it > 0:
			r = self.b - self.A @ xk
			res = float(cp.linalg.norm(r).get())
			elapsed = time.perf_counter() - self.start
			etr = (self.maxiter - self.it) * (elapsed / max(self.it, 1))

			print(
				f"  Iter {self.it}/{self.maxiter} | "
				f"||r||={res:.3e} rel={res/self.bnorm:.3e} "
				f"ETR={etr/60:.1f} min",
				flush=True
			)
		self.it += 1


# =================================================
# Solver
# =================================================

class Quad8StokesPenaltySolverGPU:

	def __init__(self, mesh_file, mu=1e-3, penalty=1e7, rtol=1e-8, maxiter=20000, verbose=True):
		self.mesh_file = Path(mesh_file)
		self.mu = mu
		self.penalty = penalty
		self.rtol = rtol
		self.maxiter = maxiter
		self.verbose = verbose


	def _log(self, msg):
		if self.verbose:
			elapsed = time.perf_counter() - self._t0
			print(f"[{elapsed:8.2f}s] {msg}", flush=True)


	def load_mesh(self):
		with h5py.File(self.mesh_file, "r") as f:
			self.x = cp.asarray(f["x"], dtype=cp.float64)
			self.y = cp.asarray(f["y"], dtype=cp.float64)
			self.quad8 = cp.asarray(f["quad8"], dtype=cp.int32) - 1

		self.Nnds = self.x.size
		self.Nels = self.quad8.shape[0]


	def assemble(self):
		self._log("Assembling global system (GPU kernel)")

		NNZ = 256 * self.Nels
		rows = cp.empty(NNZ, cp.int32)
		cols = cp.empty(NNZ, cp.int32)
		vals = cp.empty(NNZ, cp.float64)

		kernel = cp.RawKernel(CUDA_SRC, "quad8_stokes_penalty_assemble")
		threads = 128
		blocks = (self.Nels + threads - 1) // threads

		kernel(
			(blocks,), (threads,),
			(self.x, self.y, self.quad8, self.Nels, self.mu, self.penalty, rows, cols, vals)
		)

		self.K = cpsparse.coo_matrix(
			(vals, (rows, cols)), shape=(2*self.Nnds, 2*self.Nnds)
		).tocsr()

		self.f = cp.zeros(2*self.Nnds)


	def solve(self):
		self._log("Preparing CG solve")

		diag = self.K.diagonal()
		Dinv = 1.0 / cp.sqrt(cp.abs(diag) + 1e-14)

		D = cpsparse.diags(Dinv)
		A = D @ self.K @ D
		b = self.f * Dinv

		diagA = A.diagonal()
		diagA_safe = cp.where(cp.abs(diagA) > 1e-14, diagA, 1.0)

		M = cpsplg.LinearOperator(A.shape, lambda v: v / diagA_safe)

		monitor = CGMonitor(A, b, self.maxiter)

		self._log("Starting CG solve")

		u_eq, info = cpsplg.cg(
			A,
			b,
			tol=self.rtol,
			maxiter=self.maxiter,
			M=M,
			callback=monitor
		)

		self.u = u_eq * Dinv
		self.converged = (info == 0)

		self._log(f"CG completed (converged={self.converged})")


	def run(self):
		self._t0 = time.perf_counter()

		self._log("Loading mesh")
		self.load_mesh()
		self._log(f"Mesh loaded ({self.Nels} elements, {self.Nnds} nodes)")

		self.assemble()
		self.solve()

		return self.u


# =================================================
# Main
# =================================================

if __name__ == "__main__":
	solver = Quad8StokesPenaltySolverGPU(
		"exported_mesh_v6.h5",
		verbose=True
	)
	u = solver.run()
