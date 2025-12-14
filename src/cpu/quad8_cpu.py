"""
QUAD8 FEM - CPU baseline

Key properties:
- Dynamic mesh size (any Quad-8 Excel mesh)
- LIL assembly → CSR once
- Robin BCs assembled BEFORE Dirichlet
- Dirichlet enforced by elimination
- Jacobi-preconditioned CG solver
- Iteration progress printed
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

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

# -------------------------------------------------
# Physics parameters
# -------------------------------------------------
P0   = 101328.8281
RHO  = 0.6125
GAMA = 2.5

# -------------------------------------------------
# Input
# -------------------------------------------------
MESH_FILE = PROJECT_ROOT / "data/input/converted_mesh_v3.xlsx"

coord = pd.read_excel(MESH_FILE, sheet_name="coord")
conec = pd.read_excel(MESH_FILE, sheet_name="conec")

x = coord["X"].to_numpy(dtype=float) / 1000.0
y = coord["Y"].to_numpy(dtype=float) / 1000.0
quad8 = conec.iloc[:, :8].to_numpy(dtype=int) - 1

Nnds = x.size
Nels = quad8.shape[0]

print(f"Loaded mesh: {Nnds} nodes, {Nels} Quad-8 elements")

# -------------------------------------------------
# Global system (use lil_matrix for dynamic assembly)
# -------------------------------------------------
Kg = lil_matrix((Nnds, Nnds), dtype=np.float64)
fg = np.zeros(Nnds, dtype=np.float64)

# -------------------------------------------------
# Element assembly
# -------------------------------------------------
print("Assembling global system...")
for e in range(Nels):
    edofs = quad8[e]
    XN = np.column_stack((x[edofs], y[edofs]))
    Ke, fe = Elem_Quad8(XN, fL=0.0)

    for i in range(8):
        fg[edofs[i]] += fe[i]
        for j in range(8):
            Kg[edofs[i], edofs[j]] += Ke[i, j]

    if (e + 1) % 50000 == 0:
        print(f"  {e + 1}/{Nels} elements assembled")

# -------------------------------------------------
# Apply Robin boundary conditions
# -------------------------------------------------
tol = 1e-9
x_min = x.min()
boundary_nodes = set(np.where(np.abs(x - x_min) < tol)[0].tolist())

robin_edges = []
for e in range(Nels):
    n = quad8[e]
    edges = [
        (n[0], n[4], n[1]),
        (n[1], n[5], n[2]),
        (n[2], n[6], n[3]),
        (n[3], n[7], n[0]),
    ]
    for edge in edges:
        if all(k in boundary_nodes for k in edge):
            robin_edges.append(edge)

print(f"Applying boundary conditions ({len(robin_edges)} Robin edges)...")

for (n1, n2, n3) in robin_edges:
    He, Pe = Robin_quadr(
        x[n1], y[n1],
        x[n2], y[n2],
        x[n3], y[n3],
        p=0.0,
        gama=GAMA
    )

    ed = [n1, n2, n3]
    for i in range(3):
        fg[ed[i]] += Pe[i]
        for j in range(3):
            Kg[ed[i], ed[j]] += He[i, j]

# -------------------------------------------------
# Apply Dirichlet BCs (maximum-x boundary)
# -------------------------------------------------
x_max = x.max()
exit_nodes = np.where(np.abs(x - x_max) < tol)[0]

for n in exit_nodes:
    Kg[n, :] = 0.0
    Kg[:, n] = 0.0
    Kg[n, n] = 1.0
    fg[n] = 0.0

# -------------------------------------------------
# Fix unused nodes (set to Dirichlet BC)
# -------------------------------------------------
used_nodes_set = set(quad8.flatten())
unused_count = 0
for n in range(Nnds):
    if n not in used_nodes_set:
        Kg[n, :] = 0.0
        Kg[:, n] = 0.0
        Kg[n, n] = 1.0
        fg[n] = 0.0
        unused_count += 1

if unused_count > 0:
    print(f"Fixed {unused_count} unused nodes")

# -------------------------------------------------
# Convert to CSR
# -------------------------------------------------
print("Converting to CSR format...")
Kg = csr_matrix(Kg)

# -------------------------------------------------
# Jacobi preconditioner
# -------------------------------------------------
diag = Kg.diagonal().astype(np.float64, copy=True)
diag[diag == 0.0] = 1.0
M_inv = np.reciprocal(diag)

def precond(v):
    return M_inv * v

M = LinearOperator(Kg.shape, precond)

# -------------------------------------------------
# CG progress monitor
# -------------------------------------------------
class CGMonitor:
    def __init__(self, every=10):
        self.it = 0
        self.every = every

    def __call__(self, rk):
        self.it += 1
        if self.it % self.every == 0:
            residual = np.linalg.norm(rk)
            print(f"  CG iteration {self.it}, residual = {residual:.3e}")

# -------------------------------------------------
# Solve with Conjugate Gradient
# -------------------------------------------------
print("Solving linear system (PCG)...")

monitor = CGMonitor(every=50)

u, info = cg(
    Kg,
    fg,
    rtol=1e-8,
    maxiter=5000,
    M=M,
    callback=monitor
)

if info == 0:
    print(f"✓ CG converged in {monitor.it} iterations")
elif info > 0:
    print(f"✗ CG did not converge (max iterations reached)")
else:
    print(f"✗ CG error (info={info})")

# -------------------------------------------------
# Solution diagnostics
# -------------------------------------------------
print(f"\nSolution statistics:")
print(f"  u range: [{u.min():.6e}, {u.max():.6e}]")
print(f"  u mean:  {u.mean():.6e}")
print(f"  u std:   {u.std():.6e}")

# -------------------------------------------------
# Post-processing
# -------------------------------------------------
print("Computing velocity field...")

abs_vel = np.zeros(Nels)
vel = np.zeros((Nels, 2))

for e in range(Nels):
    edofs = quad8[e]
    XN = np.column_stack((x[edofs], y[edofs]))

    xp, _ = Genip2DQ(4)
    v_ip = np.zeros(4)

    for ip in range(4):
        B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
        grad = B.T @ u[edofs]
        vel[e, 0] = grad[0]
        vel[e, 1] = grad[1]
        v_ip[ip] = np.linalg.norm(grad)

    abs_vel[e] = v_ip.mean()

pressure = P0 - RHO * abs_vel**2

# -------------------------------------------------
# Generate visualizations
# -------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "data/output/figures"
output_files = generate_all_visualizations(
    x, y, quad8, u, abs_vel, pressure,
    OUTPUT_DIR,
    implementation_name="CPU"
)

print(f"Visualizations saved to {OUTPUT_DIR}")

# -------------------------------------------------
# Export results
# -------------------------------------------------
outfile = PROJECT_ROOT / "data/output/Results_quad8_CPU.xlsx"
export_results_to_excel(
    outfile,
    x, y, quad8,
    u, vel, abs_vel, pressure,
    implementation_name="CPU"
)

print(f"Results written to {outfile}")
print("\n✓ Simulation complete")
