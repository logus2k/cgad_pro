"""
QUAD8 FEM — GPU-accelerated version using CuPy

Dynamic version:
- Loads arbitrary Quad-8 meshes from Excel
- Geometry-based BC detection (same as CPU)
- COO-only sparse assembly (GPU-friendly)
- Linear solve on CPU for stability and fair comparison
"""

from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd

from elem_quad8_gpu import Elem_Quad8
from genip2dq_gpu import Genip2DQ
from shape_n_der8_gpu import Shape_N_Der8
from robin_quadr_gpu import Robin_quadr

# -------------------------------------------------
# Configuration (physics)
# -------------------------------------------------
P0   = 101328.8281
RHO  = 0.6125
GAMA = 2.5

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = Path(__file__).resolve().parent.parent
MESH_FILE = HERE / "data/input/mesh_data_quad8.xlsx"
OUT_FILE  = HERE / "data/output/Results_quad8_GPU.xlsx"

# -------------------------------------------------
# Input (CPU → GPU)
# -------------------------------------------------
coord = pd.read_excel(MESH_FILE, sheet_name="coord")
conec = pd.read_excel(MESH_FILE, sheet_name="conec")

x_cpu = coord["X"].to_numpy(dtype=float) / 1000.0
y_cpu = coord["Y"].to_numpy(dtype=float) / 1000.0
quad8 = conec.iloc[:, :8].to_numpy(dtype=int) - 1

Nnds = x_cpu.size
Nels = quad8.shape[0]

print(f"Loaded mesh: {Nnds} nodes, {Nels} elements")

x = cp.asarray(x_cpu)
y = cp.asarray(y_cpu)

# -------------------------------------------------
# Geometry-based BC detection (CPU → GPU)
# -------------------------------------------------
tol = 1e-9

x_max = x_cpu.max()
x_min = x_cpu.min()

exit_nodes = np.where(np.abs(x_cpu - x_max) < tol)[0]
boundary_nodes = set(np.where(np.abs(x_cpu - x_min) < tol)[0].tolist())

exit_nodes_gpu = cp.asarray(exit_nodes, dtype=cp.int32)

# Detect Robin edges
robin_edges = []
for e in range(Nels):
    nodes = quad8[e]
    edges = [
        (nodes[0], nodes[4], nodes[1]),
        (nodes[1], nodes[5], nodes[2]),
        (nodes[2], nodes[6], nodes[3]),
        (nodes[3], nodes[7], nodes[0]),
    ]
    for ed in edges:
        if all(n in boundary_nodes for n in ed):
            robin_edges.append(ed)

print(f"Dirichlet nodes: {len(exit_nodes)}")
print(f"Robin edges: {len(robin_edges)}")

# -------------------------------------------------
# Global RHS (GPU)
# -------------------------------------------------
fg_gpu = cp.zeros(Nnds, dtype=cp.float64)

# -------------------------------------------------
# Element assembly (COO triplets)
# -------------------------------------------------
rows = []
cols = []
vals = []

for e in range(Nels):
    edofs = quad8[e]
    XN = cp.column_stack((x[edofs], y[edofs]))

    Ke, fe = Elem_Quad8(XN, fL=0.0)
    fg_gpu[edofs] += fe

    for i in range(8):
        for j in range(8):
            rows.append(edofs[i])
            cols.append(edofs[j])
            vals.append(Ke[i, j])

# -------------------------------------------------
# Robin BCs (into COO)
# -------------------------------------------------
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
        fg_gpu[ed[i]] += Pe[i]
        for j in range(3):
            rows.append(ed[i])
            cols.append(ed[j])
            vals.append(He[i, j])

# -------------------------------------------------
# Dirichlet BCs via COO filtering (GPU-safe)
# -------------------------------------------------
rows_cp = cp.asarray(rows, dtype=cp.int32)
cols_cp = cp.asarray(cols, dtype=cp.int32)
vals_cp = cp.asarray(vals, dtype=cp.float64)

exit_mask = cp.zeros(Nnds, dtype=cp.bool_)
exit_mask[exit_nodes_gpu] = True

keep = ~(exit_mask[rows_cp] | exit_mask[cols_cp])

rows_cp = rows_cp[keep]
cols_cp = cols_cp[keep]
vals_cp = vals_cp[keep]

# Add identity rows for Dirichlet nodes
rows_cp = cp.concatenate([rows_cp, exit_nodes_gpu])
cols_cp = cp.concatenate([cols_cp, exit_nodes_gpu])
vals_cp = cp.concatenate([vals_cp, cp.ones(exit_nodes_gpu.size)])

fg_gpu[exit_nodes_gpu] = 0.0

# -------------------------------------------------
# Build sparse matrix ONCE
# -------------------------------------------------
Kg_gpu = cpsparse.coo_matrix(
    (vals_cp, (rows_cp, cols_cp)),
    shape=(Nnds, Nnds)
).tocsr()

# -------------------------------------------------
# Solve (CPU on purpose)
# -------------------------------------------------
Kg_cpu = Kg_gpu.get()
fg_cpu = fg_gpu.get()

u_cpu = spsolve(csr_matrix(Kg_cpu), fg_cpu)

# -------------------------------------------------
# Post-processing (GPU)
# -------------------------------------------------
u = cp.asarray(u_cpu)

abs_vel = cp.zeros(Nels, dtype=cp.float64)
vel = cp.zeros((Nels, 2), dtype=cp.float64)

for e in range(Nels):
    edofs = quad8[e]
    XN = cp.column_stack((x[edofs], y[edofs]))

    xp, _ = Genip2DQ(4)
    v_ip = cp.zeros(4, dtype=cp.float64)

    for ip in range(4):
        B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
        grad = B.T @ u[edofs]

        vel[e, 0] = grad[0]
        vel[e, 1] = grad[1]
        v_ip[ip] = cp.linalg.norm(grad)

    abs_vel[e] = cp.mean(v_ip)

pressure = P0 - RHO * abs_vel**2

# -------------------------------------------------
# Export (GPU → CPU → Excel)
# -------------------------------------------------
u_cpu = cp.asnumpy(u)
vel_cpu = cp.asnumpy(vel)
abs_cpu = cp.asnumpy(abs_vel)
p_cpu = cp.asnumpy(pressure)

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    header = [
        "#NODE", "VELOCITY POTENTIAL", "",
        "#ELEMENT", "U m/s (x  vel)", "V m/s (y vel)",
        "|V| m/s", "Pressure (Pa)"
    ]
    pd.DataFrame([header]).to_excel(writer, index=False, header=False)

    pd.DataFrame({
        "#NODE": np.arange(1, Nnds + 1),
        "VELOCITY POTENTIAL": u_cpu
    }).to_excel(writer, startrow=1, startcol=0,
                index=False, header=False)

    pd.DataFrame({
        "#ELEMENT": np.arange(1, Nels + 1),
        "Ux": vel_cpu[:, 0],
        "Uy": vel_cpu[:, 1],
        "|V|": abs_cpu,
        "Pressure": p_cpu
    }).to_excel(writer, startrow=1, startcol=3,
                index=False, header=False)

print(f"GPU results written to: {OUT_FILE}")
