"""
QUAD8 FEM — GPU-accelerated version using CuPy

Changes vs CPU version:
- numpy  -> cupy
- scipy.sparse -> cupyx.scipy.sparse
- assembly + post-processing on GPU
- solve kept on CPU for numerical stability & fair comparison
- COO assembly (correct FEM pattern), CSR only after assembly
"""

from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Input (CPU → GPU)
# -------------------------------------------------
coord = pd.read_excel(HERE / "../data/input/mesh_data_quad8.xlsx", sheet_name="coord", header=None)
conec = pd.read_excel(HERE / "../data/input/mesh_data_quad8.xlsx", sheet_name="conec", header=None)

x_cpu = coord.iloc[:, 0].to_numpy() / 1000.0
y_cpu = coord.iloc[:, 1].to_numpy() / 1000.0
quad8 = conec.to_numpy(dtype=int) - 1  # zero-based

Nnds = len(x_cpu)
Nels = quad8.shape[0]

x = cp.asarray(x_cpu)
y = cp.asarray(y_cpu)

# -------------------------------------------------
# Global RHS (GPU)
# -------------------------------------------------
fg_gpu = cp.zeros(Nnds, dtype=cp.float64)

# -------------------------------------------------
# Element assembly (COO triplets on GPU)
# -------------------------------------------------
from elem_quad8_gpu import Elem_Quad8
from genip2dq_gpu import Genip2DQ
from shape_n_der8_gpu import Shape_N_Der8
from robin_quadr_gpu import Robin_quadr

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
# Robin BCs (also into COO — NO CSR MODIFICATION)
# -------------------------------------------------
robin_sides = [
    [18, 6, 5],
    [5, 7, 17],
    [17, 114, 115],
    [115, 116, 123]
]

for ed in robin_sides:
    He, Pe = Robin_quadr(
        x[ed[0]], y[ed[0]],
        x[ed[1]], y[ed[1]],
        x[ed[2]], y[ed[2]],
        p=0.0, gama=2.5
    )

    for i in range(3):
        fg_gpu[ed[i]] += Pe[i]
        for j in range(3):
            rows.append(ed[i])
            cols.append(ed[j])
            vals.append(He[i, j])

# -------------------------------------------------
# Build sparse matrix ONCE (COO → CSR)
# -------------------------------------------------
Kg_gpu = cpsparse.coo_matrix(
    (cp.asarray(vals), (cp.asarray(rows), cp.asarray(cols))),
    shape=(Nnds, Nnds)
).tocsr()

# -------------------------------------------------
# Dirichlet BCs (safe on CSR)
# -------------------------------------------------
"""
exit_nodes = cp.asarray([32, 29, 33, 200, 198, 199])  # zero-based

Kg_gpu[exit_nodes, :] = 0
Kg_gpu[:, exit_nodes] = 0
Kg_gpu[exit_nodes, exit_nodes] = 1.0
fg_gpu[exit_nodes] = 0.0
"""

exit_nodes = cp.asarray([32, 29, 33, 200, 198, 199])

exit_mask = cp.zeros(Nnds, dtype=cp.bool_)
exit_mask[exit_nodes] = True

rows_cp = cp.asarray(rows)
cols_cp = cp.asarray(cols)
vals_cp = cp.asarray(vals)

# Remove any entry touching Dirichlet nodes
keep = ~(exit_mask[rows_cp] | exit_mask[cols_cp])

rows_cp = rows_cp[keep]
cols_cp = cols_cp[keep]
vals_cp = vals_cp[keep]

# Add identity entries for Dirichlet nodes
rows_cp = cp.concatenate([rows_cp, exit_nodes])
cols_cp = cp.concatenate([cols_cp, exit_nodes])
vals_cp = cp.concatenate([vals_cp, cp.ones(len(exit_nodes))])

# Build CSR once
Kg_gpu = cpsparse.coo_matrix(
    (vals_cp, (rows_cp, cols_cp)),
    shape=(Nnds, Nnds)
).tocsr()

fg_gpu[exit_nodes] = 0.0

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

pressure = 101328.8281 - 0.6125 * abs_vel**2

# -------------------------------------------------
# Export (GPU → CPU → Excel)
# -------------------------------------------------
u_cpu = cp.asnumpy(u)
vel_cpu = cp.asnumpy(vel)
abs_cpu = cp.asnumpy(abs_vel)
p_cpu = cp.asnumpy(pressure)

outfile = HERE / "../data/output/Results_quad8_GPU.xlsx"

with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
    header = [
        "#NODE", "VELOCITY POTENTIAL", "",
        "#ELEMENT", "U m/s (x  vel)", "V m/s (y vel)",
        "|V| m/s", "Pressure (Pa)"
    ]
    pd.DataFrame([header]).to_excel(writer, index=False, header=False)

    pd.DataFrame(
        np.column_stack([np.arange(1, Nnds + 1), u_cpu])
    ).to_excel(writer, startrow=1, startcol=0,
               index=False, header=False)

    pd.DataFrame(
        np.column_stack([
            np.arange(1, Nels + 1),
            vel_cpu[:, 0],
            vel_cpu[:, 1],
            abs_cpu,
            p_cpu
        ])
    ).to_excel(writer, startrow=1, startcol=3,
               index=False, header=False)
