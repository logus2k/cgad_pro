from pathlib import Path
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# -------------------------------------------------
# FEM helper functions (ported separately)
# -------------------------------------------------
from elem_quad8_cpu import Elem_Quad8
from robin_quadr_cpu import Robin_quadr
from genip2dq_cpu import Genip2DQ
from shape_n_der8_cpu import Shape_N_Der8


# -------------------------------------------------
# Input
# -------------------------------------------------

CURRENT_PATH = Path(__file__).resolve().parent.parent
xlsx_path = CURRENT_PATH / "../data/input/mesh_data_quad8.xlsx"

coord = pd.read_excel(xlsx_path, sheet_name="coord", header=None)
conec = pd.read_excel(xlsx_path, sheet_name="conec", header=None)

x = coord.iloc[:201, 0].to_numpy(dtype=float) / 1000.0
y = coord.iloc[:201, 1].to_numpy(dtype=float) / 1000.0
quad8 = conec.iloc[:52, :8].to_numpy(dtype=int) - 1  # 0-based indexing

Nels = quad8.shape[0]
Nnds = x.shape[0]


# -------------------------------------------------
# Global system (SPARSE)
# -------------------------------------------------
Kg = lil_matrix((Nnds, Nnds), dtype=float)
fg = np.zeros(Nnds)


# -------------------------------------------------
# Element assembly
# -------------------------------------------------
for i in range(Nels):
    edofs = quad8[i]
    XN = np.column_stack((x[edofs], y[edofs]))

    Ke, fe = Elem_Quad8(XN, fL=0.0)

    for a in range(8):
        fg[edofs[a]] += fe[a]
        for b in range(8):
            Kg[edofs[a], edofs[b]] += Ke[a, b]


# -------------------------------------------------
# Dirichlet BCs (elimination)
# -------------------------------------------------
exit_nodes = np.array([33, 30, 34, 201, 199, 200]) - 1  # 0-based

for j in exit_nodes:
    Kg[j, :] = 0.0
    Kg[:, j] = 0.0
    Kg[j, j] = 1.0
    fg[j] = 0.0


# -------------------------------------------------
# Robin BCs
# -------------------------------------------------
p_robin = 0.0
gama = 2.5

robin_sides = [
    [19, 7, 6],
    [6, 8, 18],
    [18, 115, 116],
    [116, 117, 124],
]

for side in robin_sides:
    ed = np.array(side) - 1
    He, Pe = Robin_quadr(
        x[ed[0]], y[ed[0]],
        x[ed[1]], y[ed[1]],
        x[ed[2]], y[ed[2]],
        p_robin, gama
    )

    for a in range(3):
        fg[ed[a]] += Pe[a]
        for b in range(3):
            Kg[ed[a], ed[b]] += He[a, b]


# -------------------------------------------------
# Diagnostics
# -------------------------------------------------
Kg = Kg.tocsr()

print("NaNs in Kg:", np.isnan(Kg.data).any())
print("Infs in Kg:", np.isinf(Kg.data).any())
print("Rank approx:", np.linalg.matrix_rank(Kg.toarray()))
print("Condition estimate:", np.linalg.cond(Kg.toarray()))


# -------------------------------------------------
# Solve
# -------------------------------------------------
u = spsolve(Kg, fg)


# -------------------------------------------------
# Post-processing
# -------------------------------------------------
abs_vel_nds = np.zeros(Nnds)
abs_vel = np.zeros(Nels)
pressure = np.zeros(Nels)
vel = np.zeros((Nels, 2))

for i in range(Nels):
    edofs = quad8[i]
    XN = np.column_stack((x[edofs], y[edofs]))

    xp, _ = Genip2DQ(nip=4)
    abs_vel_ip = np.zeros(len(xp))

    for ip, (csi, eta) in enumerate(xp):
        B, psi, _ = Shape_N_Der8(XN, csi, eta)
        gradu = B.T @ u[edofs]

        vel[i, 0] = gradu[0]
        vel[i, 1] = gradu[1]
        abs_vel_ip[ip] = np.linalg.norm(gradu)

    abs_vel[i] = abs_vel_ip.mean()
    pressure[i] = 101328.8281 - 0.6125 * abs_vel[i] ** 2
    abs_vel_nds[edofs] = abs_vel[i]


# -------------------------------------------------
# Excel export (exact MATLAB-style layout)
# -------------------------------------------------
outfile = CURRENT_PATH / "../data/output/Results_quad8_CPU.xlsx"

u = spsolve(Kg, fg)
u = np.asarray(u).ravel()

node_block = np.column_stack([
    np.arange(1, Nnds + 1),
    u
])

elem_block = np.column_stack([
    np.arange(1, Nels + 1),
    vel[:, 0],
    vel[:, 1],
    abs_vel,
    pressure
])

with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
    header = [
        "#NODE", "VELOCITY POTENTIAL", "",
        "#ELEMENT", "U m/s (x  vel)", "V m/s (y vel)", "|V| m/s", "Pressure (Pa)"
    ]
    pd.DataFrame([header]).to_excel(writer, index=False, header=False)

    pd.DataFrame(node_block).to_excel(
        writer, startrow=1, startcol=0, index=False, header=False
    )

    pd.DataFrame(elem_block).to_excel(
        writer, startrow=1, startcol=3, index=False, header=False
    )
