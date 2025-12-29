import numpy as np
from pathlib import Path
import h5py

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

# Load mesh
mesh_file = PROJECT_ROOT / Path("src/app/client/mesh/s_duct.h5")
with h5py.File(mesh_file, 'r') as f:
    x = np.array(f['x'], dtype=np.float64)
    y = np.array(f['y'], dtype=np.float64)
    quad8 = np.array(f['quad8'], dtype=np.int32)

print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
print(f"y range: [{y.min():.6f}, {y.max():.6f}]")

# Test element 0
e = 0
edofs = quad8[e]
print(f"\nElement 0 DOFs: {edofs}")
for i in range(8):
    print(f"  Node {edofs[i]}: ({x[edofs[i]]:.6f}, {y[edofs[i]]:.6f})")

# Compute Jacobian at center (csi=0, eta=0)
XN = np.array([[x[edofs[i]], y[edofs[i]]] for i in range(8)])

# Dpsi at (0,0)
Dpsi = np.array([
    [0.0, 0.0],   # node 0
    [0.0, 0.0],   # node 1  
    [0.0, 0.0],   # node 2
    [0.0, 0.0],   # node 3
    [0.0, 0.0],   # node 4: csi*(eta-1) = 0
    [0.5, 0.0],   # node 5: (1-eta^2)/2 = 0.5
    [0.0, 0.0],   # node 6
    [-0.5, 0.0],  # node 7: (eta^2-1)/2 = -0.5
])

# Actually compute properly at (0,0)
csi, eta = 0.0, 0.0
Dpsi = np.zeros((8, 2))
Dpsi[0, 0] = (2*csi + eta) * (1 - eta) / 4
Dpsi[1, 0] = (2*csi - eta) * (1 - eta) / 4
Dpsi[2, 0] = (2*csi + eta) * (1 + eta) / 4
Dpsi[3, 0] = (2*csi - eta) * (1 + eta) / 4
Dpsi[4, 0] = csi * (eta - 1)
Dpsi[5, 0] = (1 - eta*eta) / 2
Dpsi[6, 0] = -csi * (1 + eta)
Dpsi[7, 0] = (eta*eta - 1) / 2

Dpsi[0, 1] = (2*eta + csi) * (1 - csi) / 4
Dpsi[1, 1] = (2*eta - csi) * (1 + csi) / 4
Dpsi[2, 1] = (2*eta + csi) * (1 + csi) / 4
Dpsi[3, 1] = (2*eta - csi) * (1 - csi) / 4
Dpsi[4, 1] = (csi*csi - 1) / 2
Dpsi[5, 1] = -(1 + csi) * eta
Dpsi[6, 1] = (1 - csi*csi) / 2
Dpsi[7, 1] = (csi - 1) * eta

J = Dpsi.T @ XN  # 2x2 Jacobian
detJ = np.linalg.det(J)

print(f"\nJacobian at (0,0):\n{J}")
print(f"det(J) = {detJ:.6e}")
