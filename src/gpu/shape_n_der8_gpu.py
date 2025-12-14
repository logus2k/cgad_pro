import cupy as cp

def Shape_N_Der8(XN, csi, eta):
    psi = cp.zeros(8)

    # Shape functions
    psi[0] = (csi-1)*(eta+csi+1)*(1-eta)/4
    psi[1] = (1+csi)*(1-eta)*(csi-eta-1)/4
    psi[2] = (1+csi)*(1+eta)*(csi+eta-1)/4
    psi[3] = (csi-1)*(csi-eta+1)*(1+eta)/4
    psi[4] = (1-csi*csi)*(1-eta)/2
    psi[5] = (1+csi)*(1-eta*eta)/2
    psi[6] = (1-csi*csi)*(1+eta)/2
    psi[7] = (1-csi)*(1-eta*eta)/2

    # Derivatives wrt (csi, eta)
    Dpsi = cp.zeros((8, 2))

    Dpsi[0,0] = (2*csi+eta)*(1-eta)/4
    Dpsi[1,0] = (2*csi-eta)*(1-eta)/4
    Dpsi[2,0] = (2*csi+eta)*(1+eta)/4
    Dpsi[3,0] = (2*csi-eta)*(1+eta)/4
    Dpsi[4,0] = csi*(eta-1)
    Dpsi[5,0] = (1-eta*eta)/2
    Dpsi[6,0] = -csi*(1+eta)
    Dpsi[7,0] = (eta*eta-1)/2

    Dpsi[0,1] = (2*eta+csi)*(1-csi)/4
    Dpsi[1,1] = (2*eta-csi)*(1+csi)/4
    Dpsi[2,1] = (2*eta+csi)*(1+csi)/4
    Dpsi[3,1] = (2*eta-csi)*(1-csi)/4
    Dpsi[4,1] = (csi*csi-1)/2
    Dpsi[5,1] = -(1+csi)*eta
    Dpsi[6,1] = (1-csi*csi)/2
    Dpsi[7,1] = (csi-1)*eta

    # Jacobian
    jaco = XN.T @ Dpsi
    Detj = cp.linalg.det(jaco)
    Invj = cp.linalg.inv(jaco)

    B = Dpsi @ Invj

    return B, psi, Detj
