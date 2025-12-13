import cupy as cp
from genip2dq_gpu import Genip2DQ
from shape_n_der8_gpu import Shape_N_Der8

def Elem_Quad8(XN, fL):
    Ke = cp.zeros((8,8))
    fe = cp.zeros(8)

    xp, wp = Genip2DQ(9)

    for ip in range(9):
        csi, eta = xp[ip]
        B, psi, Detj = Shape_N_Der8(XN, csi, eta)

        wip = wp[ip] * Detj
        Ke += wip * (B @ B.T)
        fe += fL * wip * psi

    return Ke, fe
