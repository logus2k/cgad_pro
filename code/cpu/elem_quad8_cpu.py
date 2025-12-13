import numpy as np

from genip2dq_cpu import Genip2DQ
from shape_n_der8_cpu import Shape_N_Der8


def Elem_Quad8(XN, fL):
    """
    Parameters
    ----------
    XN : (8,2) ndarray
        Coordinates of the 8-node quadrilateral element
    fL : float
        Body/load term

    Returns
    -------
    Ke : (8,8) ndarray
        Element stiffness matrix
    fe : (8,) ndarray
        Element load vector
    """

    Ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)

    nip = 9
    xp, wp = Genip2DQ(nip)

    for ip in range(nip):
        csi = xp[ip, 0]
        eta = xp[ip, 1]

        B, psi, Detj = Shape_N_Der8(XN, csi, eta)

        wip = wp[ip] * Detj

        # Ke += wip * B * B'
        Ke += wip * (B @ B.T)

        # fe += fL * wip * psi
        fe += fL * wip * psi

    return Ke, fe
