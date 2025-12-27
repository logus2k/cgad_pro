"""
Shape functions and derivatives for 8-node quadrilateral - Numba optimized.

Provides shape function evaluation and Jacobian computation for Quad-8 elements.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def shape_n_der8(XN, csi, eta):
    """
    Shape functions and derivatives for 8-node quadrilateral element.
    
    Parameters
    ----------
    XN : (8, 2) ndarray
        Coordinates of the element nodes
    csi, eta : float
        Parametric coordinates
    
    Returns
    -------
    B : (8, 2) ndarray
        Derivatives of shape functions w.r.t x and y
    psi : (8,) ndarray
        Shape functions
    Detj : float
        Determinant of the Jacobian
    """
    
    # -------------------------------------------------
    # 1) Shape functions (psi)
    # -------------------------------------------------
    psi = np.zeros(8, dtype=np.float64)
    
    psi[0] = (csi - 1) * (eta + csi + 1) * (1 - eta) / 4
    psi[1] = (1 + csi) * (1 - eta) * (csi - eta - 1) / 4
    psi[2] = (1 + csi) * (1 + eta) * (csi + eta - 1) / 4
    psi[3] = (csi - 1) * (csi - eta + 1) * (1 + eta) / 4
    psi[4] = (1 - csi * csi) * (1 - eta) / 2
    psi[5] = (1 + csi) * (1 - eta * eta) / 2
    psi[6] = (1 - csi * csi) * (1 + eta) / 2
    psi[7] = (1 - csi) * (1 - eta * eta) / 2
    
    # -------------------------------------------------
    # 2) Parametric derivatives Dpsi (8x2)
    # -------------------------------------------------
    Dpsi = np.zeros((8, 2), dtype=np.float64)
    
    # d/d(csi)
    Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
    Dpsi[1, 0] = (2 * csi - eta) * (1 - eta) / 4
    Dpsi[2, 0] = (2 * csi + eta) * (1 + eta) / 4
    Dpsi[3, 0] = (2 * csi - eta) * (1 + eta) / 4
    Dpsi[4, 0] = csi * (eta - 1)
    Dpsi[5, 0] = (1 - eta * eta) / 2
    Dpsi[6, 0] = -csi * (1 + eta)
    Dpsi[7, 0] = (eta * eta - 1) / 2
    
    # d/d(eta)
    Dpsi[0, 1] = (2 * eta + csi) * (1 - csi) / 4
    Dpsi[1, 1] = (2 * eta - csi) * (1 + csi) / 4
    Dpsi[2, 1] = (2 * eta + csi) * (1 + csi) / 4
    Dpsi[3, 1] = (2 * eta - csi) * (1 - csi) / 4
    Dpsi[4, 1] = (csi * csi - 1) / 2
    Dpsi[5, 1] = -(1 + csi) * eta
    Dpsi[6, 1] = (1 - csi * csi) / 2
    Dpsi[7, 1] = (csi - 1) * eta
    
    # -------------------------------------------------
    # 3) Jacobian matrix (2x2): jaco = XN.T @ Dpsi
    # -------------------------------------------------
    jaco = np.zeros((2, 2), dtype=np.float64)
    for i in range(8):
        jaco[0, 0] += XN[i, 0] * Dpsi[i, 0]
        jaco[0, 1] += XN[i, 0] * Dpsi[i, 1]
        jaco[1, 0] += XN[i, 1] * Dpsi[i, 0]
        jaco[1, 1] += XN[i, 1] * Dpsi[i, 1]
    
    # -------------------------------------------------
    # 4) Determinant
    # -------------------------------------------------
    Detj = jaco[0, 0] * jaco[1, 1] - jaco[0, 1] * jaco[1, 0]
    
    # -------------------------------------------------
    # 5) Inverse Jacobian (2x2)
    # -------------------------------------------------
    inv_det = 1.0 / Detj
    Invj = np.zeros((2, 2), dtype=np.float64)
    Invj[0, 0] =  jaco[1, 1] * inv_det
    Invj[0, 1] = -jaco[0, 1] * inv_det
    Invj[1, 0] = -jaco[1, 0] * inv_det
    Invj[1, 1] =  jaco[0, 0] * inv_det
    
    # -------------------------------------------------
    # 6) Derivatives w.r.t x, y: B = Dpsi @ Invj
    # -------------------------------------------------
    B = np.zeros((8, 2), dtype=np.float64)
    for i in range(8):
        B[i, 0] = Dpsi[i, 0] * Invj[0, 0] + Dpsi[i, 1] * Invj[1, 0]
        B[i, 1] = Dpsi[i, 0] * Invj[0, 1] + Dpsi[i, 1] * Invj[1, 1]
    
    return B, psi, Detj
