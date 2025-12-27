"""
Element stiffness matrix for Quad-8 elements - Numba optimized.

Computes element stiffness matrix Ke and load vector fe using
9-point Gauss quadrature.
"""

import numpy as np
from numba import njit

from genip2dq_numba import genip2dq_9
from shape_n_der8_numba import shape_n_der8


@njit(cache=True)
def elem_quad8(XN, fL):
    """
    Compute element stiffness matrix and load vector for Quad-8 element.
    
    Parameters
    ----------
    XN : (8, 2) ndarray
        Coordinates of the 8-node quadrilateral element
    fL : float
        Body/load term
    
    Returns
    -------
    Ke : (8, 8) ndarray
        Element stiffness matrix
    fe : (8,) ndarray
        Element load vector
    """
    
    Ke = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)
    
    # Get 9-point quadrature rule
    xp, wp = genip2dq_9()
    
    # Integration loop
    for ip in range(9):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        
        B, psi, Detj = shape_n_der8(XN, csi, eta)
        
        wip = wp[ip] * Detj
        
        # Ke += wip * B @ B.T
        for i in range(8):
            fe[i] += fL * wip * psi[i]
            for j in range(8):
                Ke[i, j] += wip * (B[i, 0] * B[j, 0] + B[i, 1] * B[j, 1])
    
    return Ke, fe


@njit(cache=True)
def elem_quad8_velocity(XN, u_e):
    """
    Compute velocity at element centroid from nodal potentials.
    
    Parameters
    ----------
    XN : (8, 2) ndarray
        Coordinates of the 8-node quadrilateral element
    u_e : (8,) ndarray
        Nodal potential values
    
    Returns
    -------
    vel_x : float
        Average x-velocity component
    vel_y : float
        Average y-velocity component
    abs_vel : float
        Average velocity magnitude
    """
    
    # Get 4-point quadrature for post-processing
    G = np.sqrt(1.0 / 3.0)
    xp = np.array([
        [-G, -G],
        [ G, -G],
        [ G,  G],
        [-G,  G],
    ], dtype=np.float64)
    
    vel_x_sum = 0.0
    vel_y_sum = 0.0
    v_mag_sum = 0.0
    
    for ip in range(4):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        
        B, _, _ = shape_n_der8(XN, csi, eta)
        
        # Compute gradient: grad = B.T @ u_e
        grad_x = 0.0
        grad_y = 0.0
        for i in range(8):
            grad_x += B[i, 0] * u_e[i]
            grad_y += B[i, 1] * u_e[i]
        
        # Velocity is negative gradient: v = -grad(u)
        vel_x_sum += -grad_x
        vel_y_sum += -grad_y
        v_mag_sum += np.sqrt(grad_x * grad_x + grad_y * grad_y)
    
    return vel_x_sum / 4.0, vel_y_sum / 4.0, v_mag_sum / 4.0
