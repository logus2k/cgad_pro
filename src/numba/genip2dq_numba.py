"""
Gauss-Legendre integration points - Numba optimized.

Provides 2D quadrature rules for Quad-8 element integration.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def genip2dq_4():
    """
    Generate 2x2 Gauss-Legendre integration points and weights.
    
    Returns
    -------
    xp : (4, 2) ndarray
        Integration points [csi, eta]
    wp : (4,) ndarray
        Integration weights
    """
    G = np.sqrt(1.0 / 3.0)
    
    xp = np.array([
        [-G, -G],
        [ G, -G],
        [ G,  G],
        [-G,  G],
    ], dtype=np.float64)
    
    wp = np.ones(4, dtype=np.float64)
    
    return xp, wp


@njit(cache=True)
def genip2dq_9():
    """
    Generate 3x3 Gauss-Legendre integration points and weights.
    
    Returns
    -------
    xp : (9, 2) ndarray
        Integration points [csi, eta]
    wp : (9,) ndarray
        Integration weights
    """
    G = np.sqrt(0.6)
    
    xp = np.array([
        [-G, -G],
        [ 0, -G],
        [ G, -G],
        [-G,  0],
        [ 0,  0],
        [ G,  0],
        [-G,  G],
        [ 0,  G],
        [ G,  G],
    ], dtype=np.float64)
    
    wp = np.array(
        [25, 40, 25, 40, 64, 40, 25, 40, 25],
        dtype=np.float64
    ) / 81.0
    
    return xp, wp


@njit(cache=True)
def genip1d_3():
    """
    Generate 3-point 1D Gauss-Legendre integration points and weights.
    Used for edge integration (Robin BC).
    
    Returns
    -------
    xi : (3,) ndarray
        Integration points
    wi : (3,) ndarray
        Integration weights
    """
    G = np.sqrt(0.6)
    
    xi = np.array([-G, 0.0, G], dtype=np.float64)
    wi = np.array([5.0, 8.0, 5.0], dtype=np.float64) / 9.0
    
    return xi, wi
