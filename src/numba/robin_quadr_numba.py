"""
Robin boundary condition for quadratic edge - Numba optimized.

Computes boundary stiffness and load contributions for 3-node edges.
"""

import numpy as np
from numba import njit

from genip2dq_numba import genip1d_3


@njit(cache=True)
def robin_quadr(x1, y1, x2, y2, x3, y3, p, gama):
    """
    Robin boundary contribution for a quadratic edge (3 nodes).
    
    Parameters
    ----------
    x1, y1, x2, y2, x3, y3 : float
        Coordinates of the three boundary nodes
    p : float
        Robin p coefficient
    gama : float
        Robin gamma coefficient
    
    Returns
    -------
    He : (3, 3) ndarray
        Boundary stiffness contribution
    Pe : (3,) ndarray
        Boundary load contribution
    """
    
    He = np.zeros((3, 3), dtype=np.float64)
    Pe = np.zeros(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    
    # Get 3-point 1D quadrature
    xi, wi = genip1d_3()
    
    # Integration loop
    for ip in range(3):
        csi = xi[ip]
        
        # Shape functions (quadratic edge)
        b[0] = 0.5 * csi * (csi - 1.0)
        b[1] = 1.0 - csi * csi
        b[2] = 0.5 * csi * (csi + 1.0)
        
        # Derivatives of shape functions
        db0 = csi - 0.5
        db1 = -2.0 * csi
        db2 = csi + 0.5
        
        # Derivatives of x and y along edge
        xx = db0 * x1 + db1 * x2 + db2 * x3
        yy = db0 * y1 + db1 * y2 + db2 * y3
        
        # Jacobian (edge length scaling)
        jaco = np.sqrt(xx * xx + yy * yy)
        
        # Weighted contributions
        wip = jaco * wi[ip]
        wipp = wip * p
        wipg = wip * gama
        
        # Assemble: He += wipp * outer(b, b), Pe += wipg * b
        for i in range(3):
            Pe[i] += wipg * b[i]
            for j in range(3):
                He[i, j] += wipp * b[i] * b[j]
    
    return He, Pe
