import numpy as np


def Genip1D(nip):
    """
    1D Gauss–Legendre integration rules.

    Parameters
    ----------
    nip : int
        Number of integration points (2 or 3)

    Returns
    -------
    xi : (nip,) ndarray
        Integration points
    wi : (nip,) ndarray
        Integration weights
    """
    if nip == 2:
        # 2-point, degree 3
        G = np.sqrt(1.0 / 3.0)
        xi = np.array([-G, G], dtype=float)
        wi = np.array([1.0, 1.0], dtype=float)

    elif nip == 3:
        # 3-point, degree 5
        G = np.sqrt(0.6)
        xi = np.array([-G, 0.0, G], dtype=float)
        wi = np.array([5.0, 8.0, 5.0], dtype=float) / 9.0

    else:
        raise ValueError("Genip1D: nip must be 2 or 3")

    return xi, wi


def Robin_quadr(x1, y1, x2, y2, x3, y3, p, gama):
    """
    Robin boundary contribution for a quadratic edge (3 nodes).

    Parameters
    ----------
    x1,y1,x2,y2,x3,y3 : float
        Coordinates of the three boundary nodes
    p : float
        Robin p coefficient
    gama : float
        Robin gamma coefficient

    Returns
    -------
    He : (3,3) ndarray
        Boundary stiffness contribution
    Pe : (3,) ndarray
        Boundary load contribution
    """

    # -------------------------------------------------
    # Initializations
    # -------------------------------------------------
    b = np.zeros(3, dtype=float)
    He = np.zeros((3, 3), dtype=float)
    Pe = np.zeros(3, dtype=float)

    # -------------------------------------------------
    # Integration points
    # -------------------------------------------------
    nip = 3
    xi, wi = Genip1D(nip)

    # -------------------------------------------------
    # Integration loop
    # -------------------------------------------------
    for ip in range(nip):
        csi = xi[ip]

        # Shape functions (quadratic edge)
        b[0] = 0.5 * csi * (csi - 1.0)
        b[1] = 1.0 - csi * csi
        b[2] = 0.5 * csi * (csi + 1.0)

        # Derivatives of shape functions
        db0 =  csi - 0.5
        db1 = -2.0 * csi
        db2 =  csi + 0.5

        # Derivatives of x and y
        xx = db0 * x1 + db1 * x2 + db2 * x3
        yy = db0 * y1 + db1 * y2 + db2 * y3

        # Jacobian (edge length scaling)
        jaco = np.sqrt(xx * xx + yy * yy)

        # Weighted contributions
        wip  = jaco * wi[ip]
        wipp = wip * p
        wipg = wip * gama

        # Assemble
        He += wipp * np.outer(b, b)
        Pe += wipg * b

    return He, Pe
