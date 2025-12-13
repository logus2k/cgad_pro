import cupy as cp

def Robin_quadr(x1, y1, x2, y2, x3, y3, p, gama):
    # -------------------------------------------------
    # Initialization (GPU arrays)
    # -------------------------------------------------
    b  = cp.zeros(3, dtype=cp.float64)
    He = cp.zeros((3, 3), dtype=cp.float64)
    Pe = cp.zeros(3, dtype=cp.float64)

    # -------------------------------------------------
    # 1D Gauss–Legendre (3-point)
    # -------------------------------------------------
    G = cp.sqrt(0.6)

    # build from Python floats, then scale on GPU
    xi = cp.array([-1.0, 0.0, 1.0], dtype=cp.float64) * G
    wi = cp.array([5.0, 8.0, 5.0], dtype=cp.float64) / 9.0

    # -------------------------------------------------
    # Integration loop
    # -------------------------------------------------
    for ip in range(3):
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
        jaco = cp.sqrt(xx * xx + yy * yy)

        # Weighted contributions
        wip  = jaco * wi[ip]
        wipp = wip * p
        wipg = wip * gama

        # Assemble (GPU-safe)
        He += wipp * cp.outer(b, b)
        Pe += wipg * b

    return He, Pe
