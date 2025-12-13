import numpy as np


def Genip2DQ(nip):
    """
    Generate 2D Gauss–Legendre integration points and weights
    using tensor product of 1D rules.

    Parameters
    ----------
    nip : int
        Number of integration points (4 or 9)

    Returns
    -------
    xp : (nip, 2) ndarray
        Integration points [csi, eta]
    wp : (nip,) ndarray
        Integration weights
    """

    if nip == 4:
        # 2x2 Gauss points (degree 3)
        G = np.sqrt(1.0 / 3.0)

        xp = np.array([
            [-G, -G],
            [ G, -G],
            [ G,  G],
            [-G,  G],
        ], dtype=float)

        wp = np.ones(4, dtype=float)

    elif nip == 9:
        # 3x3 Gauss points (degree 5)
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
        ], dtype=float)

        wp = np.array(
            [25, 40, 25, 40, 64, 40, 25, 40, 25],
            dtype=float
        ) / 81.0

    else:
        raise ValueError("Genip2DQ: nip must be 4 or 9")

    return xp, wp
