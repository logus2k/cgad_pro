import cupy as cp

def Genip2DQ(nip):
    if nip == 4:
        G = cp.sqrt(1.0 / 3.0)

        # build from Python floats, then scale on GPU
        xp = G * cp.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0]
        ], dtype=cp.float64)

        wp = cp.ones(4, dtype=cp.float64)

    elif nip == 9:
        G = cp.sqrt(0.6)

        xp = cp.array([
            [-1.0, -1.0], [ 0.0, -1.0], [ 1.0, -1.0],
            [-1.0,  0.0], [ 0.0,  0.0], [ 1.0,  0.0],
            [-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0]
        ], dtype=cp.float64) * G

        wp = cp.array(
            [25, 40, 25, 40, 64, 40, 25, 40, 25],
            dtype=cp.float64
        ) / 81.0

    else:
        raise ValueError(f"Unsupported nip={nip}. Only 4 or 9 are allowed.")

    return xp, wp
