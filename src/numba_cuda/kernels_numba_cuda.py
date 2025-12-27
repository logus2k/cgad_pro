"""
Numba CUDA Kernels for Quad-8 FEM Solver.

Provides @cuda.jit GPU kernels as alternative to CuPy RawKernel.
Demonstrates Python-syntax GPU programming.

Note: Pylance shows false positive type errors for cuda.local.array()
and other Numba CUDA APIs due to incomplete type stubs. The code
runs correctly at runtime.
"""
# pyright: reportArgumentType=false
# pyright: reportOptionalSubscript=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false

import numpy as np
from numba import cuda
import math


# =============================================================================
# Assembly Kernel
# =============================================================================

@cuda.jit
def quad8_assembly_kernel(x, y, quad8, xp, wp, vals_out, fg_out):
    """
    Compute element stiffness matrices for all elements in parallel.
    
    Each thread handles one element.
    
    Parameters
    ----------
    x, y : (Nnds,) device arrays
        Nodal coordinates
    quad8 : (Nels, 8) device array
        Element connectivity
    xp : (9, 2) device array
        Integration points
    wp : (9,) device array
        Integration weights
    vals_out : (Nels * 64,) device array
        Output: flattened Ke values (8x8 per element)
    fg_out : (Nnds,) device array
        Output: global force vector (atomic updates)
    """
    
    e = cuda.grid(1)
    
    if e >= quad8.shape[0]:
        return
    
    # Get element DOFs
    edofs = cuda.local.array(8, dtype=np.int32)
    for i in range(8):
        edofs[i] = quad8[e, i]
    
    # Get nodal coordinates
    XN = cuda.local.array((8, 2), dtype=np.float64)
    for i in range(8):
        XN[i, 0] = x[edofs[i]]
        XN[i, 1] = y[edofs[i]]
    
    # Initialize element matrices
    Ke = cuda.local.array((8, 8), dtype=np.float64)
    fe = cuda.local.array(8, dtype=np.float64)
    
    for i in range(8):
        fe[i] = 0.0
        for j in range(8):
            Ke[i, j] = 0.0
    
    # Integration loop (9 points)
    for ip in range(9):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        w = wp[ip]
        
        # Shape functions
        psi = cuda.local.array(8, dtype=np.float64)
        psi[0] = (csi - 1) * (eta + csi + 1) * (1 - eta) / 4
        psi[1] = (1 + csi) * (1 - eta) * (csi - eta - 1) / 4
        psi[2] = (1 + csi) * (1 + eta) * (csi + eta - 1) / 4
        psi[3] = (csi - 1) * (csi - eta + 1) * (1 + eta) / 4
        psi[4] = (1 - csi * csi) * (1 - eta) / 2
        psi[5] = (1 + csi) * (1 - eta * eta) / 2
        psi[6] = (1 - csi * csi) * (1 + eta) / 2
        psi[7] = (1 - csi) * (1 - eta * eta) / 2
        
        # Shape function derivatives (parametric)
        Dpsi = cuda.local.array((8, 2), dtype=np.float64)
        
        Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
        Dpsi[1, 0] = (2 * csi - eta) * (1 - eta) / 4
        Dpsi[2, 0] = (2 * csi + eta) * (1 + eta) / 4
        Dpsi[3, 0] = (2 * csi - eta) * (1 + eta) / 4
        Dpsi[4, 0] = csi * (eta - 1)
        Dpsi[5, 0] = (1 - eta * eta) / 2
        Dpsi[6, 0] = -csi * (1 + eta)
        Dpsi[7, 0] = (eta * eta - 1) / 2
        
        Dpsi[0, 1] = (2 * eta + csi) * (1 - csi) / 4
        Dpsi[1, 1] = (2 * eta - csi) * (1 + csi) / 4
        Dpsi[2, 1] = (2 * eta + csi) * (1 + csi) / 4
        Dpsi[3, 1] = (2 * eta - csi) * (1 - csi) / 4
        Dpsi[4, 1] = (csi * csi - 1) / 2
        Dpsi[5, 1] = -(1 + csi) * eta
        Dpsi[6, 1] = (1 - csi * csi) / 2
        Dpsi[7, 1] = (csi - 1) * eta
        
        # Jacobian matrix
        J00 = 0.0
        J01 = 0.0
        J10 = 0.0
        J11 = 0.0
        for i in range(8):
            J00 += XN[i, 0] * Dpsi[i, 0]
            J01 += XN[i, 0] * Dpsi[i, 1]
            J10 += XN[i, 1] * Dpsi[i, 0]
            J11 += XN[i, 1] * Dpsi[i, 1]
        
        # Determinant and inverse
        Detj = J00 * J11 - J01 * J10
        
        if Detj <= 1.0e-12:
            return
        
        inv_det = 1.0 / Detj
        Invj00 = J11 * inv_det
        Invj01 = -J01 * inv_det
        Invj10 = -J10 * inv_det
        Invj11 = J00 * inv_det
        
        # B matrix (derivatives w.r.t. physical coordinates)
        B = cuda.local.array((8, 2), dtype=np.float64)
        for i in range(8):
            B[i, 0] = Dpsi[i, 0] * Invj00 + Dpsi[i, 1] * Invj10
            B[i, 1] = Dpsi[i, 0] * Invj01 + Dpsi[i, 1] * Invj11
        
        # Accumulate Ke and fe
        wip = w * Detj
        fL = 0.0  # Body force term
        
        for i in range(8):
            fe[i] += fL * wip * psi[i]
            for j in range(8):
                Ke[i, j] += wip * (B[i, 0] * B[j, 0] + B[i, 1] * B[j, 1])
    
    # Store Ke values (flattened)
    base_idx = e * 64
    k = 0
    for i in range(8):
        for j in range(8):
            vals_out[base_idx + k] = Ke[i, j]
            k += 1
    
    # Atomic update to global force vector
    for i in range(8):
        cuda.atomic.add(fg_out, edofs[i], fe[i])


# =============================================================================
# Post-Processing Kernel (Velocity Computation)
# =============================================================================

@cuda.jit
def quad8_postprocess_kernel(u, x, y, quad8, xp, vel_out, abs_vel_out):
    """
    Compute velocity field for all elements in parallel.
    
    Each thread handles one element.
    
    Parameters
    ----------
    u : (Nnds,) device array
        Solution vector (potential)
    x, y : (Nnds,) device arrays
        Nodal coordinates
    quad8 : (Nels, 8) device array
        Element connectivity
    xp : (4, 2) device array
        Integration points (4-point rule for post-processing)
    vel_out : (Nels, 2) device array
        Output: velocity vectors
    abs_vel_out : (Nels,) device array
        Output: velocity magnitudes
    """
    
    e = cuda.grid(1)
    
    if e >= quad8.shape[0]:
        return
    
    # Get element DOFs
    edofs = cuda.local.array(8, dtype=np.int32)
    for i in range(8):
        edofs[i] = quad8[e, i]
    
    # Get nodal coordinates and solution
    XN = cuda.local.array((8, 2), dtype=np.float64)
    u_e = cuda.local.array(8, dtype=np.float64)
    for i in range(8):
        XN[i, 0] = x[edofs[i]]
        XN[i, 1] = y[edofs[i]]
        u_e[i] = u[edofs[i]]
    
    # Accumulate velocity over integration points
    vel_x_sum = 0.0
    vel_y_sum = 0.0
    v_mag_sum = 0.0
    
    # 4-point integration
    for ip in range(4):
        csi = xp[ip, 0]
        eta = xp[ip, 1]
        
        # Shape function derivatives (parametric)
        Dpsi = cuda.local.array((8, 2), dtype=np.float64)
        
        Dpsi[0, 0] = (2 * csi + eta) * (1 - eta) / 4
        Dpsi[1, 0] = (2 * csi - eta) * (1 - eta) / 4
        Dpsi[2, 0] = (2 * csi + eta) * (1 + eta) / 4
        Dpsi[3, 0] = (2 * csi - eta) * (1 + eta) / 4
        Dpsi[4, 0] = csi * (eta - 1)
        Dpsi[5, 0] = (1 - eta * eta) / 2
        Dpsi[6, 0] = -csi * (1 + eta)
        Dpsi[7, 0] = (eta * eta - 1) / 2
        
        Dpsi[0, 1] = (2 * eta + csi) * (1 - csi) / 4
        Dpsi[1, 1] = (2 * eta - csi) * (1 + csi) / 4
        Dpsi[2, 1] = (2 * eta + csi) * (1 + csi) / 4
        Dpsi[3, 1] = (2 * eta - csi) * (1 - csi) / 4
        Dpsi[4, 1] = (csi * csi - 1) / 2
        Dpsi[5, 1] = -(1 + csi) * eta
        Dpsi[6, 1] = (1 - csi * csi) / 2
        Dpsi[7, 1] = (csi - 1) * eta
        
        # Jacobian matrix
        J00 = 0.0
        J01 = 0.0
        J10 = 0.0
        J11 = 0.0
        for i in range(8):
            J00 += XN[i, 0] * Dpsi[i, 0]
            J01 += XN[i, 0] * Dpsi[i, 1]
            J10 += XN[i, 1] * Dpsi[i, 0]
            J11 += XN[i, 1] * Dpsi[i, 1]
        
        # Determinant and inverse
        Detj = J00 * J11 - J01 * J10
        inv_det = 1.0 / Detj
        
        Invj00 = J11 * inv_det
        Invj01 = -J01 * inv_det
        Invj10 = -J10 * inv_det
        Invj11 = J00 * inv_det
        
        # Compute gradient
        grad_x = 0.0
        grad_y = 0.0
        for i in range(8):
            B_i0 = Dpsi[i, 0] * Invj00 + Dpsi[i, 1] * Invj10
            B_i1 = Dpsi[i, 0] * Invj01 + Dpsi[i, 1] * Invj11
            grad_x += B_i0 * u_e[i]
            grad_y += B_i1 * u_e[i]
        
        # Velocity is negative gradient
        vel_x_sum += -grad_x
        vel_y_sum += -grad_y
        v_mag_sum += math.sqrt(grad_x * grad_x + grad_y * grad_y)
    
    # Store averaged values
    vel_out[e, 0] = vel_x_sum / 4.0
    vel_out[e, 1] = vel_y_sum / 4.0
    abs_vel_out[e] = v_mag_sum / 4.0


# =============================================================================
# Helper Functions
# =============================================================================

def get_gauss_points_9():
    """Return 3x3 Gauss-Legendre points and weights."""
    G = np.sqrt(0.6)
    xp = np.array([
        [-G, -G], [0.0, -G], [G, -G],
        [-G, 0.0], [0.0, 0.0], [G, 0.0],
        [-G, G], [0.0, G], [G, G]
    ], dtype=np.float64)
    
    wp = np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float64) / 81.0
    
    return xp, wp


def get_gauss_points_4():
    """Return 2x2 Gauss-Legendre points."""
    G = np.sqrt(1.0 / 3.0)
    xp = np.array([
        [-G, -G], [G, -G], [G, G], [-G, G]
    ], dtype=np.float64)
    
    return xp
