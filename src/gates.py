"""
Native trapped-ion gates for qudit systems of any dimension d.

Gate set:
  - XY_gate: rotation in the XY plane of a (j,k) subspace
  - Z_gate: relative phase between levels j and k
  - MS_gate: Molmer-Sorensen entangling gate on a (j,k) transition

All gates are unitary and PennyLane autograd-compatible.
Works for any d >= 2 and any adjacent or non-adjacent (j,k) transition.
"""
import pennylane as qml
from pennylane import numpy as np


def XY_gate(phi: float, alpha: float, level_j: int, level_k: int, d: int = 3):
    """
    XY rotation in the (level_j, level_k) subspace.

    U = proj_rest + cos(alpha/2) * I_jk
        - i sin(alpha/2) [cos(phi) X_jk + sin(phi) Y_jk]

    Acts as identity on all levels outside {j, k}.

    Args:
        phi: rotation axis angle in the XY plane
        alpha: rotation angle
        level_j, level_k: active levels (0-indexed)
        d: qudit dimension
    """
    I_jk = np.zeros((d, d), dtype=complex)
    I_jk[level_j, level_j] = 1.0
    I_jk[level_k, level_k] = 1.0

    X_jk = np.zeros((d, d), dtype=complex)
    X_jk[level_j, level_k] = 1.0
    X_jk[level_k, level_j] = 1.0

    Y_jk = np.zeros((d, d), dtype=complex)
    Y_jk[level_j, level_k] = -1j
    Y_jk[level_k, level_j] = 1j

    proj_rest = np.eye(d, dtype=complex) - I_jk

    c = qml.math.cos(alpha / 2) + 0j
    s = qml.math.sin(alpha / 2) + 0j
    term_x = (-1j * s * qml.math.cos(phi)) * X_jk
    term_y = (-1j * s * qml.math.sin(phi)) * Y_jk

    return proj_rest + c * I_jk + term_x + term_y


def Z_gate(theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Relative phase gate between levels j and k.

    U = proj_rest + e^{i theta/2} |j><j| + e^{-i theta/2} |k><k|

    Acts as identity on all levels outside {j, k}.

    Args:
        theta: phase angle
        level_j, level_k: active levels
        d: qudit dimension
    """
    proj_j = np.zeros((d, d), dtype=complex)
    proj_j[level_j, level_j] = 1.0

    proj_k = np.zeros((d, d), dtype=complex)
    proj_k[level_k, level_k] = 1.0

    proj_rest = np.eye(d, dtype=complex) - proj_j - proj_k

    p_j = qml.math.exp(1j * theta / 2)
    p_k = qml.math.exp(-1j * theta / 2)

    return proj_rest + p_j * proj_j + p_k * proj_k


def _build_ms_masks(level_j, level_k, d):
    """
    Build mask matrices for the MS gate on (level_j, level_k) in dimension d.

    For two qudits with basis |ab> (index a*d + b):
      - Both a,b in {j,k}: active MS subspace (cos/sin mixing)
      - Exactly one in {j,k}: spectator-active phase p1
      - Neither in {j,k}: double spectator, identity

    Returns:
        (M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s): six d²×d² mask matrices
    """
    dim2 = d * d
    active = {level_j, level_k}
    j, k = level_j, level_k

    M_c = np.zeros((dim2, dim2), dtype=complex)
    M_p1 = np.zeros((dim2, dim2), dtype=complex)
    M_p0 = np.zeros((dim2, dim2), dtype=complex)
    M_s_minus = np.zeros((dim2, dim2), dtype=complex)
    M_s_plus = np.zeros((dim2, dim2), dtype=complex)
    M_s = np.zeros((dim2, dim2), dtype=complex)

    for a in range(d):
        for b in range(d):
            idx = a * d + b
            a_in = a in active
            b_in = b in active
            if a_in and b_in:
                M_c[idx, idx] = 1.0
            elif a_in or b_in:
                M_p1[idx, idx] = 1.0
            else:
                M_p0[idx, idx] = 1.0

    # |jj> <-> |kk> coupling
    idx_jj = j * d + j
    idx_kk = k * d + k
    M_s_minus[idx_jj, idx_kk] = 1.0
    M_s_plus[idx_kk, idx_jj] = 1.0

    # |jk> <-> |kj> coupling
    idx_jk = j * d + k
    idx_kj = k * d + j
    M_s[idx_jk, idx_kj] = 1.0
    M_s[idx_kj, idx_jk] = 1.0

    return M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s


def MS_gate(phi: float, theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Molmer-Sorensen entangling gate on the (level_j, level_k) transition.

    Two-qudit gate (d²×d² matrix). Analytically pre-solved to avoid
    autograd issues with matrix exponentiation.

    Works for any dimension d and any (j,k) transition. The mask matrices
    are computed from the index classification of two-qudit basis states.

    Args:
        phi: MS phase
        theta: MS angle
        level_j, level_k: transition levels
        d: qudit dimension
    """
    c = qml.math.exp(-1j * theta / 2) * qml.math.cos(theta / 2) + 0j
    s = -1j * qml.math.exp(-1j * theta / 2) * qml.math.sin(theta / 2) + 0j
    p1 = qml.math.exp(-1j * theta / 4) + 0j

    s_minus = s * qml.math.exp(-1j * 2 * phi)
    s_plus = s * qml.math.exp(1j * 2 * phi)

    M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s = _build_ms_masks(level_j, level_k, d)

    return (c * M_c + p1 * M_p1 + M_p0
            + s_minus * M_s_minus + s_plus * M_s_plus + s * M_s)


def CSUM_gate(d: int = 3):
    """
    Controlled-SUM gate: |a,b> -> |a, (a+b) mod d>.

    Generalization of CNOT for qudits. Fixed gate (no parameters).
    Works for any d >= 2.

    Args:
        d: qudit dimension
    """
    dim2 = d * d
    U = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            idx_in = a * d + b
            idx_out = a * d + ((a + b) % d)
            U[idx_out, idx_in] = 1.0
    return U


def CSUB_gate(d: int = 3):
    """
    Controlled-SUB gate: |a,b> -> |a, (b-a) mod d>.
    Written by Ulrich. Equivalent to CSUM†.

    Args:
        d: qudit dimension
    """
    dim2 = d * d
    U = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            idx_in = a * d + b
            idx_out = a * d + ((b - a) % d)
            U[idx_out, idx_in] = 1.0
    return U


def light_shift_gate(theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Light-shift entangling gate (diagonal ZZ-type interaction).
    Written by Ulrich. Geometric phase from off-resonant laser coupling.

    Acts on the (level_j, level_k) subspace of two qudits.
    U|a,b> = exp(-i*theta/2 * Z_jk(a) * Z_jk(b)) |a,b>

    where Z_jk has +1 on level_j, -1 on level_k, 0 elsewhere.

    One continuous parameter. Acts as identity at theta=0.

    Args:
        theta: interaction angle
        level_j, level_k: active levels
        d: qudit dimension
    """
    dim2 = d * d
    diag = np.ones(dim2, dtype=complex)
    for a in range(d):
        for b in range(d):
            z_a = 0.0
            if a == level_j:
                z_a = 1.0
            elif a == level_k:
                z_a = -1.0
            z_b = 0.0
            if b == level_j:
                z_b = 1.0
            elif b == level_k:
                z_b = -1.0
            diag[a * d + b] = qml.math.exp(-1j * theta / 2 * z_a * z_b)
    return qml.math.diag(diag)
