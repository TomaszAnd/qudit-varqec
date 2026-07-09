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

    # gate form verified against Ringbauer et al., Nat. Phys. 18, 1053 (2022),
    # Eq. (2) (arXiv:2109.06903, label eq:entOps):
    #   MS(theta, phi) = expm(-i(theta/4)(sigma_phi (x) I + I (x) sigma_phi)^2)
    # Numerically identical to the matrix exponential to <1e-15 for
    # d in {2,3,5} and all level pairs, including the single-active spectator
    # phase e^{-i theta/4}; at d=2 reduces to the standard qubit MS up to the
    # documented global phase e^{-i theta/2}.

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


def CEX_gate(d: int = 3, c: int = 1, t1: int = 0, t2: int = 1):
    """
    Controlled-EXchange gate: an EMBEDDED-qubit entangler.

    Fixed (parameter-free) two-qudit permutation. With the target's control
    level fixed at |c>, it swaps the two target levels |t1> <-> |t2>; every
    other basis state is left unchanged (identity). Basis index |a,b> = a*d + b.

        |c, t1> <-> |c, t2>
        |a, b>  ->  |a, b>   for all (a, b) not in {(c, t1), (c, t2)}

    It is an involution (a single basis transposition, CEX = CEX^dag = CEX^-1).
    Unlike the genuine-qudit CSUM (|i,j> -> |i, j+i mod d>, which entangles ALL
    levels and makes |00>+|11>+...+|(d-1)(d-1)>), CEX acts only inside the 2-level
    target subspace {|t1>, |t2>}, producing an EMBEDDED-qubit Bell state |00>+|11>.

    At d=2, c=1, t1=0, t2=1 it is exactly the qubit CNOT (control |1>, flip target).
    For d=3, c=1, t1=0, t2=1 it swaps indices 3 (|10>) and 4 (|11>), identity
    elsewhere.

    Library-only / non-default: the trained ansatz uses the physical XX/YY
    Molmer-Sorensen entangler (see MS_gate). CEX is provided for the
    embedded-qubit vs genuine-qudit gate-set comparison.

    Args:
        d: qudit dimension (>= 2)
        c: control level (target levels swap only when control is in this level)
        t1, t2: the two target levels that are exchanged (t1 != t2)
    """
    assert d >= 2, f"d must be >= 2, got {d}"
    assert 0 <= c < d, f"control level c must be in [0, d), got {c}"
    assert 0 <= t1 < d and 0 <= t2 < d and t1 != t2, \
        f"target levels must be distinct and in [0, d), got t1={t1}, t2={t2}"
    dim2 = d * d
    U = np.eye(dim2, dtype=complex)
    i1 = c * d + t1
    i2 = c * d + t2
    U[i1, i1] = 0.0
    U[i2, i2] = 0.0
    U[i1, i2] = 1.0  # |c,t2> -> |c,t1>
    U[i2, i1] = 1.0  # |c,t1> -> |c,t2>
    return U


def zz_subspace_gate(theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Subspace ZZ gate on the (level_j, level_k) levels (paper Eq. (4)).

    Formerly `light_shift_gate` — renamed because it is NOT the physical
    light-shift gate of Hrmo et al. (see `ls_global_gate`); it is identical
    to Ulrich's `entangling_MS_gate`, which is not an MS gate either.

    Diagonal ZZ-type interaction restricted to two chosen levels. Written
    by Ulrich; used in the §VII gate-set ablations of the paper.

    U|a,b> = exp(-i*theta/2 * Z_jk(a) * Z_jk(b)) |a,b>

    where Z_jk has +1 on level_j, -1 on level_k, 0 elsewhere. The gate
    acts non-trivially only on basis states where both qudits are in
    {level_j, level_k}; spectator levels accumulate no phase.

    NOTE: This is a level-restricted approximation, not the native qudit
    light-shift gate of Hrmo et al. 2023 (arXiv:2206.04104, Eq. 3). For
    the genuine high-dimensional Hrmo gate that acts on all level pairs
    of the q^2 Hilbert space simultaneously, see `ls_global_gate`.
    The two coincide at d = 2 but diverge for d >= 3.

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


def ls_global_gate(theta: float, d: int = 3):
    """
    Native qudit light-shift gate (Hrmo et al. 2023, Eq. 3).

    Formerly `light_shift_gate_hrmo`.

    Reference: H. Hrmo et al., "Native qudit entanglement in a trapped
    ion quantum processor," arXiv:2206.04104, Eq. 3 (page 3).

    Acts on the full d^2-dimensional two-qudit Hilbert space:

        G(theta) |j,j>           = |j,j>            for every j
        G(theta) |j,k>           = e^{i theta} |j,k>  for j != k

    Diagonal q^2 x q^2 matrix. Non-trivial phase e^{i theta} on every
    off-diagonal-product computational basis state (the q^2 - q states
    |j,k> with j != k); identity on the q diagonal states |j,j>. No
    level-pair argument — unlike the level-restricted `zz_subspace_gate`,
    this acts on all level pairs simultaneously and generates genuine
    high-dimensional entanglement in a single application.

    For d = 2 reduces to a global CZ-equivalent phase; for d >= 3 it
    diverges from the level-restricted approximation.

    One continuous parameter. Acts as identity at theta = 0 (mod 2*pi).

    Args:
        theta: phase angle applied to every |j,k> with j != k
        d: qudit dimension
    """
    dim2 = d * d
    diag = np.ones(dim2, dtype=complex)
    phase = qml.math.exp(1j * theta)
    for a in range(d):
        for b in range(d):
            if a != b:
                diag[a * d + b] = phase
    return qml.math.diag(diag)


# ── Deprecated aliases (pre-rename names; kept for historical scripts) ──
light_shift_gate = zz_subspace_gate
light_shift_gate_hrmo = ls_global_gate
