"""
Correlated noise model for trapped-ion qudit gates.

Implements the dephasing noise from Appendix J of arXiv:2310.12110v3 (Meth et al. 2024).
During a C-ROT gate acting on levels (i,j) of a d-dimensional qudit:
- Control qudit: level i is unaffected, all other levels pick up phase 2*Phi
- Target qudit: level i gets phase Phi, level j gets 2*Phi, spectators get 3*Phi
- Other qudits in register: standard Gaussian dephasing on all levels

where Phi ~ N(0, sigma_p^2) and eta = exp(-sigma_p^2).

All Kraus operators are DIAGONAL matrices. We store them as 1D arrays
(the diagonal) for efficiency.
"""
import numpy as np
from typing import List, Tuple
from math import factorial, sqrt


def _kraus_diagonal(d: int, coupling_strengths: np.ndarray, eta: float, n_max: int) -> List[np.ndarray]:
    """
    Compute Kraus operator diagonals for Gaussian phase noise.

    For each Kraus index n = 0, 1, ..., n_max:
        E_n[k,k] = (f_k * sqrt(-2 * ln(eta)))^n * eta^{f_k^2} / sqrt(n!)

    where f_k is the coupling strength for level k.

    Args:
        d: qudit dimension
        coupling_strengths: array of shape (d,) with coupling strength f_k for each level
        eta: dephasing parameter, eta = exp(-sigma_p^2), range (0, 1]
        n_max: maximum Kraus index

    Returns:
        List of d-element complex arrays (diagonals of Kraus operators)
    """
    assert 0 < eta <= 1, f"eta must be in (0, 1], got {eta}"
    assert len(coupling_strengths) == d

    sigma_p2 = -np.log(eta)  # sigma_p^2 = -ln(eta)
    base = coupling_strengths * np.sqrt(2 * sigma_p2)  # f_k * sqrt(2*sigma_p^2)
    decay = eta ** (coupling_strengths ** 2)  # eta^{f_k^2}

    diags = []
    for n in range(n_max + 1):
        diag = (base ** n) * decay / sqrt(factorial(n))
        diags.append(diag.astype(complex))

    return diags


def control_qudit_kraus(d: int, control_level: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the CONTROL qudit during C-ROT.

    "Physical" model (Model B): level-dependent coupling f_k = k.
    Higher levels dephase faster — standard Gaussian phase damping where
    higher motional states couple more strongly to the environment.

    Coupling: f_k = k for k != control_level, f_control = 0.
    For d=5, ctrl=0: f = {0, 1, 2, 3, 4}.

    See also: control_qudit_kraus_simplified() for the literal Eq. J3 model.

    Args:
        d: qudit dimension
        control_level: level i (unaffected by noise)
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of d-element complex arrays (Kraus operator diagonals)
    """
    coupling = np.arange(d, dtype=float)
    coupling[control_level] = 0.0
    return _kraus_diagonal(d, coupling, eta, n_max)


def control_qudit_kraus_simplified(d: int, control_level: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the CONTROL qudit during C-ROT.

    "Simplified" model (Model A): literal Eq. J3 of arXiv:2310.12110v3.
    ALL non-control levels get the SAME phase 2*Phi, i.e. uniform coupling f=2.

    Coupling: f_k = 2 for k != control_level, f_control = 0.
    For d=5, ctrl=0: f = {0, 2, 2, 2, 2}.

    This matches the paper's description "c_i|i> + e^{i2Phi} sum_{j!=i} c_j|j>"
    where ALL non-control levels pick up the same phase factor.

    Args:
        d: qudit dimension
        control_level: level i (unaffected by noise)
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of d-element complex arrays (Kraus operator diagonals)
    """
    coupling = np.full(d, 2.0)
    coupling[control_level] = 0.0
    return _kraus_diagonal(d, coupling, eta, n_max)


def target_qudit_kraus(d: int, control_level: int, target_level: int,
                       eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the TARGET qudit during C-ROT.

    From Eq. J4 of arXiv:2310.12110v3:
    Three tiers of coupling:
        f(k) = 1 if k = control_level (i)
        f(k) = 2 if k = target_level (j)
        f(k) = 3 if k not in {i, j} (spectator levels)

    Args:
        d: qudit dimension
        control_level: level i
        target_level: level j
        eta: dephasing parameter
        n_max: truncation order

    Returns:
        List of d-element complex arrays
    """
    coupling = np.full(d, 3.0)  # default: spectator coupling = 3
    coupling[control_level] = 1.0
    coupling[target_level] = 2.0

    return _kraus_diagonal(d, coupling, eta, n_max)


def spectator_qudit_kraus(d: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on SPECTATOR qudits (not involved in gate).

    Standard Gaussian dephasing on all levels:
    D_n = sum_k [k*sqrt(2*sigma_p^2)]^n * eta^{k^2} / sqrt(n!) * |k><k|

    Args:
        d: qudit dimension
        eta: dephasing parameter
        n_max: truncation order

    Returns:
        List of d-element complex arrays
    """
    coupling = np.arange(d, dtype=float)
    return _kraus_diagonal(d, coupling, eta, n_max)


def subspace_depolarizing_kraus(d: int, level_i: int, level_j: int,
                                 p: float) -> List[np.ndarray]:
    """
    Kraus operators for depolarizing noise restricted to the 2D subspace {|i>, |j>}.

    From Martin Ringbauer's experimental characterization:
    "Gate errors are generally depolarizing in the subspace where the gate acts"

    The noise channel is:
        N(rho) = (1-p)*rho + (p/3)*(X_{ij}*rho*X_{ij} + Y_{ij}*rho*Y_{ij} + Z_{ij}*rho*Z_{ij})
    restricted to the gate subspace, with identity on spectator levels.

    Args:
        d: qudit dimension
        level_i: first level of gate subspace
        level_j: second level of gate subspace
        p: depolarizing error probability (0 = no error, 1 = full depolarizing)

    Returns:
        List of 4 d x d complex matrices (K_0 = no-error, K_1,2,3 = Pauli errors in subspace)
    """
    assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
    assert 0 <= level_i < d and 0 <= level_j < d and level_i != level_j

    # Build projectors
    P_sub = np.zeros((d, d), dtype=complex)
    P_sub[level_i, level_i] = 1.0
    P_sub[level_j, level_j] = 1.0
    P_perp = np.eye(d, dtype=complex) - P_sub

    # No-error Kraus: sqrt(1-p) on subspace, identity on complement
    K0 = np.sqrt(1 - p) * P_sub + P_perp

    # Subspace Pauli X: |i><j| + |j><i|
    X_sub = np.zeros((d, d), dtype=complex)
    X_sub[level_i, level_j] = 1.0
    X_sub[level_j, level_i] = 1.0
    K1 = np.sqrt(p / 3) * X_sub

    # Subspace Pauli Y: -i|i><j| + i|j><i|
    Y_sub = np.zeros((d, d), dtype=complex)
    Y_sub[level_i, level_j] = -1j
    Y_sub[level_j, level_i] = 1j
    K2 = np.sqrt(p / 3) * Y_sub

    # Subspace Pauli Z: |i><i| - |j><j|
    Z_sub = np.zeros((d, d), dtype=complex)
    Z_sub[level_i, level_i] = 1.0
    Z_sub[level_j, level_j] = -1.0
    K3 = np.sqrt(p / 3) * Z_sub

    return [K0, K1, K2, K3]


def combined_gate_kraus(d: int, control_level: int, target_level: int,
                        eta_dephasing: float, p_depolarizing: float,
                        n_max_dephasing: int = 5, noise_model: str = "physical") -> dict:
    """
    Complete noise model for a C-ROT gate: dephasing + subspace depolarizing.

    Returns Kraus operators for each qudit role:
    - control_qudit: dephasing Kraus ops (diagonal, from Eq. J3)
    - target_qudit: dephasing Kraus ops (diagonal, from Eq. J4)
    - gate_subspace: subspace depolarizing Kraus ops (NOT diagonal, full d x d)
    - spectator_qudit: dephasing Kraus ops (diagonal)

    The full channel per gate is the composition:
        N_gate = N_subspace_depol o N_dephasing

    Args:
        d: qudit dimension
        control_level: level i (control of C-ROT)
        target_level: level j (target of C-ROT)
        eta_dephasing: dephasing parameter eta = exp(-sigma_p^2)
        p_depolarizing: depolarizing probability in gate subspace
        n_max_dephasing: truncation order for dephasing Kraus series
        noise_model: "physical" (f_k=k, level-dependent) or "simplified" (f=2, Eq. J3 literal)

    Returns:
        dict with keys:
          'control_dephasing': list of diagonal arrays (n_max+1 operators)
          'target_dephasing': list of diagonal arrays (n_max+1 operators)
          'subspace_depol': list of d x d matrices (4 operators: I, X, Y, Z in subspace)
          'spectator_dephasing': list of diagonal arrays (n_max+1 operators)
    """
    if noise_model == "simplified":
        ctrl_kraus = control_qudit_kraus_simplified(d, control_level, eta_dephasing, n_max_dephasing)
    elif noise_model == "physical":
        ctrl_kraus = control_qudit_kraus(d, control_level, eta_dephasing, n_max_dephasing)
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    return {
        'control_dephasing': ctrl_kraus,
        'target_dephasing': target_qudit_kraus(d, control_level, target_level, eta_dephasing, n_max_dephasing),
        'subspace_depol': subspace_depolarizing_kraus(d, control_level, target_level, p_depolarizing),
        'spectator_dephasing': spectator_qudit_kraus(d, eta_dephasing, n_max_dephasing),
    }


def verify_kraus_completeness(diags: List[np.ndarray], atol: float = 1e-10) -> Tuple[bool, float]:
    """
    Check that sum_n E_n^dag E_n ~ I for diagonal Kraus operators.

    Since E_n is diagonal with diagonal d_n, E_n^dag E_n has diagonal |d_n|^2.
    Completeness means sum_n |d_n[k]|^2 = 1 for all k.

    Args:
        diags: list of 1D arrays (Kraus operator diagonals)
        atol: absolute tolerance

    Returns:
        (is_complete, max_deviation): tuple of (bool, float)
    """
    sum_sq = np.zeros_like(diags[0], dtype=float)
    for d in diags:
        sum_sq += np.abs(d) ** 2

    deviation = np.max(np.abs(sum_sq - 1.0))
    return deviation < atol, float(deviation)


def verify_matrix_kraus_completeness(kraus_ops: List[np.ndarray], atol: float = 1e-10) -> Tuple[bool, float]:
    """
    Check that sum_k K_k^dag K_k = I for full matrix Kraus operators.

    Args:
        kraus_ops: list of d x d complex matrices
        atol: absolute tolerance

    Returns:
        (is_complete, max_deviation): tuple of (bool, float)
    """
    d = kraus_ops[0].shape[0]
    total = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        total += K.conj().T @ K
    deviation = np.max(np.abs(total - np.eye(d)))
    return deviation < atol, float(deviation)
