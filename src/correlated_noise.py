"""
Correlated trapped-ion noise model (Meth et al. arXiv:2310.12110v3, App. J).

Sections:
  1. Single-qudit Kraus operators (dephasing + subspace depolarizing)
  2. Combined gate noise (dephasing + depolarizing composition)
  3. Correlated error set construction (tensor products of Kraus operators)

Merged from: trapped_ion_noise.py, correlated_error_sets.py
"""
import numpy as np
from itertools import product as cartesian_product
from typing import List, Tuple
from math import factorial, sqrt
from src.loss import estimate_memory_mb


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


# ═══════════════════════════════════════════════════════════════════════
# Correlated error set construction (from correlated_error_sets.py)
# ═══════════════════════════════════════════════════════════════════════

def _single_qudit_error_products(kraus_diags: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute all error products E_alpha^dag E_beta for a single qudit's Kraus operators.

    Since operators are diagonal: (E_alpha^dag E_beta)[k,k] = conj(E_alpha[k,k]) * E_beta[k,k]
    = conj(diag_alpha[k]) * diag_beta[k]

    Returns unique products (deduplicated by value).
    """
    products = []
    seen = set()
    for da in kraus_diags:
        for db in kraus_diags:
            prod = np.conj(da) * db
            key = tuple(np.round(prod, decimals=12))
            if key not in seen:
                seen.add(key)
                products.append(prod)
    return products


def build_correlated_error_set(
    n_qudits: int,
    d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float,
    n_max: int = 2,
    truncation_threshold: float = 1e-8,
    noise_model: str = "physical"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build VarQEC error set for the correlated trapped-ion noise channel.

    For each gate acting on (qudit_a, qudit_b) at levels (ctrl_level, tgt_level):
    - qudit_a gets control_qudit_kraus (physical) or control_qudit_kraus_simplified
    - qudit_b gets target_qudit_kraus (same in both models)
    - all other qudits get spectator_qudit_kraus (same in both models)

    All operators are diagonal -> stored as 1D vectors of length d^n_qudits.

    Args:
        n_qudits: number of physical qudits
        d: qudit dimension (3, 5, or 7)
        gate_pairs: list of (qudit_a, qudit_b, ctrl_level, tgt_level)
        eta: noise parameter eta = exp(-sigma_p^2)
        n_max: Kraus truncation order (2-3 is usually sufficient)
        truncation_threshold: drop operators with max|diag| below this
        noise_model: "physical" (f_k=k) or "simplified" (f=2, Eq. J3 literal)

    Returns:
        E_detect: list of 1D arrays of length d^n_qudits (error operator diagonals)
        E_correct: same list (for this noise model, detect = correct)
    """
    if noise_model == "simplified":
        _control_kraus_fn = control_qudit_kraus_simplified
    elif noise_model == "physical":
        _control_kraus_fn = control_qudit_kraus
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    dim_full = d ** n_qudits
    all_error_diags = set()

    # Identity is always in the error set
    identity_diag = np.ones(dim_full, dtype=complex)
    all_error_diags.add(tuple(np.round(identity_diag, decimals=12)))

    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        # Get single-qudit Kraus diagonals for each role
        per_qudit_products = []
        for q in range(n_qudits):
            if q == qudit_a:
                kraus = _control_kraus_fn(d, ctrl_level, eta, n_max)
            elif q == qudit_b:
                kraus = target_qudit_kraus(d, ctrl_level, tgt_level, eta, n_max)
            else:
                kraus = spectator_qudit_kraus(d, eta, n_max)
            per_qudit_products.append(_single_qudit_error_products(kraus))

        # Tensor product of single-qudit error products
        for combo in cartesian_product(*per_qudit_products):
            # Tensor product of diagonals = kronecker product of 1D vectors
            full_diag = combo[0]
            for q_diag in combo[1:]:
                full_diag = np.kron(full_diag, q_diag)

            # Truncation: skip near-identity or near-zero operators
            if np.max(np.abs(full_diag - identity_diag)) < truncation_threshold:
                continue  # too close to identity, skip
            if np.max(np.abs(full_diag)) < truncation_threshold:
                continue  # negligible operator

            key = tuple(np.round(full_diag, decimals=12))
            all_error_diags.add(key)

    # Convert back to arrays, add identity
    error_list = [identity_diag]
    for key in all_error_diags:
        diag = np.array(key, dtype=complex)
        if not np.allclose(diag, identity_diag):
            error_list.append(diag)

    print(f"Correlated error set ({noise_model}): {len(error_list)} operators for {n_qudits} qudits (d={d})")

    # For channel-adaptive codes, E_detect = E_correct
    return error_list, error_list


def _compose_kraus_single_qudit(dephasing_diags: List[np.ndarray],
                                 depol_matrices: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose dephasing and subspace depolarizing Kraus ops for one qudit.

    The composed channel is N_depol o N_dephasing, so:
        K_{n,k} = S_k @ diag(D_n)
    for dephasing index n and depolarizing index k.

    Returns list of d x d matrices.
    """
    composed = []
    for D_n in dephasing_diags:
        D_matrix = np.diag(D_n)
        for S_k in depol_matrices:
            composed.append(S_k @ D_matrix)
    return composed


def _matrix_hash(M, decimals=10):
    """Memory-efficient hash for matrices. Uses only non-zero elements."""
    nz = np.nonzero(np.abs(M) > 1e-14)
    if len(nz[0]) == 0:
        return (0,)
    vals = M[nz]
    return (tuple(nz[0]) + tuple(nz[1])
            + tuple(np.round(vals.real, decimals))
            + tuple(np.round(vals.imag, decimals)))


def _single_qudit_matrix_error_products(kraus_matrices: List[np.ndarray],
                                         truncation_threshold: float = 1e-8) -> List[np.ndarray]:
    """
    Compute unique error products E_alpha^dag E_beta for full matrix Kraus operators.

    Returns list of d x d matrices (deduplicated).
    """
    products = []
    seen = set()
    for Ka in kraus_matrices:
        Ka_dag = Ka.conj().T
        for Kb in kraus_matrices:
            M = Ka_dag @ Kb
            if np.max(np.abs(M)) < truncation_threshold:
                continue
            key = _matrix_hash(M)
            if key not in seen:
                seen.add(key)
                products.append(M)
    return products


def build_combined_error_set(
    n_qudits: int,
    d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta_dephasing: float,
    p_depolarizing: float,
    n_max: int = 2,
    truncation_threshold: float = 1e-8,
    noise_model: str = "physical"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build VarQEC error set for the COMBINED trapped-ion noise channel
    (spectator dephasing + subspace depolarizing).

    This is the physically correct noise model from arXiv:2310.12110v3.

    Error products E_alpha^dag E_beta include both diagonal (dephasing) and
    non-diagonal (depolarizing) components. Stored as full d^n x d^n matrices.

    Args:
        n_qudits: number of physical qudits
        d: qudit dimension (3, 5, or 7)
        gate_pairs: list of (qudit_a, qudit_b, ctrl_level, tgt_level)
        eta_dephasing: noise parameter for dephasing
        p_depolarizing: noise parameter for subspace depolarizing
        n_max: Kraus truncation order for dephasing
        truncation_threshold: drop negligible operators
        noise_model: "physical" (f_k=k) or "simplified" (f=2, Eq. J3 literal)

    Returns:
        E_detect, E_correct: lists of d^n x d^n complex matrices
    """
    dim_full = d ** n_qudits

    # Memory guard: estimate worst case for full-system operators
    n_products_per_gate_qudit = ((n_max + 1) * 4) ** 2
    n_products_per_spectator = (n_max + 1) ** 2
    max_products = max(n_products_per_gate_qudit, n_products_per_spectator)
    max_combos = max_products ** n_qudits
    est_mb = estimate_memory_mb(min(max_combos, 10000), dim_full)
    if est_mb > 4000:
        raise MemoryError(
            f"Estimated {est_mb:.0f} MB for {n_qudits} qudits (d={d}). "
            f"Reduce n_max (currently {n_max}) or n_qudits, or increase truncation_threshold."
        )

    if noise_model == "simplified":
        _control_kraus_fn = control_qudit_kraus_simplified
    elif noise_model == "physical":
        _control_kraus_fn = control_qudit_kraus
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    identity = np.eye(dim_full, dtype=complex)

    # Collect all unique full-system error operators using memory-efficient hash
    all_errors = {}
    all_errors[_matrix_hash(identity)] = identity

    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        # Build single-qudit error products for each qudit
        per_qudit_products = []
        for q in range(n_qudits):
            if q == qudit_a:
                deph = _control_kraus_fn(d, ctrl_level, eta_dephasing, n_max)
                depol = subspace_depolarizing_kraus(d, ctrl_level, tgt_level, p_depolarizing)
                composed = _compose_kraus_single_qudit(deph, depol)
                products = _single_qudit_matrix_error_products(composed, truncation_threshold)
            elif q == qudit_b:
                deph = target_qudit_kraus(d, ctrl_level, tgt_level, eta_dephasing, n_max)
                depol = subspace_depolarizing_kraus(d, ctrl_level, tgt_level, p_depolarizing)
                composed = _compose_kraus_single_qudit(deph, depol)
                products = _single_qudit_matrix_error_products(composed, truncation_threshold)
            else:
                # Spectator: dephasing only, diagonal
                deph_diags = spectator_qudit_kraus(d, eta_dephasing, n_max)
                # Convert to matrix products for uniform handling
                products = _single_qudit_matrix_error_products(
                    [np.diag(diag) for diag in deph_diags], truncation_threshold)
            per_qudit_products.append(products)

        # Tensor product across qudits
        for combo in cartesian_product(*per_qudit_products):
            full_op = combo[0]
            for q_op in combo[1:]:
                full_op = np.kron(full_op, q_op)

            # Skip near-identity
            if np.max(np.abs(full_op - identity)) < truncation_threshold:
                continue
            # Skip negligible
            if np.max(np.abs(full_op)) < truncation_threshold:
                continue

            key = _matrix_hash(full_op)
            if key not in all_errors:
                all_errors[key] = full_op

    error_list = list(all_errors.values())
    print(f"Combined error set ({noise_model}): {len(error_list)} operators for {n_qudits} qudits (d={d})")

    # For channel-adaptive codes, E_detect = E_correct
    return error_list, error_list
