"""
Build VarQEC error sets from the correlated trapped-ion noise model.

For channel-adaptive codes (VarQEC paper Sec 5.3, Eq. 19), the error set is:
    E = {E_alpha^dag E_beta | E_alpha, E_beta are Kraus operators of the noise channel}

Two error set builders:
1. build_correlated_error_set(): dephasing-only noise. All operators diagonal (1D vectors).
2. build_combined_error_set(): dephasing + subspace depolarizing. Full d^n x d^n matrices
   (depolarizing introduces off-diagonal elements).
"""
import numpy as np
from typing import List, Tuple
from itertools import product as cartesian_product
from src.trapped_ion_noise import (
    control_qudit_kraus, control_qudit_kraus_simplified,
    target_qudit_kraus, spectator_qudit_kraus,
    subspace_depolarizing_kraus
)
from src.kl_loss_fast import estimate_memory_mb


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
