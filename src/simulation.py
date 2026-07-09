"""
Logical error rate simulation for VarQEC codes.

Monte Carlo simulation: encode a logical state, apply noise, project back
onto the code space, and check whether a logical error has occurred.

For a [[n,k,d]] code with K = d^k codewords {|psi_k>}:
1. Prepare a random logical state |L> = sum_k alpha_k |psi_k>
2. Apply noise: rho -> N(|L><L|) = sum_a K_a |L><L| K_a†
3. Project onto code space: P_code = sum_k |psi_k><psi_k|
4. Measure fidelity of decoded state with original
5. Count logical errors (fidelity < threshold)
"""
import numpy as np
from typing import List, Callable, Dict, Optional, Tuple


def build_code_projector(code_states: np.ndarray) -> np.ndarray:
    """
    Build projector onto the code space.

    Args:
        code_states: (K, dim) array of orthonormal codeword vectors

    Returns:
        (dim, dim) projector matrix P = sum_k |psi_k><psi_k|
    """
    # P[i,j] = sum_k psi_k[i] * conj(psi_k[j]) = (C^T @ conj(C))[i,j]
    return code_states.T @ np.conj(code_states)


def apply_diagonal_noise(state: np.ndarray, kraus_diags: List[np.ndarray],
                          rng: np.random.Generator) -> np.ndarray:
    """
    Apply a diagonal noise channel to a state vector by sampling one Kraus operator.

    The channel N(rho) = sum_n E_n rho E_n† is simulated by:
    1. Computing probability p_n = <psi|E_n†E_n|psi> = sum_k |E_n[k]|^2 |psi[k]|^2
    2. Sampling n with probability p_n
    3. Returning E_n|psi> / ||E_n|psi>||

    Args:
        state: (dim,) complex state vector
        kraus_diags: list of (dim,) diagonal Kraus operators
        rng: random number generator

    Returns:
        (dim,) noisy state vector (normalized)
    """
    probs = np.array([np.sum(np.abs(d)**2 * np.abs(state)**2) for d in kraus_diags])
    probs = probs / probs.sum()  # normalize for numerical safety

    idx = rng.choice(len(kraus_diags), p=probs)
    noisy = kraus_diags[idx] * state  # element-wise for diagonal
    norm = np.linalg.norm(noisy)
    if norm < 1e-15:
        return state  # fallback if collapsed
    return noisy / norm


def apply_matrix_noise(state: np.ndarray, kraus_ops: List[np.ndarray],
                        rng: np.random.Generator) -> np.ndarray:
    """
    Apply a noise channel with full matrix Kraus operators by sampling.

    Args:
        state: (dim,) complex state vector
        kraus_ops: list of (dim, dim) Kraus operator matrices
        rng: random number generator

    Returns:
        (dim,) noisy state vector (normalized)
    """
    probs = np.array([np.real(np.vdot(state, K.conj().T @ K @ state)) for K in kraus_ops])
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()

    idx = rng.choice(len(kraus_ops), p=probs)
    noisy = kraus_ops[idx] @ state
    norm = np.linalg.norm(noisy)
    if norm < 1e-15:
        return state
    return noisy / norm


def apply_pauli_noise(state: np.ndarray, error_ops: List[np.ndarray],
                       p_total: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply Pauli noise: with probability p_total, apply a random error from error_ops.
    With probability 1-p_total, do nothing.

    Args:
        state: (dim,) state vector
        error_ops: list of (dim, dim) Pauli error operators (excluding identity)
        p_total: total error probability
        rng: random number generator

    Returns:
        (dim,) noisy state vector
    """
    if rng.random() > p_total:
        return state.copy()
    idx = rng.integers(0, len(error_ops))
    noisy = error_ops[idx] @ state
    norm = np.linalg.norm(noisy)
    return noisy / norm if norm > 1e-15 else state


def code_space_fidelity(state: np.ndarray, code_states: np.ndarray,
                         logical_coeffs: np.ndarray) -> float:
    """
    Compute fidelity of a (potentially noisy) state with the intended logical state.

    The logical state is |L> = sum_k alpha_k |psi_k>.
    Fidelity = |<L|state>|^2.

    Args:
        state: (dim,) state vector after noise
        code_states: (K, dim) codeword matrix
        logical_coeffs: (K,) coefficients alpha_k

    Returns:
        fidelity in [0, 1]
    """
    logical_state = logical_coeffs @ code_states  # (dim,)
    return float(np.abs(np.vdot(logical_state, state))**2)


def simulate_logical_error_rate(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_shots: int = 1000,
    fidelity_threshold: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Monte Carlo simulation of logical error rate for a given code and noise channel.

    For each shot:
    1. Pick a random logical state |L> = sum_k alpha_k |psi_k>
    2. Apply noise: |L'> = noise_fn(|L>)
    3. Compute fidelity F = |<L|L'>|^2
    4. If F < fidelity_threshold, count as logical error

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        n_shots: number of Monte Carlo samples
        fidelity_threshold: fidelity below this counts as a logical error
        seed: random seed

    Returns:
        dict with 'logical_error_rate', 'mean_fidelity', 'std_fidelity', 'n_shots'
    """
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    fidelities = np.zeros(n_shots)

    for shot in range(n_shots):
        # Random logical state (Haar-random coefficients)
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)

        logical_state = coeffs @ code_states  # (dim,)

        # Apply noise
        noisy_state = noise_fn(logical_state, rng)

        # Fidelity with original logical state
        fidelities[shot] = float(np.abs(np.vdot(logical_state, noisy_state))**2)

    n_errors = np.sum(fidelities < fidelity_threshold)
    return {
        'logical_error_rate': float(n_errors / n_shots),
        'mean_fidelity': float(np.mean(fidelities)),
        'std_fidelity': float(np.std(fidelities)),
        'n_shots': n_shots,
    }


def sweep_logical_error_rate(
    code_states: np.ndarray,
    noise_fn_factory: Callable[[float], Callable],
    physical_error_rates: List[float],
    n_shots: int = 1000,
    fidelity_threshold: float = 0.5,
    seed: int = 42,
    label: str = ""
) -> Dict[str, any]:
    """
    Sweep logical error rate over physical error rates for threshold-style plots.

    Args:
        code_states: (K, dim) codeword matrix
        noise_fn_factory: function(p) -> noise_fn, where noise_fn(state, rng) -> noisy_state
        physical_error_rates: list of physical error rates to sweep
        n_shots: shots per error rate point
        fidelity_threshold: fidelity threshold for logical error
        seed: random seed
        label: label for this sweep

    Returns:
        dict with 'physical_rates', 'logical_rates', 'mean_fidelities', 'label'
    """
    logical_rates = []
    mean_fids = []

    for p in physical_error_rates:
        noise_fn = noise_fn_factory(p)
        result = simulate_logical_error_rate(
            code_states, noise_fn, n_shots=n_shots,
            fidelity_threshold=fidelity_threshold, seed=seed
        )
        logical_rates.append(result['logical_error_rate'])
        mean_fids.append(result['mean_fidelity'])
        if label:
            print(f"  {label}: p={p:.4f} -> LER={result['logical_error_rate']:.4f}, "
                  f"F_mean={result['mean_fidelity']:.4f}")

    return {
        'physical_rates': physical_error_rates,
        'logical_rates': logical_rates,
        'mean_fidelities': mean_fids,
        'label': label,
    }


# === Noise channel factories ===

def make_dephasing_noise_fn(error_ops: List[np.ndarray], p_per_error: float):
    """
    Create a noise function for standard Pauli dephasing.

    Args:
        error_ops: list of Pauli Z-type error matrices
        p_per_error: probability per individual error

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    p_total = min(1.0, p_per_error * len(error_ops))

    def noise_fn(state, rng):
        return apply_pauli_noise(state, error_ops, p_total, rng)
    return noise_fn


def make_depolarizing_noise_fn(error_ops: List[np.ndarray], p_per_error: float):
    """
    Create a noise function for standard Pauli depolarizing.

    Args:
        error_ops: list of all non-identity Pauli error matrices
        p_per_error: probability per individual error

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    p_total = min(1.0, p_per_error * len(error_ops))

    def noise_fn(state, rng):
        return apply_pauli_noise(state, error_ops, p_total, rng)
    return noise_fn


def make_pauli_dephasing_noise_fn(single_qudit_errors: List[np.ndarray],
                                  n_qudits: int, dim_qudit: int, p: float):
    """
    Apply independent per-qudit dephasing noise.

    For each qudit independently:
    - With probability (1-p): do nothing
    - With probability p: apply a random Z-type Pauli (IZ, ZI, or ZZ) to that qudit

    This matches the noise model that dephasing error sets are built for.

    Args:
        single_qudit_errors: list of single-qudit Z-type Pauli matrices (d x d)
        n_qudits: number of qudits
        dim_qudit: local qudit dimension
        p: per-qudit error probability
    """
    d = dim_qudit
    n_errors = len(single_qudit_errors)

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudits):
            if rng.random() < p:
                # Pick a random single-qudit error
                idx = rng.integers(0, n_errors)
                E_local = single_qudit_errors[idx]
                # Apply to qudit q via tensor reshape
                shape = [d] * n_qudits
                s_tensor = s.reshape(shape)
                # Contract E_local on axis q
                s_tensor = np.tensordot(E_local, s_tensor, axes=([1], [q]))
                s_tensor = np.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = np.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


def make_pauli_depolarizing_noise_fn(single_qudit_errors: List[np.ndarray],
                                     n_qudits: int, dim_qudit: int, p: float):
    """
    Apply independent per-qudit depolarizing noise.

    For each qudit independently:
    - With probability (1-p): do nothing
    - With probability p: apply a random Pauli (all 15 non-identity) to that qudit

    Args:
        single_qudit_errors: list of single-qudit Pauli matrices (d²-1 = 15 for d=4)
        n_qudits: number of qudits
        dim_qudit: local qudit dimension
        p: per-qudit error probability
    """
    d = dim_qudit
    n_errors = len(single_qudit_errors)

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudits):
            if rng.random() < p:
                idx = rng.integers(0, n_errors)
                E_local = single_qudit_errors[idx]
                shape = [d] * n_qudits
                s_tensor = s.reshape(shape)
                s_tensor = np.tensordot(E_local, s_tensor, axes=([1], [q]))
                s_tensor = np.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = np.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


def make_correlated_dephasing_noise_fn(
    n_qudits: int, d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float, n_max: int = 5,
    noise_model: str = "physical"
):
    """
    Create a noise function for correlated trapped-ion dephasing.

    For each gate, applies Kraus noise to the full system by tensoring
    single-qudit Kraus operators.

    Args:
        n_qudits: number of qudits
        d: qudit dimension
        gate_pairs: list of (qudit_a, qudit_b, ctrl_level, tgt_level)
        eta: dephasing parameter
        n_max: Kraus truncation order
        noise_model: "physical" or "simplified"

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    from src.correlated_noise import (
        control_qudit_kraus, control_qudit_kraus_simplified,
        target_qudit_kraus, spectator_qudit_kraus
    )

    if noise_model == "simplified":
        ctrl_fn = control_qudit_kraus_simplified
    else:
        ctrl_fn = control_qudit_kraus

    # Pre-build per-gate, per-qudit Kraus diag lists
    gate_kraus = []
    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        per_qudit = []
        for q in range(n_qudits):
            if q == qudit_a:
                per_qudit.append(ctrl_fn(d, ctrl_level, eta, n_max))
            elif q == qudit_b:
                per_qudit.append(target_qudit_kraus(d, ctrl_level, tgt_level, eta, n_max))
            else:
                per_qudit.append(spectator_qudit_kraus(d, eta, n_max))
        gate_kraus.append(per_qudit)

    def noise_fn(state, rng):
        s = state.copy()
        for per_qudit in gate_kraus:
            # For each gate, sample one Kraus index per qudit
            # Build the full-system diagonal = kron of per-qudit diags
            # Sample proportional to probability
            n_ops_per_qudit = [len(kd) for kd in per_qudit]

            # Compute all combo probabilities efficiently:
            # For diagonal Kraus, we can sample each qudit independently
            # because the full Kraus is the tensor product
            full_diag = np.ones(d**n_qudits, dtype=complex)
            for q in range(n_qudits):
                kd_list = per_qudit[q]
                # Compute prob for each Kraus index on this qudit
                # Reshape state to access qudit q
                shape = [d] * n_qudits
                s_tensor = np.abs(s.reshape(shape))**2
                # Sum over all other qudits to get marginal probs for qudit q
                axes_to_sum = tuple(i for i in range(n_qudits) if i != q)
                marginal = np.sum(s_tensor, axis=axes_to_sum)  # shape (d,)

                q_probs = np.array([np.sum(np.abs(kd)**2 * marginal) for kd in kd_list])
                q_probs = q_probs / q_probs.sum()

                idx = rng.choice(len(kd_list), p=q_probs)
                chosen_diag = kd_list[idx]

                # Apply to qudit q: multiply state by chosen_diag on qudit axis
                s_reshaped = s.reshape(shape)
                # Expand chosen_diag to broadcast on axis q
                expand_shape = [1] * n_qudits
                expand_shape[q] = d
                s_reshaped = s_reshaped * chosen_diag.reshape(expand_shape)
                s = s_reshaped.reshape(-1)

            norm = np.linalg.norm(s)
            if norm > 1e-15:
                s = s / norm
        return s

    return noise_fn


# === Detection and correction simulation functions ===

def simulate_ler_with_detection(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_shots: int = 1000,
    detection_threshold: float = 0.1,
    seed: int = 42
) -> Dict[str, float]:
    """
    For d=2 codes: apply noise, project onto code space, detect errors.

    If the projection norm^2 < detection_threshold, the error is detected and
    the shot is discarded (post-selection). For undetected shots, compute
    fidelity with the original logical state.

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        n_shots: number of Monte Carlo samples
        detection_threshold: projection norm^2 below this = detected
        seed: random seed

    Returns:
        dict with 'detected_fraction', 'undetected_error_rate',
        'post_selected_fidelity', 'mean_raw_fidelity', 'n_shots'
    """
    from src.simulation import detection_decoder

    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    n_detected = 0
    post_selected_fidelities = []
    raw_fidelities = []

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states

        noisy_state = noise_fn(logical_state, rng)

        # Raw fidelity (no decoding)
        raw_fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        raw_fidelities.append(raw_fid)

        # Detection decoder
        decoded, detected = detection_decoder(noisy_state, code_states, detection_threshold)

        if detected:
            n_detected += 1
        else:
            fid = float(np.abs(np.vdot(logical_state, decoded))**2)
            post_selected_fidelities.append(fid)

    detected_frac = n_detected / n_shots
    if post_selected_fidelities:
        ps_fid = float(np.mean(post_selected_fidelities))
        ps_err = float(np.mean([f < 0.5 for f in post_selected_fidelities]))
    else:
        ps_fid = 0.0
        ps_err = 1.0

    return {
        'detected_fraction': detected_frac,
        'undetected_error_rate': ps_err,
        'post_selected_fidelity': ps_fid,
        'mean_raw_fidelity': float(np.mean(raw_fidelities)),
        'n_shots': n_shots,
    }


def simulate_ler_with_correction(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    error_ops: List[np.ndarray],
    n_shots: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    For d>=3 codes: apply noise, measure syndrome, apply correction.

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        error_ops: list of correctable error operators (for syndrome decoding)
        n_shots: number of Monte Carlo samples
        seed: random seed

    Returns:
        dict with 'logical_error_rate', 'mean_fidelity', 'mean_raw_fidelity', 'n_shots'
    """
    from src.simulation import lookup_table_decoder

    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    corrected_fidelities = []
    raw_fidelities = []

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states

        noisy_state = noise_fn(logical_state, rng)

        raw_fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        raw_fidelities.append(raw_fid)

        corrected = lookup_table_decoder(noisy_state, code_states, error_ops)
        fid = float(np.abs(np.vdot(logical_state, corrected))**2)
        corrected_fidelities.append(fid)

    return {
        'logical_error_rate': float(np.mean([f < 0.5 for f in corrected_fidelities])),
        'mean_fidelity': float(np.mean(corrected_fidelities)),
        'mean_raw_fidelity': float(np.mean(raw_fidelities)),
        'n_shots': n_shots,
    }


def simulate_decoder_comparison(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    error_ops: List[np.ndarray],
    n_shots: int = 500,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compare all decoders on the same noisy states for fair comparison.

    Returns dict keyed by decoder name, each with 'ler' and 'mean_fidelity'.
    """
    from src.simulation import (
        projection_decoder, nearest_codeword_decoder, lookup_table_decoder
    )

    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    results = {
        'no_decoding': {'fids': []},
        'projection': {'fids': []},
        'nearest_cw': {'fids': []},
        'lookup_table': {'fids': []},
    }

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states
        noisy_state = noise_fn(logical_state, rng)

        raw_fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        results['no_decoding']['fids'].append(raw_fid)

        proj = projection_decoder(noisy_state, code_states)
        results['projection']['fids'].append(
            float(np.abs(np.vdot(logical_state, proj))**2))

        ncw = nearest_codeword_decoder(noisy_state, code_states)
        results['nearest_cw']['fids'].append(
            float(np.abs(np.vdot(logical_state, ncw))**2))

        lut = lookup_table_decoder(noisy_state, code_states, error_ops)
        results['lookup_table']['fids'].append(
            float(np.abs(np.vdot(logical_state, lut))**2))

    out = {}
    for name, data in results.items():
        fids = data['fids']
        out[name] = {
            'ler': float(np.mean([f < 0.5 for f in fids])),
            'mean_fidelity': float(np.mean(fids)),
        }
    return out


def simulate_raw_fidelity(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_shots: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute mean fidelity after noise with NO decoding.
    Shows how much the code space is disturbed by the noise.

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        n_shots: number of Monte Carlo samples
        seed: random seed

    Returns:
        dict with 'mean_fidelity', 'std_fidelity', 'n_shots'
    """
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    fidelities = []

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states

        noisy_state = noise_fn(logical_state, rng)
        fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        fidelities.append(fid)

    return {
        'mean_fidelity': float(np.mean(fidelities)),
        'std_fidelity': float(np.std(fidelities)),
        'n_shots': n_shots,
    }


# ═══════════════════════════════════════════════════════════════════════
# Decoders (merged from decoders.py)
# ═══════════════════════════════════════════════════════════════════════

def _build_decoder_projector(code_states: np.ndarray) -> np.ndarray:
    """Build projector P = sum_k |psi_k><psi_k|."""
    return code_states.T @ np.conj(code_states)


def projection_decoder(noisy_state: np.ndarray,
                       code_states: np.ndarray) -> np.ndarray:
    """Project noisy state onto code space, renormalize."""
    P = _build_decoder_projector(code_states)
    projected = P @ noisy_state
    norm = np.linalg.norm(projected)
    if norm < 1e-15:
        return noisy_state
    return projected / norm


def detection_decoder(noisy_state: np.ndarray,
                      code_states: np.ndarray,
                      detection_threshold: float = 0.1
                      ) -> Tuple[Optional[np.ndarray], bool]:
    """For distance-2: project, discard if projection norm^2 < threshold."""
    P = _build_decoder_projector(code_states)
    projected = P @ noisy_state
    proj_norm_sq = np.real(np.vdot(projected, projected))
    if proj_norm_sq < detection_threshold:
        return None, True
    return projected / np.sqrt(proj_norm_sq), False


def nearest_codeword_decoder(noisy_state: np.ndarray,
                             code_states: np.ndarray) -> np.ndarray:
    """Return single codeword with maximum overlap (hard decision)."""
    overlaps = np.abs(np.conj(code_states) @ noisy_state)**2
    return code_states[np.argmax(overlaps)].copy()


def lookup_table_decoder(noisy_state: np.ndarray,
                         code_states: np.ndarray,
                         error_ops: List[np.ndarray]) -> np.ndarray:
    """Try E†|noisy> for each E, pick best projection onto code space."""
    K = code_states.shape[0]
    best_overlap = -1.0
    best_correction = None
    for E in error_ops:
        error_states = (E @ code_states.T).T
        overlap = sum(np.abs(np.vdot(error_states[k], noisy_state))**2 for k in range(K))
        if overlap > best_overlap:
            best_overlap = overlap
            best_correction = E
    if best_correction is None:
        return noisy_state
    corrected = best_correction.conj().T @ noisy_state
    return projection_decoder(corrected, code_states)


syndrome_based_decoder = lookup_table_decoder


# ═══════════════════════════════════════════════════════════════════════
# Factored LER simulation (for n >= 7 where dense E_corr is infeasible)
# ═══════════════════════════════════════════════════════════════════════

def _apply_factored_op(state, op, q, n_qudit, d):
    """Apply a d×d operator to qudit q of an n-qudit state vector."""
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    s = np.tensordot(op, s, axes=([1], [q]))
    s = np.moveaxis(s, 0, q)
    return s.reshape(-1)


def simulate_ler_with_correction_factored(
    code_states, noise_fn, single_errors, n_qudit, d,
    n_shots=2000, seed=42
):
    """
    LER simulation with factored single-qudit lookup-table decoder.

    Avoids building dense d^n x d^n correction operators AND the dense
    projector. Uses overlap with individual codewords instead:
      overlap = sum_k |<psi_k|corrected>|^2

    The correction set is {I} + {E_q for q in [n], E in single_errors}.

    Returns dict with 'logical_error_rate', 'mean_fidelity', 'mean_raw_fidelity'.
    """
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    logical_errors = 0
    fids_raw = []
    fids_corr = []

    for shot in range(n_shots):
        alpha = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        alpha /= np.linalg.norm(alpha)
        logical_state = alpha @ code_states

        noisy = noise_fn(logical_state, rng)
        fids_raw.append(float(np.abs(np.vdot(logical_state, noisy)) ** 2))

        # Overlap with code space via individual codewords (no dense projector)
        def _code_overlap(state):
            return sum(np.abs(np.vdot(code_states[k], state)) ** 2 for k in range(K))

        best_overlap = _code_overlap(noisy)
        best_corrected = noisy

        for q in range(n_qudit):
            for E in single_errors:
                corrected = _apply_factored_op(noisy, E.conj().T, q, n_qudit, d)
                overlap = _code_overlap(corrected)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_corrected = corrected

        # Project onto code space via codeword overlaps
        coeffs = np.array([np.vdot(code_states[k], best_corrected) for k in range(K)])
        projected = coeffs @ code_states
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected /= norm
        fid = float(np.abs(np.vdot(logical_state, projected)) ** 2)
        fids_corr.append(fid)
        if fid < 0.5:
            logical_errors += 1

    return {
        'logical_error_rate': float(logical_errors / n_shots),
        'mean_fidelity': float(np.mean(fids_corr)),
        'mean_raw_fidelity': float(np.mean(fids_raw)),
        'n_shots': n_shots,
    }
