"""
Decoders for VarQEC codes.

Four decoders:
1. projection_decoder: Simple projection onto code space + renormalize.
2. detection_decoder: For d=2 codes. Project; if projection norm is small, discard.
3. nearest_codeword_decoder: Pick the single codeword with maximum overlap (hard decision).
4. lookup_table_decoder: For d>=3 codes. Try E†|noisy> for each correctable error E,
   pick the one that projects best onto code space (maximum-likelihood).
"""
import numpy as np
from typing import Optional, Tuple, List


def _build_code_projector(code_states: np.ndarray) -> np.ndarray:
    """Build projector P = sum_k |psi_k><psi_k|."""
    # P[i,j] = sum_k psi_k[i] * conj(psi_k[j]) = (C^T @ conj(C))[i,j]
    return code_states.T @ np.conj(code_states)


def projection_decoder(noisy_state: np.ndarray,
                       code_states: np.ndarray) -> np.ndarray:
    """
    Simple projection decoder: project noisy state onto code space, renormalize.
    Always returns a state (no discard).

    Args:
        noisy_state: (dim,) complex state vector
        code_states: (K, dim) orthonormal codeword matrix

    Returns:
        (dim,) decoded state vector (projected + renormalized)
    """
    P = _build_code_projector(code_states)
    projected = P @ noisy_state
    norm = np.linalg.norm(projected)
    if norm < 1e-15:
        return noisy_state  # fallback: state is orthogonal to code space
    return projected / norm


def detection_decoder(noisy_state: np.ndarray,
                      code_states: np.ndarray,
                      detection_threshold: float = 0.1
                      ) -> Tuple[Optional[np.ndarray], bool]:
    """
    For distance-2 codes: project onto code space.
    If projection norm^2 is below threshold, declare "error detected" (discard).

    Args:
        noisy_state: (dim,) complex state vector
        code_states: (K, dim) orthonormal codeword matrix
        detection_threshold: if ||P|noisy>||^2 < threshold, error detected

    Returns:
        (decoded_state_or_None, detected_flag)
        - If detected: (None, True) — this shot should be discarded
        - If not detected: (projected_state, False)
    """
    P = _build_code_projector(code_states)
    projected = P @ noisy_state
    proj_norm_sq = np.real(np.vdot(projected, projected))

    if proj_norm_sq < detection_threshold:
        return None, True  # error detected, discard

    return projected / np.sqrt(proj_norm_sq), False


def nearest_codeword_decoder(noisy_state: np.ndarray,
                             code_states: np.ndarray) -> np.ndarray:
    """
    Return the single codeword |psi_k> with maximum overlap |<psi_k|noisy>|^2.

    This is a "hard decision" decoder that picks the most likely codeword
    and loses all superposition information. For K=4 codewords, checks 4 overlaps.

    Args:
        noisy_state: (dim,) complex state vector
        code_states: (K, dim) orthonormal codeword matrix

    Returns:
        (dim,) the codeword with maximum overlap (a copy)
    """
    overlaps = np.abs(np.conj(code_states) @ noisy_state)**2  # (K,)
    best_k = np.argmax(overlaps)
    return code_states[best_k].copy()


def lookup_table_decoder(noisy_state: np.ndarray,
                         code_states: np.ndarray,
                         error_ops: List[np.ndarray]) -> np.ndarray:
    """
    Lookup table decoder: try E†|noisy> for each E in error_ops,
    pick the one that projects best onto code space.

    This is a maximum-likelihood decoder when error_ops includes
    all correctable errors. For a [[5,1,3]] code, error_ops is
    the set of weight-1 errors (16 for dephasing, 76 for depolarizing).

    Complexity: O(|error_ops| x K x dim) per shot.

    Args:
        noisy_state: (dim,) complex state vector
        code_states: (K, dim) orthonormal codeword matrix
        error_ops: list of correctable error operators (including identity)

    Returns:
        (dim,) corrected state vector
    """
    K = code_states.shape[0]
    best_overlap = -1.0
    best_correction = None

    for E in error_ops:
        # Error subspace: {E|psi_k>}
        # Overlap of noisy state with this subspace
        error_states = (E @ code_states.T).T  # (K, dim)
        overlap = 0.0
        for k in range(K):
            overlap += np.abs(np.vdot(error_states[k], noisy_state))**2

        if overlap > best_overlap:
            best_overlap = overlap
            best_correction = E

    if best_correction is None:
        return noisy_state

    # Apply correction: E†|noisy> then project onto code space
    corrected = best_correction.conj().T @ noisy_state
    return projection_decoder(corrected, code_states)


# Backward compatibility alias
syndrome_based_decoder = lookup_table_decoder
