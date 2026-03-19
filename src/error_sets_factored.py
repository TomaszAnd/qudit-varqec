"""
Factored error set representation for VarQEC.

Instead of storing full d^n × d^n matrices (37 GB for depolarizing d=3),
each error is stored as a list of (qudit_index, d×d matrix) pairs.
Identity qudits are omitted. Memory: O(n_errors × max_weight × d²).

For depolarizing d=3 with n=5, d=4:
  Dense: 2326 × 1024 × 1024 × 16 bytes = 37 GB
  Factored: 2326 × 2 × (1 int + 4×4×16 bytes) ≈ 600 KB
"""
import itertools
import numpy as np
from src.pauli_ops import single_qudit_paulis, single_qudit_dephasing_paulis


def build_error_sets_factored(n_qudits, distance, single_errs_fn=single_qudit_paulis, dim_qudit=4):
    """
    Build error sets in factored form.

    Each error is a list of (qudit_idx, pauli_matrix) tuples.
    The identity error is represented as an empty list [].

    Returns:
        E_det_factors: list of factored errors for detection
        E_corr_factors: list of factored errors for correction
    """
    single_errs = single_errs_fn()

    E_det_factors = [[]]  # identity = empty list
    E_corr_factors = [[]]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudits), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                factors = [(qudit_subset[i], np.array(err_choice[i]))
                           for i in range(w)]
                E_det_factors.append(factors)
                if w <= max_corr:
                    E_corr_factors.append(factors)

    return E_det_factors, E_corr_factors


def build_dephasing_error_sets_factored(n_qudits, distance):
    return build_error_sets_factored(n_qudits, distance,
                                     single_errs_fn=single_qudit_dephasing_paulis)


def apply_factored_error(state, factors, n_qudits, dim_qudit):
    """
    Apply a factored error to a state vector using tensor contractions.

    Args:
        state: (dim,) state vector where dim = dim_qudit^n_qudits
        factors: list of (qudit_idx, d×d matrix) tuples. Empty = identity.
        n_qudits: number of qudits
        dim_qudit: local qudit dimension

    Returns:
        (dim,) resulting state vector
    """
    if not factors:
        return state  # identity

    s = state
    shape = tuple([dim_qudit] * n_qudits)
    for qudit_idx, op in factors:
        s_tensor = s.reshape(shape)
        s_tensor = np.tensordot(op, s_tensor, axes=([1], [qudit_idx]))
        s_tensor = np.moveaxis(s_tensor, 0, qudit_idx)
        s = s_tensor.reshape(-1)
    return s


def apply_factored_error_dag(state, factors, n_qudits, dim_qudit):
    """
    Apply E† (conjugate transpose of factored error) to state.
    For Pauli operators (hermitian + unitary), E† = E, but this handles the general case.
    """
    if not factors:
        return state

    s = state
    shape = tuple([dim_qudit] * n_qudits)
    # Apply in reverse order with conjugate transpose
    for qudit_idx, op in reversed(factors):
        op_dag = op.conj().T
        s_tensor = s.reshape(shape)
        s_tensor = np.tensordot(op_dag, s_tensor, axes=([1], [qudit_idx]))
        s_tensor = np.moveaxis(s_tensor, 0, qudit_idx)
        s = s_tensor.reshape(-1)
    return s


def factored_to_dense(factors, n_qudits, dim_qudit):
    """Convert factored error to dense matrix (for testing/validation only)."""
    dim = dim_qudit ** n_qudits
    Id = np.eye(dim_qudit, dtype=complex)

    if not factors:
        return np.eye(dim, dtype=complex)

    # Build the tensor product operator
    factor_dict = {q: op for q, op in factors}
    op = np.array([[1.0]], dtype=complex)
    for q in range(n_qudits):
        if q in factor_dict:
            op = np.kron(op, factor_dict[q])
        else:
            op = np.kron(op, Id)
    return op
