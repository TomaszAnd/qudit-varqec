"""
Hardware error basis for trapped-ion qudits of any dimension d.

For dimension d, single-qudit errors come from d-1 adjacent transitions:
  - Z-type (AC Stark shifts): phase flip on level k, one per transition
  - X-type (laser fluctuations): swap adjacent levels, one per transition
  - Y-type (optional): quadrature component, one per transition

All operators are unitary (act as identity on spectator levels).
"""
import itertools
from pennylane import numpy as np


def qudit_hardware_error_basis(d: int):
    """
    Single-qudit hardware errors for trapped-ion systems of dimension d.

    Returns 2(d-1) unitary error operators:
      - Z_{k}: phase flip on level k for transition (k-1, k), k=1..d-1
      - X_{k-1,k}: swap levels k-1 and k, identity elsewhere

    For d=3: returns [Z_1, Z_2, X_01, X_12] (4 errors)
    For d=4: returns [Z_1, Z_2, Z_3, X_01, X_12, X_23] (6 errors)
    """
    errors = []

    # Z-type: phase flip on level k (spectator dephasing from AC Stark)
    for k in range(1, d):
        Z = np.eye(d, dtype=complex)
        Z[k, k] = -1.0
        errors.append(Z)

    # X-type: swap adjacent levels (laser fluctuation)
    for k in range(d - 1):
        X = np.eye(d, dtype=complex)
        X[k, k] = 0.0
        X[k + 1, k + 1] = 0.0
        X[k, k + 1] = 1.0
        X[k + 1, k] = 1.0
        errors.append(X)

    return errors


def qudit_extended_hardware_error_basis(d: int):
    """
    Extended error basis including Y-type errors.

    Returns 3(d-1) operators: Z, X, and Y for each adjacent transition.
    """
    errors = qudit_hardware_error_basis(d)

    # Y-type: quadrature component of laser fluctuation
    for k in range(d - 1):
        Y = np.eye(d, dtype=complex)
        Y[k, k] = 0.0
        Y[k + 1, k + 1] = 0.0
        Y[k, k + 1] = -1j
        Y[k + 1, k] = 1j
        errors.append(Y)

    return errors


def _is_proportional_to_identity(M, d):
    """Check if M = c * I for some scalar c with |c| = 1."""
    if not np.allclose(M - np.diag(np.diag(M)), 0, atol=1e-12):
        return False
    diag = np.diag(M)
    return np.allclose(diag, diag[0] * np.ones(d), atol=1e-12)


def _is_in_set(M, op_set, d):
    """Check if M is proportional (global phase) to any operator in op_set."""
    for E in op_set:
        prod = M @ E.conj().T
        if _is_proportional_to_identity(prod, d):
            return True
    return False


def close_error_basis(single_errors):
    """
    Close a set of single-qudit error operators under pairwise products.

    For distance-3 codes, E_corr has weight <= 1. The KL conditions require
    detecting all M = E_a^dag E_b, which for same-qudit errors produces
    operators outside the original basis (hardware errors are not a group).

    Returns: deduplicated list of NEW operators only (excluding originals
    and anything proportional to identity).
    """
    d = single_errors[0].shape[0]
    cross_products = []

    for Ea in single_errors:
        for Eb in single_errors:
            M = Ea @ Eb
            if _is_proportional_to_identity(M, d):
                continue
            if _is_in_set(M, single_errors, d):
                continue
            if _is_in_set(M, cross_products, d):
                continue
            cross_products.append(M)

    return cross_products


def _embed_single_qudit(E, q, n_qudit, d):
    """Embed single-qudit operator E on qudit q in n_qudit system."""
    Id = np.eye(d, dtype=complex)
    op = 1
    for i in range(n_qudit):
        op = np.kron(op, E if i == q else Id)
    return op


def build_native_error_set(n_qudit: int, d: int, distance: int,
                           extended: bool = False, closed: bool = False):
    """
    Build detection and correction error sets for n_qudit qudits of dimension d.

    Args:
        n_qudit: number of physical qudits
        d: qudit dimension (3, 4, 5, 7, ...)
        distance: code distance
        extended: if True, include Y-type errors
        closed: if True, add cross-product operators to E_det (required for
                distance >= 3 with non-group error bases like hardware errors)

    Returns:
        E_det: detection error operators (weight up to distance-1)
        E_corr: correction error operators (weight up to floor((distance-1)/2))
    """
    Id = np.eye(d, dtype=complex)

    id_full = Id
    for _ in range(1, n_qudit):
        id_full = np.kron(id_full, Id)

    E_det = [id_full]
    E_corr = [id_full]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    if extended:
        single_errs = qudit_extended_hardware_error_basis(d)
    else:
        single_errs = qudit_hardware_error_basis(d)

    # Standard weight-1 through weight-max_det construction
    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                op = 1
                idx_choice = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx_choice])
                        idx_choice += 1
                    else:
                        op = np.kron(op, Id)
                E_det.append(op)
                if w <= max_corr:
                    E_corr.append(op)

    # Cross-product closure: add weight-1 same-qudit products to E_det
    if closed and max_corr >= 1:
        cross_products = close_error_basis(single_errs)
        for q in range(n_qudit):
            for E in cross_products:
                op = _embed_single_qudit(E, q, n_qudit, d)
                E_det.append(op)

    return E_det, E_corr


def build_native_error_set_factored(n_qudit: int, d: int, distance: int,
                                     extended: bool = False, closed: bool = False):
    """
    Build error set in factored form for memory efficiency.

    Each error is a list of (qudit_idx, d×d_matrix) tuples.
    Weight-0 (identity): empty list []
    Weight-1: [(q, E_single)]
    Weight-2: [(q1, E1), (q2, E2)]

    Memory: O(n_qudit × d²) per error vs O(d^{2n}) for dense.

    Returns:
        E_det_factors: list of factor lists
        E_corr_factors: list of factor lists
    """
    if extended:
        single_errs = qudit_extended_hardware_error_basis(d)
    else:
        single_errs = qudit_hardware_error_basis(d)

    E_det_factors = [[]]  # identity = empty factor list
    E_corr_factors = [[]]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                factors = [(qudit_subset[i], err_choice[i]) for i in range(w)]
                E_det_factors.append(factors)
                if w <= max_corr:
                    E_corr_factors.append(factors)

    if closed and max_corr >= 1:
        cross_products = close_error_basis(single_errs)
        for q in range(n_qudit):
            for E in cross_products:
                E_det_factors.append([(q, E)])

    return E_det_factors, E_corr_factors


# ── Hardware noise (merged from native_noise.py) ──────────────────────

def make_hardware_noise_fn(d: int, n_qudit: int, p: float):
    """
    Independent per-qudit hardware noise.

    For each qudit independently:
    - With prob (1-p): no error
    - With prob p: apply a random hardware error

    Args:
        d: qudit dimension
        n_qudit: number of qudits
        p: per-qudit error probability

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    import numpy
    single_errors = [numpy.array(E) for E in qudit_hardware_error_basis(d)]
    n_errors = len(single_errors)
    shape = [d] * n_qudit

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudit):
            if rng.random() < p:
                idx = rng.integers(n_errors)
                E = single_errors[idx]
                s_tensor = s.reshape(shape)
                s_tensor = numpy.tensordot(E, s_tensor, axes=([1], [q]))
                s_tensor = numpy.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = numpy.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


# ── Backward compatibility aliases ────────────────────────────────────

def hardware_error_basis():
    """Single-qutrit (d=3) hardware errors. Backward-compat alias."""
    return qudit_hardware_error_basis(d=3)


def extended_hardware_error_basis():
    """Extended qutrit (d=3) errors. Backward-compat alias."""
    return qudit_extended_hardware_error_basis(d=3)


def build_qutrit_error_set(n_qutrit: int, distance: int, extended: bool = False):
    """Build error sets for qutrits (d=3). Backward-compat alias."""
    return build_native_error_set(n_qutrit, d=3, distance=distance, extended=extended)


# ── ErrorModel class ─────────────────────────────────────────────────

class ErrorModel:
    """
    Hardware error model for trapped-ion qudits.

    Bundles qudit dimension, system size, and error basis into one object.
    Provides methods to build dense or factored error sets and noise functions.

    Usage:
        model = ErrorModel(d=3, n_qudit=5, distance=3, closed=True)
        E_det, E_corr = model.build_dense()
        E_det_f, E_corr_f = model.build_factored()
        noise_fn = model.make_noise_fn(p=0.05)
    """

    def __init__(self, d: int, n_qudit: int, distance: int = 2,
                 extended: bool = False, closed: bool = False):
        self.d = d
        self.n_qudit = n_qudit
        self.distance = distance
        self.extended = extended
        self.closed = closed

        if extended:
            self.single_errors = qudit_extended_hardware_error_basis(d)
        else:
            self.single_errors = qudit_hardware_error_basis(d)

        self.cross_products = close_error_basis(self.single_errors) if closed else []

    @property
    def dim(self) -> int:
        return self.d ** self.n_qudit

    @property
    def K(self) -> int:
        return self.d

    @property
    def n_single_errors(self) -> int:
        return len(self.single_errors)

    def build_dense(self):
        """Build dense (dim x dim matrix) error sets. Returns (E_det, E_corr)."""
        return build_native_error_set(
            self.n_qudit, self.d, self.distance,
            extended=self.extended, closed=self.closed)

    def build_factored(self):
        """Build factored error sets. Returns (E_det_factors, E_corr_factors)."""
        return build_native_error_set_factored(
            self.n_qudit, self.d, self.distance,
            extended=self.extended, closed=self.closed)

    def make_noise_fn(self, p: float):
        """Create independent per-qudit hardware noise function."""
        return make_hardware_noise_fn(self.d, self.n_qudit, p)

    def __repr__(self):
        cross = f"+{len(self.cross_products)} cross" if self.cross_products else ""
        return (f"ErrorModel(d={self.d}, n={self.n_qudit}, dist={self.distance}, "
                f"errors={self.n_single_errors}{cross})")
