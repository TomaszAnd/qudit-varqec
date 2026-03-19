"""
Knill-Laflamme loss functions for VarQEC (Cao et al. arXiv:2204.03560).

Two families of loss functions, both enforcing the KL conditions:

1. Correction-based (kl_loss_fast, kl_loss_minibatch, kl_loss_diagonal_minibatch):
   Term 1: Σ_{E∈E_det} Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|²  (off-diagonal)
   Term 2: Σ_{M∈E_a†E_b} (K/4) Var_k(⟨ψ_k|M|ψ_k⟩)  (diagonal variance)
   Used for d=2 (Term 1 only) and dephasing d=3. O(|E_corr|²) for Term 2.

2. Detection-based (kl_loss_detection_*_minibatch, paper Eq. 16):
   Σ_{E∈E_det} [Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|² + (K/4) Var_k(⟨ψ_k|E|ψ_k⟩)]
   Single sum over E_det, no E_a†E_b products. O(|E_det|).
   Used for depolarizing d=3 and correlated d=3.

The (K/4) factor: paper Eq. 16 has (1/4)Σ_j|...-mean|² = (K/4)*Var (since Var = (1/K)Σ).

Variants handle different error representations:
  - Dense matrices (dephasing): kl_loss_fast, kl_loss_detection_minibatch
  - Factored single-qudit ops (depolarizing d=3): kl_loss_detection_factored_minibatch
  - 1D diagonal vectors (correlated): kl_loss_diagonal_minibatch, kl_loss_detection_diagonal_minibatch
"""
import numpy as np
import pennylane as qml


def estimate_memory_mb(n_operators, dim):
    """Estimate memory for storing n_operators dense complex128 matrices of size dim x dim."""
    bytes_per_matrix = dim * dim * 16  # complex128 = 16 bytes
    total_bytes = n_operators * bytes_per_matrix
    return total_bytes / (1024 ** 2)


def check_memory_budget(n_operators, dim, budget_mb=4000, label=""):
    """Raise error if estimated memory exceeds budget. (Internal utility.)"""
    est = estimate_memory_mb(n_operators, dim)
    if est > budget_mb:
        raise MemoryError(
            f"{label}: {n_operators} operators x {dim}x{dim} complex128 "
            f"= {est:.0f} MB exceeds {budget_mb} MB budget. "
            f"Reduce n_max, increase truncation_threshold, or use smaller system."
        )
    return est


def precompute_error_products(E_corr):
    """
    Precompute all E_a† @ E_b products. Call ONCE before training loop.
    These are constant — they don't depend on variational parameters.

    For |E_corr|=16 (dephasing d=3), this precomputes 256 matrices.
    For |E_corr|=76 (depolarizing d=3), this precomputes 5776 matrices.

    Returns:
        M_products: list of (dim, dim) matrices [Ea†Eb for all pairs]
    """
    M_products = []
    for Ea in E_corr:
        Ea_dag = np.conj(Ea.T)
        for Eb in E_corr:
            M_products.append(Ea_dag @ Eb)
    return M_products


def precompute_error_products_dedup(E_corr):
    """
    Precompute UNIQUE E_a†E_b products with deduplication.
    For |E_corr|=76 (depolarizing d=3), 5776 products reduce to far fewer unique ones.

    Returns:
        M_products: list of unique (dim, dim) matrices
    """
    M_products = []
    seen = {}

    for Ea in E_corr:
        Ea_dag = np.conj(Ea.T)
        for Eb in E_corr:
            M = Ea_dag @ Eb
            key = tuple(np.round(M.ravel(), decimals=10))
            if key not in seen:
                seen[key] = len(M_products)
                M_products.append(M)

    n_total = len(E_corr) ** 2
    print(f"Deduplicated: {n_total} products -> {len(M_products)} unique")
    return M_products


def kl_loss_fast(params, encoder_fn, E_det, M_products, K, distance):
    """
    Fast KL loss with precomputed error products and vectorized operations.

    Args:
        params: variational parameters
        encoder_fn: PennyLane QNode encoder
        E_det: list of detection error operators
        M_products: PRECOMPUTED list of E_a†E_b matrices (from precompute_error_products)
        K: number of codewords (typically 4)
        distance: code distance

    Returns:
        loss value (PennyLane autograd-compatible)
    """
    # Generate code states and stack into matrix
    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    # code_states shape: (K, dim)

    loss = 0.0

    # === Term 1: Off-diagonal (orthogonality) ===
    # For each error E, compute <ψ_i|E|ψ_j> for all i<j
    for E in E_det:
        # Apply E to all codewords at once: E @ code_states.T -> (dim, K)
        E_applied = qml.math.tensordot(E, qml.math.transpose(code_states), axes=1)
        # Inner products: code_states.conj() @ E_applied -> (K, K)
        overlaps = qml.math.tensordot(qml.math.conj(code_states), E_applied, axes=[[1], [0]])
        # Sum |overlaps[i,j]|^2 for i < j (only 6 pairs for K=4)
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + qml.math.abs(overlaps[i, j]) ** 2

    # === Term 2: Diagonal variance (KL condition) ===
    if distance >= 3:
        for M in M_products:
            # Apply M to all codewords: M @ code_states.T -> (dim, K)
            M_applied = qml.math.tensordot(M, qml.math.transpose(code_states), axes=1)
            # Diagonal expectations: <ψ_k|M|ψ_k> = sum over dim of conj(ψ_k) * (M @ ψ_k)
            vals = qml.math.sum(qml.math.conj(code_states) * qml.math.transpose(M_applied), axis=1)
            loss = loss + (K / 4) * qml.math.var(vals)

    return loss


def kl_loss_diagonal(params, encoder_fn, E_det_diags, K, distance):
    """
    LEGACY: Full (non-minibatch) diagonal KL loss. Use kl_loss_diagonal_minibatch
    or kl_loss_detection_diagonal_minibatch instead.

    KL loss for DIAGONAL error operators (stored as 1D vectors).

    For diagonal error E = diag(v):
        <psi_i|E|psi_j> = sum_k v[k] * conj(psi_i[k]) * psi_j[k]

    This is O(dim) per inner product instead of O(dim^2) for dense matrices.
    No dense matrices are ever created.

    Memory: O(K x dim + n_errors x dim) instead of O(n_errors x dim^2)
    For dim=1024, 400 diagonal errors: 400 x 1024 x 16 bytes = 6.5 MB
    vs dense: 400 x 1024 x 1024 x 16 = 6.7 GB  (1000x less memory!)

    Args:
        params: variational parameters
        encoder_fn: PennyLane QNode
        E_det_diags: list of 1D arrays (diagonal of each error operator)
        K: number of codewords
        distance: code distance (only Term 1 for d=2)
    """
    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    # code_states shape: (K, dim)
    loss = 0.0

    # Term 1: off-diagonal
    for v in E_det_diags:
        # For diagonal E=diag(v): E|psi_j> = v * psi_j (element-wise)
        # <psi_i|E|psi_j> = sum_k conj(psi_i[k]) * v[k] * psi_j[k]
        v_tensor = qml.math.convert_like(v, code_states)
        weighted_states = v_tensor[None, :] * code_states  # (K, dim)
        # overlaps[i,j] = <psi_i|E|psi_j>
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(weighted_states),
            axes=[[1], [0]]
        )
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + qml.math.abs(overlaps[i, j]) ** 2

    # Term 2: diagonal variance (only for distance >= 3)
    # For diagonal errors, M = E_a^dag E_b is also diagonal: diag(conj(v_a) * v_b)
    # Compute M products on the fly to avoid storing them
    if distance >= 3:
        for va in E_det_diags:
            va_conj = np.conj(va)
            for vb in E_det_diags:
                m_diag = va_conj * vb  # diagonal of E_a^dag E_b
                m_tensor = qml.math.convert_like(m_diag, code_states)
                # <psi_k|M|psi_k> = sum_l m[l] * |psi_k[l]|^2
                weighted = m_tensor[None, :] * code_states  # (K, dim)
                vals = qml.math.sum(qml.math.conj(code_states) * weighted, axis=1)
                loss = loss + (K / 4) * qml.math.var(vals)

    return loss


def kl_loss_diagonal_minibatch(params, encoder_fn, E_det_diags, K, distance,
                                batch_fraction=0.3, rng=None):
    """
    Minibatch KL loss for DIAGONAL error operators (stored as 1D vectors).

    Same math as kl_loss_diagonal but samples a fraction of errors for Term 1
    and a fraction of (a, b) pairs for Term 2, with unbiased scaling.

    For 25k diagonal operators, the full Term 2 has 625M pairs — completely
    intractable. This samples batch_fraction of E_det for Term 1 and
    batch_fraction^2 worth of pairs for Term 2.

    Args:
        params: variational parameters
        encoder_fn: PennyLane QNode
        E_det_diags: list of 1D arrays (diagonal of each error operator)
        K: number of codewords
        distance: code distance
        batch_fraction: fraction of errors to sample (0.3 = 30%)
        rng: numpy random Generator
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0
    n_total = len(E_det_diags)

    # Term 1: sample batch_fraction of E_det
    n_sample = max(1, int(n_total * batch_fraction))
    indices = rng.choice(n_total, n_sample, replace=False)
    scale_det = n_total / n_sample

    for idx in indices:
        v = E_det_diags[idx]
        v_tensor = qml.math.convert_like(v, code_states)
        weighted_states = v_tensor[None, :] * code_states
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(weighted_states),
            axes=[[1], [0]]
        )
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + scale_det * qml.math.abs(overlaps[i, j]) ** 2

    # Term 2: sample batch_fraction of rows AND columns independently
    if distance >= 3:
        n_sample_a = max(1, int(n_total * batch_fraction))
        n_sample_b = max(1, int(n_total * batch_fraction))
        indices_a = rng.choice(n_total, n_sample_a, replace=False)
        indices_b = rng.choice(n_total, n_sample_b, replace=False)
        scale_m = (n_total / n_sample_a) * (n_total / n_sample_b)

        for ia in indices_a:
            va_conj = np.conj(E_det_diags[ia])
            for ib in indices_b:
                m_diag = va_conj * E_det_diags[ib]
                m_tensor = qml.math.convert_like(m_diag, code_states)
                weighted = m_tensor[None, :] * code_states
                vals = qml.math.sum(qml.math.conj(code_states) * weighted, axis=1)
                loss = loss + scale_m * (K / 4) * qml.math.var(vals)

    return loss


def kl_loss_minibatch(params, encoder_fn, E_det, M_products, K, distance,
                       batch_fraction=0.2, rng=None):
    """
    KL loss with mini-batch sampling of error operators.

    From VarQEC paper Sec 4: "Within each iteration, we sample a subset E_S in E"
    This gives ~5x speedup per step with batch_fraction=0.2.

    Uses unbiased scaling so E[loss_minibatch] = loss_full.

    Args:
        params: variational parameters
        encoder_fn: PennyLane QNode encoder
        E_det: list of detection error operators
        M_products: PRECOMPUTED list of E_a†E_b matrices
        K: number of codewords (typically 4)
        distance: code distance
        batch_fraction: fraction of errors to sample (0.2 = 20%)
        rng: numpy random Generator (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    # Sample subset of detection errors
    n_sample_det = max(1, int(len(E_det) * batch_fraction))
    indices_det = rng.choice(len(E_det), n_sample_det, replace=False)
    scale_det = len(E_det) / n_sample_det

    for idx in indices_det:
        E = E_det[idx]
        E_applied = qml.math.tensordot(E, qml.math.transpose(code_states), axes=1)
        overlaps = qml.math.tensordot(qml.math.conj(code_states), E_applied, axes=[[1], [0]])
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + scale_det * qml.math.abs(overlaps[i, j]) ** 2

    # Sample subset of M products
    if distance >= 3 and M_products:
        n_sample_m = max(1, int(len(M_products) * batch_fraction))
        indices_m = rng.choice(len(M_products), n_sample_m, replace=False)
        scale_m = len(M_products) / n_sample_m

        for idx in indices_m:
            M = M_products[idx]
            M_applied = qml.math.tensordot(M, qml.math.transpose(code_states), axes=1)
            vals = qml.math.sum(qml.math.conj(code_states) * qml.math.transpose(M_applied), axis=1)
            loss = loss + scale_m * (K / 4) * qml.math.var(vals)

    return loss


def apply_single_qudit_op(state, op, qudit_idx, n_qudits, d):
    """
    Apply a d x d operator to qudit qudit_idx of an n_qudits system.

    Reshapes the state vector to (d, d, ..., d), applies the operator
    as a tensor contraction on the qudit_idx axis, then flattens back.

    This is O(d^n * d) instead of O(d^{2n}) for a full matrix multiply.
    """
    shape = tuple([d] * n_qudits)
    state_tensor = state.reshape(shape)
    result = np.tensordot(op, state_tensor, axes=([1], [qudit_idx]))
    result = np.moveaxis(result, 0, qudit_idx)
    return result.reshape(-1)


def _apply_factored_qml(state, factors, n_qudits, d):
    """Apply factored error to a PennyLane autograd state via tensor contractions."""
    if not factors:
        return state
    s = state
    shape = tuple([d] * n_qudits)
    for qudit_idx, op in factors:
        s_tensor = qml.math.reshape(s, shape)
        # tensordot with numpy op on autograd state
        op_tensor = qml.math.convert_like(op, s_tensor)
        s_tensor = qml.math.tensordot(op_tensor, s_tensor, axes=([1], [qudit_idx]))
        # moveaxis(src=0, dst=qudit_idx): after tensordot, new axis is at 0
        perm = list(range(1, qudit_idx + 1)) + [0] + list(range(qudit_idx + 1, n_qudits))
        s_tensor = qml.math.transpose(s_tensor, perm)
        s = qml.math.reshape(s_tensor, (-1,))
    return s


def kl_loss_factored_minibatch(params, encoder_fn, E_det_factors, E_corr_factors,
                                K, distance, n_qudits, dim_qudit,
                                batch_fraction_det=0.1, batch_fraction_corr=0.1,
                                rng=None):
    """
    LEGACY: Correction-based factored loss with quadratic E_a†E_b Term 2.
    Use kl_loss_detection_factored_minibatch instead (same results, no E_a†E_b products).

    KL loss where errors are stored in FACTORED form (list of single-qudit ops)
    instead of full dim^n × dim^n matrices.

    E_det_factors: list of lists, each inner list is [(qudit_idx, 4x4 matrix), ...]
                   Empty list = identity.
    E_corr_factors: same format.

    Memory: O(n_errors × max_weight × d²) instead of O(n_errors × d^{2n})
    For depolarizing d=3: ~600 KB instead of 37 GB.
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    n_det = len(E_det_factors)
    n_sample_det = max(1, int(n_det * batch_fraction_det))
    idx_det = rng.choice(n_det, n_sample_det, replace=False)
    scale_det = n_det / n_sample_det

    for i in idx_det:
        factors = E_det_factors[i]
        # Apply E to each codeword
        E_codewords = qml.math.stack([
            _apply_factored_qml(code_states[k], factors, n_qudits, dim_qudit)
            for k in range(K)
        ])  # (K, dim)
        # Overlaps: <psi_i|E|psi_j>
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(E_codewords), axes=[[1], [0]])
        for ci in range(K):
            for cj in range(ci + 1, K):
                loss = loss + scale_det * qml.math.abs(overlaps[ci, cj]) ** 2

    # Term 2: sample (a,b) pairs from E_corr, compute M = E_a†E_b on the fly
    if distance >= 3 and len(E_corr_factors) > 0:
        n_corr = len(E_corr_factors)
        n_sample_a = max(1, int(n_corr * batch_fraction_corr))
        n_sample_b = max(1, int(n_corr * batch_fraction_corr))
        idx_a = rng.choice(n_corr, n_sample_a, replace=False)
        idx_b = rng.choice(n_corr, n_sample_b, replace=False)
        scale_m = (n_corr / n_sample_a) * (n_corr / n_sample_b)

        for ia in idx_a:
            factors_a = E_corr_factors[ia]
            for ib in idx_b:
                factors_b = E_corr_factors[ib]
                # <psi_k|E_a†E_b|psi_k> = <E_a psi_k|E_b psi_k>
                vals_list = []
                for k in range(K):
                    Ea_psi = _apply_factored_qml(code_states[k], factors_a, n_qudits, dim_qudit)
                    Eb_psi = _apply_factored_qml(code_states[k], factors_b, n_qudits, dim_qudit)
                    val = qml.math.sum(qml.math.conj(Ea_psi) * Eb_psi)
                    vals_list.append(val)
                vals = qml.math.stack(vals_list)
                loss = loss + scale_m * (K / 4) * qml.math.var(vals)

    return loss


def kl_loss_detection_minibatch(params, encoder_fn, E_det, K, distance,
                                batch_fraction=0.2, rng=None):
    """
    Detection-style KL loss (VarQEC paper Eq. 16, ℓ2 norm).

    For each error E in E_det:
      - Off-diagonal: Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|²
      - Diagonal variance (d≥3): (K/4) * Var_k(⟨ψ_k|E|ψ_k⟩)

    NO E_a†E_b products needed. Single sum over E_det.
    Mathematically equivalent to correction-based loss but O(|E_det|) not O(|E_corr|²).
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    n_total = len(E_det)
    n_sample = max(1, int(n_total * batch_fraction))
    indices = rng.choice(n_total, n_sample, replace=False)
    scale = n_total / n_sample

    for idx in indices:
        E = E_det[idx]
        E_applied = qml.math.tensordot(E, qml.math.transpose(code_states), axes=1)
        overlaps = qml.math.tensordot(qml.math.conj(code_states), E_applied, axes=[[1], [0]])
        # Off-diagonal
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + scale * qml.math.abs(overlaps[i, j]) ** 2
        # Diagonal variance (d≥3)
        if distance >= 3:
            diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
            loss = loss + scale * (K / 4) * qml.math.real(qml.math.var(diag_vals))

    return loss


def kl_loss_detection_factored_minibatch(params, encoder_fn, E_det_factors, K, distance,
                                          n_qudits, dim_qudit,
                                          batch_fraction=0.2, rng=None):
    """
    Detection-style KL loss for FACTORED errors (depolarizing d=3).

    Same as kl_loss_detection_minibatch but errors are stored as factored
    single-qudit ops instead of dense matrices. Memory: ~600 KB vs 37 GB.
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    n_det = len(E_det_factors)
    n_sample = max(1, int(n_det * batch_fraction))
    idx_det = rng.choice(n_det, n_sample, replace=False)
    scale = n_det / n_sample

    for i in idx_det:
        factors = E_det_factors[i]
        E_codewords = qml.math.stack([
            _apply_factored_qml(code_states[k], factors, n_qudits, dim_qudit)
            for k in range(K)
        ])
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(E_codewords), axes=[[1], [0]])
        # Off-diagonal
        for ci in range(K):
            for cj in range(ci + 1, K):
                loss = loss + scale * qml.math.abs(overlaps[ci, cj]) ** 2
        # Diagonal variance (d≥3)
        if distance >= 3:
            diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
            loss = loss + scale * (K / 4) * qml.math.real(qml.math.var(diag_vals))

    return loss


def kl_loss_detection_diagonal_minibatch(params, encoder_fn, E_det_diags, K, distance,
                                          batch_fraction=0.01, rng=None):
    """
    Detection-style KL loss for DIAGONAL error operators.

    Same as kl_loss_detection_minibatch but errors are 1D diagonals.
    NO quadratic E_a†E_b loop — single sum over E_det.
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0
    n_total = len(E_det_diags)

    n_sample = max(1, int(n_total * batch_fraction))
    indices = rng.choice(n_total, n_sample, replace=False)
    scale = n_total / n_sample

    for idx in indices:
        v = E_det_diags[idx]
        v_tensor = qml.math.convert_like(v, code_states)
        weighted_states = v_tensor[None, :] * code_states
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(weighted_states),
            axes=[[1], [0]]
        )
        # Off-diagonal
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + scale * qml.math.abs(overlaps[i, j]) ** 2
        # Diagonal variance (d≥3)
        if distance >= 3:
            diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
            loss = loss + scale * (K / 4) * qml.math.real(qml.math.var(diag_vals))

    return loss


def save_varqec_result(filepath, params, losses, noise_type, distance, n_layers, metadata=None):
    """Save trained VarQEC parameters and training history."""
    result = {
        'params': np.array(params),
        'losses': np.array(losses),
        'noise_type': noise_type,
        'distance': distance,
        'n_layers': n_layers,
        'final_loss': float(losses[-1]) if losses else None,
        'converged': float(losses[-1]) < 1e-6 if losses else False,
    }
    if metadata:
        result.update(metadata)
    np.savez(filepath, **result)
    print(f"Saved VarQEC result to {filepath} (loss={result['final_loss']:.2e})")


def load_varqec_result(filepath):
    """Load trained VarQEC parameters."""
    data = np.load(filepath, allow_pickle=True)
    return dict(data)
