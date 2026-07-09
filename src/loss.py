"""
Knill-Laflamme loss functions for VarQEC (Cao et al. arXiv:2204.03560).

Two families of loss functions, both enforcing the KL conditions:

1. Correction-based (kl_loss_fast, kl_loss_minibatch, kl_loss_diagonal_minibatch):
   Term 1: Σ_{E∈E_det} Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|²  (off-diagonal)
   Term 2: Σ_{M∈E_a†E_b} (K/4) Var_k(⟨ψ_k|M|ψ_k⟩)  (diagonal variance)
   Used for d=2 (Term 1 only) and dephasing d=3. O(|E_corr|²) for Term 2.

2. Detection-based (kl_loss_detection_*_minibatch, Cao et al. Eq. 16 = our paper Eq. 10):
   Σ_{E∈E_det} [Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|² + (K/4) Var_k(⟨ψ_k|E|ψ_k⟩)]
   Single sum over E_det, no E_a†E_b products. O(|E_det|).
   Used for depolarizing d=3 and correlated d=3.

The (K/4) factor: Cao et al. (arXiv:2204.03560) Eq. (16) — our paper's Eq. (10) —
has (1/4)Σ_j|...-mean|² = (K/4)*Var (since Var = (1/K)Σ). "Eq. 16" throughout this
module refers to Cao et al.'s numbering, not our paper's.

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


def _kl_overlap_loss(overlaps, K, distance, scale=1.0):
    """Accumulate KL loss from an overlap matrix overlaps[i,j] = <psi_i|E|psi_j>."""
    loss = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            loss = loss + scale * qml.math.abs(overlaps[i, j]) ** 2
    if distance >= 3:
        diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
        loss = loss + scale * (K / 4) * qml.math.real(qml.math.var(diag_vals))
    return loss


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
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

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
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

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
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

    return loss


# ── Round-12 additions: stratified + Hoeffding minibatch loss ────────
# Code addition. The existing kl_loss_detection_*_minibatch functions
# are preserved unchanged. These new functions take the wire-grouped
# detection-set structure produced by ErrorModel.build_grouped(), which
# the existing uniform-sampling minibatch losses do NOT use.


def _stratified_pick(group_sizes, fraction_per_group, rng,
                     adaptive=False, prev_contributions=None):
    """Pick a stratified sample of error indices from a wire-grouped E_det.

    Args:
        group_sizes: list of N_g, one entry per wire group.
        fraction_per_group: target fraction of errors to sample per group,
            rounded up. Minimum 1 per non-empty group.
        rng: numpy random Generator.
        adaptive: if True, use prev_contributions to keep top-half of
            (group, intra-group-index) tuples by contribution and resample
            the bottom half uniformly. Biased estimator — use cautiously.
        prev_contributions: list aligned with group_sizes; entry g is a
            length-N_g array of per-error contributions from the previous
            step (positive floats). Required if adaptive=True.

    Returns:
        picks: list of np.int64 arrays; picks[g] are the chosen intra-group
            indices for group g.
        scales: list of floats; scales[g] = N_g / len(picks[g]) for the
            per-group inverse-probability weight.
    """
    picks, scales = [], []
    for g, n_g in enumerate(group_sizes):
        if n_g == 0:
            picks.append(np.zeros(0, dtype=np.int64))
            scales.append(0.0)
            continue
        n_keep = max(1, int(np.ceil(fraction_per_group * n_g)))
        n_keep = min(n_keep, n_g)
        if adaptive and prev_contributions is not None and n_g > 1:
            n_top = n_keep // 2
            n_bot = n_keep - n_top
            contrib = np.asarray(prev_contributions[g], dtype=float)
            top_idx = np.argsort(-contrib)[:n_top]
            remaining = np.setdiff1d(np.arange(n_g), top_idx,
                                     assume_unique=False)
            bot_idx = rng.choice(remaining, size=n_bot, replace=False) \
                if n_bot > 0 else np.zeros(0, dtype=np.int64)
            idx = np.concatenate([top_idx, bot_idx]).astype(np.int64)
        else:
            idx = rng.choice(n_g, n_keep, replace=False).astype(np.int64)
        picks.append(idx)
        scales.append(n_g / float(n_keep))
    return picks, scales


def kl_loss_detection_stratified(params, encoder_fn, E_det_grouped, K, distance,
                                 fraction_per_group=0.3, rng=None,
                                 adaptive=False, prev_contributions=None,
                                 return_contributions=False,
                                 n_qudit=None, dim_qudit=None):
    """Stratified-wire-group minibatch KL loss.

    Within each wire group of `E_det_grouped` (output of
    `ErrorModel.build_grouped`), samples `fraction_per_group` of the
    errors (rounded up; minimum 1 per non-empty group). The per-group
    estimator is inverse-probability-weighted (N_g / n_g_sampled) so
    that the FULL loss expectation is recovered in the infinite-sample
    limit. Hoeffding-style concentration bounds apply to the resulting
    gradient estimate.

    Structural property: every wire group is represented every step.
    The existing `kl_loss_detection_minibatch` uniform-sampling loss
    can miss entire wire groups in a single step, which produces
    gradient-variance dominated by group-coverage noise rather than
    per-error noise — Tomek's observation that
    `batch_fraction = 0.2` failed at n = 5.

    The `adaptive=True` option is a separate, BIASED variant that
    keeps the top half of each group's sampled errors (by previous-step
    contribution) deterministically and resamples the bottom half
    uniformly. Trades unbiasedness for faster per-step convergence
    when the loss landscape is dominated by a small subset of errors.
    In the infinite-sample limit (`fraction_per_group >= 1`) it
    recovers the full loss.

    Args:
        params: variational parameters.
        encoder_fn: callable `(params, code_ind) -> statevector`.
        E_det_grouped: list of dicts, one per wire group:
            {'wires': tuple of qudit indices, 'matrices': (n_g, d^|wires|, d^|wires|)
            complex array, 'inverse_perm': (n_qudit,) int array}.
        K: number of codewords.
        distance: code distance; if >= 3 the diagonal variance term is
            included.
        fraction_per_group: target sampling fraction per group (0, 1].
        rng: numpy random Generator (default new RNG).
        adaptive: enable adaptive top-half-keep / bottom-half-resample.
        prev_contributions: required when adaptive=True; list aligned
            with groups, entry g is an (n_g,) array of last-step
            squared-overlap contributions.
        return_contributions: if True, also return the per-error
            contributions list (suitable to pass back as
            prev_contributions on the next step).
        n_qudit, dim_qudit: required for the local-matrix application.
            Inferred from the first group's `inverse_perm` if None.

    Returns:
        loss (scalar) if return_contributions is False;
        else (loss, contributions) where contributions is a list of
        length len(E_det_grouped), each entry an (n_g,) array of
        unweighted per-error contributions for the sampled errors;
        unsampled errors carry their previous value (or 0 if first
        step). The contributions are unscaled so they can be compared
        across groups.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not E_det_grouped:
        return (0.0, []) if return_contributions else 0.0

    if n_qudit is None:
        n_qudit = len(E_det_grouped[0]['inverse_perm'])
    if dim_qudit is None:
        m0 = E_det_grouped[0]['matrices'][0]
        n_wires = len(E_det_grouped[0]['wires'])
        dim_qudit = int(round(m0.shape[0] ** (1.0 / max(1, n_wires))))

    group_sizes = [int(g['matrices'].shape[0]) for g in E_det_grouped]
    picks, scales = _stratified_pick(
        group_sizes, fraction_per_group, rng,
        adaptive=adaptive, prev_contributions=prev_contributions)

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])

    loss = 0.0
    out_contribs = ([np.asarray(prev_contributions[g], dtype=float).copy()
                     if (prev_contributions is not None
                         and prev_contributions[g] is not None
                         and len(prev_contributions[g]) == group_sizes[g])
                     else np.zeros(group_sizes[g])
                     for g in range(len(E_det_grouped))]
                    if return_contributions else None)

    for g, group in enumerate(E_det_grouped):
        if picks[g].size == 0:
            continue
        wires = tuple(int(w) for w in group['wires'])
        inv_perm = tuple(int(p) for p in group['inverse_perm'])
        matrices = group['matrices']
        scale_g = scales[g]
        for idx in picks[g]:
            M = matrices[int(idx)]
            E_states = qml.math.stack([
                _apply_local_dense(code_states[k], M, wires, inv_perm,
                                   n_qudit, dim_qudit)
                for k in range(K)
            ])
            overlaps = qml.math.tensordot(
                qml.math.conj(code_states),
                qml.math.transpose(E_states), axes=[[1], [0]])
            contrib_term1 = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    contrib_term1 = contrib_term1 + qml.math.abs(
                        overlaps[i, j]) ** 2
            loss = loss + scale_g * contrib_term1
            if distance >= 3:
                diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
                loss = loss + scale_g * (K / 4) * qml.math.real(
                    qml.math.var(diag_vals))
            if out_contribs is not None:
                # Unweighted per-error contribution: Term 1 + (K/4)*Var diag.
                # We need a plain-numpy view for sorting purposes; convert.
                t1 = float(qml.math.toarray(contrib_term1)) \
                    if hasattr(qml.math, "toarray") \
                    else float(np.real(np.asarray(contrib_term1)))
                if distance >= 3:
                    diag_vals_np = np.asarray(
                        [complex(overlaps[k, k]) for k in range(K)])
                    t2 = (K / 4.0) * float(np.var(diag_vals_np.real)) \
                        + (K / 4.0) * float(np.var(diag_vals_np.imag))
                else:
                    t2 = 0.0
                out_contribs[g][int(idx)] = t1 + t2

    if return_contributions:
        return loss, out_contribs
    return loss


def _apply_local_dense(state, matrix, wires, inverse_perm, n_qudit, d):
    """Apply a (d^|wires|, d^|wires|) matrix to the chosen wires of a
    statevector. Ported from src.jax_backend._apply_local_matrix (the
    grouped-error vmap kernel) for the PennyLane / numpy-autograd path.
    Used by kl_loss_detection_stratified and kl_loss_detection_hoeffding.
    """
    shape = tuple([d] * n_qudit)
    s = qml.math.reshape(state, shape)
    w = len(wires)
    if w == 0:
        # No-op on the wires, but still scale by the (1x1) scalar.
        scalar = matrix[0, 0] if hasattr(matrix, "__getitem__") else matrix
        return scalar * state
    matrix_tensor = qml.math.reshape(matrix, tuple([d] * (2 * w)))
    contract_lhs = list(range(w, 2 * w))
    s = qml.math.tensordot(matrix_tensor, s,
                           axes=[contract_lhs, list(wires)])
    s = qml.math.transpose(s, list(inverse_perm))
    return qml.math.reshape(s, (-1,))


def kl_loss_detection_hoeffding(params, encoder_fn, E_det_grouped, K, distance,
                                n_samples=2000, rng=None,
                                return_bound=False, delta=0.05,
                                n_qudit=None, dim_qudit=None):
    """Uniform random sample of n_samples errors from the flat E_det.

    Pulls samples across all wire groups proportional to each group's
    size (so the sample is i.i.d. uniform over the flat E_det list).
    The estimator is scaled by |E_det| / n_samples for unbiased
    recovery of the full loss expectation.

    If return_bound is True, also returns the Hoeffding 95% confidence
    half-width

        epsilon = sqrt(log(2 / delta) / (2 * n_samples)) * R

    where R is the empirical range of per-error contributions in the
    sample (a conservative substitute for the unknown true max-min,
    valid under the assumption that the sample range upper-bounds the
    population range with high probability).

    Args:
        params, encoder_fn, E_det_grouped, K, distance: as in
            kl_loss_detection_stratified.
        n_samples: total number of error draws.
        rng: numpy random Generator.
        return_bound: return (loss, epsilon) instead of loss.
        delta: confidence level for the Hoeffding bound (delta = 0.05
            → 95%).
    """
    if rng is None:
        rng = np.random.default_rng()
    group_sizes = [int(g['matrices'].shape[0]) for g in E_det_grouped]
    n_total = int(sum(group_sizes))
    if n_total == 0:
        return (0.0, 0.0) if return_bound else 0.0

    if n_qudit is None:
        n_qudit = len(E_det_grouped[0]['inverse_perm'])
    if dim_qudit is None:
        m0 = E_det_grouped[0]['matrices'][0]
        n_wires = len(E_det_grouped[0]['wires'])
        dim_qudit = int(round(m0.shape[0] ** (1.0 / max(1, n_wires))))

    flat_idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    # Map flat indices into (group, intra) coordinates.
    boundaries = np.cumsum([0] + group_sizes)
    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])

    loss = 0.0
    scale = n_total / float(len(flat_idx))
    per_sample_contribs = []
    for fi in flat_idx:
        g = int(np.searchsorted(boundaries[1:], fi, side='right'))
        intra = int(fi - boundaries[g])
        group = E_det_grouped[g]
        wires = tuple(int(w) for w in group['wires'])
        inv_perm = tuple(int(p) for p in group['inverse_perm'])
        M = group['matrices'][intra]
        E_states = qml.math.stack([
            _apply_local_dense(code_states[k], M, wires, inv_perm,
                               n_qudit, dim_qudit)
            for k in range(K)
        ])
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states),
            qml.math.transpose(E_states), axes=[[1], [0]])
        t = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                t = t + qml.math.abs(overlaps[i, j]) ** 2
        if distance >= 3:
            diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
            t = t + (K / 4) * qml.math.real(qml.math.var(diag_vals))
        per_sample_contribs.append(float(t) if hasattr(t, '__float__')
                                   else float(np.real(np.asarray(t))))
        loss = loss + scale * t

    if return_bound and per_sample_contribs:
        R = float(np.max(per_sample_contribs) - np.min(per_sample_contribs))
        eps = float(np.sqrt(np.log(2.0 / delta) / (2.0 * len(flat_idx))) * R)
        # Bound applies to mean estimate; we are summing scale * mean,
        # so propagate by the n_total factor.
        epsilon_total = eps * n_total
        return loss, epsilon_total
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
