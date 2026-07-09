"""JAX backend for VarQEC native-gate codes.

Provides a JIT-compiled general-d encoder (`create_jax_encoder`), the
wire-grouped vmap loss (`create_jax_loss_vmap` + the `build_varqec_loss`
factory that composes it with the encoder), and the Adam training loop
(`train_jax`). Uses reverse-mode AD via `jax.grad` — roughly 3-10x faster
than parameter-shift on the cells we train.

The original encoder and error-set / training infrastructure are main's
(Tomek). The wire-grouped vmap loss was ported from Ulrich's
`Qutrit_NativeGates` branch (qutrit-only, module globals) and generalized
to arbitrary d and n_qudit; shape parameters are now explicit. See
`_apply_local_matrix` and `create_jax_loss_vmap` for the ported kernels
and the attribution notes inline.

Design decisions worth noting:
- Statevector dtype is complex128 (`jax_enable_x64=True`). BF16/complex64
  would break KL residual checks on distance >= 3.
- The encoder uses `jax.checkpoint` per layer so memory stays O(d**n)
  rather than O(L * d**n) on the backward pass.
- Errors are grouped by wire signature so a single `jax.vmap` applies
  every error in a group in one tensordot; this avoids the `lax.switch`
  dispatch-table blow-up that broke n>=8 in the pre-merge factored path.

Requires: jax, jaxlib, optax.
"""
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import numpy as np

jax.config.update("jax_enable_x64", True)


# ── JAX gate implementations ──────────────────────────────────────────

def _xy_gate_jax(phi, alpha, level_j, level_k, d):
    """XY gate in JAX (matches native_gates.XY_gate)."""
    I_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    I_jk = I_jk.at[level_j, level_j].set(1.0)
    I_jk = I_jk.at[level_k, level_k].set(1.0)

    X_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    X_jk = X_jk.at[level_j, level_k].set(1.0)
    X_jk = X_jk.at[level_k, level_j].set(1.0)

    Y_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    Y_jk = Y_jk.at[level_j, level_k].set(-1j)
    Y_jk = Y_jk.at[level_k, level_j].set(1j)

    proj_rest = jnp.eye(d, dtype=jnp.complex128) - I_jk
    c = jnp.cos(alpha / 2) + 0j
    s = jnp.sin(alpha / 2) + 0j
    return proj_rest + c * I_jk + (-1j * s * jnp.cos(phi)) * X_jk + (-1j * s * jnp.sin(phi)) * Y_jk


def _z_gate_jax(theta, level_j, level_k, d):
    """Z gate in JAX (matches native_gates.Z_gate)."""
    proj_j = jnp.zeros((d, d), dtype=jnp.complex128).at[level_j, level_j].set(1.0)
    proj_k = jnp.zeros((d, d), dtype=jnp.complex128).at[level_k, level_k].set(1.0)
    proj_rest = jnp.eye(d, dtype=jnp.complex128) - proj_j - proj_k
    return proj_rest + jnp.exp(1j * theta / 2) * proj_j + jnp.exp(-1j * theta / 2) * proj_k


def _ms_gate_jax(phi, theta, level_j, level_k, masks):
    """MS gate in JAX using precomputed masks.

    # gate form verified against Ringbauer et al., Nat. Phys. 18, 1053 (2022),
    # Eq. (2) (arXiv:2109.06903, eq:entOps) — same closed form as
    # src.gates.MS_gate; see that docstring for the verification note.
    """
    M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s = masks
    c = jnp.exp(-1j * theta / 2) * jnp.cos(theta / 2) + 0j
    s = -1j * jnp.exp(-1j * theta / 2) * jnp.sin(theta / 2) + 0j
    p1 = jnp.exp(-1j * theta / 4) + 0j
    s_minus = s * jnp.exp(-1j * 2 * phi)
    s_plus = s * jnp.exp(1j * 2 * phi)
    return c * M_c + p1 * M_p1 + M_p0 + s_minus * M_s_minus + s_plus * M_s_plus + s * M_s


# ── JAX state-vector operations ───────────────────────────────────────

def _apply_single_gate_jax(state, U, q, n_qudit, d):
    """Apply d×d gate to qudit q."""
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    s = jnp.tensordot(U, s, axes=[[1], [q]])
    perm = list(range(1, q + 1)) + [0] + list(range(q + 1, n_qudit))
    s = jnp.transpose(s, perm)
    return s.reshape(-1)


def _apply_two_gate_jax(state, U, q1, q2, n_qudit, d):
    """Apply d²×d² gate to qudits q1, q2."""
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    U_4d = U.reshape(d, d, d, d)
    s = jnp.tensordot(U_4d, s, axes=[[2, 3], [q1, q2]])
    remaining = sorted(set(range(n_qudit)) - {q1, q2})
    perm = [0] * n_qudit
    perm[q1] = 0
    perm[q2] = 1
    for i, r in enumerate(remaining):
        perm[r] = i + 2
    s = jnp.transpose(s, perm)
    return s.reshape(-1)


def _apply_factored_error_jax(state, factors, n_qudit, d):
    """Apply factored error (list of (qudit_idx, op) pairs) to state."""
    s = state
    for q, op in factors:
        s = _apply_single_gate_jax(s, op, q, n_qudit, d)
    return s


def _apply_local_matrix(state, matrix, wires, inverse_perm, d, n_qudit):
    """Apply a (d**w x d**w) matrix to the selected wires of a statevector.

    Ported from Ulrich's Updated_Qutrit_VQC.py apply_local_matrix
    (qutrit-only, module globals). Generalized to any d and n_qudit;
    shape parameters now explicit.

    The matrix acts on `wires = (w0, w1, ..., w_{k-1})` with w = len(wires);
    `inverse_perm` is the length-n_qudit permutation that restores original
    wire order after the wires axes are moved to the front by the contraction.
    For wires=() and matrix=[[1.0]] the call is a no-op (scalar broadcast).

    Vmap-friendly over `matrix` with `in_axes=(None, 0, None, None, None, None)`.
    """
    shape = (d,) * n_qudit
    state_tensor = jnp.reshape(state, shape)
    w = len(wires)
    matrix_tensor = jnp.reshape(matrix, (d,) * (2 * w))
    axes = list(range(w, 2 * w))
    contracted = jnp.tensordot(matrix_tensor, state_tensor,
                               axes=(axes, list(wires)))
    state_tensor = jnp.transpose(contracted, inverse_perm)
    return jnp.reshape(state_tensor, (d ** n_qudit,))


def create_jax_loss_vmap(E_det_grouped, K, d, n_qudit, distance):
    """Build a jit-compiled KL detection loss using vmap over grouped errors.

    Ported from Ulrich's Updated_Qutrit_VQC.py build_loss_func
    (qutrit-only, module globals). Generalized to any d and n_qudit;
    shape parameters now explicit.

    Algorithm. Errors sharing a wire signature are stacked and applied by a
    single `jax.vmap`. Term 1 (off-diagonal) vmaps only over the matrix
    axis and iterates the K*(K-1)/2 codeword pairs in Python (K is small).
    Term 2 (distance >= 3) uses a double vmap, outer over matrices, inner
    over K codewords, to compute all <psi_k|E|psi_k> expectations in one
    shot. The whole closure is `@jax.jit` so the Python group-iteration is
    traced away at compile time; there is no `lax.scan` and no `lax.switch`.

    Args:
        E_det_grouped: list of dicts from ErrorModel.build_grouped().
        K: number of codewords (= d for one logical qudit).
        d: qudit dimension.
        n_qudit: number of physical qudits.
        distance: code distance. Term 2 is included iff distance >= 3.

    Returns:
        loss_fn(code_states) -> scalar, where code_states has shape
        (K, d**n_qudit).
    """
    prepared = []
    for g in E_det_grouped:
        wires = g['wires']
        inverse_perm = g['inverse_perm']
        matrices = jnp.asarray(g['matrices'], dtype=jnp.complex128)
        prepared.append((wires, inverse_perm, matrices))

    vmap_apply_E = jax.vmap(
        _apply_local_matrix,
        in_axes=(None, 0, None, None, None, None))
    vmap_apply_M = jax.vmap(
        jax.vmap(_apply_local_matrix,
                 in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None))

    @jit
    def loss_fn(code_states):
        loss = jnp.array(0.0, dtype=jnp.float64)
        for wires, inverse_perm, matrices in prepared:
            # Term 1: Σ_{E in group} Σ_{i<j} |<psi_i|E|psi_j>|^2
            for i in range(K):
                for j in range(i + 1, K):
                    E_cj = vmap_apply_E(
                        code_states[j], matrices, wires, inverse_perm,
                        d, n_qudit)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj, axis=1)
                    loss = loss + jnp.sum(jnp.abs(inner) ** 2)

            if distance >= 3:
                # Term 2: (K/4) Σ_{E in group} Var_k <psi_k|E|psi_k>
                # Prefactor K/4 follows Cao et al. arXiv:2204.03560 Eq. 16.
                # Ulrich's Updated_Qutrit_VQC.py had K/3 -- both yield loss=0
                # at the KL fixed point, but K/4 is the paper's choice and
                # gives different gradient balance between off-diagonal and
                # diagonal-variance terms. Using K/4 here to stay consistent
                # with main's existing src/loss.py.
                M_psi = vmap_apply_M(
                    code_states, matrices, wires, inverse_perm,
                    d, n_qudit)
                vals = jnp.sum(
                    jnp.conj(code_states)[None, :, :] * M_psi, axis=2)
                loss = loss + (K / 4) * jnp.sum(
                    jnp.real(jnp.var(vals, axis=1)))
        return loss

    return loss_fn


# ── JAX encoder factory ───────────────────────────────────────────────

def create_jax_encoder(n_qudit, d, connections=None, use_scan=True):
    """
    Create a JAX encoder for native trapped-ion gates.

    Args:
        n_qudit: number of physical qudits
        d: qudit dimension
        connections: qudit connectivity (default: ring)
        use_scan: if True (default), use jax.lax.scan + checkpoint for fast
                  compile and low memory. False unrolls all layers (easier to debug).

    Returns: (encoder_fn, connections, params_per_layer)
    """
    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]

    n_transitions = d - 1
    n_conn = len(connections)
    params_per_layer = (5 * n_qudit + 2 * n_conn) * n_transitions
    dim = d ** n_qudit

    from src.gates import _build_ms_masks
    ms_masks = {}
    for t in range(n_transitions):
        masks_np = _build_ms_masks(t, t + 1, d)
        ms_masks[(t, t+1)] = tuple(jnp.array(m, dtype=jnp.complex128) for m in masks_np)

    connections_tuple = tuple(tuple(c) for c in connections)

    def _apply_layer(state, layer_p):
        """Apply one ansatz layer (shared between scan and unrolled paths)."""
        pi = 0
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
        for q1, q2 in connections_tuple:
            for t in range(n_transitions):
                U = _ms_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, ms_masks[(t, t+1)])
                state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
                pi += 2
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _z_gate_jax(layer_p[pi], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 1
        return state

    def _init_state(code_ind):
        state = jnp.zeros(dim, dtype=jnp.complex128)
        return state.at[code_ind * d ** (n_qudit - 1)].set(1.0)

    if use_scan:
        @jax.checkpoint
        def _checkpointed_layer(state, layer_p):
            return _apply_layer(state, layer_p)

        def encoder(params, code_ind):
            state = _init_state(code_ind)
            state, _ = jax.lax.scan(
                lambda s, p: (_checkpointed_layer(s, p), None), state, params)
            return state
    else:
        def encoder(params, code_ind):
            state = _init_state(code_ind)
            for l in range(params.shape[0]):
                state = _apply_layer(state, params[l])
            return state

    return encoder, connections, params_per_layer


def create_jax_encoder_scan(n_qudit, d, connections=None):
    """Backward compat: use create_jax_encoder(use_scan=True) instead."""
    return create_jax_encoder(n_qudit, d, connections, use_scan=True)


# ── JAX loss functions ────────────────────────────────────────────────

def create_jax_loss(encoder_fn, E_det_np, K, distance):
    """
    Create a JIT-compiled detection-style KL loss function.

    Uses a loop over errors inside @jit (faster than einsum for >50 errors
    due to better memory locality). JAX traces the loop at compile time.
    """
    E_det_jax = jnp.stack([jnp.array(E, dtype=jnp.complex128) for E in E_det_np])
    n_errors = len(E_det_np)

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])
        loss = 0.0

        for e_idx in range(n_errors):
            E = E_det_jax[e_idx]
            E_applied = E @ code_states.T  # (dim, K)
            overlaps = jnp.conj(code_states) @ E_applied  # (K, K)

            for i in range(K):
                for j in range(i + 1, K):
                    loss = loss + jnp.abs(overlaps[i, j]) ** 2

            if distance >= 3:
                diag_vals = jnp.array([overlaps[k, k] for k in range(K)])
                loss = loss + (K / 4) * jnp.real(jnp.var(diag_vals))

        return loss

    return loss_fn


def create_jax_loss_factored(encoder_fn, E_det_factors, n_qudit, d, K, distance):
    """
    JIT-compiled loss using factored error application.

    For n>=7 where dense error matrices are infeasible.
    Each error is a list of (qudit_idx, d×d matrix) pairs.
    Memory: O(d^n) per state vs O(d^{2n}) per dense error operator.
    """
    # Convert factors to JAX arrays
    E_det_jax_factors = []
    for factors in E_det_factors:
        jax_factors = [(q, jnp.array(op, dtype=jnp.complex128)) for q, op in factors]
        E_det_jax_factors.append(jax_factors)

    n_errors = len(E_det_factors)
    mask_upper = jnp.triu(jnp.ones((K, K), dtype=bool), k=1)

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])
        loss = 0.0

        for e_idx in range(n_errors):
            factors = E_det_jax_factors[e_idx]
            E_states = jnp.stack([
                _apply_factored_error_jax(code_states[k], factors, n_qudit, d)
                for k in range(K)
            ])
            overlaps = jnp.conj(code_states) @ E_states.T  # (K, K)

            # Off-diagonal
            loss = loss + jnp.sum(jnp.abs(overlaps[mask_upper]) ** 2)

            # Diagonal variance
            if distance >= 3:
                diag_vals = jnp.diag(overlaps)
                loss = loss + (K / 4) * jnp.real(jnp.var(diag_vals))

        return loss

    return loss_fn


# ── VarQEC loss factory (vmap path, the only JAX path) ───────────────

def build_varqec_loss(encoder_fn, K, d, n_qudit, distance, E_det_grouped):
    """Return a params-based VarQEC loss for the JAX vmap path.

    Composes the encoder with `create_jax_loss_vmap` (the Ulrich port) into
    a single `@jax.jit`-decorated closure that takes params. Callers build
    the grouped-error representation with `ErrorModel.build_grouped()`.

    Args:
        encoder_fn: (params, code_ind) -> statevector.
        K: number of codewords.
        d, n_qudit: qudit dimension and count.
        distance: code distance.
        E_det_grouped: output of ErrorModel.build_grouped().

    Returns:
        loss_fn(params) -> scalar.
    """
    if E_det_grouped is None:
        raise ValueError("build_varqec_loss requires E_det_grouped")
    core = create_jax_loss_vmap(E_det_grouped, K, d, n_qudit, distance)
    enc_vmapped = jax.vmap(encoder_fn, in_axes=(None, 0))

    @jit
    def loss_fn(params):
        code_states = enc_vmapped(params, jnp.arange(K))
        return core(code_states)

    return loss_fn


# ── Round-12 additions: weighted-vmap loss for stratified sampling ────


def create_jax_loss_vmap_weighted(E_det_grouped, K, d, n_qudit, distance):
    """Like create_jax_loss_vmap, but each group accepts a per-error
    weight vector. weights_per_group is a tuple aligned with
    E_det_grouped: weights_per_group[g] has shape (n_g,).

    Per-error contribution is scaled by weights_per_group[g][e]. The
    same Term 1 and Term 2 structure as create_jax_loss_vmap. Setting
    weights to ones recovers the full loss exactly.

    Stratified sampling at fraction f_g per group: weights[e] =
        (N_g / n_g_sampled) on chosen errors, 0 elsewhere
    → unbiased estimator of the full loss. Hoeffding-style uniform
    sampling sets weights = (N_total / n_samples) on chosen errors
    across all groups, 0 elsewhere.
    """
    prepared = []
    for g in E_det_grouped:
        wires = g['wires']
        inverse_perm = g['inverse_perm']
        matrices = jnp.asarray(g['matrices'], dtype=jnp.complex128)
        prepared.append((wires, inverse_perm, matrices))

    vmap_apply_E = jax.vmap(
        _apply_local_matrix,
        in_axes=(None, 0, None, None, None, None))
    vmap_apply_M = jax.vmap(
        jax.vmap(_apply_local_matrix,
                 in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None))

    @jit
    def loss_fn(code_states, weights_per_group):
        loss = jnp.array(0.0, dtype=jnp.float64)
        for idx, (wires, inverse_perm, matrices) in enumerate(prepared):
            w = weights_per_group[idx]  # (n_g,)
            # Term 1: Σ_e w_e Σ_{i<j} |<psi_i|E_e|psi_j>|^2
            for i in range(K):
                for j in range(i + 1, K):
                    E_cj = vmap_apply_E(
                        code_states[j], matrices, wires, inverse_perm,
                        d, n_qudit)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj, axis=1)
                    loss = loss + jnp.sum(w * (jnp.abs(inner) ** 2))

            if distance >= 3:
                M_psi = vmap_apply_M(
                    code_states, matrices, wires, inverse_perm,
                    d, n_qudit)
                vals = jnp.sum(
                    jnp.conj(code_states)[None, :, :] * M_psi, axis=2)
                # weighted variance contribution per error
                var_per_err = jnp.real(jnp.var(vals, axis=1))
                loss = loss + (K / 4) * jnp.sum(w * var_per_err)
        return loss

    return loss_fn


def create_jax_loss_vmap_weighted_v2(E_det_grouped, K, d, n_qudit, distance):
    """Cross-group batched version of create_jax_loss_vmap_weighted.

    Same contract: loss_fn(code_states, weights_per_group) with
    weights_per_group a tuple aligned with E_det_grouped, ones = full loss.

    Instead of one vmap dispatch per wire group (16 at n=5, 29 at n=7),
    groups are batched by arity (= len(wires)): every group's matrix stack
    in an arity class is applied in a single batched einsum GEMM — with an
    explicit (G, n_g, a, a) group axis when stack sizes are uniform (the
    campaign bases; broadcasting S avoids a gather whose backward is a
    large scatter-add — measured 0.48x vs 1.7-1.9x at n=7), else along a
    flat op axis via an op->group gather. This is exact because both loss
    terms are additive over ops (Term 1 and Term 2 are sums of
    w_e * f(E_e)); per-group structure only determines WHICH wires each
    op acts on.

    The wire structure is handled by pre-permuting the codewords once per
    group: <phi|E|psi> is invariant under any relabeling of the state
    axes, so with psi_g = transpose(psi, wires + rest).reshape(a, b)
    (a = d**w, b = d**(n-w)) the overlap is vdot(phi_g, M @ psi_g) — no
    inverse_perm transpose-back is needed at all. Agreement with the v1
    path is pinned to 1e-12 (loss) / 1e-10 (grad) by
    tests/test_loss_fastpath.py.
    """
    # Group the wire groups into arity classes; record, per class, the
    # member group indices (for weight concatenation), each member's
    # front-wires permutation, the concatenated matrix stack, and the
    # op -> class-local group index gather map.
    classes = {}
    for gi, g in enumerate(E_det_grouped):
        wires = tuple(int(w) for w in g['wires'])
        classes.setdefault(len(wires), []).append((gi, wires, g['matrices']))

    prepared = []
    for w_arity in sorted(classes):
        members = classes[w_arity]
        a = d ** w_arity
        group_indices = [gi for gi, _, _ in members]
        perms = [wires + tuple(q for q in range(n_qudit) if q not in wires)
                 for _, wires, _ in members]
        sizes = [int(np.asarray(m).shape[0]) for _, _, m in members]
        if len(set(sizes)) == 1:
            # Uniform stack sizes (the campaign bases): keep the group
            # axis explicit, (G, n_g, a, a) — broadcasting S over the op
            # axis avoids a gather whose backward is a large scatter-add.
            mats = jnp.stack(
                [jnp.asarray(m, dtype=jnp.complex128).reshape(-1, a, a)
                 for _, _, m in members])
            op_to_group = None
        else:
            mats = jnp.concatenate(
                [jnp.asarray(m, dtype=jnp.complex128).reshape(-1, a, a)
                 for _, _, m in members], axis=0)
            op_to_group = jnp.asarray(
                np.repeat(np.arange(len(members)), sizes), dtype=jnp.int32)
        prepared.append((w_arity, group_indices, perms, mats, op_to_group))

    iu, ju = np.triu_indices(K, k=1)
    diag_idx = jnp.arange(K)

    @jit
    def loss_fn(code_states, weights_per_group):
        loss = jnp.array(0.0, dtype=jnp.float64)
        psi_t = code_states.reshape((K,) + (d,) * n_qudit)
        for w_arity, group_indices, perms, mats, op_to_group in prepared:
            a = d ** w_arity
            # Per-group permuted codewords: (G, K, a, b)
            S = jnp.stack([
                jnp.transpose(psi_t, (0,) + tuple(q + 1 for q in perm))
                .reshape(K, a, -1)
                for perm in perms])
            w_cat = jnp.concatenate(
                [weights_per_group[gi] for gi in group_indices])
            if op_to_group is None:
                # mats: (G, n_g, a, a); broadcast S over the op axis
                applied = jnp.einsum('geab,gkbc->gekac', mats, S)
                overlaps = jnp.einsum('giab,gejab->geij',
                                      jnp.conj(S), applied)
                overlaps = overlaps.reshape(-1, K, K)
            else:
                gathered = S[op_to_group]  # (E, K, a, b)
                applied = jnp.einsum('eab,ekbc->ekac', mats, gathered)
                overlaps = jnp.einsum('eiab,ejab->eij',
                                      jnp.conj(gathered), applied)
            # overlaps[e, i, j] = <psi_i | E_e | psi_j>, e class-flat
            # Term 1: Σ_e w_e Σ_{i<j} |<psi_i|E_e|psi_j>|^2
            t1 = jnp.sum(jnp.abs(overlaps[:, iu, ju]) ** 2, axis=1)
            loss = loss + jnp.sum(w_cat * t1)
            if distance >= 3:
                # Term 2: (K/4) Σ_e w_e Var_k <psi_k|E_e|psi_k>
                diag = overlaps[:, diag_idx, diag_idx]
                var_per_err = jnp.real(jnp.var(diag, axis=1))
                loss = loss + (K / 4) * jnp.sum(w_cat * var_per_err)
        return loss

    return loss_fn


def build_varqec_loss_weighted(encoder_fn, K, d, n_qudit, distance,
                               E_det_grouped, fast=True):
    """Params-based weighted KL loss for the JAX vmap path.

    Returns a callable `loss_fn(params, weights_per_group)` where
    weights_per_group is a tuple of jnp arrays, one per wire group,
    each of shape (n_g,). Setting all weights to 1 recovers the full
    loss; stratified sampling uses zero-or-N_g/n_g_sampled weights.

    fast=True (default) selects the cross-group batched core
    (create_jax_loss_vmap_weighted_v2, measured 1.7-1.9x at n=7 and
    2.1x at n=5 on value_and_grad — audit/07 Part 4); fast=False keeps
    the original per-group vmap core for debugging. The two agree to
    1e-12 (loss) / 1e-10 (grad) — pinned by tests/test_loss_fastpath.py.
    """
    if E_det_grouped is None:
        raise ValueError("build_varqec_loss_weighted requires E_det_grouped")
    make_core = (create_jax_loss_vmap_weighted_v2 if fast
                 else create_jax_loss_vmap_weighted)
    core = make_core(E_det_grouped, K, d, n_qudit, distance)
    enc_vmapped = jax.vmap(encoder_fn, in_axes=(None, 0))

    @jit
    def loss_fn(params, weights_per_group):
        code_states = enc_vmapped(params, jnp.arange(K))
        return core(code_states, weights_per_group)

    return loss_fn


def stratified_weights(group_sizes, fraction_per_group, rng, dtype=None):
    """Build a stratified-sampling weights tuple. For each group g of
    size N_g, samples n_g_sampled = max(1, ceil(f * N_g)) errors and
    returns weights of shape (N_g,) with N_g/n_g_sampled on those
    positions and 0 elsewhere. Tuple is suitable for passing to
    `build_varqec_loss_weighted`.

    Note: this per-wire-group uniform subsampling with
    inverse-inclusion-probability scaling is the same equivalent scheme
    independently implemented by U. Holzer
    (Tensorform_Wire-grouped-Batching_Random_Seeding.py); the two were
    verified mechanically equivalent and unbiased in audit/04 (H6), and
    unbiasedness is pinned by tests/test_sampling_unbiased.py. Paper-safe
    name: "uniform wire-group subsampling" — distinct from the R14
    Meth-weighted "Stratified" and "Importance-weighted" samplers in
    src/sampling/stratified_importance.py.
    """
    if dtype is None:
        dtype = jnp.float64
    weights = []
    for n_g in group_sizes:
        if n_g == 0:
            weights.append(jnp.zeros((0,), dtype=dtype))
            continue
        n_keep = max(1, int(np.ceil(fraction_per_group * n_g)))
        n_keep = min(n_keep, n_g)
        idx = rng.choice(n_g, n_keep, replace=False)
        w = np.zeros(n_g, dtype=float)
        w[idx] = float(n_g) / float(n_keep)
        weights.append(jnp.asarray(w, dtype=dtype))
    return tuple(weights)


def hoeffding_weights(group_sizes, n_samples, rng, dtype=None):
    """Build a Hoeffding-uniform weights tuple. Samples n_samples
    errors uniformly from the flat E_det (proportional to each group's
    size) and assigns weight N_total / n_samples to chosen errors,
    0 elsewhere.
    """
    if dtype is None:
        dtype = jnp.float64
    group_sizes = np.asarray(group_sizes, dtype=int)
    n_total = int(group_sizes.sum())
    if n_total == 0:
        return tuple(jnp.zeros((0,), dtype=dtype) for _ in group_sizes)
    n_samples = min(int(n_samples), n_total)
    flat_idx = rng.choice(n_total, n_samples, replace=False)
    boundaries = np.concatenate([[0], np.cumsum(group_sizes)])
    weights = []
    scale = float(n_total) / float(n_samples)
    for g, n_g in enumerate(group_sizes):
        mask = (flat_idx >= boundaries[g]) & (flat_idx < boundaries[g + 1])
        w = np.zeros(n_g, dtype=float)
        w[(flat_idx[mask] - boundaries[g])] = scale
        weights.append(jnp.asarray(w, dtype=dtype))
    return tuple(weights)


# ── JAX training loop ─────────────────────────────────────────────────

def train_jax(loss_fn, n_layers, params_per_layer, n_steps=2000,
              lr=0.05, lr_switch=0.01, seed=0):
    """
    Train with JAX + optax Adam.

    Returns:
        (best_params, losses): best parameters (numpy) and loss history
    """
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (n_layers, params_per_layer),
                               minval=0.0, maxval=2 * np.pi)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)
    val_grad_fn = value_and_grad(loss_fn)

    best_loss = 1e10
    best_theta = theta
    losses = []
    lr_switched = False

    for step in range(n_steps):
        loss_val, grads = val_grad_fn(theta)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        lv = float(loss_val)
        losses.append(lv)

        if lv < best_loss:
            best_loss = lv
            best_theta = theta

        if not lr_switched and lv < 0.1:
            optimizer = optax.adam(lr_switch)
            opt_state = optimizer.init(theta)
            lr_switched = True

        if step % 50 == 0:
            print(f"  step {step:4d} | loss={lv:.4e}")

        if lv < 1e-6:
            print(f"  CONVERGED at step {step}")
            break

    return np.array(best_theta), losses
