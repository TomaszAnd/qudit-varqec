"""
JAX backend for VarQEC native-gate codes.

Provides JIT-compiled encoder, vectorized loss function, and training loop.
Uses jax.grad (reverse-mode AD) instead of parameter-shift for ~3-10x speedup.

Requires: jax, jaxlib, optax
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
    """MS gate in JAX using precomputed masks."""
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


def create_jax_loss_scan(encoder_fn, E_det_np, K, distance):
    """
    JIT loss with lax.scan over errors. Compile time O(1 error) not O(n_errors).

    Critical for n>=7 where n_errors > 400. The scan body is compiled once and
    iterated over the leading axis of the stacked E_det tensor at runtime.
    """
    E_det_jax = jnp.stack([jnp.array(E, dtype=jnp.complex128) for E in E_det_np])

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])

        def error_contribution(carry, E):
            E_applied = E @ code_states.T
            overlaps = jnp.conj(code_states) @ E_applied

            loss_e = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    loss_e = loss_e + jnp.abs(overlaps[i, j]) ** 2

            if distance >= 3:
                diag_vals = jnp.diag(overlaps)
                loss_e = loss_e + (K / 4) * jnp.real(jnp.var(diag_vals))

            return carry + loss_e, None

        total_loss, _ = jax.lax.scan(error_contribution, 0.0, E_det_jax)
        return total_loss

    return loss_fn


def create_jax_loss_scan_minibatch(encoder_fn, E_det_np, K, distance, batch_size=100):
    """
    Scan loss with stochastic minibatching. Requires a PRNG key argument.

    Usage:
        loss_fn = create_jax_loss_scan_minibatch(enc, E_det, K, dist, batch_size=80)
        key = jax.random.PRNGKey(0)
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            loss, grads = jax.value_and_grad(loss_fn)(params, subkey)
    """
    E_det_jax = jnp.stack([jnp.array(E, dtype=jnp.complex128) for E in E_det_np])
    n_errors = len(E_det_np)
    scale = n_errors / batch_size

    @jit
    def loss_fn(params, key):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])

        indices = jax.random.choice(key, n_errors, shape=(batch_size,), replace=False)
        E_batch = E_det_jax[indices]

        def error_contribution(carry, E):
            E_applied = E @ code_states.T
            overlaps = jnp.conj(code_states) @ E_applied

            loss_e = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    loss_e = loss_e + jnp.abs(overlaps[i, j]) ** 2

            if distance >= 3:
                diag_vals = jnp.diag(overlaps)
                loss_e = loss_e + (K / 4) * jnp.real(jnp.var(diag_vals))

            return carry + loss_e, None

        total_loss, _ = jax.lax.scan(error_contribution, 0.0, E_batch)
        return scale * total_loss

    return loss_fn


def create_jax_loss_factored_scan(encoder_fn, E_det_factors, n_qudit, d, K, distance):
    """
    Factored loss with scan over errors. Memory O(d^n), compile O(1 error).

    Uses jax.lax.switch to dispatch gate application to the correct qudit
    (avoids traced-integer-as-axis error inside scan).
    """
    max_weight = max((len(f) for f in E_det_factors), default=0)
    if max_weight == 0:
        max_weight = 1

    n_errors = len(E_det_factors)

    padded_qudits = np.zeros((n_errors, max_weight), dtype=np.int32)
    padded_ops = np.zeros((n_errors, max_weight, d, d), dtype=np.complex128)
    padded_mask = np.ones((n_errors, max_weight), dtype=bool)

    for e_idx, factors in enumerate(E_det_factors):
        for f_idx, (q, op) in enumerate(factors):
            padded_qudits[e_idx, f_idx] = q
            padded_ops[e_idx, f_idx] = np.array(op)
            padded_mask[e_idx, f_idx] = False
        for f_idx in range(len(factors), max_weight):
            padded_ops[e_idx, f_idx] = np.eye(d)

    padded_qudits_jax = jnp.array(padded_qudits)
    padded_ops_jax = jnp.array(padded_ops, dtype=jnp.complex128)
    padded_mask_jax = jnp.array(padded_mask)

    # Precompile one branch per qudit for lax.switch
    def _make_apply_q(q_fixed):
        def apply_q(s, op):
            return _apply_single_gate_jax(s, op, q_fixed, n_qudit, d)
        return apply_q

    apply_branches = [_make_apply_q(q) for q in range(n_qudit)]

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])

        def error_contribution(carry, error_data):
            qudits, ops, mask = error_data

            def apply_padded_factors(state):
                s = state
                for f_idx in range(max_weight):
                    op = ops[f_idx]
                    q = qudits[f_idx]
                    is_id = mask[f_idx]
                    new_s = jax.lax.switch(q, apply_branches, s, op)
                    s = jnp.where(is_id, s, new_s)
                return s

            E_states = jnp.stack([apply_padded_factors(code_states[k]) for k in range(K)])
            overlaps = jnp.conj(code_states) @ E_states.T

            loss_e = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    loss_e = loss_e + jnp.abs(overlaps[i, j]) ** 2

            if distance >= 3:
                diag_vals = jnp.diag(overlaps)
                loss_e = loss_e + (K / 4) * jnp.real(jnp.var(diag_vals))

            return carry + loss_e, None

        total_loss, _ = jax.lax.scan(
            error_contribution, 0.0,
            (padded_qudits_jax, padded_ops_jax, padded_mask_jax)
        )
        return total_loss

    return loss_fn


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
