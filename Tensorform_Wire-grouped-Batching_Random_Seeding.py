import warnings, itertools

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
# ERROR SET GENERATION (Standard NumPy is fine here as it's static)
import numpy as onp  # Original numpy for static setup

import time
from datetime import datetime

import os

def log_time(message):
    """Prints a formatted timestamp alongside a custom message."""
    current_time = datetime.now().strftime('%H:%M:%S')
    print(f"[{current_time}] {message}")

# Record the absolute start of the script
script_start_time = time.time()
log_time("PROGRAM START: Initializing Quantum Environment")

# Crucial: Enable 64-bit precision for complex quantum state calculations
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. PHYSICAL SETUP & CONSTANTS
# ==========================================
N_QUTRIT = 5            # Number of physical qutrits
LOGICAL_QUTRIT = 1      # Number of logical qutrits being encoded
DIM_QUTRIT = 3          # Dimension of the system (3 = qutrit)
DISTANCE = 3           # Distance 2 = Error Detection

K = DIM_QUTRIT ** LOGICAL_QUTRIT

# Qutrit connections
connections = list(itertools.combinations(range(N_QUTRIT), 2))
ring_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]   # ring
star_connections = [[0, 1], [0, 2], [0, 3], [0, 4]]   # star
chord_connections = [[0, 2], [2, 4], [4, 1], [1, 3], [3, 0]]    # chords
cascade_connections = []
for i in range(N_QUTRIT - 1):
    for j in range(i + 1, N_QUTRIT):
        cascade_connections.append([i, j])

#PARAMS_PER_LAYER = 70   # For MS-layer 70, for LS-layer 60 (can be used hardcoded or with get_params_per_layer function)

# ==========================================
# 2. HARDWARE ERROR MODEL
# ==========================================
# Returns the 8 Gell-Mann matrices for qutrit error channels
def hardware_error_basis():
    errors = []

    # Amplitude Errors (Bit Flips / Population Transfer)
    # 0 <-> 1 subspace (symmetric and antisymmetric)
    L1 = onp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    L2 = onp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)

    # 0 <-> 2 subspace (symmetric and antisymmetric)
    L4 = onp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    L5 = onp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)

    # 1 <-> 2 subspace (symmetric and antisymmetric)
    L6 = onp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    L7 = onp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)

    errors.extend([L1, L2, L4, L5, L6, L7])

    # Phase Errors (Dephasing / AC Stark Shifts)

    # Relative phase shift between 0 and 1
    L3 = onp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)

    # Symmetric phase shift of 0 and 1 relative to 2
    L8 = (1 / onp.sqrt(3)) * onp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)

    errors.extend([L3, L8])

    return errors

# Building up error set, yet to complete
def build_error_set(n_qutrit: int, distance: int):
    dim_qutrit = 3
    I3 = onp.eye(dim_qutrit, dtype=complex)

    # Store errors as tuples: (matrix, target_qutrits)
    E_det = [(I3, (0,))] # Identity (no error) acts on wire 0 (arbitrary, has no effect)
    E_corr = [(I3, (0,))]

    max_det = distance - 1
    max_corr = (distance - 1) // 2
    single_errs = hardware_error_basis()

    for w in range(1, max_det + 1):
        for qutrit_subset in itertools.combinations(range(n_qutrit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                # Combine only the active errors (e.g., a 9x9 matrix for a weight-2 error)
                op = err_choice[0]
                for e in err_choice[1:]:
                    op = onp.kron(op, e)

                E_det.append((op, qutrit_subset))
                if w <= max_corr:
                    E_corr.append((op, qutrit_subset))

    return E_det, E_corr


# Generate the error tracking lists before training begins
E_det_onp, E_corr_onp = build_error_set(N_QUTRIT, DISTANCE)


def group_by_wires(error_list):
    """
    Groups local error matrices by their target wires and precomputes
    the inverse transposition order for JAX tensordot.
    """
    grouped = {}
    for mat, wires in error_list:
        if wires not in grouped:
            grouped[wires] = []
        grouped[wires].append(mat)

    result = {}
    for wires, mats in grouped.items():
        # 1. Stack the matrices and cast to JAX
        stacked_mats = jnp.array(onp.stack(mats).astype(onp.complex128))

        # 2. Precompute the inverse_order for this specific wire combination
        leftover_wires = [i for i in range(N_QUTRIT) if i not in wires]
        new_order = list(wires) + leftover_wires
        inverse_order = tuple(new_order.index(i) for i in range(N_QUTRIT))

        # Store both the matrices and the precalculated order
        result[wires] = (stacked_mats, inverse_order)

    return result

def apply_local_matrix(state, matrix, wires, inverse_order, d=3):
    """Applies a small matrix to specific wires of a state vector via tensor contraction"""
    # 1. Reshape the high number-element vector into a N_QUTRIT-D tensor (3, 3,... , 3)
    state_tensor = jnp.reshape(state, (d,) * N_QUTRIT)

    # 2. Reshape the matrix into a multi-dim tensor
    w = len(wires)
    matrix_tensor = jnp.reshape(matrix, (d,) * (2*w))

    # 3. Contract the matrix with the state tensor on the target wires. Because the matrix was built with np.kron,
    # the "input" columns we need to contract with the state are the odd indices.
    axes = list(range(w, 2*w))
    contracted = jnp.tensordot(matrix_tensor, state_tensor, axes=(axes, wires))

    state_tensor = jnp.transpose(contracted, inverse_order)

    # 4. Flatten back to a high number-element vector
    return jnp.reshape(state_tensor, (d ** N_QUTRIT,))


# --- NEW PRECOMPUTATION CODE ---
def precompute_error_products_dedup(E_corr):
    """Precompute UNIQUE E_a†E_b products with deduplication."""
    M_products = []
    seen = {}
    for Ea, wiresA in E_corr:
        Ea_dag = onp.conj(Ea.T)
        qa =wiresA[0]
        for Eb, wiresB in E_corr:
            qb = wiresB[0]

            if qa == qb:
                M_mat = Ea_dag @ Eb
                M_wires = (qa,)
            else:
                M_wires = tuple(sorted((qa, qb)))
                if qa < qb:
                    M_mat = onp.kron(Ea_dag, Eb)
                else:
                    M_mat = onp.kron(Eb, Ea_dag)

            # Rounding to handle floating point noise
            key = (M_wires, tuple(onp.round(M_mat.ravel(), decimals=10)))
            if key not in seen:
                seen[key] = len(M_products)
                M_products.append((M_mat, M_wires))

    n_total = len(E_corr) ** 2
    print(f"Deduplicated: {n_total} products -> {len(M_products)} unique")
    return M_products

M_products_onp = precompute_error_products_dedup(E_corr_onp)

print(f"E_det size = {len(E_det_onp)}, Unique M matrices = {len(M_products_onp)}")

# ==========================================
# 3. NATIVE HARDWARE GATES
# ==========================================
# Native single-qutrit operation driving transitions in the XY plane
def XY_gate(phi, alpha, level_j, level_k, d=3):
    U = jnp.eye(d, dtype=jnp.complex128)

    c = jnp.cos(alpha / 2)
    s = jnp.sin(alpha / 2)

    # Set the diagonal elements for the active subspace
    U = U.at[level_j, level_j].set(c)
    U = U.at[level_k, level_k].set(c)

    # Set the off-diagonal elements with the geometric phase
    U = U.at[level_j, level_k].set(-1j * s * jnp.exp(-1j * phi))
    U = U.at[level_k, level_j].set(-1j * s * jnp.exp(1j * phi))

    return U

# Native single-qutrit operation for phase shifts between two populations
def Z_gate(theta, level_j, level_k, d=3):
    Z_jk = jnp.eye(d, dtype=jnp.complex128)
    Z_jk = Z_jk.at[level_j, level_j].multiply(jnp.exp(1j * theta / 2))
    Z_jk = Z_jk.at[level_k, level_k].multiply(jnp.exp(-1j * theta / 2))
    return Z_jk

# Native Molmer-Sorensen entangling gate
def MS_gate(theta, phi, level_j, level_k, d=3):
    # 1. Build the base sigma_phi generator
    sigma_phi = jnp.zeros((d, d), dtype=jnp.complex128)
    sigma_phi = sigma_phi.at[level_j, level_k].set(jnp.exp(-1j * phi))
    sigma_phi = sigma_phi.at[level_k, level_j].set(jnp.exp(1j * phi))

    # 2. Build the term_sum matrix (let's call it S)
    I_d = jnp.eye(d, dtype=jnp.complex128)
    S = jnp.kron(sigma_phi, I_d) + jnp.kron(I_d, sigma_phi)

    # 3. Calculate exact matrix powers (lightning fast in JAX)
    S2 = S @ S
    S4 = S2 @ S2

    # 4. Calculate polynomial coefficients analytically
    exp_theta = jnp.exp(-1j * theta)
    exp_theta_4 = jnp.exp(-1j * (theta / 4.0))

    c0 = 1.0
    c1 = (-exp_theta + 16.0 * exp_theta_4 - 15.0) / 12.0
    c2 = (exp_theta - 4.0 * exp_theta_4 + 3.0) / 12.0

    # 5. Combine and return
    I_dd = jnp.eye(d * d, dtype=jnp.complex128)
    return c0 * I_dd + c1 * S2 + c2 * S4

# Native parameterized CSUM gate
def CSUM_gate(theta, level_j, level_k, d=3):
    """
    Generalized CSUM gate targeting a specific 2-level subspace.
    Sandwiches a targeted ZZ interaction with a targeted 2D QFT (Hadamard).
    """
    # 1. Embed a 2D QFT (Hadamard) into the targeted subspace
    F_jk = jnp.eye(d, dtype=jnp.complex128)

    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    F_jk = F_jk.at[level_j, level_j].set(inv_sqrt2)
    F_jk = F_jk.at[level_j, level_k].set(inv_sqrt2)
    F_jk = F_jk.at[level_k, level_j].set(inv_sqrt2)
    F_jk = F_jk.at[level_k, level_k].set(-inv_sqrt2)

    # 2. Tensor product (I_d \otimes F_jk) to target the second qudit
    I_d = jnp.eye(d, dtype=jnp.complex128)
    I_Fjk = jnp.kron(I_d, F_jk)
    I_Fjk_dag = jnp.conj(I_Fjk.T)

    # 3. Natively construct the targeted ZZ interaction
    # Create the local Z-type operator for the targeted subspace
    Z_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    Z_jk = Z_jk.at[level_j, level_j].set(1.0)
    Z_jk = Z_jk.at[level_k, level_k].set(-1.0)

    # The two-qudit ZZ interaction operator
    ZZ_jk = jnp.kron(Z_jk, Z_jk)

    # Calculate the geometric phase accumulation as a diagonal matrix
    exponent = -1j * (theta / 2) * ZZ_jk
    ZZ_interaction = jnp.diag(jnp.exp(jnp.diag(exponent)))

    # 4. Sandwich the ZZ interaction
    return I_Fjk_dag @ ZZ_interaction @ I_Fjk

# Native Entangling MS Gate (Geometric Phase)
def entangling_MS_gate(theta, level_j, level_k, d=3):
    """
    Creates a ZZ-type geometric phase entanglement between two qudits
    on the specified subspace (level_j and level_k).
    """
    # Create the local Z-type operator for the targeted subspace
    Z_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    Z_jk = Z_jk.at[level_j, level_j].set(1.0)
    Z_jk = Z_jk.at[level_k, level_k].set(-1.0)

    # The two-qudit ZZ interaction operator
    ZZ_jk = jnp.kron(Z_jk, Z_jk)

    # Calculate the geometric phase accumulation
    exponent = -1j * (theta / 2) * ZZ_jk

    # Because this matrix is purely diagonal, we can compute the matrix
    # exponential instantly by just exponentiating the diagonal elements
    return jnp.diag(jnp.exp(jnp.diag(exponent)))

# Native Entangling LS Gate
def entangling_LS_gate(theta, d=3):
    """
    JAX-native implementation of the Hrmo et al. global light-shift gate.
    Applies e^{i theta} to all |j,k> (j != k) and Identity to |j,j>.
    """
    # 1. Fill the entire 9-element array with the phase e^{i theta}
    phase = jnp.exp(1j * theta)
    diag_elements = jnp.full(d * d, phase, dtype=jnp.complex128)

    # 2. Overwrite the |j,j> states (00, 11, 22) back to exactly 1.0
    # JAX strictly requires the .at[].set() syntax for array updates!
    for j in range(d):
        idx = j * d + j  # For d=3, this hits indices 0, 4, 8
        diag_elements = diag_elements.at[idx].set(1.0 + 0.0j)

    # 3. Convert the 1D array into the 9x9 diagonal matrix
    return jnp.diag(diag_elements)

# ==========================================
# 4. QML ARCHITECTURE (ANSATZ A - MS)
# ==========================================
# Initialize the native qutrit simulator
dev = qml.device("default.qutrit", wires=N_QUTRIT)

# Calculates the number of parameters needed per layer based on physical number of qutrits and entanglement layout
def get_params_per_layer(n_qutrit):
    """
    Dynamically calculates the exact number of parameters needed per layer
    based on the physical qutrits and the entanglement graph.
    """
    # 1. XY Single Qutrit Gates: 4 gates per qutrit (2 params each)
    n_xy_params = 8 * n_qutrit

    # 2. CSUM Entangling Gates: 2 gates per connection (1 param each)
    n_csum_params = 2 * len(star_connections)

    # 3. Z-Corrections: 2 gates per qutrit (1 param each)
    n_z_params = 2 * n_qutrit

    # 4. Light-Shift (LS) Gates: 2 gates per qutrit (1 param each)
    n_ls_params = len(ring_connections)

    # 5. MS Entangling Gates: 1 gate per connection (2 params each)
    n_ms_params = 2 * len(ring_connections)

    params = n_xy_params + n_z_params + n_ls_params #+ n_csum_params #+ n_ms_params

    print(f"--- Layer Parameter Breakdown ---")
    print(f"Number of parameters Layer:")
    print(f"XY Params: {n_xy_params} | CSUM Params: {n_csum_params} | MS Params: {n_ms_params} | LS Params: {n_ls_params}| Z Params: {n_z_params}")
    print(f"Total Params Per Layer: {params}\n")

    return params

PARAMS_PER_LAYER = get_params_per_layer(N_QUTRIT)

# Encoder QNODE
# NOTE: diff_method="backprop" is required here to tell PennyLane to trace
# the gradients through the internal statevector array math using JAX.
@qml.qnode(dev, interface="jax", diff_method="backprop")
def encoder(params, code_ind):
    # params = 2D array of shape (n_layer, 36)
    # code_ind = Logical State to encode (0, 1, 2)
    n_layer = params.shape[0]

    # State Preparation
    if code_ind == 1:
        qml.QutritUnitary(XY_gate(0.0, jnp.pi, 0, 1), wires=0)
    elif code_ind == 2:
        qml.QutritUnitary(XY_gate(0.0, jnp.pi, 0, 1), wires=0)
        qml.QutritUnitary(XY_gate(0.0, jnp.pi, 1, 2), wires=0)

    # Ansatz Layers
    for l in range(n_layer):
        layer_p = params[l]
        param_idx = 0

        # 1. Pre-Mixing: Single Qutrit XY Layer (40 params)
        for q in range(N_QUTRIT):
            p = layer_p[param_idx: param_idx + 8]
            param_idx += 8
            for U in [
                XY_gate(p[0], p[1], 0, 1), XY_gate(p[2], p[3], 1, 2),
                XY_gate(p[4], p[5], 0, 1), XY_gate(p[6], p[7], 1, 2)
            ]:
                qml.QutritUnitary(U, wires=q)

        # 2. Spreading Phase: CSUM Star Graph (8 params)
        # Mimics the CNOT cascade from the data qutrit to ancillas
        """if l == 0:
            for q1, q2 in star_connections:
                qml.QutritUnitary(CSUM_gate(layer_p[param_idx], 0, 1), wires=[q1, q2])
                qml.QutritUnitary(CSUM_gate(layer_p[param_idx + 1], 1, 2), wires=[q1, q2])
                param_idx += 2
        else:
            param_idx += 2 * len(star_connections)"""

        # 3. Correlation Phase A: LS Ring Graph (20 params)
        # Builds the cyclic stabilizer correlations
        for q1, q2 in ring_connections:
            qml.QutritUnitary(entangling_LS_gate(layer_p[param_idx], d=3), wires=[q1, q2])
            param_idx += 1

        # 4. Correlation Phase B: MS Ring Graph (10 params)
        # Finishes geometric phase accumulation
        """for q1, q2 in ring_connections:
            qml.QutritUnitary(entangling_MS_gate(layer_p[param_idx], 0, 1), wires=[q1, q2])
            qml.QutritUnitary(entangling_MS_gate(layer_p[param_idx + 1], 1, 2), wires=[q1, q2])
            param_idx += 2"""

        # 5. Fine-Tuning: Z-Gate Correction Layer (10 params)
        for q in range(N_QUTRIT):
            z_01, z_12 = layer_p[param_idx], layer_p[param_idx + 1]
            param_idx += 2
            qml.QutritUnitary(Z_gate(z_01, 0, 1), wires=q)
            qml.QutritUnitary(Z_gate(z_12, 1, 2), wires=q)

    return qml.state()


# ==========================================
# 5. COST FUNCTION (KNILL-LAFLAMME) - BATCHED TENSORFORM
# ==========================================
def build_loss_func(E_det, M_prods, sample_fraction):
    """
    A factory function that groups errors by wire signature and compiles
    highly optimized batched tensor contractions with stochastic mini-batching.
    """
    # 1. Group errors by their target wires
    E_det_grouped = group_by_wires(E_det)
    M_prods_grouped = group_by_wires(M_prods)

    # 2. Setup VMAPs for batch processing
    vmap_apply_E = jax.vmap(apply_local_matrix, in_axes=(None, 0, None, None))
    vmap_apply_M = jax.vmap(jax.vmap(apply_local_matrix, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))

    @jax.jit
    def loss_func(params, key):
        # Stack states into a single tensor: shape (K, 243)
        code_states = jnp.stack([encoder(params, k) for k in range(K)])
        loss = jnp.array(0.0, dtype=jnp.float64)

        # Split the PRNG key for the number of groups we need to sample
        num_groups = len(E_det_grouped) + len(M_prods_grouped)
        keys = jax.random.split(key, num_groups)
        key_idx = 0

        # --- PRE-SAMPLE DETECTABILITY ERRORS ---
        # We sample outside the i,j loops to ensure we evaluate the
        # exact same subset of errors against all logical state pairs.
        sampled_E_groups = {}
        for E_wires, (E_mats, inv_order) in E_det_grouped.items():
            N_total = E_mats.shape[0]
            N_sample = max(1, int(N_total * sample_fraction))

            if sample_fraction < 1.0:
                indices = jax.random.choice(keys[key_idx], N_total, shape=(N_sample,), replace=False)
                sampled_E_groups[E_wires] = (E_mats[indices], inv_order, N_total / N_sample)
                key_idx += 1
            else:
                sampled_E_groups[E_wires] = (E_mats, inv_order, 1.0)

        # 1. Detectability Loop (Batched & Sampled)
        for i in range(K):
            for j in range(i + 1, K):
                for E_wires, (sampled_mats, inv_order, scale) in sampled_E_groups.items():
                    E_cj_batch = vmap_apply_E(code_states[j], sampled_mats, E_wires, inv_order)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj_batch, axis=1)

                    # Scale the loss up to compensate for dropped errors
                    loss += scale * jnp.sum(jnp.abs(inner) ** 2)

        # 2. Correctability Loop (Batched & Sampled)
        if DISTANCE >= 3:
            for M_wires, (M_mats, inv_order) in M_prods_grouped.items():
                N_total = M_mats.shape[0]
                N_sample = max(1, int(N_total * sample_fraction))

                if sample_fraction < 1.0:
                    indices = jax.random.choice(keys[key_idx], N_total, shape=(N_sample,), replace=False)
                    sampled_mats = M_mats[indices]
                    scale = N_total / N_sample
                    key_idx += 1
                else:
                    sampled_mats = M_mats
                    scale = 1.0

                M_v_batch = vmap_apply_M(code_states, sampled_mats, M_wires, inv_order)
                vals = jnp.sum(jnp.conj(code_states) * M_v_batch, axis=2)

                # Scale the variance penalty
                loss += (K / 3) * scale * jnp.sum(jnp.var(vals, axis=1))

        return loss

    return loss_func


# ==========================================
# 6. STOCHASTIC SEED RACING LOOP (BATCHING + SPEED)
# ==========================================
n_layer = 2  # <-- SET YOUR DESIRED NUMBER OF LAYERS HERE
STEPS = 15000  # Maximum steps per seed
NUM_SEEDS = 100000  # How many different random initializations to race
n_params = n_layer * PARAMS_PER_LAYER
SAMPLE_FRAC = 0.02  # Evaluating 20% of errors per step

print("--- TRAINING SETUP ---")
print(f"Physical Qutrits: {N_QUTRIT} | Logical Qutrits: {LOGICAL_QUTRIT}")
print(f"Distance: {DISTANCE} | Layers: {n_layer} | Total Params: {n_params}")
print(f"Batching: Evaluating {SAMPLE_FRAC * 100}% of errors per step")
print(f"Racing {NUM_SEEDS} seeds to find the fastest convergence.")
print("----------------------\n")

# Generate and compile the STOCHASTIC loss function
custom_loss_func = build_loss_func(E_det_onp, M_products_onp, sample_fraction=SAMPLE_FRAC)

log_time("Triggering XLA Compilation (Stochastic)...")
start_compile = time.time()
val_grad_fn = jax.jit(jax.value_and_grad(custom_loss_func))

# Dummy compile call to set JAX static shapes
# We must pass both a dummy theta AND a dummy PRNG key for the batching
dummy_key = jax.random.PRNGKey(0)
dummy_theta = jax.random.uniform(dummy_key, (n_layer, PARAMS_PER_LAYER), minval=0.0, maxval=2 * jnp.pi)
first_loss, _ = val_grad_fn(dummy_theta, dummy_key)
first_loss.block_until_ready()
compile_time = time.time() - start_compile
log_time(f"XLA Compilation Complete! Took: {compile_time:.2f} seconds\n")

# --- SPEED TRACKING VARIABLES ---
global_fastest_steps = 400  # Starting with your baseline record
global_best_theta = None
winning_seed = -1

# --- THE SEED RACE LOOP ---
for seed in range(NUM_SEEDS):
    #seed = seed +
    log_time(f"--- STARTING RACE FOR SEED {seed} ---")

    # Generate a base key for this specific seed run
    base_key = jax.random.PRNGKey(seed)

    # Split the base key: one for initializing theta, one for the batching stream
    init_key, batch_key = jax.random.split(base_key)

    theta = jax.random.uniform(init_key, (n_layer, PARAMS_PER_LAYER), minval=0.0, maxval=2 * jnp.pi)

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(theta)
    lr_switched_1 = False
    lr_switched_2 = False

    for step in range(STEPS):
        # MERCY RULE: If this seed is already slower than our current record, kill it.
        if step >= global_fastest_steps:
            print(f"Seed {seed} eliminated (Passed current record of {global_fastest_steps} steps).")
            break

        # Split the batch_key to get a fresh, random subset of errors for THIS step
        batch_key, step_key = jax.random.split(batch_key)

        # Pass the step_key into the compiled gradient function
        loss, grads = val_grad_fn(theta, step_key)

        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        # Dynamic Learning Rate Switch (using your updated thresholds)
        if (not lr_switched_1) and loss < 0.5:
            optimizer = optax.adam(learning_rate=0.01)
            opt_state = optimizer.init(theta)
            lr_switched_1 = True

        if (not lr_switched_2) and loss < 1e-3:
            optimizer = optax.adam(learning_rate=0.001)
            opt_state = optimizer.init(theta)
            lr_switched_2 = True

        if step % 500 == 0:
            print(f"Seed {seed} | Step {step:04d} | loss={loss:.6e}")

        # SUCCESS CONDITION (HIT TARGET LOSS)
        if loss < 1e-6:
            print(f"\n>>> TARGET REACHED! Seed {seed} hit 1e-6 at step {step} <<<")

            # Check if this is a new speed record
            if step < global_fastest_steps:
                print(f"*** NEW RECORD! Beating previous best of {global_fastest_steps} steps. ***")
                global_fastest_steps = step
                global_best_theta = jnp.copy(theta)
                winning_seed = seed
            break  # Stop training this seed and move to the next competitor

    print(f"--- Seed {seed} race concluded. ---\n")

# ==========================================
# Final Block
# ==========================================
if winning_seed != -1:
    log_time(f"PROGRAM END: Stochastic Seed Race Complete.")
    log_time(f"WINNING SEED: {winning_seed} with an incredible {global_fastest_steps} steps!")

    theta = global_best_theta
    print(f"\nFastest Theta Array (from Seed {winning_seed}):")
    print(onp.array2string(onp.array(theta), separator=', '))
else:
    print("PROGRAM END: No seeds reached the target loss within the step limit.")
    print(f"The baseline record of {global_fastest_steps} steps remains unbroken.")

total_script_time = time.time() - script_start_time
log_time(f"Total Script Runtime: {total_script_time / 60:.2f} minutes")


# 1. Generate the logical basis states using your best parameters
"""print("\nGenerating logical basis states...")
logical_states = []
for k in range(K):
    state_k = encoder(best_theta, k)
    logical_states.append(onp.array(state_k)) # Cast JAX array to standard NumPy

code_states_array = onp.stack(logical_states)

# 2. Save the states and parameters to an .npz file
save_path = "my_513_LS_random_code.npz"
onp.savez(save_path,
          code_states=code_states_array,
          theta=onp.array(best_theta),
          losses= onp.array(loss_history))

print(f"Successfully saved code states to {save_path}")
print(f"Shape of code states: {code_states_array.shape}")"""
