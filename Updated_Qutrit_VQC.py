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
DISTANCE = 3            # Distance 2 = Error Detection

K = DIM_QUTRIT ** LOGICAL_QUTRIT

# Full connectivity between n qutrits
connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [0, 3], [1, 3], [1, 4], [2, 4]]
ms_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]   # ring
ls_connections = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4]]   # star
csum_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]   # ring

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

# Native Entangling Light-Shift Gate (Geometric Phase)
def entangling_LS_gate(theta, level_j, level_k, d=3):
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

# ==========================================
# 4. QML ARCHITECTURE (ANSATZ A - MS)
# ==========================================
# Initialize the native qutrit simulator
dev = qml.device("default.qutrit", wires=N_QUTRIT)

# Calculates the number of parameters needed per layer based on physical number of qutrits and entanglement layout
def get_params_per_layer(n_qutrit, connections):
    """
    Dynamically calculates the exact number of parameters needed per layer
    based on the physical qutrits and the entanglement graph.
    """
    # 1. XY Single Qutrit Gates: 4 gates per qutrit (2 params each)
    n_xy_params = 8 * n_qutrit

    # 2. CSUM Entangling Gates: 2 gates per connection (1 param each)
    n_csum_params = 2 * len(csum_connections)

    # 3. Z-Corrections: 2 gates per qutrit (1 param each)
    n_z_params = 2 * n_qutrit

    # 4. Light-Shift (LS) Gates: 2 gates per qutrit (1 param each)
    n_ls_params = 2 * len(ls_connections)

    # 5. MS Entangling Gates: 2 gates per connection (2 params each)
    #n_ms_params = 4 * len(ms_connections)

    #ms_total = n_xy_params + n_ms_params + n_z_params
    ls_total = n_xy_params + n_ls_params + n_z_params
    csum_total = n_xy_params + n_csum_params + n_z_params

    max_params = max(csum_total, ls_total)

    print(f"--- Layer Parameter Breakdown ---")
    #print(f"Number of parameters per MS-Gate-Layer:")
    #print(f"XY Params: {n_xy_params} | MS Params: {n_ms_params} | Z Params: {n_z_params}")
    #print(f"Total Params Per MS-Gate-Layer: {ms_total}\n")
    print(f"Number of parameters per LS-Gate-Layer:")
    print(f"XY Params: {n_xy_params} | LS Params: {n_ls_params} | Z Params: {n_z_params}")
    print(f"Total Params Per LS-Gate-Layer: {ls_total}\n")
    print(f"Number of parameters per CSUM-Gate-Layer:")
    print(f"XY Params: {n_xy_params} | CSUM Params: {n_csum_params} | Z Params: {n_z_params}")
    print(f"Total Params Per LS-Gate-Layer: {csum_total}\n")

    return max_params

PARAMS_PER_LAYER = get_params_per_layer(N_QUTRIT, connections)

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

        # 1. Single Qutrit Layer
        for q in range(N_QUTRIT):
            p = layer_p[param_idx: param_idx + 8]
            param_idx += 8
            for U in [
                XY_gate(p[0], p[1], 0, 1), XY_gate(p[2], p[3], 1, 2),
                XY_gate(p[4], p[5], 0, 1), XY_gate(p[6], p[7], 1, 2)
            ]:
                qml.QutritUnitary(U, wires=q)

        # 2. Entangling Layer (CSUM/LS-Gates)
        if l % 2 == 0:
            # EVEN LAYERS: CSUM Mixing (Needs 20 params)
            for q1, q2 in csum_connections:
                theta_01 = layer_p[param_idx]
                param_idx += 1
                qml.QutritUnitary(CSUM_gate(theta_01, 0, 1), wires=[q1, q2])

                theta_12 = layer_p[param_idx]
                param_idx += 1
                qml.QutritUnitary(CSUM_gate(theta_12, 1, 2), wires=[q1, q2])
        else:
            # ODD LAYERS: LS Phase Shifting (Needs 10 params)
            for q1, q2 in ls_connections:
                # LS gates only take a theta, so we only pull 1 param per gate
                theta_ls_01 = layer_p[param_idx]
                param_idx += 1
                qml.QutritUnitary(entangling_LS_gate(theta_ls_01, 0, 1), wires=[q1, q2])

                theta_ls_12 = layer_p[param_idx]
                param_idx += 1
                qml.QutritUnitary(entangling_LS_gate(theta_ls_12, 1, 2), wires=[q1, q2])

        # 3. Z-Gate Layer
        for q in range(N_QUTRIT):
            z_01, z_12 = layer_p[param_idx], layer_p[param_idx + 1]
            param_idx += 2
            qml.QutritUnitary(Z_gate(z_01, 0, 1), wires=q)
            qml.QutritUnitary(Z_gate(z_12, 1, 2), wires=q)

    return qml.state()


# ==========================================
# 5. COST FUNCTION (KNILL-LAFLAMME) - BATCHED TENSORFORM
# ==========================================
def build_loss_func(E_det, M_prods):
    """
    A factory function that groups errors by wire signature and compiles
    highly optimized batched tensor contractions.
    """
    # 1. Group errors by their target wires
    E_det_grouped = group_by_wires(E_det)
    M_prods_grouped = group_by_wires(M_prods)

    # 2. Setup VMAPs for batch processing
    # vmap_E applies N matrices to 1 state -> Output shape: (N_matrices, 243)
    vmap_apply_E = jax.vmap(apply_local_matrix, in_axes=(None, 0, None, None))

    # vmap_M applies N matrices to K states -> Output shape: (N_matrices, K_states, 243)
    # We double-vmap: once over the states, and once over the batch of matrices
    vmap_apply_M = jax.vmap(jax.vmap(apply_local_matrix, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))

    @jax.jit
    def loss_func(params):
        # Stack states into a single tensor: shape (K, 243)
        code_states = jnp.stack([encoder(params, k) for k in range(K)])
        loss = jnp.array(0.0, dtype=jnp.float64)

        # 1. Detectability Loop (Batched)
        for i in range(K):
            for j in range(i + 1, K):
                # Python loop over the GROUPS (only ~15 iterations!)
                for E_wires, (E_mats, inv_order) in E_det_grouped.items():
                    # Apply all matrices in this group to state j at once
                    # E_mats shape is (N, 3, 3) or (N, 9, 9)
                    E_cj_batch = vmap_apply_E(code_states[j], E_mats, E_wires, inv_order)

                    # Compute inner products for the whole batch
                    # Multiply element-wise and sum over the state dimension (axis=1)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj_batch, axis=1)
                    loss += jnp.sum(jnp.abs(inner) ** 2)

        # 2. Correctability Loop (Batched)
        if DISTANCE >= 3:
            for M_wires, (M_mats, inv_order) in M_prods_grouped.items():
                # Apply all matrices in this group to all K states at once
                M_v_batch = vmap_apply_M(code_states, M_mats, M_wires, inv_order)

                # Calculate <v | M | v> for all states and all matrices
                # Sum over the state dimension (axis=2)
                vals = jnp.sum(jnp.conj(code_states) * M_v_batch, axis=2)

                # Calculate variance across the K states (axis=1), then sum up the penalties
                loss += (K / 3) * jnp.sum(jnp.var(vals, axis=1))

        return loss

    return loss_func


# ==========================================
# 6. TRAINING LOOP (Optimized for Heterogeneous Ansatz)
# ==========================================
n_layer = 4
STEPS = 10000  # How many steps to train
best_loss = 1e7
best_theta = None
n_params = n_layer * PARAMS_PER_LAYER

# Initialize random gate angles
key = jax.random.PRNGKey(0)
theta = jax.random.uniform(key, (n_layer, PARAMS_PER_LAYER), minval=0.0, maxval=2 * jnp.pi)

print("--- TRAINING SETUP ---")
print(f"Physical Qutrits: {N_QUTRIT} | Logical Qutrits: {LOGICAL_QUTRIT}")
print(f"Distance: {DISTANCE} | Layers: {n_layer} | Total Params: {n_params}")
print(f"Ansatz: Alternating MS (Ring) and LS (Star)")
print("----------------------\n")

# 1. Learning Rate Scheduler
# Starts at 0.05, halves every 1500 steps.
# This prevents the optimizer from "bouncing" at the 3.218 floor.
lr_schedule = optax.exponential_decay(
    init_value=0.05,
    transition_steps=1500,
    decay_rate=0.5
)

# 2. Initialize Optax Adam optimizer with the schedule
optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(theta)

log_time("Starting Error Grouping and Graph Tracing...")
start_trace = time.time()

# Generate the specialized, compiled loss function
custom_loss_func = build_loss_func(E_det_onp, M_products_onp)

trace_time = time.time() - start_trace
log_time(f"Graph Tracing Complete! Took: {trace_time:.2f} seconds")

log_time("Triggering XLA Compilation (This may take a while)...")
start_compile = time.time()

# Wrap the value_and_grad call with JIT compilation
val_grad_fn = jax.jit(jax.value_and_grad(custom_loss_func))

# Trigger the compilation and WAIT for it to finish (using dummy call)
# We use .block_until_ready() to get an accurate timing of the compile
first_loss, _ = val_grad_fn(theta)
first_loss.block_until_ready()

compile_time = time.time() - start_compile
log_time(f"XLA Compilation Complete! Took: {compile_time:.2f} seconds")

# --- TRAINING LOOP STARTS HERE ---
log_time(f"Starting {STEPS} Step Training Loop...")
start_training_time = time.time()

for step in range(STEPS):
    # Perform a single optimization step
    loss, grads = val_grad_fn(theta)

    # Update parameters using Optax
    updates, opt_state = optimizer.update(grads, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    # Track the best global solution found
    if loss < best_loss:
        best_loss = loss
        best_theta = jnp.copy(theta)

    # Progress Logging
    if step % 10 == 0:
        # We don't use block_until_ready here to keep speed high,
        # but the print provides a steady heartbeat of the optimization.
        print(f"step {step:04d} | current_loss={loss:.6e} | best_loss={best_loss:.6e}")

    # Convergence Check
    if loss < 1e-6:
        print(f"\nSUCCESS! Valid Code found at step {step} with loss {loss:.6e}")
        break

# Final Block to ensure hardware is finished before logging total time
best_loss.block_until_ready()

total_script_time = time.time() - script_start_time
log_time(f"PROGRAM END: Training finished with best loss {best_loss:.6e}")
log_time(f"Total Script Runtime: {total_script_time / 60:.2f} minutes")

# Reassign best theta for potential saving/analysis
theta = best_theta
print(f"\nBest Theta Array:\n{best_theta}")
