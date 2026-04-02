import warnings, itertools

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
# ERROR SET GENERATION (Standard NumPy is fine here as it's static)
import numpy as onp  # Original numpy for static setup

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

# ==========================================
# 2. HARDWARE ERROR MODEL
# ==========================================
# Returns the 8 Gell-Mann matrices for qutrit error channels
def hardware_error_basis():
    errors = []

    # Amplitude Errors (Bit Flips / Population Transfer)
    # 0 <-> 1 subspace (symmetric and antisymmetric)
    #L1 = onp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    #L2 = onp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)

    # 0 <-> 2 subspace (symmetric and antisymmetric)
    #L4 = onp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    #L5 = onp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)

    # 1 <-> 2 subspace (symmetric and antisymmetric)
    #L6 = onp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    #L7 = onp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)

    #errors.extend([L1, L2, L4, L5, L6, L7])

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

    # 1. Build the "No Error" full system identity matrix
    id_full = I3
    for _ in range(1, n_qutrit):
        id_full = onp.kron(id_full, I3)

    E_det, E_corr = [id_full], [id_full]
    max_det = distance - 1
    max_corr = (distance - 1) // 2

    # 2. Fetch our hardware-specific errors
    single_errs = hardware_error_basis()

    # 3. Place every physical error on every possible qutrit wire one by one
    for w in range(1, max_det + 1):
        for qutrit_subset in itertools.combinations(range(n_qutrit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                op = 1
                idx_choice = 0
                # Smashing the 3x3 matrices together into the full system size
                for q in range(n_qutrit):
                    if q in qutrit_subset:
                        op = onp.kron(op, err_choice[idx_choice])
                        idx_choice += 1
                    else:
                        op = onp.kron(op, I3)
                E_det.append(op)
                if w <= max_corr:
                    E_corr.append(op)

    return E_det, E_corr

# Generate the error tracking lists before training begins
E_det_onp, E_corr_onp = build_error_set(N_QUTRIT, DISTANCE)

# --- NEW PRECOMPUTATION CODE ---
def precompute_error_products_dedup(E_corr):
    """Precompute UNIQUE E_a†E_b products with deduplication."""
    M_products = []
    seen = {}
    for Ea in E_corr:
        Ea_dag = onp.conj(Ea.T)
        for Eb in E_corr:
            M = Ea_dag @ Eb
            # Rounding to handle floating point noise
            key = tuple(onp.round(M.ravel(), decimals=10))
            if key not in seen:
                seen[key] = len(M_products)
                M_products.append(M)

    n_total = len(E_corr) ** 2
    print(f"Deduplicated: {n_total} products -> {len(M_products)} unique")
    return M_products

M_products_onp = precompute_error_products_dedup(E_corr_onp)

# Cast the optimized error sets to JAX arrays
E_det_jax = jnp.array(E_det_onp, dtype=jnp.complex128)
# We now cast your precomputed M matrices instead of E_corr
M_products_jax = jnp.array(M_products_onp, dtype=jnp.complex128)

print(f"E_det size = {len(E_det_jax)}, Unique M matrices = {len(M_products_jax)}")

# ==========================================
# 3. NATIVE HARDWARE GATES
# ==========================================
# Native single-qutrit operation driving transitions in the XY plane
def XY_gate(phi, alpha, level_j, level_k, d=3):
    X_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    Y_jk = jnp.zeros((d, d), dtype=jnp.complex128)

    # JAX immutable assignment
    X_jk = X_jk.at[level_j, level_k].set(1.0)
    X_jk = X_jk.at[level_k, level_j].set(1.0)

    Y_jk = Y_jk.at[level_j, level_k].set(-1j)
    Y_jk = Y_jk.at[level_k, level_j].set(1j)

    exponent = -1j * (alpha / 2) * (jnp.cos(phi) * X_jk + jnp.sin(phi) * Y_jk)
    return jax.scipy.linalg.expm(exponent)

# Native single-qutrit operation for phase shifts between two populations
def Z_gate(theta, level_j, level_k, d=3):
    Z_jk = jnp.eye(d, dtype=jnp.complex128)
    Z_jk = Z_jk.at[level_j, level_j].multiply(jnp.exp(1j * theta / 2))
    Z_jk = Z_jk.at[level_k, level_k].multiply(jnp.exp(-1j * theta / 2))
    return Z_jk

# Native Molmer-Sorensen entangling gate
def MS_gate(theta, phi, level_j, level_k, d=3):
    sigma_phi = jnp.zeros((d, d), dtype=jnp.complex128)
    sigma_phi = sigma_phi.at[level_j, level_k].set(jnp.exp(-1j * phi))
    sigma_phi = sigma_phi.at[level_k, level_j].set(jnp.exp(1j * phi))

    I_d = jnp.eye(d, dtype=jnp.complex128)
    term_sum = jnp.kron(sigma_phi, I_d) + jnp.kron(I_d, sigma_phi)

    # jnp.linalg.matrix_power is safe for integer powers in JAX
    exponent = -1j * (theta / 4) * jnp.linalg.matrix_power(term_sum, 2)
    return jax.scipy.linalg.expm(exponent)

# Native parameterized ZZ interaction between levels 1 and 2 of two qutrits
def ZZ_12_gate(theta, d=3):
    # Generates the diagonal ZZ_12 matrix
    diag_vals = jnp.array([
        1.0, 1.0, 1.0,
        1.0, jnp.exp(-1j * theta / 2), jnp.exp(1j * theta / 2),
        1.0, jnp.exp(1j * theta / 2), jnp.exp(-1j * theta / 2)
    ], dtype=jnp.complex128)
    return jnp.diag(diag_vals)

# Native parameterized CSUM gate
def CSUM_gate(theta, d=3):
    # 1. Define the 3-dimensional QFT matrix (F_3)
    omega = jnp.exp(2j * jnp.pi / 3)
    F_3 = (1 / jnp.sqrt(3)) * jnp.array([
        [1, 1, 1],
        [1, omega, omega ** 2],
        [1, omega ** 2, omega]
    ], dtype=jnp.complex128)

    # 2. Tensor product (1 \otimes F_3)
    I_3 = jnp.eye(3, dtype=jnp.complex128)
    I_F3 = jnp.kron(I_3, F_3)
    I_F3_dag = jnp.conj(I_F3.T)

    # 3. Sandwich the ZZ_12 gate: (1 \otimes F_3^\dagger) @ ZZ_12 @ (1 \otimes F_3)
    ZZ = ZZ_12_gate(theta, d)
    return I_F3_dag @ ZZ @ I_F3

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

# Full connectivity between n qutrits
connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
# Calculates the number of parameters needed per layer based on physical number of qutrits and entanglement layout
def get_params_per_layer(n_qutrit, connections):
    """
    Dynamically calculates the exact number of parameters needed per layer
    based on the physical qutrits and the entanglement graph.
    """
    # 1. XY Single Qutrit Gates: 4 gates per qutrit (2 params each)
    n_xy_params = 8 * n_qutrit

    # 2. CSUM Entangling Gates: 1 gate per connection (1 param each)
    #n_csum_params = 1 * len(connections)

    # 3. Z-Corrections: 2 gates per qutrit (1 param each)
    n_z_params = 2 * n_qutrit

    # 4. Light-Shift (LS) Gates: 2 gates per qutrit (1 param each)
    #n_ls_params = 2 * len(connections)

    # 5. MS Entangling Gates: 1 gate per connection (2 params each)
    n_ms_params = 2 * len(connections)

    total = n_xy_params + n_ms_params + n_z_params

    print(f"--- Layer Parameter Breakdown ---")
    print(f"XY Params: {n_xy_params} | MS Params: {n_ms_params} | Z Params: {n_z_params}")
    print(f"Total Params Per Layer: {total}\n")

    return total

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

        # 2. Entangling Layer (MS-Gates)
        for q1, q2 in connections:
            phi, theta = layer_p[param_idx], layer_p[param_idx + 1]
            param_idx += 2
            qml.QutritUnitary(MS_gate(theta, phi, 0, 1), wires=[q1, q2])

        # 3. Z-Gate Layer
        for q in range(N_QUTRIT):
            z_01, z_12 = layer_p[param_idx], layer_p[param_idx + 1]
            param_idx += 2
            qml.QutritUnitary(Z_gate(z_01, 0, 1), wires=q)
            qml.QutritUnitary(Z_gate(z_12, 1, 2), wires=q)

    return qml.state()

# ==========================================
# 5. COST FUNCTION (KNILL-LAFLAMME)
# ==========================================
def loss_func(params):
    # Stack states into a single JAX tensor for fast operations
    code_states = jnp.stack([encoder(params, k) for k in range(K)])
    loss = jnp.array(0.0, dtype=jnp.float64)

    # 1. Detectability Loop
    # Checks: <c_i | E | c_j> = 0
    for i in range(K):
        for j in range(i + 1, K):
            for E in E_det_jax:  # Iterate over the list
                inner = jnp.dot(jnp.conj(code_states[i]), E @ code_states[j])
                loss += jnp.abs(inner) ** 2

    # 2. Correctability Loop (Activates for Distance >= 3)
    # Ensures errors deform the logical state space uniformly.
    if DISTANCE >= 3:
        # Replaced the nested Ea/Eb loops with your single deduplicated list!
        for M in M_products_jax:
            vals = jnp.array([jnp.dot(jnp.conj(v), M @ v) for v in code_states])
            loss += (K / 3) * jnp.var(vals)

    return loss

# ==========================================
# 6. TRAINING LOOP
# ==========================================
n_layer = 5           # <-- SET YOUR DESIRED NUMBER OF LAYERS HERE
STEPS = 10000         # How many steps to train
best_loss = 1e7
best_theta = None
n_params = n_layer * PARAMS_PER_LAYER

# Initialize random gate angles
# JAX uses explicit PRNG keys instead of global seeds
key = jax.random.PRNGKey(0)
theta = jax.random.uniform(key, (n_layer, PARAMS_PER_LAYER), minval=0.0, maxval=2 * jnp.pi)

print("--- TRAINING SETUP ---")
print(f"Physical Qutrits: {N_QUTRIT} | Logical Qutrits: {LOGICAL_QUTRIT}")
print(f"Distance: {DISTANCE} | Layers: {n_layer} | Total Params: {n_params}")
print("----------------------\n")

# Initialize Optax optimizer
optimizer = optax.adam(learning_rate=0.05)
opt_state = optimizer.init(theta)
lr_switched = False

# jax.value_and_grad returns both the loss value and the gradients
val_grad_fn = jax.value_and_grad(loss_func)

for step in range(STEPS):
    loss, grads = val_grad_fn(theta)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    if loss < best_loss:
        best_loss = loss
        best_theta = jnp.copy(theta)

    # Dynamic Learning Rate Switch
    if (not lr_switched) and loss < 0.1:
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(theta)
        lr_switched = True
        print(f"\n[step {step}] loss={loss:.3e}  ->  switch lr -> 0.01\n")

    if step % 10 == 0:
        print(f"step {step:04d} | current_loss={loss:.6e} | best_loss={best_loss:.6e}")

    if loss < 1e-6:
        print(f"\nSUCCESS! Code found at step {step} with loss {loss:.6e}")
        break

theta = best_theta
