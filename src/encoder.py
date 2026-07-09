"""
Native trapped-ion encoder for qudits of any dimension d.

For d=3: uses PennyLane's default.qutrit device (QNode-based).
For d>3: manual state-vector simulation with autograd differentiation
         (no PennyLane qudit device available).

Both paths produce identical encoder signatures: encoder(params, code_ind) -> state.
"""
import pennylane as qml
from pennylane import numpy as np
from src.gates import XY_gate, Z_gate, MS_gate


def _apply_single_qudit_gate(state, U, q, n_qudit, d):
    """Apply d x d gate U on qudit q of an n_qudit state vector."""
    shape = tuple([d] * n_qudit)
    s = qml.math.reshape(state, shape)
    s = qml.math.tensordot(U, s, axes=[[1], [q]])
    perm = list(range(1, q + 1)) + [0] + list(range(q + 1, n_qudit))
    s = qml.math.transpose(s, perm)
    return qml.math.reshape(s, (-1,))


def _apply_two_qudit_gate(state, U, q1, q2, n_qudit, d):
    """Apply d^2 x d^2 gate U on qudits q1, q2 of an n_qudit state vector."""
    shape = tuple([d] * n_qudit)
    s = qml.math.reshape(state, shape)

    # Reshape U to (d, d, d, d): U[a_out, b_out, a_in, b_in]
    U_4d = qml.math.reshape(U, (d, d, d, d))

    # Contract input axes of U with state axes q1, q2
    s = qml.math.tensordot(U_4d, s, axes=[[2, 3], [q1, q2]])
    # Result axes: [a_out, b_out, remaining_0, remaining_1, ...]

    # Move a_out to position q1, b_out to position q2
    remaining = sorted(set(range(n_qudit)) - {q1, q2})
    perm = [0] * n_qudit
    perm[q1] = 0
    perm[q2] = 1
    for i, r in enumerate(remaining):
        perm[r] = i + 2

    s = qml.math.transpose(s, perm)
    return qml.math.reshape(s, (-1,))


def _ansatz_layer_manual(state, layer_p, n_qudit, d, connections):
    """Apply one ansatz layer using manual state-vector operations."""
    n_transitions = d - 1
    param_idx = 0

    # 1. Single-qudit rotations: 2 rounds of XY per transition
    for q in range(n_qudit):
        for t in range(n_transitions):
            U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U, q, n_qudit, d)
            param_idx += 2
        for t in range(n_transitions):
            U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U, q, n_qudit, d)
            param_idx += 2

    # 2. MS entangling gates on all transitions
    for q1, q2 in connections:
        for t in range(n_transitions):
            U_MS = MS_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_two_qudit_gate(state, U_MS, q1, q2, n_qudit, d)
            param_idx += 2

    # 3. Z corrections on all transitions
    for q in range(n_qudit):
        for t in range(n_transitions):
            U_z = Z_gate(layer_p[param_idx], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U_z, q, n_qudit, d)
            param_idx += 1

    return state


def _create_manual_encoder(n_qudit, d, connections):
    """Create encoder using manual state-vector simulation (for d > 3)."""
    dim = d ** n_qudit

    def encoder(params, code_ind):
        n_layer = params.shape[0]

        # State preparation: |code_ind, 0, ..., 0>
        state = np.zeros(dim, dtype=complex)
        state[int(code_ind) * d ** (n_qudit - 1)] = 1.0

        for l in range(n_layer):
            state = _ansatz_layer_manual(state, params[l], n_qudit, d, connections)

        return state

    return encoder


def _create_qutrit_qnode_encoder(n_qudit, connections):
    """Create encoder using PennyLane default.qutrit device (d=3 only)."""
    dev = qml.device("default.qutrit", wires=n_qudit)

    @qml.qnode(dev, interface="autograd")
    def encoder(params, code_ind):
        n_layer = params.shape[0]

        # State preparation: chain of XY(pi) gates to reach |code_ind>
        for t in range(int(code_ind)):
            U_prep = XY_gate(0.0, np.pi, t, t + 1, d=3)
            qml.QutritUnitary(U_prep, wires=0)

        for l in range(n_layer):
            layer_p = params[l]
            param_idx = 0

            # 1. Single-qutrit rotations
            for q in range(n_qudit):
                for t in range(2):
                    U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U, wires=q)
                    param_idx += 2
                for t in range(2):
                    U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U, wires=q)
                    param_idx += 2

            # 2. MS entangling gates on both (0,1) and (1,2)
            for q1, q2 in connections:
                for t in range(2):
                    U_MS = MS_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U_MS, wires=[q1, q2])
                    param_idx += 2

            # 3. Z corrections
            for q in range(n_qudit):
                for t in range(2):
                    U_z = Z_gate(layer_p[param_idx], t, t + 1, d=3)
                    qml.QutritUnitary(U_z, wires=q)
                    param_idx += 1

        return qml.state()

    return encoder


def create_native_encoder(n_qudit: int, d: int, connections=None, force_manual: bool = False):
    """
    Create a native trapped-ion encoder for qudits of dimension d.

    Ansatz per layer (for d-1 transitions per qudit):
      1. Single-qudit: 2 rounds x (d-1) XY gates x 2 params = 4(d-1) per qudit
      2. MS entangling: (d-1) MS gates x 2 params per connection = 2(d-1) per conn
      3. Z corrections: (d-1) Z gates x 1 param per qudit = (d-1) per qudit

    Total params per layer = (5*n_qudit + 2*n_conn) * (d-1)

    Args:
        n_qudit: number of physical qudits
        d: qudit dimension (2, 3, 4, 5, 7, ...)
        connections: list of [q1, q2] pairs; defaults to ring connectivity
        force_manual: if True, use manual state-vector encoder even for d=3
                      (faster gradients via autograd vs parameter-shift)

    Returns:
        (encoder_fn, connections, params_per_layer)
    """
    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]

    n_transitions = d - 1
    n_conn = len(connections)
    params_per_layer = (5 * n_qudit + 2 * n_conn) * n_transitions

    if d == 3 and not force_manual:
        encoder = _create_qutrit_qnode_encoder(n_qudit, connections)
    else:
        encoder = _create_manual_encoder(n_qudit, d, connections)

    return encoder, connections, params_per_layer
