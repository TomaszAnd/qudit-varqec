"""
Abstract ququart pipeline (2-qubit-per-ququart encoding).

Used to load and evaluate the 6 original VarQEC codes trained with
abstract G_theta gates and Pauli error bases. For new training, use
the native_*.py modules instead.

Merged from: pauli_ops.py, error_sets.py, error_sets_factored.py, gates.py, encoder.py
"""
import itertools
import pennylane as qml
from pennylane import numpy as np
import numpy as onp


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Pauli operators (from pauli_ops.py)
# ═══════════════════════════════════════════════════════════════════════

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single_qudit_paulis():
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    single_errs = []
    for p0 in 'IXYZ':
        for p1 in 'IXYZ':
            if p0 == 'I' and p1 == 'I':
                continue
            single_errs.append(qml.math.kron(pauli_map[p0], pauli_map[p1]))
    assert len(single_errs) == 15
    return single_errs


def single_qudit_dephasing_paulis():
    pauli_map = {'I': I, 'Z': Z}
    dephasing_errs = []
    for p0 in 'IZ':
        for p1 in 'IZ':
            if p0 == 'I' and p1 == 'I':
                continue
            dephasing_errs.append(qml.math.kron(pauli_map[p0], pauli_map[p1]))
    assert len(dephasing_errs) == 3
    return dephasing_errs


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Error sets — dense (from error_sets.py)
# ═══════════════════════════════════════════════════════════════════════

def build_error_sets(n_qudits, distance, single_errs_fn=single_qudit_paulis, dim_qudit=4):
    single_errs = single_errs_fn()
    Id = np.eye(dim_qudit, dtype=complex)

    id_full = Id
    for _ in range(1, n_qudits):
        id_full = np.kron(id_full, Id)

    E_det = [id_full]
    E_corr = [id_full]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudits), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                op = 1
                idx_choice = 0
                for q in range(n_qudits):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx_choice])
                        idx_choice += 1
                    else:
                        op = np.kron(op, Id)
                E_det.append(op)
                if w <= max_corr:
                    E_corr.append(op)

    return E_det, E_corr


def build_dephasing_error_sets(n_qudits, distance):
    return build_error_sets(n_qudits, distance, single_errs_fn=single_qudit_dephasing_paulis)


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Error sets — factored (from error_sets_factored.py)
# ═══════════════════════════════════════════════════════════════════════

def build_error_sets_factored(n_qudits, distance, single_errs_fn=single_qudit_paulis, dim_qudit=4):
    single_errs = single_errs_fn()
    E_det_factors = [[]]
    E_corr_factors = [[]]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudits), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                factors = [(qudit_subset[i], onp.array(err_choice[i])) for i in range(w)]
                E_det_factors.append(factors)
                if w <= max_corr:
                    E_corr_factors.append(factors)

    return E_det_factors, E_corr_factors


def build_dephasing_error_sets_factored(n_qudits, distance):
    return build_error_sets_factored(n_qudits, distance,
                                     single_errs_fn=single_qudit_dephasing_paulis)


def apply_factored_error(state, factors, n_qudits, dim_qudit):
    if not factors:
        return state
    s = state
    shape = tuple([dim_qudit] * n_qudits)
    for qudit_idx, op in factors:
        s_tensor = s.reshape(shape)
        s_tensor = onp.tensordot(op, s_tensor, axes=([1], [qudit_idx]))
        s_tensor = onp.moveaxis(s_tensor, 0, qudit_idx)
        s = s_tensor.reshape(-1)
    return s


def apply_factored_error_dag(state, factors, n_qudits, dim_qudit):
    if not factors:
        return state
    s = state
    shape = tuple([dim_qudit] * n_qudits)
    for qudit_idx, op in reversed(factors):
        op_dag = op.conj().T
        s_tensor = s.reshape(shape)
        s_tensor = onp.tensordot(op_dag, s_tensor, axes=([1], [qudit_idx]))
        s_tensor = onp.moveaxis(s_tensor, 0, qudit_idx)
        s = s_tensor.reshape(-1)
    return s


def factored_to_dense(factors, n_qudits, dim_qudit):
    dim = dim_qudit ** n_qudits
    Id = onp.eye(dim_qudit, dtype=complex)
    if not factors:
        return onp.eye(dim, dtype=complex)
    factor_dict = {q: op for q, op in factors}
    op = onp.array([[1.0]], dtype=complex)
    for q in range(n_qudits):
        op = onp.kron(op, factor_dict.get(q, Id))
    return op


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Gates (from gates.py)
# ═══════════════════════════════════════════════════════════════════════

def G_theta_unitary(theta, d=4):
    indices = np.arange(d)
    condition = indices[:, None] != indices
    phase_matrix = qml.math.where(condition, qml.math.exp(1j * theta), 1.0)
    phases_flat = qml.math.flatten(phase_matrix)
    return qml.math.diag(phases_flat)


def ent_layer(conns, phi_vec):
    for (q1, q2), phi in zip(conns, phi_vec):
        U = G_theta_unitary(phi)
        wires = [2 * q1, 2 * q1 + 1, 2 * q2, 2 * q2 + 1]
        qml.QubitUnitary(U, wires=wires)


def single_qudit_layer(p, q_idx):
    q0, q1 = 2 * q_idx, 2 * q_idx + 1
    qml.RX(p[0], wires=q0); qml.RZ(p[1], wires=q0); qml.RX(p[2], wires=q0)
    qml.RX(p[3], wires=q1); qml.RZ(p[4], wires=q1); qml.RX(p[5], wires=q1)
    qml.CNOT(wires=[q1, q0])
    qml.RZ(p[6], wires=q0)
    qml.RY(p[7], wires=q1)
    qml.CNOT(wires=[q0, q1])
    qml.RY(p[8], wires=q1)
    qml.CNOT(wires=[q1, q0])
    qml.RX(p[9], wires=q0); qml.RZ(p[10], wires=q0); qml.RX(p[11], wires=q0)
    qml.RX(p[12], wires=q1); qml.RZ(p[13], wires=q1); qml.RX(p[14], wires=q1)


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Encoder (from encoder.py)
# ═══════════════════════════════════════════════════════════════════════

def create_encoder(n_qudits=5, dim_qudit=4):
    n_qubits = n_qudits * int(np.log2(dim_qudit))
    connections = [[0, i] for i in range(1, n_qudits)]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def encoder(params, code_ind):
        n_layer = params.shape[0]
        bits = np.binary_repr(code_ind, width=int(np.log2(dim_qudit)))
        for b_idx, b in enumerate(bits):
            if b == "1":
                qml.PauliX(wires=b_idx)
        split = n_qudits * 15
        for l in range(n_layer):
            layer_p = params[l]
            for q in range(n_qudits):
                single_qudit_layer(layer_p[q * 15:(q + 1) * 15], q)
            ent_layer(connections, layer_p[split:])
        return qml.state()

    return encoder, connections
