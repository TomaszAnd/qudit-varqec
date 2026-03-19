# Provenance: Extracted from Phase.py:22-51 (original VarQEC ququart implementation)
# Defines the 15 single-qudit Pauli operators for d=4 (ququarts encoded as 2 qubits)
# and the 3 dephasing-only (Z-type) operators.

import pennylane as qml
from pennylane import numpy as np

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
