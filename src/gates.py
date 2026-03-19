# Provenance: Extracted from Phase.py:131-168
# G_theta_unitary: diagonal entangling gate motivated by light-shift interactions
# single_qudit_layer: 15-parameter SU(4) decomposition (2xRX-RZ-RX + CNOTs)
# These gates assume the 2-qubit-per-qudit encoding (d=4 only).

import pennylane as qml
from pennylane import numpy as np


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
    q0, q1 = 2*q_idx, 2*q_idx + 1
    qml.RX(p[0], wires=q0); qml.RZ(p[1], wires=q0); qml.RX(p[2], wires=q0)
    qml.RX(p[3], wires=q1); qml.RZ(p[4], wires=q1); qml.RX(p[5], wires=q1)
    qml.CNOT(wires=[q1, q0])
    qml.RZ(p[6], wires=q0)
    qml.RY(p[7], wires=q1)
    qml.CNOT(wires=[q0, q1])
    qml.RY(p[8], wires=q1)
    qml.CNOT(wires=[q1, q0])
    qml.RX(p[9],  wires=q0);  qml.RZ(p[10], wires=q0); qml.RX(p[11], wires=q0)
    qml.RX(p[12], wires=q1);  qml.RZ(p[13], wires=q1); qml.RX(p[14], wires=q1)
