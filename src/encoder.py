# Provenance: Extracted from Phase.py:170-198
# Original used module-level globals (N_QUDITS, N_QUBITS, connections, dev).
# Refactored as create_encoder() factory function that returns (encoder_qnode, connections).
# The encoder maps K=d computational basis states through the variational circuit.

import pennylane as qml
from pennylane import numpy as np
from src.gates import single_qudit_layer, ent_layer


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
                single_qudit_layer(layer_p[q*15:(q+1)*15], q)
            ent_layer(connections, layer_p[split:])

        return qml.state()

    return encoder, connections
