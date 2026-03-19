# Provenance: Extracted from Phase.py:55-118 and test2.py:40-72
# Unified build_error_sets() with configurable single_errs_fn parameter.
# Original Phase.py had separate build_dephasing_error_sets() with hardcoded 5-fold kron;
# now delegates to build_error_sets() with single_qudit_dephasing_paulis.

import itertools
from pennylane import numpy as np
from src.pauli_ops import single_qudit_paulis, single_qudit_dephasing_paulis


def build_error_sets(n_qudits, distance, single_errs_fn=single_qudit_paulis, dim_qudit=4):
    single_errs = single_errs_fn()
    Id = np.eye(dim_qudit, dtype=complex)

    id_full = Id
    for _ in range(1, n_qudits):
        id_full = np.kron(id_full, Id)

    E_det  = [id_full]
    E_corr = [id_full]

    max_det  = distance - 1
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
