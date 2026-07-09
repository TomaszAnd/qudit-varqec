"""Shor-Laflamme weight-enumerator regressions.

Two invariants:
  1. [[5,1,3]]_Z3 is a perfect distance-3 code: over the FULL generalized-Pauli
     basis, A_j = B_j for j <= 2 AND A_1 = A_2 = 0 (pure code).
  2. A converged trained distance-3 code satisfies the KL identity B_j ~= A_j
     over its TRAINED detection basis (the hardware basis), computed via the
     wire-grouped enumerator path (no dense projector).

Note (audit finding, 2026-07-04): trained codes do NOT have A_1 = A_2 = 0 over
the full Pauli basis — they are distance-3 with respect to the hardware error
basis, not all Paulis (measured A_2 ~ 0.04-0.55 across checkpoints). The
correct trained-code invariant is B_j - A_j -> 0, with the residual set by the
final training loss; r14_8's "A_1 = A_2 = 0" refers to its count_preserving
statistic, a different quantity.
"""
import itertools
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N9_NPZ = os.path.join(
    REPO, "results/round14_scoping/r14_3_fullbatch_meth/"
          "d3_n9_dist3_4L_meth_physical_seed0.npz")


def _generalized_paulis(d):
    X = np.roll(np.eye(d), 1, axis=0).astype(complex)
    Z = np.diag(np.exp(2j * np.pi * np.arange(d) / d))
    return [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            for a in range(d) for b in range(d) if (a, b) != (0, 0)]


def _enumerators_grouped(code_states, n, d, weight, ops):
    """Wire-grouped A_w, B_w: A_w = (1/K^2) sum_E |Tr(E P)|^2 with
    Tr(E P) = sum_k <psi_k|E|psi_k>, B_w = (1/K) sum_E sum_jk |<psi_j|E|psi_k>|^2.
    Local applications only — no d^n x d^n projector."""
    from src.decoders._common import apply_local_np
    K = code_states.shape[0]
    A = 0.0
    B = 0.0
    for sub in itertools.combinations(range(n), weight):
        left = [i for i in range(n) if i not in sub]
        order = list(sub) + left
        inv = tuple(order.index(i) for i in range(n))
        for choice in itertools.product(ops, repeat=weight):
            E = choice[0]
            for e in choice[1:]:
                E = np.kron(E, e)
            E_psi = np.stack([apply_local_np(code_states[k], E, sub, inv,
                                             d=d, n_qudit=n) for k in range(K)])
            M = np.conj(code_states) @ E_psi.T
            A += abs(np.trace(M)) ** 2
            B += float(np.sum(np.abs(M) ** 2))
    return A / K ** 2, B / K


class TestPerfectCode513:
    def test_A_equals_B_and_pure_up_to_weight2(self):
        from src.catalog import five_qudit_code_states
        from src.analysis import compute_weight_enumerators
        cs = five_qudit_code_states(3)
        A, B = compute_weight_enumerators(cs, 5, 3, _generalized_paulis(3),
                                          max_weight=2)
        assert np.allclose(A - B, 0.0, atol=1e-10)
        assert A[1] < 1e-12 and A[2] < 1e-12  # pure code

    def test_grouped_path_matches_dense_path(self):
        from src.catalog import five_qudit_code_states
        from src.analysis import compute_weight_enumerators
        cs = five_qudit_code_states(3)
        paulis = _generalized_paulis(3)
        A_dense, B_dense = compute_weight_enumerators(cs, 5, 3, paulis,
                                                      max_weight=1)
        A1, B1 = _enumerators_grouped(cs, 5, 3, 1, paulis)
        assert abs(A1 - A_dense[1]) < 1e-10
        assert abs(B1 - B_dense[1]) < 1e-10


class TestTrainedCodeKLIdentity:
    @pytest.mark.skipif(not os.path.exists(N9_NPZ),
                        reason="R14-3 checkpoint not present (results/ artifact)")
    def test_n9_meth_code_B_equals_A_over_hardware_basis(self):
        from src.decoders._common import encoder_forward
        from src.errors import qudit_hardware_error_basis
        n = 9
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, d, nq, K = encoder_forward(N9_NPZ, conns)
        hw = [np.asarray(m) for m in qudit_hardware_error_basis(3)]
        a1, b1 = _enumerators_grouped(cs, n, 3, 1, hw)
        # final_loss ~2e-5 -> weight-1 KL residual ~1e-4 (measured 1.26e-4)
        assert b1 - a1 < 1e-3, f"weight-1 KL residual grew: {b1 - a1:.3e}"
        a2, b2 = _enumerators_grouped(cs, n, 3, 2, hw)
        # weight-2 residual is larger (Meth weighting deprioritizes low-weight
        # products; measured 1.34e-2) — bound catches gross regressions only
        assert b2 - a2 < 5e-2, f"weight-2 KL residual grew: {b2 - a2:.3e}"
