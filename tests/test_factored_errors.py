"""Tests for factored error representation."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.legacy.ququart_pipeline import build_error_sets, build_dephasing_error_sets
from src.legacy.ququart_pipeline import (
    build_error_sets_factored, build_dephasing_error_sets_factored,
    apply_factored_error, apply_factored_error_dag, factored_to_dense
)
from src.loss import _apply_factored_qml


class TestFactoredErrorSets:
    def test_same_count_depolarizing_d2(self):
        E_det, E_corr = build_error_sets(5, 2)
        E_det_f, E_corr_f = build_error_sets_factored(5, 2)
        assert len(E_det_f) == len(E_det)
        assert len(E_corr_f) == len(E_corr)

    def test_same_count_depolarizing_d3(self):
        """Check counts without materializing 37 GB dense error set."""
        from math import comb
        E_det_f, E_corr_f = build_error_sets_factored(5, 3)
        # Analytical: sum_{w=0}^{2} C(5,w)*15^w = 1 + 75 + 2250 = 2326
        expected_det = sum(comb(5, w) * 15**w for w in range(3))
        expected_corr = sum(comb(5, w) * 15**w for w in range(2))
        assert len(E_det_f) == expected_det
        assert len(E_corr_f) == expected_corr

    def test_same_count_dephasing_d3(self):
        E_det, E_corr = build_dephasing_error_sets(5, 3)
        E_det_f, E_corr_f = build_dephasing_error_sets_factored(5, 3)
        assert len(E_det_f) == len(E_det)
        assert len(E_corr_f) == len(E_corr)

    def test_identity_is_empty_list(self):
        E_det_f, _ = build_error_sets_factored(5, 2)
        assert E_det_f[0] == []

    def test_weight1_has_one_factor(self):
        E_det_f, _ = build_error_sets_factored(5, 2)
        # E_det_f[1] through E_det_f[75] are weight-1
        for i in range(1, 76):
            assert len(E_det_f[i]) == 1

    def test_weight2_has_two_factors(self):
        E_det_f, _ = build_error_sets_factored(5, 3)
        # Weight-2 errors start after identity (1) + weight-1 (75) = index 76
        assert len(E_det_f[76]) == 2


class TestFactoredToDense:
    def test_identity(self):
        dense = factored_to_dense([], 5, 4)
        assert np.allclose(dense, np.eye(1024))

    def test_matches_dense_d2(self):
        """Factored errors should produce identical dense matrices."""
        E_det, _ = build_error_sets(5, 2)
        E_det_f, _ = build_error_sets_factored(5, 2)
        for i in range(len(E_det)):
            dense_from_factored = factored_to_dense(E_det_f[i], 5, 4)
            assert np.allclose(dense_from_factored, E_det[i]), f"Mismatch at index {i}"


class TestApplyFactoredError:
    def test_identity(self):
        state = np.random.randn(1024) + 1j * np.random.randn(1024)
        result = apply_factored_error(state, [], 5, 4)
        assert np.allclose(result, state)

    def test_matches_dense(self):
        """Applying factored error should give same result as dense matrix multiply."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        state /= np.linalg.norm(state)

        E_det, _ = build_error_sets(5, 2)
        E_det_f, _ = build_error_sets_factored(5, 2)

        for i in [0, 1, 10, 50, 75]:  # sample a few
            dense_result = E_det[i] @ state
            factored_result = apply_factored_error(state, E_det_f[i], 5, 4)
            assert np.allclose(dense_result, factored_result, atol=1e-12), \
                f"Mismatch at index {i}"

    def test_dag_matches_dense(self):
        """E† applied via factored should match dense."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)

        E_det, _ = build_error_sets(5, 2)
        E_det_f, _ = build_error_sets_factored(5, 2)

        for i in [0, 1, 10, 50]:
            dense_result = E_det[i].conj().T @ state
            factored_result = apply_factored_error_dag(state, E_det_f[i], 5, 4)
            assert np.allclose(dense_result, factored_result, atol=1e-12)

    def test_weight2_matches_dense(self):
        """Weight-2 factored errors should match dense (built from factored_to_dense)."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)

        E_det_f, _ = build_error_sets_factored(5, 3)

        # Test a few weight-2 errors (indices > 75)
        for i in [76, 100, 200, 500]:
            dense = factored_to_dense(E_det_f[i], 5, 4)
            dense_result = dense @ state
            factored_result = apply_factored_error(state, E_det_f[i], 5, 4)
            assert np.allclose(dense_result, factored_result, atol=1e-12), \
                f"Mismatch at index {i}"


class TestApplyFactoredQml:
    """Test PennyLane autograd version matches dense matrix multiply."""

    def test_matches_dense_all_qudit_indices(self):
        """Verify _apply_factored_qml for single-qudit ops on all 5 qudit positions."""
        from pennylane import numpy as pnp

        rng = np.random.default_rng(42)
        state = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        state /= np.linalg.norm(state)
        state_pnp = pnp.array(state, requires_grad=False)

        E_det_f, _ = build_error_sets_factored(5, 2)

        # Weight-1 errors: indices 1..75 (15 per qudit × 5 qudits)
        # Test all 5 qudit positions (3 errors each to be thorough)
        for qudit_idx in range(5):
            start = 1 + qudit_idx * 15
            for offset in [0, 7, 14]:  # first, middle, last Pauli on this qudit
                i = start + offset
                factors = E_det_f[i]
                assert len(factors) == 1
                assert factors[0][0] == qudit_idx

                dense = factored_to_dense(factors, 5, 4)
                expected = dense @ state

                # numpy version
                got_np = apply_factored_error(state, factors, 5, 4)
                assert np.allclose(expected, got_np, atol=1e-12), \
                    f"numpy mismatch at qudit_idx={qudit_idx}, offset={offset}"

                # PennyLane autograd version
                got_qml = np.array(_apply_factored_qml(state_pnp, factors, 5, 4))
                assert np.allclose(expected, got_qml, atol=1e-12), \
                    f"QML mismatch at qudit_idx={qudit_idx}, offset={offset}"

    def test_weight2_matches_dense(self):
        """Verify _apply_factored_qml for weight-2 errors."""
        from pennylane import numpy as pnp

        rng = np.random.default_rng(42)
        state = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        state /= np.linalg.norm(state)
        state_pnp = pnp.array(state, requires_grad=False)

        E_det_f, _ = build_error_sets_factored(5, 3)

        for i in [76, 100, 200, 500, 1000]:
            factors = E_det_f[i]
            dense = factored_to_dense(factors, 5, 4)
            expected = dense @ state

            got_qml = np.array(_apply_factored_qml(state_pnp, factors, 5, 4))
            assert np.allclose(expected, got_qml, atol=1e-12), \
                f"QML mismatch at index {i}, factors on qudits {[q for q, _ in factors]}"

    def test_identity(self):
        """Empty factors = identity."""
        from pennylane import numpy as pnp

        state = pnp.array(np.random.randn(1024) + 1j * np.random.randn(1024),
                          requires_grad=False)
        result = _apply_factored_qml(state, [], 5, 4)
        assert np.allclose(np.array(result), np.array(state))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
