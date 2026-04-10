"""Tests for detection-style KL loss functions."""
import numpy as np
import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pennylane import numpy as pnp
from src.legacy.ququart_pipeline import build_dephasing_error_sets
from src.legacy.ququart_pipeline import create_encoder
from src.loss import (
    kl_loss_fast, kl_loss_detection_minibatch,
    kl_loss_detection_diagonal_minibatch,
    kl_loss_detection_factored_minibatch,
    precompute_error_products, load_varqec_result,
)
from src.legacy.ququart_pipeline import build_dephasing_error_sets_factored


class TestDetectionLossEquivalence:
    """Detection loss should agree with correction loss on converged codes."""

    def test_dephasing_d3_both_small(self):
        """Both losses should be very small on converged dephasing d=3."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "params", "dephasing_d3_2layer_seed2.npz")
        if not os.path.exists(path):
            import pytest
            pytest.skip("dephasing d=3 trained params not found")

        data = load_varqec_result(path)
        theta = pnp.array(data['params'], requires_grad=False)
        E_det, E_corr = build_dephasing_error_sets(5, 3)
        M_products = precompute_error_products(E_corr)
        encoder, _ = create_encoder(5, 4)

        old_loss = float(kl_loss_fast(theta, encoder, E_det, M_products, 4, 3))
        rng = np.random.default_rng(0)
        new_loss = float(kl_loss_detection_minibatch(
            theta, encoder, E_det, 4, 3, batch_fraction=1.0, rng=rng))

        assert old_loss < 0.01, f"Old loss too large: {old_loss}"
        assert new_loss < 0.01, f"New loss too large: {new_loss}"

    def test_d2_detection_matches_correction(self):
        """For d=2, detection loss == correction loss (no diagonal term)."""
        E_det, E_corr = build_dephasing_error_sets(5, 2)
        encoder, _ = create_encoder(5, 4)
        np.random.seed(42)
        theta = pnp.array(np.random.uniform(0, 2 * np.pi, (1, 79)), requires_grad=False)

        old_loss = float(kl_loss_fast(theta, encoder, E_det, [], 4, 2))
        rng = np.random.default_rng(0)
        new_loss = float(kl_loss_detection_minibatch(
            theta, encoder, E_det, 4, 2, batch_fraction=1.0, rng=rng))

        assert abs(old_loss - new_loss) < 1e-10, \
            f"d=2 losses differ: old={old_loss}, new={new_loss}"


class TestDetectionDiagonal:
    """Detection diagonal loss should produce reasonable values."""

    def test_not_nan(self):
        """Loss should not be NaN or Inf."""
        from src.correlated_noise import build_correlated_error_set
        encoder, _ = create_encoder(5, 4)
        gate_pairs = [(i, i + 1, 0, 1) for i in range(4)]
        E_det, _ = build_correlated_error_set(5, 4, gate_pairs, eta=0.95, n_max=2,
                                               noise_model="simplified")
        np.random.seed(42)
        theta = pnp.array(np.random.uniform(0, 2 * np.pi, (3, 79)), requires_grad=False)
        rng = np.random.default_rng(0)
        loss = float(kl_loss_detection_diagonal_minibatch(
            theta, encoder, E_det, 4, 3, batch_fraction=0.01, rng=rng))
        assert np.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss >= 0, f"Loss is negative: {loss}"


class TestDetectionFactored:
    """Detection factored loss should produce reasonable values."""

    def test_matches_dense_detection(self):
        """Factored detection loss should match dense detection loss for dephasing."""
        E_det, _ = build_dephasing_error_sets(5, 3)
        E_det_f, _ = build_dephasing_error_sets_factored(5, 3)
        encoder, _ = create_encoder(5, 4)
        np.random.seed(42)
        theta = pnp.array(np.random.uniform(0, 2 * np.pi, (2, 79)), requires_grad=False)

        rng1 = np.random.default_rng(0)
        dense_loss = float(kl_loss_detection_minibatch(
            theta, encoder, E_det, 4, 3, batch_fraction=1.0, rng=rng1))

        rng2 = np.random.default_rng(0)
        factored_loss = float(kl_loss_detection_factored_minibatch(
            theta, encoder, E_det_f, 4, 3, 5, 4, batch_fraction=1.0, rng=rng2))

        assert abs(dense_loss - factored_loss) / max(dense_loss, 1e-15) < 1e-6, \
            f"Dense={dense_loss}, Factored={factored_loss}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
