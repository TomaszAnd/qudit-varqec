"""
Unit tests for VarQEC core library.
Run: python3 -m pytest tests/test_src.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.legacy.ququart_pipeline import single_qudit_paulis, single_qudit_dephasing_paulis, I, X, Y, Z


class TestPauliOps:
    def test_pauli_count(self):
        assert len(single_qudit_paulis()) == 15

    def test_dephasing_count(self):
        assert len(single_qudit_dephasing_paulis()) == 3

    def test_paulis_are_unitary(self):
        for P in single_qudit_paulis():
            assert np.allclose(P @ P.conj().T, np.eye(4)), "Pauli should be unitary"

    def test_paulis_are_hermitian(self):
        for P in single_qudit_paulis():
            assert np.allclose(P, P.conj().T), "Pauli should be Hermitian"

    def test_paulis_traceless(self):
        for P in single_qudit_paulis():
            assert abs(np.trace(P)) < 1e-12, "Non-identity Pauli should be traceless"

    def test_dephasing_subset_of_full(self):
        """Dephasing paulis should be a subset of full paulis."""
        full = [p.tolist() for p in single_qudit_paulis()]
        for dp in single_qudit_dephasing_paulis():
            assert dp.tolist() in full, "Dephasing Pauli not found in full set"


class TestErrorSets:
    def test_depolarizing_sizes(self):
        from src.legacy.ququart_pipeline import build_error_sets
        Ed, Ec = build_error_sets(5, 3)
        assert len(Ed) == 2326, f"Expected 2326, got {len(Ed)}"
        assert len(Ec) == 76, f"Expected 76, got {len(Ec)}"

    def test_dephasing_sizes(self):
        from src.legacy.ququart_pipeline import build_dephasing_error_sets
        Ed, Ec = build_dephasing_error_sets(5, 3)
        assert len(Ed) == 106, f"Expected 106, got {len(Ed)}"
        assert len(Ec) == 16, f"Expected 16, got {len(Ec)}"

    def test_identity_in_sets(self):
        from src.legacy.ququart_pipeline import build_error_sets
        Ed, Ec = build_error_sets(5, 3)
        assert np.allclose(Ed[0], np.eye(1024)), "First element should be identity"

    def test_custom_dim(self):
        """Test with dim_qudit=4 and 3 qudits."""
        from src.legacy.ququart_pipeline import build_error_sets
        Ed, Ec = build_error_sets(3, 2, dim_qudit=4)
        assert Ed[0].shape == (64, 64), f"3 ququarts: expected 64x64, got {Ed[0].shape}"

    def test_distance_2_no_correction(self):
        """Distance 2: E_corr should only contain identity."""
        from src.legacy.ququart_pipeline import build_dephasing_error_sets
        Ed, Ec = build_dephasing_error_sets(5, 2)
        assert len(Ec) == 1, "Distance 2: E_corr should be {I} only"
        assert len(Ed) > 1, "Distance 2: E_det should have weight-1 errors"


class TestEncoder:
    def test_create_encoder(self):
        from src.legacy.ququart_pipeline import create_encoder
        enc, conn = create_encoder(5, 4)
        assert callable(enc)
        assert len(conn) == 4  # star topology: 0-1, 0-2, 0-3, 0-4

    def test_encoder_returns_state(self):
        from src.legacy.ququart_pipeline import create_encoder
        from pennylane import numpy as pnp
        enc, conn = create_encoder(3, 4)  # 3 qudits, smaller
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, 3*15 + 2)), requires_grad=False)
        state = enc(params, 0)
        assert abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10, "State should be normalized"


class TestGates:
    def test_g_theta_unitary(self):
        from src.legacy.ququart_pipeline import G_theta_unitary
        import pennylane.numpy as pnp
        U = G_theta_unitary(pnp.array(0.5), d=4)
        assert U.shape == (16, 16)

    def test_g_theta_identity_at_zero(self):
        from src.legacy.ququart_pipeline import G_theta_unitary
        import pennylane.numpy as pnp
        U = G_theta_unitary(pnp.array(0.0), d=4)
        assert np.allclose(U, np.eye(16)), "G(0) should be identity"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
