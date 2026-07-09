"""
Timing benchmarks for VarQEC components.
Run: python3 -m pytest tests/test_benchmarks.py -v -s
"""
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_error_set_construction_time():
    """Benchmark error set construction."""
    from src.legacy.ququart_pipeline import build_error_sets, build_dephasing_error_sets

    t0 = time.time()
    Ed, Ec = build_dephasing_error_sets(5, 3)
    t_deph = time.time() - t0

    t0 = time.time()
    Ed, Ec = build_error_sets(5, 3)
    t_depol = time.time() - t0

    print(f"\n  Dephasing error sets:    {t_deph:.3f}s (106 + 16 operators)")
    print(f"  Depolarizing error sets: {t_depol:.3f}s (2326 + 76 operators)")
    assert t_deph < 30, "Dephasing should build in <30s"
    assert t_depol < 120, "Depolarizing should build in <120s"


def test_kl_loss_single_eval_time():
    """Benchmark a single KL loss evaluation (dephasing, 2 layers)."""
    from src.legacy.ququart_pipeline import build_dephasing_error_sets
    from src.legacy.ququart_pipeline import create_encoder
    from src.loss import kl_loss_fast, precompute_error_products
    from pennylane import numpy as pnp

    Ed, Ec = build_dephasing_error_sets(5, 3)
    M_products = precompute_error_products(Ec)
    encoder, _ = create_encoder(5, 4)

    params = pnp.array(np.random.uniform(0, 2*np.pi, (2, 79)), requires_grad=False)

    t0 = time.time()
    loss = kl_loss_fast(params, encoder, Ed, M_products, K=4, distance=3)
    t_eval = time.time() - t0

    print(f"\n  KL loss (dephasing, 2 layers): {t_eval:.3f}s, loss={float(loss):.4e}")
    assert t_eval < 300, "Single dephasing loss eval should be <300s"


def test_correlated_error_set_time():
    """Benchmark correlated error set construction."""
    from src.correlated_noise import build_correlated_error_set

    t0 = time.time()
    Ed, Ec = build_correlated_error_set(
        n_qudits=2, d=3, gate_pairs=[(0, 1, 0, 1)], eta=0.95, n_max=2
    )
    t_small = time.time() - t0

    t0 = time.time()
    Ed2, Ec2 = build_correlated_error_set(
        n_qudits=3, d=3, gate_pairs=[(0, 1, 0, 1), (1, 2, 0, 1)], eta=0.95, n_max=2
    )
    t_medium = time.time() - t0

    print(f"\n  Correlated errors (n=2, d=3, 1 gate): {t_small:.3f}s, {len(Ed)} operators")
    print(f"  Correlated errors (n=3, d=3, 2 gates): {t_medium:.3f}s, {len(Ed2)} operators")


def test_kraus_generation_time():
    """Benchmark Kraus operator generation."""
    from src.correlated_noise import control_qudit_kraus, verify_kraus_completeness

    for d in [3, 5, 7]:
        t0 = time.time()
        for _ in range(100):
            diags = control_qudit_kraus(d=d, control_level=0, eta=0.95, n_max=10)
        t_per = (time.time() - t0) / 100
        print(f"\n  Kraus gen d={d}: {t_per*1000:.2f}ms per call")

    # Completeness check timing
    diags = control_qudit_kraus(d=5, control_level=0, eta=0.95, n_max=15)
    t0 = time.time()
    for _ in range(1000):
        verify_kraus_completeness(diags)
    t_per = (time.time() - t0) / 1000
    print(f"  Completeness check d=5: {t_per*1000:.3f}ms per call")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
