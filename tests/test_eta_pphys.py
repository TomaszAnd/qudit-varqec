"""Pin the eta -> p_phys conversion.

NOISE-FIX NOTE (Gaussian correction, src/correlated_noise.py): the default
correlated channel now uses the Gaussian-averaged dephasing e^{-1/2 sigma_p^2 Delta_f^2}
(agrees with Meth arXiv:2310.12110v3), i.e. exactly HALF the exponent of the
pre-fix factor-2 channel. Consequence: at a FIXED eta the channel is weaker, so
F_avg rises. At the paper operating point eta = 0.9296 (sigma_p^2 = 0.073):

    corrected  F_avg_per_gate = 0.6244,  p_phys = 1 - F_avg = 0.3756
    (pre-fix)  F_avg_per_gate = 0.4411,  p_phys = 0.5589   <- effective 2*sigma_p^2

To reproduce the paper's p_phys ~ 0.56 operating point under the corrected channel
one must set eta = e^{-2*0.073} = 0.8642 (the old 2*sigma_p^2 is the new sigma_p^2
at that eta). The live Monte-Carlo test below is re-pinned to the corrected value.

Definition: single entangling gate-pair channel on the FULL n=9, d=3 register
(2 active + 7 spectators), noise_model='physical', n_max=5; F_avg is the
Monte-Carlo average of |<psi|N(psi)>|^2 over Haar-random register states.
The 0.44-vs-0.56 confusion was a misreading of the fidelity column as the
infidelity; this test pins the number so it cannot silently recur.
"""
import os

import numpy as np
import pytest

ETA = 0.9296
# Corrected (Gaussian) channel values at ETA=0.9296:
F_AVG_EXPECTED = 0.6244
P_PHYS_EXPECTED = 0.3756
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/round14_scoping/r14_7g_eta_to_pphys.csv")


def test_pphys_at_operating_point_monte_carlo():
    from src.simulation import make_correlated_dephasing_noise_fn
    d, n = 3, 9
    dim = d ** n
    # one gate pair (0,1) at levels (0,1) — same as r14_7g's gp_one
    nf = make_correlated_dephasing_noise_fn(
        n_qudits=n, d=d, gate_pairs=[(0, 1, 0, 1)], eta=ETA, n_max=5,
        noise_model='physical')
    rng = np.random.default_rng(0)
    acc = 0.0
    n_states = 120  # seeded; sd of the mean ~0.006 at this count
    for _ in range(n_states):
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        acc += float(np.abs(np.vdot(psi, nf(psi, rng))) ** 2)
    f_avg = acc / n_states
    p_phys = 1.0 - f_avg
    assert abs(f_avg - F_AVG_EXPECTED) < 0.03, f"F_avg drifted: {f_avg:.4f}"
    assert abs(p_phys - P_PHYS_EXPECTED) < 0.03, f"p_phys drifted: {p_phys:.4f}"
    # corrected (Gaussian) channel is weaker at eta=0.9296: F_avg > 0.5 > p_phys
    assert f_avg > 0.5 > p_phys


def test_csv_row_pins_resolved_values():
    # PRE-FIX ARTIFACT: r14_7g_eta_to_pphys.csv was generated under the old factor-2
    # channel; its F_avg=0.4411 / p_phys=0.5589 are the effective-2*sigma_p^2 numbers.
    # Kept pinned to prevent silent edits; the CSV is scheduled for regeneration under
    # the corrected channel in Stage J (see docs/FOLLOWUP.md). This test asserts the
    # historical values, NOT the corrected operating point above.
    if not os.path.exists(CSV):
        pytest.skip("r14_7g_eta_to_pphys.csv not present (results/ artifact)")
    import csv
    with open(CSV) as f:
        rows = {float(r['eta']): r for r in csv.DictReader(f)}
    row = rows[ETA]
    assert abs(float(row['F_avg_per_gate']) - 0.44108398420425265) < 1e-12
    assert abs(float(row['p_phys_per_gate']) - 0.5589160157957473) < 1e-12
