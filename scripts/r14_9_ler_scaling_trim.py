#!/usr/bin/env python3
"""R14-9 §A (trimmed) — lower-noise LER scaling, contention-tolerant variant.

Re-scopes the R14-8 §C scan to three low-noise points at n_shots=5000:
  η ∈ {0.97, 0.98, 0.99}  on R14-3b/seed1 + R12-a2a, MAP decoder.

Expected resolution at n=5000:
  η=0.97 (LER~0.04, ~200 errors), η=0.98 (~0.015, ~75), η=0.99 (~0.004, ~20)
all give usable CIs for a slope fit. Combined with the committed
η ∈ {0.94, 0.95, 0.96} points, the threshold plot has 6 points spanning
p_phys ≈ 0.10–0.50.

Output appends to results/round14_scoping/r14_8_ler_scaling.csv (the R14-8
file the threshold plot already reads); rows with the same (code, eta) as the
existing R12 η=0.97 row are overwritten via fresh header. nice -19.
"""
from __future__ import annotations
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

# Re-scopes R14-8 §C to fixed n_shots=5000 — shares the r14_6 sweep loop
# (audit/01 §A1 duplicated-logic 3). Note the base r14_8_ler_scaling.py keeps
# its own copy because its per-eta adaptive shot count genuinely diverges.
from src.decoders.ler_driver import run_map_ler_sweep

CODES = {
    'R12-a2a': 'results/saved_params/all_to_all/d3_n9_dist3_4L_best3s_seed1.npz',
    'R14-3b/seed1': 'results/round14_scoping/r14_3b_meth_physical_strat_is/d3_n9_dist3_4L_meth_physical_seed1.npz',
}
ETAS = [0.97, 0.98, 0.99]
N_SHOTS = 5000
OUT = "results/round14_scoping/r14_8_ler_scaling.csv"


def main():
    run_map_ler_sweep(CODES, ETAS, os.path.join(REPO, OUT), N_SHOTS)


if __name__ == "__main__":
    main()
