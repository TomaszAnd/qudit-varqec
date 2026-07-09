#!/usr/bin/env python3
"""R14-10 §0.5 step 1 — sub-100k LER validation for R14-7a / R14-7b.

The two sub-100k-op-EV samplers (R14-7a Hoeffding/stratified-IS truncated to
1100 steps = 99k op-EVs/seed; R14-7b full-basis IS, 63 op-EVs/step = 94k/seed)
finished training but were never LER-validated. This closes the deferred
measurement so the 5-sampler operational-equivalence claim rests on measured
LER, not just matched training loss.

Exact replica of scripts/r14_6_r14_3b_ler.py: same simulate_ler_with_weighted_map,
same Meth-prior MAP correction set (max_weight=2), same paired-noise per-cell
seeding cell_seed(eta), n_shots=2000, channel (b) η∈{0.94,0.95,0.96}. Rows are
therefore directly comparable to r14_6_r14_3b_ler.csv / r14_7e_fullbatch_ler.csv.

nice -19; appends per-cell so partial progress survives.
"""
from __future__ import annotations
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Exact-replica-of-r14_6 sweep loop, now shared (audit/01 §A1 duplicated-logic 3).
from src.decoders.ler_driver import run_map_ler_sweep

CODES = {
    'R14-7a/seed0': 'results/round14_scoping/r14_7a_hoeffding/d3_n9_dist3_4L_meth_physical_seed0.npz',
    'R14-7a/seed1': 'results/round14_scoping/r14_7a_hoeffding/d3_n9_dist3_4L_meth_physical_seed1.npz',
    'R14-7a/seed2': 'results/round14_scoping/r14_7a_hoeffding/d3_n9_dist3_4L_meth_physical_seed2.npz',
    'R14-7b/seed0': 'results/round14_scoping/r14_7b_full_is/d3_n9_dist3_4L_meth_physical_seed0.npz',
    'R14-7b/seed1': 'results/round14_scoping/r14_7b_full_is/d3_n9_dist3_4L_meth_physical_seed1.npz',
    'R14-7b/seed2': 'results/round14_scoping/r14_7b_full_is/d3_n9_dist3_4L_meth_physical_seed2.npz',
}
ETAS = [0.94, 0.95, 0.96]
N_SHOTS = 2000
OUT = "results/round14_scoping/r14_10_r14_7ab_ler.csv"


def main():
    run_map_ler_sweep(CODES, ETAS, os.path.join(REPO, OUT), N_SHOTS)


if __name__ == "__main__":
    main()
