#!/usr/bin/env python3
"""R14-6 §C — R14-3b LER validation under MAP, channel (b) η∈{0.94,0.95,0.96}.

The missing experiment: R14-3a was head-to-head tested in R14-5; R14-3b
(stratified-IS-trained) needs the same MAP/channel-(b) measurement to confirm
operational equivalence (matched LER, not just matched training loss).

Same `simulate_ler_with_weighted_map`, same Meth-prior MAP correction set, same
paired-noise per-cell seeding (cell_seed(b,η)) as R14-5's sweep, so the R14-3b
rows are directly comparable to r14_5_map_cells.csv. n_shots=2000.

Channel (c) and Petz skipped (Outcome I + Petz saturation already established).
nice -19; appends per-cell so partial progress survives.
"""
from __future__ import annotations
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Shared channel-(b) MAP-LER sweep loop (audit/01 §A1 duplicated-logic 3).
from src.decoders.ler_driver import run_map_ler_sweep

CODES = {
    'R14-3b/seed0': 'results/round14_scoping/r14_3b_meth_physical_strat_is/d3_n9_dist3_4L_meth_physical_seed0.npz',
    'R14-3b/seed1': 'results/round14_scoping/r14_3b_meth_physical_strat_is/d3_n9_dist3_4L_meth_physical_seed1.npz',
    'R14-3b/seed2': 'results/round14_scoping/r14_3b_meth_physical_strat_is/d3_n9_dist3_4L_meth_physical_seed2.npz',
}
ETAS = [0.94, 0.95, 0.96]
N_SHOTS = 2000
OUT = "results/round14_scoping/r14_6_r14_3b_ler.csv"


def main():
    run_map_ler_sweep(CODES, ETAS, os.path.join(REPO, OUT), N_SHOTS)


if __name__ == "__main__":
    main()
