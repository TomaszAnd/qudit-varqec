#!/usr/bin/env python3
"""R14-7 §D — missing R14-3a MAP cells at η∈{0.95,0.96} (R14-5 left R14-3a MAP
at η=0.94 only). Closes the R14-6 LER-plot asymmetry. 3 seeds × 2 η = 6 cells.

Identical method/seeding to r14_6_r14_3b_ler.py. η=0.94 R14-3a MAP already in
r14_5_map_cells.csv (not re-run). Output kept separate so R14-5's CSV is intact.
"""
from __future__ import annotations
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

# Identical method/seeding to r14_6_r14_3b_ler.py — shared sweep loop
# (audit/01 §A1 duplicated-logic 3).
from src.decoders.ler_driver import run_map_ler_sweep

CODES = {
    'R14-3a/seed0': 'results/round14_scoping/r14_3a_meth_physical/d3_n9_dist3_4L_meth_physical_seed0.npz',
    'R14-3a/seed1': 'results/round14_scoping/r14_3a_meth_physical/d3_n9_dist3_4L_meth_physical_seed1.npz',
    'R14-3a/seed2': 'results/round14_scoping/r14_3a_meth_physical/d3_n9_dist3_4L_meth_physical_seed2.npz',
}
ETAS = [0.95, 0.96]  # 0.94 already in r14_5_map_cells.csv
N_SHOTS = 2000
OUT = "results/round14_scoping/r14_7d_r14_3a_map_full.csv"


def main():
    run_map_ler_sweep(CODES, ETAS, os.path.join(REPO, OUT), N_SHOTS)


if __name__ == "__main__":
    main()
