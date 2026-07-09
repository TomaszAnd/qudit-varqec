#!/usr/bin/env python3
"""R14-7 §E.2 — R14-3-fullbatch LER under MAP, channel (b) η∈{0.94,0.95,0.96}.

The central comparison's reference: measures the no-sampling (full-batch)
code's operational LER so "smart sampling matches naive sampling" can be
quantified. Identical method/seeding to r14_6_r14_3b_ler.py (paired noise,
Meth-prior MAP, n=2000), so rows are directly comparable to
r14_5_map_cells.csv (R14-3a) and r14_6_r14_3b_ler.csv (R14-3b).
"""
from __future__ import annotations
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

# Identical method/seeding to r14_6_r14_3b_ler.py — shared sweep loop
# (audit/01 §A1 duplicated-logic 3).
from src.decoders.ler_driver import run_map_ler_sweep

CODES = {
    'R14-3-fb/seed0': 'results/round14_scoping/r14_3_fullbatch_meth/d3_n9_dist3_4L_meth_physical_seed0.npz',
    'R14-3-fb/seed1': 'results/round14_scoping/r14_3_fullbatch_meth/d3_n9_dist3_4L_meth_physical_seed1.npz',
    'R14-3-fb/seed2': 'results/round14_scoping/r14_3_fullbatch_meth/d3_n9_dist3_4L_meth_physical_seed2.npz',
}
ETAS = [0.94, 0.95, 0.96]
N_SHOTS = 2000
OUT = "results/round14_scoping/r14_7e_fullbatch_ler.csv"


def main():
    run_map_ler_sweep(CODES, ETAS, os.path.join(REPO, OUT), N_SHOTS)


if __name__ == "__main__":
    main()
