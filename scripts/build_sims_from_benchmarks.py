#!/usr/bin/env python3
"""Rebuild results/simulations/ler_{name}_hardware.npz from the committed
benchmark CSVs (results/benchmarks_30k), so notebooks/analyze_codes.py plots
Fig 8-11 at the paper's shot budget (qutrit family + [[5,1,3]]: 30k shots;
q=4,5 codes: 100k shots) and reproduces the printed LER values exactly.

This is a pure FORMAT CONVERSION of committed benchmark data -- no simulation
is run. The benchmark CSVs are the same i.i.d.-per-qudit-p experiment the
paper's Table II / Fig 8-11 numbers come from (verified: identical raw fidelity
and code as the prior lower-shot sims, higher shot count).

Usage:  PYTHONPATH=. python3 scripts/build_sims_from_benchmarks.py
"""
import csv
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "results", "benchmarks_30k")
SIM = os.path.join(REPO, "results", "simulations")

# benchmark CSV filename -> (catalog/sim code name, d, n)
MAP = {
    "ler_qutrit_d3_30000.csv":      ("qutrit_d3",    3, 5),
    "ler_d3_n6_dist3_30000.csv":    ("d3_n6_dist3",  3, 6),
    "ler_qutrit_n7_d3_30000.csv":   ("qutrit_n7_d3", 3, 7),
    "ler_qutrit_n8_d3_30000.csv":   ("qutrit_n8_d3", 3, 8),
    "ler_d3_n9_dist3_30000.csv":    ("d3_n9_dist3",  3, 9),
    "ler_d4_n6_dist3_100000.csv":   ("d4_n6_dist3",  4, 6),
    "ler_d5_n5_dist3_100000.csv":   ("d5_n5_dist3",  5, 5),
    "ler_ququart_n5_d3_100000.csv": ("ququart_n5_d3", 4, 5),
    "ler_five_qudit_d3_30000.csv":  ("five_qudit_d3", 3, 5),
}


def load_csv(path):
    p, ler, fc, fr, shots = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            p.append(float(row["p"]))
            ler.append(float(row["LER"]))
            fc.append(float(row["F_corr"]))
            fr.append(float(row["F_raw"]))
            shots.append(int(row["shots"]))
    return (np.array(p), np.array(ler), np.array(fc), np.array(fr), shots[0])


def main():
    os.makedirs(SIM, exist_ok=True)
    for fname, (name, d, n) in MAP.items():
        src = os.path.join(BENCH, fname)
        if not os.path.exists(src):
            print(f"  MISSING {fname}; skipped")
            continue
        p, ler, fc, fr, shots = load_csv(src)
        out = os.path.join(SIM, f"ler_{name}_hardware.npz")
        np.savez(out, p_rates=p, lers=ler, fid_corrs=fc, fid_raws=fr,
                 n_shots=shots, code=[name], distance=[3], d=[d], n_qudit=[n])
        print(f"  wrote {os.path.basename(out)}  ({shots} shots, {len(p)} points)")


if __name__ == "__main__":
    main()
