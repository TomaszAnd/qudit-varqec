#!/usr/bin/env python3
"""R14-9 §B — extend r14_7g_eta_to_pphys.csv to η ∈ {0.9296, 0.99, 0.995, 0.998}.

Same empirical-fidelity method as R14-7 §G:
- per-gate: single first-gate-pair channel × 500 random states (F_avg estimate)
- per-shot: full 36-gate-pair channel × 500 random states
- p_phys = 1 − F_avg

Writes a merged `r14_9_eta_to_pphys.csv` containing the union of old + new rows
(deduplicated by η, latest write wins). The original file is left untouched.
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.simulation import make_correlated_dephasing_noise_fn
from benchmark_ler_meth_kraus import build_gate_pairs
# F_avg here is the identical empirical-fidelity estimator defined in R14-7 §G
# (verified byte-for-byte equal on a seeded input); import it rather than recopy.
from r14_7g_eta_to_pphys import avg_fidelity as F_avg

NEW_ETAS = [0.9296, 0.99, 0.995, 0.998]
N_STATES = 500
SRC = "results/round14_scoping/r14_7g_eta_to_pphys.csv"
OUT = "results/round14_scoping/r14_9_eta_to_pphys.csv"


def main():
    d, n = 3, 9
    dim = d ** n
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    gp_full = build_gate_pairs(n, conns)
    gp_one = [gp_full[0]]
    rng = np.random.default_rng(0)

    existing = {}
    with open(os.path.join(REPO, SRC)) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            existing[float(r['eta'])] = r

    new_rows = []
    for eta in NEW_ETAS:
        t0 = time.time()
        nfg = make_correlated_dephasing_noise_fn(n_qudits=n, d=d, gate_pairs=gp_one,
                                                  eta=eta, n_max=5, noise_model='physical')
        nfs = make_correlated_dephasing_noise_fn(n_qudits=n, d=d, gate_pairs=gp_full,
                                                  eta=eta, n_max=5, noise_model='physical')
        Fg = F_avg(nfg, dim, rng, n=N_STATES)
        Fs = F_avg(nfs, dim, rng, n=N_STATES)
        row = dict(eta=eta, p_phys_per_gate=1 - Fg, p_phys_per_shot=1 - Fs,
                   F_avg_per_gate=Fg, F_avg_per_shot=Fs, n_states=N_STATES)
        new_rows.append(row)
        print(f"  η={eta}: p_phys/gate={1 - Fg:.4e}, p_phys/shot={1 - Fs:.4e} ({time.time() - t0:.0f}s)",
              flush=True)

    # merge: new rows override existing
    merged = dict(existing)
    for r in new_rows:
        merged[r['eta']] = r

    fields = ['eta', 'p_phys_per_gate', 'p_phys_per_shot', 'F_avg_per_gate',
              'F_avg_per_shot', 'n_states']
    op = os.path.join(REPO, OUT)
    with open(op, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eta in sorted(merged):
            r = merged[eta]
            w.writerow({k: r[k] for k in fields})
    print(f"wrote {op} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
