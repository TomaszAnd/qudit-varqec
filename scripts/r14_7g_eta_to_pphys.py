#!/usr/bin/env python3
"""R14-7 §G — η → physical error rate conversion via empirical channel fidelity.

The Meth η isn't a physical error rate. Convert to the standard QEC x-axis,
per-gate average gate infidelity 1 − F_avg, empirically:
  F_avg = E_ψ E_kraus |⟨ψ | N(ψ)⟩|²  (the stochastic noise_fn samples one Kraus
  outcome per call; averaging |⟨ψ|noisy⟩|² over (state, draw) pairs = F_avg).

Two channels per η:
  - per-gate: a single gate-pair's channel (1 control+target + spectators)
  - per-shot: the full 36-gate composition (what the code actually sees/shot)
Per-gate p_phys is the QEC-paper convention; per-shot is the effective strength.
n=9, d=3. nice -19.
"""
from __future__ import annotations
import csv, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.simulation import make_correlated_dephasing_noise_fn
from benchmark_ler_meth_kraus import build_gate_pairs

ETAS = [0.94, 0.95, 0.96, 0.97, 0.98]
N_STATES = 500


def avg_fidelity(noise_fn, dim, rng, n=N_STATES):
    acc = 0.0
    for _ in range(n):
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        noisy = noise_fn(psi, rng)
        acc += float(np.abs(np.vdot(psi, noisy)) ** 2)
    return acc / n


def main():
    d, n_qudit = 3, 9
    dim = d ** n_qudit
    conns = [[i, j] for i in range(n_qudit) for j in range(i + 1, n_qudit)]
    gp_full = build_gate_pairs(n_qudit, conns)            # 36 gates
    gp_one = [gp_full[0]]                                  # single gate
    rng = np.random.default_rng(0)
    out = os.path.join(REPO, "results/round14_scoping/r14_7g_eta_to_pphys.csv")
    rows = []
    for eta in ETAS:
        nf_gate = make_correlated_dephasing_noise_fn(
            n_qudits=n_qudit, d=d, gate_pairs=gp_one, eta=eta, n_max=5,
            noise_model='physical')
        nf_shot = make_correlated_dephasing_noise_fn(
            n_qudits=n_qudit, d=d, gate_pairs=gp_full, eta=eta, n_max=5,
            noise_model='physical')
        F_gate = avg_fidelity(nf_gate, dim, rng)
        F_shot = avg_fidelity(nf_shot, dim, rng)
        p_gate, p_shot = 1 - F_gate, 1 - F_shot
        print(f"  η={eta}: per-gate p_phys={p_gate:.4e}, per-shot p_phys={p_shot:.4e}",
              flush=True)
        rows.append(dict(eta=eta, p_phys_per_gate=p_gate, p_phys_per_shot=p_shot,
                         F_avg_per_gate=F_gate, F_avg_per_shot=F_shot,
                         n_states=N_STATES))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
