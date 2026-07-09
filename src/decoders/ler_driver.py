"""Shared MAP-LER sweep driver for the R14-6/7d/7e/9/10 channel-(b) experiments.

Each of those scripts measured the same quantity — a trained code's logical
error rate under the weighted-MAP decoder against the physical-mode Meth phase
channel (channel 'b'), swept over eta with paired per-cell seeding — and differed
only in which codes / eta grid / n_shots / output file it used. This module hosts
the verbatim setup-and-double-loop those scripts share so each stays a thin
configuration wrapper over it. See audit/01 §A1 duplicated-logic item 3.

r14_8_ler_scaling.py is intentionally NOT converted: it looks up an adaptive
per-eta shot count (SHOTS_BY_ETA[eta]) inside the loop, a genuine divergence
from the fixed-n_shots form shared by the other five.
"""
from __future__ import annotations
import csv
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.errors import ErrorModel
from src.decoders._common import encoder_forward
from src.decoders.priors import meth_pauli_prior
from src.decoders.weighted_map import (
    build_weighted_correction_set, simulate_ler_with_weighted_map)
from src.simulation import make_correlated_dephasing_noise_fn

FIELDS = ['code', 'channel', 'axis_label', 'axis_value', 'decoder',
          'ler_mean', 'ler_lo', 'ler_hi', 'n_shots', 'seed']


def cell_seed(eta):
    """Paired-noise per-cell seed; matches R14-5 cell_seed('b', eta)."""
    return 200000 + int(round(eta * 1000))


def _scripts_helpers():
    # build_gate_pairs and bootstrap_ler_ci are the canonical single copies in
    # the library-role benchmark scripts. Import them lazily (adding scripts/ to
    # the path here) so src/ keeps no hard module-load dependency on scripts/.
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmark_ler_meth_kraus import build_gate_pairs
    from benchmark_ler_meth_pauli import bootstrap_ler_ci
    return build_gate_pairs, bootstrap_ler_ci


def run_map_ler_sweep(codes, etas, out_path, n_shots, repo=_REPO):
    """Sweep channel-(b) weighted-MAP LER over `etas` for each code in `codes`.

    codes    : mapping of label -> npz path (relative to `repo`).
    etas      : iterable of Meth phase-noise etas.
    out_path : CSV written with the FIELDS schema (fresh header, then one row
               appended per cell so partial progress survives interruption).
    n_shots  : Monte-Carlo shots per cell (fixed across etas).

    Reproduces the inline loop of r14_6/7d/7e/9/10 byte-for-byte in its CSV
    output; prints per-cell progress to stdout.
    """
    build_gate_pairs, bootstrap_ler_ci = _scripts_helpers()
    t0 = time.time()
    d, n_qudit = 3, 9
    conns = [[i, j] for i in range(n_qudit) for j in range(i + 1, n_qudit)]
    gp = build_gate_pairs(n_qudit, conns)
    E_full = ErrorModel(d=d, n_qudit=n_qudit, distance=3,
                        closed=True).build_grouped(verbose=False)
    corr = build_weighted_correction_set(E_full, meth_pauli_prior(), n_qudit, d,
                                         max_weight=2)
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for code, path in codes.items():
        cs, *_ = encoder_forward(os.path.join(repo, path), conns)
        print(f"encoded {code}", flush=True)
        for eta in etas:
            seed = cell_seed(eta)
            nf = make_correlated_dephasing_noise_fn(
                n_qudits=n_qudit, d=d, gate_pairs=gp, eta=eta, n_max=5,
                noise_model='physical')
            tc = time.time()
            out = simulate_ler_with_weighted_map(
                cs, nf, corr, n_qudit, d, n_shots=n_shots, seed=seed)
            ler, lo, hi = bootstrap_ler_ci(out, rng=np.random.default_rng(seed))
            print(f"  {code} b eta={eta} (n={n_shots}) map: LER={ler:.4e} "
                  f"[{lo:.4e},{hi:.4e}] ({time.time()-tc:.0f}s, "
                  f"total {time.time()-t0:.0f}s)", flush=True)
            with open(out_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                    code=code, channel='b', axis_label='eta', axis_value=eta,
                    decoder='map', ler_mean=ler, ler_lo=lo, ler_hi=hi,
                    n_shots=n_shots, seed=seed))
    print(f"done; total {time.time()-t0:.0f}s; wrote {out_path}", flush=True)
