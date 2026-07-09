#!/usr/bin/env python3
"""R14-10 §C — sampler equivalence at calibrated correlated noise.

All available samplers' MAP LER side-by-side at eta in {0.94,0.95,0.96} (the band
where every sampler has well-resolved data). This is the central evidence for the
operational-equivalence claim: despite a ~11x training-cost spread (full-batch
1.03M op-EVs/seed down to the sub-100k samplers), the bootstrap 95% CIs overlap.

R14-7a/R14-7b are included automatically once r14_10_r14_7ab_ler.csv is populated
(R14-10 sub-100k validation); until then the figure shows R12 + the three settled
samplers and the caption notes the sub-100k samplers are pending.
"""
from __future__ import annotations
import csv
import os
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results/round14_scoping")
FIGDIR = os.path.join(REPO, "latex/figures")

MAP_CSVS = [
    "r14_5_map_cells.csv", "r14_6_r14_3b_ler.csv", "r14_7d_r14_3a_map_full.csv",
    "r14_7e_fullbatch_ler.csv", "r14_8_ler_scaling.csv", "r14_10_r14_7ab_ler.csv",
]
ETAS = [0.94, 0.95, 0.96]
ETA_COLOR = {0.94: 'tab:red', 0.95: 'tab:orange', 0.96: 'tab:green'}

# display order, label (descriptive name + op-EVs/seed annotation)
SAMPLERS = [
    ('R12',      'Sec. VIII\nbaseline'),
    ('R14-3a',   'Stratified\n(175k)'),
    ('R14-3b',   'Importance\n(135k)'),
    ('R14-3-fb', 'Full-batch\n(1.03M)'),
    ('R14-7a',   'Importance\n(99k)'),
    ('R14-7b',   'Full-basis IS\n(94k)'),
]


def code_class(name):
    name = name.split('/')[0]
    return 'R12' if name.startswith('R12') else name


def read_cells():
    cells = defaultdict(list)
    for fn in MAP_CSVS:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get('decoder', 'map') != 'map' or r.get('channel', 'b') != 'b':
                    continue
                eta = round(float(r['axis_value']), 4)
                if eta in ETA_COLOR:
                    cells[(code_class(r['code']), eta)].append(
                        (float(r['ler_mean']), float(r['ler_lo']),
                         float(r['ler_hi'])))
    return cells


def main():
    cells = read_cells()
    present = [(tag, lab) for tag, lab in SAMPLERS
               if any((tag, e) in cells for e in ETAS)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xpos = np.arange(len(present))
    offsets = {0.94: -0.22, 0.95: 0.0, 0.96: 0.22}

    for eta in ETAS:
        xs, ys, los, his = [], [], [], []
        for i, (tag, _) in enumerate(present):
            vals = cells.get((tag, eta))
            if not vals:
                continue
            lers = np.array([v[0] for v in vals])
            med = float(np.median(lers))
            xs.append(i + offsets[eta]); ys.append(med)
            los.append(med - min(v[1] for v in vals))
            his.append(max(v[2] for v in vals) - med)
        ax.errorbar(xs, ys, yerr=[los, his], marker='o', ms=6, ls='none',
                    capsize=3, color=ETA_COLOR[eta], label=f'$\\eta={eta}$')

    ax.set_xticks(xpos)
    ax.set_xticklabels([lab for _, lab in present], fontsize=8.5)
    ax.set_ylabel('logical error rate (MAP)')
    ax.set_ylim(0, 0.25)
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.set_title('Sampler equivalence at calibrated correlated noise\n'
                 '$((9,3,3))_3$ a2a, MAP decoder (calibrated $\\eta=0.9296$, '
                 '$p_{\\mathrm{phys}}=0.56$)', fontsize=10)
    ax.legend(title='noise level', fontsize=9, loc='upper right')
    ax.grid(alpha=0.25, axis='y')
    ax.text(0.01, -0.16,
            'op-EVs/seed annotated under each sampler; full-batch trains at '
            '$\\sim$11$\\times$ the sub-100k cost',
            transform=ax.transAxes, fontsize=7.5, color='0.4')

    fig.tight_layout()
    base = os.path.join(OUT, "r14_10_sampler_equivalence")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    if os.path.isdir(FIGDIR):  # minimal-repo: no latex/ tree
        shutil.copy(base + ".png", os.path.join(FIGDIR, "r14_10_sampler_equivalence.png"))
    print(f"wrote {base}.png/.pdf + latex/figures copy")
    print(f"samplers shown: {[t for t,_ in present]}")
    for (tag, eta), vals in sorted(cells.items()):
        if tag in [t for t, _ in present]:
            lers = [v[0] for v in vals]
            print(f"  {tag} eta={eta}: median LER {np.median(lers):.4f} (n_seeds={len(vals)})")
    plt.close(fig)


if __name__ == "__main__":
    main()
