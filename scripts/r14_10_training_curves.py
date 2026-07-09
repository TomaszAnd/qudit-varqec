#!/usr/bin/env python3
"""R14-10 §A — training curves: loss vs step AND loss vs op-EVs.

Loss-vs-step alone misleads (all samplers converge by ~step 600); panel (b)
reframes against cumulative hardware cost (cumulative operator-expectation-value
evaluations), which is the actual sampling-budget claim. Full-batch shifts right
(it pays ~684 op-EVs/step), the smart samplers shift left (63--117/step).

R12 is intentionally excluded from the loss axis: its stored objective plateaus
near 1.6e-2 (a different normalization from the KL detection loss used by the
R14 campaign, which reaches ~2e-5), so its absolute loss is not comparable. R12
appears instead in the LER figures, where it is a meaningful operational baseline.

Renders to results/round14_scoping/ and copies the PNG into latex/figures/.
"""
from __future__ import annotations
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results/round14_scoping")
FIGDIR = os.path.join(REPO, "latex/figures")

# op-EVs per training step (cross-checked against npz weights_meta / logs).
OP_EVS_PER_STEP = {
    'R14-3-fb': 684,   # full-batch over weight-<=2 closure
    'R14-3a':   117,   # stratified-10
    'R14-3b':    90,   # stratified-IS
    'R14-7a':    90,   # = R14-3b per-step, fewer steps (1100)
    'R14-7b':    63,   # full-basis IS, realized
}

SAMPLERS = {
    'R14-3a':    ('results/round14_scoping/r14_3a_meth_physical',          'tab:blue',   'Stratified'),
    'R14-3b':    ('results/round14_scoping/r14_3b_meth_physical_strat_is', 'tab:orange', 'Importance'),
    'R14-3-fb':  ('results/round14_scoping/r14_3_fullbatch_meth',          'tab:green',  'Full-batch'),
    'R14-7a':    ('results/round14_scoping/r14_7a_hoeffding',              'tab:purple', 'Importance (99k)'),
    'R14-7b':    ('results/round14_scoping/r14_7b_full_is',                'tab:brown',  'Full-basis IS'),
}
SEEDS = [0, 1, 2]
NPZ = "d3_n9_dist3_4L_meth_physical_seed{s}.npz"


def load_traces(subdir):
    traces = []
    for s in SEEDS:
        p = os.path.join(REPO, subdir, NPZ.format(s=s))
        if os.path.exists(p):
            traces.append(np.asarray(np.load(p, allow_pickle=True)['loss_trace'], float))
    return traces


def median_envelope(traces):
    """Median, min, max across seeds, clipped to the shortest trace length."""
    L = min(len(t) for t in traces)
    M = np.stack([t[:L] for t in traces])
    return np.arange(L), np.median(M, 0), M.min(0), M.max(0)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for tag, (subdir, color, label) in SAMPLERS.items():
        traces = load_traces(subdir)
        if not traces:
            print(f"  skip {tag}: no traces")
            continue
        step, med, lo, hi = median_envelope(traces)
        opev = step * OP_EVS_PER_STEP[tag]
        lbl = label
        ax1.plot(step, med, color=color, lw=1.6, label=lbl)
        ax1.fill_between(step, lo, hi, color=color, alpha=0.15, lw=0)
        # op-EV axis: drop step 0 so the log-x start is finite
        ax2.plot(opev[1:], med[1:], color=color, lw=1.6)
        ax2.fill_between(opev[1:], lo[1:], hi[1:], color=color, alpha=0.15, lw=0)
        print(f"  {tag}: {len(step)} steps, min loss {med.min():.3e}, "
              f"final op-EVs {opev[-1]:.3g}")

    # Panel (a): loss vs step
    ax1.axvline(600, color='0.4', ls='--', lw=1.0)
    ax1.text(600, 1.3e-4, ' loss-saturation\n regime', fontsize=8, color='0.3',
             va='top', ha='left')
    ax1.set_yscale('log')
    ax1.set_xlim(0, 1500)
    ax1.set_xlabel('training step')
    ax1.set_ylabel('detection loss  $\\mathcal{L}_{\\mathrm{KL}}$')
    ax1.set_title('(a) loss vs step')
    ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax1.grid(alpha=0.25, which='both')

    # Panel (b): loss vs cumulative op-EVs
    ax2.set_xscale('log')
    ax2.set_xlim(1e3, 2e6)
    for x, lab in [(1e5, '100k'), (1e6, '1M')]:
        ax2.axvline(x, color='0.6', ls=':', lw=1.0)
        ax2.text(x, 1.05e-4, f' {lab}', fontsize=8, color='0.4',
                 rotation=90, va='top', ha='left')
    ax2.set_xlabel('cumulative op-EV evaluations')
    ax2.set_title('(b) loss vs hardware cost (op-EVs)')
    ax2.grid(alpha=0.25, which='both')

    fig.tight_layout()
    base = os.path.join(OUT, "r14_10_training_curves")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    if os.path.isdir(FIGDIR):  # minimal-repo: no latex/ tree
        shutil.copy(base + ".png", os.path.join(FIGDIR, "r14_10_training_curves.png"))
    print(f"wrote {base}.png/.pdf + latex/figures copy")
    plt.close(fig)


if __name__ == "__main__":
    main()
