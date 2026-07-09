#!/usr/bin/env python3
"""Render the corrected-noise Fig 8 from fig8_weighted_train.py traces:
weighted detection loss vs training step (a) and vs cumulative op-EVs (b), one
curve per sampler. Reads results/best_practice_runs/fig8_weighted/<name>.npz.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "results", "best_practice_runs", "fig8_weighted")
ORDER = ["full_batch", "stratified_0.2", "importance_0.2", "importance_0.35"]
COLORS = {"full_batch": "tab:green", "stratified_0.2": "tab:blue",
          "importance_0.2": "tab:orange", "importance_0.35": "tab:red"}
LABELS = {"full_batch": "full-batch", "stratified_0.2": "stratified (20%)",
          "importance_0.2": "importance (20%)", "importance_0.35": "importance (35%)"}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for name in ORDER:
        p = os.path.join(D, f"{name}.npz")
        if not os.path.exists(p):
            print(f"skip {name}: no trace"); continue
        z = np.load(p)
        c = COLORS[name]
        ax1.plot(z["steps"], np.maximum(z["loss"], 1e-8), color=c, lw=1.6, label=LABELS[name])
        ax2.plot(np.maximum(z["op_evs"], 1), np.maximum(z["loss"], 1e-8), color=c, lw=1.6)
    ax1.set_yscale("log"); ax1.set_xlabel("training step")
    ax1.set_ylabel("weighted detection loss $\\mathcal{L}^{\\mathrm{Meth}}_{\\mathrm{KL}}$")
    ax1.set_title("(a) loss vs step"); ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.25, which="both")
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("cumulative op-EV evaluations")
    ax2.set_title("(b) loss vs hardware cost (op-EVs)"); ax2.grid(alpha=0.25, which="both")
    fig.suptitle("Fig 8 (corrected Meth noise): sampler loss vs op-EV, ((9,3,3))$_3$", fontsize=11)
    fig.tight_layout()
    out = os.path.join(D, "fig8_weighted.png")
    fig.savefig(out, dpi=200); print(f"wrote {out}")


if __name__ == "__main__":
    main()
