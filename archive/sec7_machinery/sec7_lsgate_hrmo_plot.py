#!/usr/bin/env python3
"""Plot Sec VII Hrmo rerun training curves.

Reads the four .npz files in results/sec7_lsgate_hrmo/ and writes
figures/sec7_hrmo_lsgate_training_curves.png — the §VII analogue with
the genuine Hrmo LS gate. Does NOT overwrite figures/training_curves.png
(the Round-11 frozen Sec VII asset).

The user spec also names figures/sec7_hrmo_lsgate_varqec_vs_stabilizer.png,
but generating that requires LER benchmarking of the Hrmo-trained codes
(per §VI.A's lookup-table decoder + Haar-random codeword sampler), which
the Round-12 spec defers to a follow-up. This script writes only the
training-curves panel.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "sec7_lsgate_hrmo")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# (variant, n) -> filename
RUNS = {
    ("pure_ls_ring", 5): "pure_ls_ring_d3_n5_dist3_4L_best3s_seed2.npz",
    ("csum_star_ls", 5): "csum_star_ls_d3_n5_dist3_4L_best3s_seed0.npz",
    ("pure_ls_ring", 7): "pure_ls_ring_d3_n7_dist3_4L_best3s_seed0.npz",
    ("csum_star_ls", 7): "csum_star_ls_d3_n7_dist3_4L_best3s_seed0.npz",
}

PRETTY = {
    ("pure_ls_ring", 5): "pure Hrmo-LS (ring), n = 5",
    ("csum_star_ls", 5): "CSUM-star + Hrmo-LS (ring), n = 5",
    ("pure_ls_ring", 7): "pure Hrmo-LS (ring), n = 7",
    ("csum_star_ls", 7): "CSUM-star + Hrmo-LS (ring), n = 7",
}

COLORS = {
    ("pure_ls_ring", 5): "#d62728",
    ("csum_star_ls", 5): "#1f77b4",
    ("pure_ls_ring", 7): "#2ca02c",
    ("csum_star_ls", 7): "#9467bd",
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for (variant, n), path in RUNS.items():
        full = os.path.join(RESULTS, path)
        if not os.path.exists(full):
            print(f"missing {full}")
            continue
        d = np.load(full, allow_pickle=True)
        seeds = []
        for key in d.files:
            if key.startswith("seed") and key.endswith("_losses"):
                seeds.append((key, d[key]))
        if not seeds:
            continue
        ax = axes[0] if n == 5 else axes[1]
        for skey, losses in seeds:
            ax.plot(np.arange(len(losses)) + 1, losses,
                    color=COLORS[(variant, n)], alpha=0.35, lw=0.9)
        # plot best-of-3 thicker
        best = min(seeds, key=lambda s: float(np.min(s[1])))
        ax.plot(np.arange(len(best[1])) + 1, best[1],
                color=COLORS[(variant, n)], lw=1.8,
                label=PRETTY[(variant, n)])

    for ax, n in zip(axes, [5, 7]):
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_xlim(0, 1500)
        ax.set_ylim(1e-7, 30)
        ax.axhline(0.05, color="grey", linestyle=":", lw=0.6,
                   label="0.05 threshold")
        ax.grid(which="both", alpha=0.3)
        ax.set_title(f"((${n},1,3$))$_3$, 4 layers, Hrmo LS")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("KL detection loss")

    fig.suptitle("§VII rerun with the genuine Hrmo et al. 2023 LS gate "
                 "(arXiv:2206.04104, Eq. 3) — three seeds per variant",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "sec7_hrmo_lsgate_training_curves.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
