#!/usr/bin/env python3
"""
Generate LER comparison plots from saved benchmark results.

Usage:
  python3 scripts/plot_ler_comparison.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "simulations")
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_ler(code_name, noise="hardware"):
    path = os.path.join(SIM_DIR, f"ler_{code_name}_{noise}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def plot_distance3_comparison():
    """Plot 2: Distance-3 correction — the thesis figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    styles = {
        'qutrit_d3': {'color': '#E53935', 'marker': 'o', 'label': '((5,3,3))₃ VarQEC native'},
        'five_qudit_d3': {'color': '#1E88E5', 'marker': 's', 'label': '[[5,1,3]]₃ stabilizer'},
    }

    for code_name, style in styles.items():
        data = load_ler(code_name)
        if data is None:
            print(f"  Skipping {code_name} (not benchmarked yet)")
            continue
        p = data['p_rates']
        ler = data['lers']
        ax.semilogy(p, np.maximum(ler, 1e-5), style['marker'] + '-',
                    color=style['color'], label=style['label'],
                    markersize=8, linewidth=2)

    # Reference line: LER = p (no coding)
    p_ref = np.linspace(0.001, 0.25, 100)
    ax.semilogy(p_ref, p_ref, 'k--', alpha=0.5, label='No coding (LER = p)')

    ax.set_xlabel('Physical error rate p', fontsize=13)
    ax.set_ylabel('Logical error rate', fontsize=13)
    ax.set_title('Distance-3 QEC: VarQEC native gates vs stabilizer code', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.22)
    ax.set_ylim(1e-5, 1)
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOT_DIR, "ler_distance3_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


def plot_distance2_comparison():
    """Plot 1: Distance-2 detection comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    styles = {
        'qutrit_d2': {'color': '#E53935', 'marker': 'o', 'label': '((3,3,2))₃ qutrit'},
        'ququart_d2': {'color': '#1E88E5', 'marker': 's', 'label': '((3,4,2))₄ ququart'},
    }

    for code_name, style in styles.items():
        data = load_ler(code_name)
        if data is None:
            continue
        p = data['p_rates']
        ax.plot(p, data['undet_errs'], style['marker'] + '-',
                color=style['color'], label=style['label'] + ' (undetected)',
                markersize=8, linewidth=2)

    p_ref = np.linspace(0.001, 0.35, 100)
    ax.plot(p_ref, p_ref, 'k--', alpha=0.5, label='No coding')

    ax.set_xlabel('Physical error rate p', fontsize=13)
    ax.set_ylabel('Undetected error rate', fontsize=13)
    ax.set_title('Distance-2 error detection under hardware noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOT_DIR, "ler_distance2_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


def plot_fidelity_comparison():
    """Plot 3: Corrected fidelity comparison for distance-3 codes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    styles = {
        'qutrit_d3': {'color': '#E53935', 'label': 'VarQEC native'},
        'five_qudit_d3': {'color': '#1E88E5', 'label': 'Stabilizer [[5,1,3]]₃'},
    }

    for code_name, style in styles.items():
        data = load_ler(code_name)
        if data is None:
            continue
        p = data['p_rates']
        ax.plot(p, data['fid_corrs'], 'o-', color=style['color'],
                label=style['label'] + ' (corrected)', markersize=8, linewidth=2)
        ax.plot(p, data['fid_raws'], 's--', color=style['color'],
                label=style['label'] + ' (raw)', markersize=6, linewidth=1, alpha=0.5)

    ax.set_xlabel('Physical error rate p', fontsize=13)
    ax.set_ylabel('Mean fidelity', fontsize=13)
    ax.set_title('Fidelity under hardware noise: raw vs corrected', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.02)
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOT_DIR, "fidelity_distance3_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating LER comparison plots...")
    plot_distance3_comparison()
    plot_distance2_comparison()
    plot_fidelity_comparison()
    print("Done.")
