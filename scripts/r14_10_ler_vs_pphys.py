#!/usr/bin/env python3
"""R14-10 §B — LER vs per-gate p_phys, redesigned (replaces the cluttered plot).

Fixes vs the broken R14-9 plot:
  * 0-error cells are plotted as Poisson 95% upper bounds (3/n_shots) with a
    downward-triangle marker, NOT connected, and EXCLUDED from every fit. The
    broken slope-9.18 fit was inflated by treating those cells as measurements.
  * slope is fit on MEASURED R14-3b MAP points only (np.polyfit on log10), with
    a quoted uncertainty -- three points is barely enough.
  * colour = code class, marker = decoder (MAP square, lookup circle); no
    legend / annotation overlap.

x-axis is the correlated-noise per-gate physical error rate p_phys = 1 - F_avg
(from r14_7g_eta_to_pphys.csv), distinct from the campaign's i.i.d. per-qudit p.
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

# MAP-decoder LER CSVs (channel b). Standard schema with a 'decoder' column.
MAP_CSVS = [
    "r14_5_map_cells.csv",
    "r14_6_r14_3b_ler.csv",
    "r14_7d_r14_3a_map_full.csv",
    "r14_7e_fullbatch_ler.csv",
    "r14_8_ler_scaling.csv",
    "r14_10_r14_7ab_ler.csv",   # optional (R14-10 sub-100k validation)
]

PALETTE = {
    'R12':      'gray',
    'R14-3a':   'tab:blue',
    'R14-3b':   'tab:orange',
    'R14-3-fb': 'tab:green',
    'R14-7a':   'tab:purple',
    'R14-7b':   'tab:brown',
}

# reader-facing legend names (internal class keys -> descriptive sampler names)
DISPLAY = {
    'R12':      'Sec. VIII baseline',
    'R14-3a':   'Stratified',
    'R14-3b':   'Importance',
    'R14-3-fb': 'Full-batch',
    'R14-7a':   'Importance (99k)',
    'R14-7b':   'Full-basis IS',
}


def code_class(name):
    name = name.split('/')[0]
    return 'R12' if name.startswith('R12') else name


def load_eta_to_pphys():
    m = {}
    with open(os.path.join(OUT, "r14_7g_eta_to_pphys.csv")) as f:
        for r in csv.DictReader(f):
            m[round(float(r['eta']), 4)] = float(r['p_phys_per_gate'])
    return m


def read_map_cells():
    """Return {(cls, eta): list of (ler, lo, hi, n_shots)} for MAP, channel b."""
    cells = defaultdict(list)
    for fn in MAP_CSVS:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get('decoder', 'map') != 'map':
                    continue
                if r.get('channel', 'b') != 'b':
                    continue
                eta = round(float(r['axis_value']), 4)
                cells[(code_class(r['code']), eta)].append(
                    (float(r['ler_mean']), float(r['ler_lo']),
                     float(r['ler_hi']), int(float(r['n_shots']))))
    return cells


def read_lookup_cells():
    """Lookup-decoder LER from the head-to-head CSV (b_kraus_phase rows)."""
    cells = defaultdict(list)
    p = os.path.join(OUT, "r14_4_head_to_head.csv")
    with open(p) as f:
        for r in csv.DictReader(f):
            if r['channel'] != 'b_kraus_phase':
                continue
            eta = round(float(r['axis_value']), 4)
            cells[(code_class(r['code']), eta)].append(
                (float(r['ler_mean']), float(r['ler_lo']),
                 float(r['ler_hi']), int(float(r['n_shots']))))
    return cells


def aggregate(cells, eta2p):
    """Per (cls, eta): median LER, CI envelope, p_phys, measured flag."""
    rows = []
    for (cls, eta), vals in cells.items():
        if eta not in eta2p:
            continue
        lers = np.array([v[0] for v in vals])
        med = float(np.median(lers))
        p = eta2p[eta]
        if med > 0:
            lo = min(v[1] for v in vals)
            hi = max(v[2] for v in vals)
            rows.append(dict(cls=cls, eta=eta, p=p, ler=med, lo=lo, hi=hi,
                             measured=True))
        else:
            n = max(v[3] for v in vals)        # tightest (largest-n) bound
            rows.append(dict(cls=cls, eta=eta, p=p, ler=3.0 / n, lo=0, hi=0,
                             measured=False))
    return rows


def main():
    eta2p = load_eta_to_pphys()
    map_rows = aggregate(read_map_cells(), eta2p)
    lookup_rows = aggregate(read_lookup_cells(), eta2p)

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- MAP: solid squares, connected within class (measured only) ---
    classes = sorted({r['cls'] for r in map_rows},
                     key=lambda c: list(PALETTE).index(c) if c in PALETTE else 99)
    for cls in classes:
        col = PALETTE.get(cls, 'k')
        meas = sorted([r for r in map_rows if r['cls'] == cls and r['measured']],
                      key=lambda r: r['p'])
        ub = [r for r in map_rows if r['cls'] == cls and not r['measured']]
        if meas:
            xs = [r['p'] for r in meas]
            ys = [r['ler'] for r in meas]
            yerr = [[y - r['lo'] for y, r in zip(ys, meas)],
                    [r['hi'] - y for y, r in zip(ys, meas)]]
            ax.errorbar(xs, ys, yerr=yerr, marker='s', ms=6, lw=1.3, capsize=2,
                        color=col, label=DISPLAY.get(cls, cls))
        for r in ub:
            ax.plot(r['p'], r['ler'], marker='v', ms=8, color=col,
                    mfc='none', mew=1.4, ls='none')

    # --- lookup: open circles, faint, no fit (decoder-gap reference) ---
    for cls in sorted({r['cls'] for r in lookup_rows}):
        col = PALETTE.get(cls, 'k')
        meas = sorted([r for r in lookup_rows if r['cls'] == cls and r['measured']],
                      key=lambda r: r['p'])
        if meas:
            ax.plot([r['p'] for r in meas], [r['ler'] for r in meas],
                    marker='o', ms=5, mfc='none', color=col, alpha=0.45,
                    lw=0.9, ls=':')

    # --- slope fit: R14-3b MAP, WELL-RESOLVED high-noise band only ---
    # Restrict to eta in {0.94,0.95,0.96} (n_shots=2000, hundreds of logical
    # errors each). The eta=0.97 cell is measured but under-resolved (~7 errors
    # in 5000 shots); including it steepens the apparent slope to ~9 -- reported
    # separately for transparency, not used for the headline fit. eta>=0.98 are
    # 0-error upper bounds and are excluded from every fit.
    FIT_ETAS = {0.94, 0.95, 0.96}

    def fit_slope(etas):
        pts = sorted([r for r in map_rows if r['cls'] == 'R14-3b'
                      and r['measured'] and r['eta'] in etas],
                     key=lambda r: r['p'])
        if len(pts) < 2:
            return None
        lp = np.log10([r['p'] for r in pts])
        ll = np.log10([r['ler'] for r in pts])
        coef, cov = np.polyfit(lp, ll, 1, cov=True)
        return coef[0], coef[1], float(np.sqrt(cov[0, 0])), lp, ll, pts

    slope_txt = ""
    res = fit_slope(FIT_ETAS)
    if res:
        slope, intc, serr, lp, ll, pts = res
        xx = np.array([min(lp), max(lp)])
        ax.plot(10 ** xx, 10 ** (intc + slope * xx), color=PALETTE['R14-3b'],
                lw=1.0, ls='--', alpha=0.7)
        slope_txt = f"Importance MAP fit: slope ${slope:.1f}\\pm{serr:.1f}$ (3 resolved pts)"
        print(f"R14-3b MAP resolved-band slope = {slope:.2f} +/- {serr:.2f} "
              f"(p {10**lp.min():.2f}-{10**lp.max():.2f}, eta 0.94-0.96)")
        # transparency: slope including the under-resolved eta=0.97 cell
        res2 = fit_slope(FIT_ETAS | {0.97})
        if res2:
            print(f"  [with under-resolved eta=0.97 (~7 errors): slope "
                  f"{res2[0]:.2f} +/- {res2[2]:.2f}]")

    # --- reference lines ---
    pp = np.array([1e-2, 1.0])
    ax.plot(pp, 0.01 * (pp / 0.10) ** 2, ls=':', color='0.6', lw=1.0,
            label='distance-3 ideal (slope 2)')
    ax.plot([1e-2, 1.0], [1e-2, 1.0], ls='--', color='0.75', lw=1.0,
            label='break-even')

    # --- calibrated annotation (top-left, away from legend at lower-right) ---
    ax.axvline(0.56, color='0.5', ls='-', lw=0.8, alpha=0.5)
    # annotation in the (empty) upper-left; legend sits lower-right, data points
    # sit at LER <= 0.2, so no overlap after the 1e-1..1e0 rescale.
    ax.text(0.105, 0.5, "Operating point\n$\\eta=0.9296$,\n$p_{\\mathrm{phys}}=0.56$",
            fontsize=8, va='center', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    ax.set_xscale('log'); ax.set_yscale('log')
    # x-axis restricted to the measured high-noise band + calibrated point. The
    # eta=0.98,0.99 0-error upper bounds (p_phys=0.19,0.105) sit at the left edge;
    # the deepest-noise eta>=0.995 bounds (p_phys<=0.04) fall off-range by design
    # (deferred to higher-shot follow-up, see caption / Sec. X A).
    ax.set_xlim(1e-1, 1e0); ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel('per-gate physical error rate  $p_{\\mathrm{phys}} = 1 - F_{\\mathrm{avg}}$')
    ax.set_ylabel('logical error rate (MAP)')
    title = 'LER vs correlated-noise $p_{\\mathrm{phys}}$ (MAP decoder)'
    if slope_txt:
        title += f'\n{slope_txt}'
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which='both')

    # legend: classes + refs; decoder shapes explained in caption
    handles, labels = ax.get_legend_handles_labels()
    # add a manual upper-bound entry
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], marker='v', mfc='none', color='0.4',
                          mew=1.4, ls='none'))
    labels.append('0-error (95% upper bnd, 3/n)')
    ax.legend(handles, labels, fontsize=7.5, loc='lower right', framealpha=0.9,
              ncol=1)

    fig.tight_layout()
    base = os.path.join(OUT, "r14_10_ler_vs_pphys")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    if os.path.isdir(FIGDIR):  # minimal-repo: no latex/ tree
        shutil.copy(base + ".png", os.path.join(FIGDIR, "r14_10_ler_vs_pphys.png"))
    print(f"wrote {base}.png/.pdf + latex/figures copy")
    plt.close(fig)


if __name__ == "__main__":
    main()
