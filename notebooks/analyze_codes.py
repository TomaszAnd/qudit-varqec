# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # VarQEC Code Analysis
#
# Comprehensive analysis of all trained VarQEC codes: training curves,
# characterization (KL residuals, weight enumerators, entanglement),
# logical error rates, and comparison to the [[5,1,3]]_{Z_3} stabilizer benchmark.

# %%
import os, sys, glob
sys.path.insert(0, '..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.catalog import load_code, list_codes, five_qudit_code_states, _CODE_REGISTRY
from src.analysis import (
    compute_kl_residuals, compute_weight_enumerators, compute_entanglement_entropy,
)
from src.errors import qudit_hardware_error_basis, ErrorModel, make_hardware_noise_fn
from src.simulation import (
    simulate_ler_with_detection, simulate_ler_with_correction,
    simulate_ler_with_correction_factored,
)
from src.errors import _embed_single_qudit

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11

PLOT_DIR = '../results/plots'
SIM_DIR = '../results/simulations'
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Unified color scheme for every plot in this notebook ──────────────
# Hue encodes the (d, distance) family; lightness encodes n (darker = larger n).
# Flagship d=3 distance-3 family uses viridis so it reads cleanly in greyscale
# printing. Stabilizer and no-coding baselines have dedicated achromatic styles.

_FAMILY_CMAP = {
    (3, 2): 'Reds',
    (3, 3): 'viridis',
    (4, 2): 'Oranges',
    (4, 3): 'Purples',
    (5, 2): 'Greens',
    (5, 3): 'BuPu',
}
_FAMILY_N_RANGE = {
    (3, 2): (3, 8),
    (3, 3): (5, 9),
    (4, 2): (3, 5),
    (4, 3): (5, 6),
    (5, 2): (3, 5),
    (5, 3): (5, 5),
}


def code_color(d, distance, n):
    """Consistent color for a code across every plot in this notebook.

    Hue = (d, distance) family. Lightness = n (darker = larger n).
    Low-n end of sequential colormaps starts at ~0.35 so pale colors
    remain visible on white; viridis is truncated to 0.2..0.9.
    """
    key = (d, distance)
    cmap_name = _FAMILY_CMAP.get(key, 'Greys')
    n_lo, n_hi = _FAMILY_N_RANGE.get(key, (n, n))
    if n_hi == n_lo:
        t = 0.7
    elif cmap_name == 'viridis':
        t = 0.2 + 0.7 * (n - n_lo) / (n_hi - n_lo)
    else:
        t = 0.35 + 0.5 * (n - n_lo) / (n_hi - n_lo)
    return plt.get_cmap(cmap_name)(t)


STABILIZER_COLOR = 'black'    # [[5,1,3]]_{Z3}, dashed
NO_CODING_COLOR = 'grey'      # dotted baseline


# Wilson 95% confidence interval for binomial rates. With 3000 shots and
# rates near zero, the difference between two LER values can be statistical
# noise; LER plots show these intervals as shaded bands or error bars.
def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial rate k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def ler_with_ci(rate, n_shots):
    """Returns (rate, lo, hi). For rate=0 the interval is one-sided."""
    k = int(round(rate * n_shots))
    lo, hi = wilson_ci(k, n_shots)
    return rate, lo, hi
os.makedirs(SIM_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Summary table

# %%
# Load every code
summary = []
for name in sorted(list_codes()):
    try:
        d = load_code(name)
        m = d['metadata']
        npz_file, _ = _CODE_REGISTRY.get(name, (None, {}))
        losses = None
        if npz_file:
            path = os.path.join('..', 'results', 'params', npz_file)
            if os.path.exists(path):
                losses = list(np.load(path, allow_pickle=True).get('losses', []))
        # Prefer min(losses) (best during training) over final_loss (last step);
        # Adam can jitter past the minimum so final_loss > min(losses).
        if losses:
            loss_val = float(min(losses))
        else:
            loss_val = float(m.get('final_loss', 0))
        summary.append({
            'name': name, 'd': m['d'], 'n': m['n_qudit'], 'K': m['K'],
            'distance': m['distance'], 'layers': m.get('n_layers', 0),
            'loss': loss_val,
            'code_states': d['code_states'], 'losses': losses,
            'abstract': m.get('abstract_gates', False),
            'analytical': m.get('type', '') == 'stabilizer',
        })
    except Exception as e:
        print(f"Skipped {name}: {e}")

print(f"{'Name':25} {'type':>6} {'d':>3} {'n':>3} {'dist':>5} "
      f"{'L':>3} {'loss':>12}")
print("-" * 68)
for s in summary:
    typ = 'stab' if s['analytical'] else ('abs' if s['abstract'] else 'nat')
    print(f"{s['name']:25} {typ:>6} {s['d']:>3} {s['n']:>3} "
          f"{s['distance']:>5} {s['layers']:>3} {s['loss']:>12.4e}")

# %% [markdown]
# ## 2. Training curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Deduplicate by (d, n, distance): keep the lowest-loss representative.
# This collapses e.g. qutrit_d2 and d3_n3_dist2 (same (3,3,2)) to a single curve.
_best_per_cell = {}
for s in summary:
    if s['abstract'] or s['analytical'] or s['losses'] is None:
        continue
    key = (s['d'], s['n'], s['distance'])
    cur = _best_per_cell.get(key)
    if cur is None or s['loss'] < cur['loss']:
        _best_per_cell[key] = s
_curves_codes = sorted(_best_per_cell.values(),
                       key=lambda s: (s['distance'], s['d'], s['n']))

for s in _curves_codes:
    ax = axes[0] if s['distance'] == 2 else axes[1]
    label = f"(({s['n']},{s['d']},{s['distance']}))_{s['d']}"
    ax.semilogy(s['losses'], color=code_color(s['d'], s['distance'], s['n']),
                alpha=0.85, linewidth=1.3, label=label)

for ax, title in zip(axes, ['Distance 2 (detection)', 'Distance 3 (correction)']):
    ax.set_xlabel('Training step')
    ax.set_ylabel('KL loss')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/training_curves.png', dpi=150)
print("Saved training_curves.png")

# %% [markdown]
# ## 3. Code characterization

# %%
characterization = []
for s in summary:
    if s['abstract'] or s['analytical']:
        continue
    d_val, n, dist = s['d'], s['n'], s['distance']
    result = {'name': s['name']}
    cs = s['code_states']

    # KL residuals (skip large systems)
    if d_val**n <= 2000:
        model = ErrorModel(d=d_val, n_qudit=n, distance=dist, closed=dist >= 3)
        _, E_corr = model.build_dense()
        off, var = compute_kl_residuals(cs, E_corr)
        result['kl_off'] = off
        result['kl_var'] = var
    else:
        result['kl_off'] = result['kl_var'] = None

    # Entanglement entropy
    if n >= 2:
        S = compute_entanglement_entropy(cs, list(range(n // 2)), n, d_val)
        result['S_mean'] = float(np.mean(S))
    else:
        result['S_mean'] = 0
    characterization.append(result)

print(f"{'Name':25} {'KL_off':>10} {'KL_var':>10} {'S_mean':>7}")
print("-" * 55)
for c in characterization:
    off = f"{c['kl_off']:.2e}" if c['kl_off'] is not None else '-'
    var = f"{c['kl_var']:.2e}" if c['kl_var'] is not None else '-'
    print(f"{c['name']:25} {off:>10} {var:>10} {c.get('S_mean', 0):>7.2f}")

# %% [markdown]
# ## 4. Logical error rates

# %%
p_rates = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
N_SHOTS = 3000

def benchmark_code(s, force=False):
    name = s['name']
    sim_path = os.path.join(SIM_DIR, f"ler_{name}_hardware.npz")
    if os.path.exists(sim_path) and not force:
        return dict(np.load(sim_path, allow_pickle=True))

    d_val, n, dist = s['d'], s['n'], s['distance']
    cs = s['code_states']
    lers, fr, fc = [], [], []

    for p in p_rates:
        noise = make_hardware_noise_fn(d_val, n, p)
        if dist == 2:
            r = simulate_ler_with_detection(cs, noise, N_SHOTS, seed=42)
            lers.append(r['undetected_error_rate'])
            fr.append(r['mean_raw_fidelity'])
            fc.append(r['post_selected_fidelity'])
        elif n >= 7:
            single = qudit_hardware_error_basis(d_val)
            r = simulate_ler_with_correction_factored(
                cs, noise, single, n, d_val, n_shots=N_SHOTS, seed=42)
            lers.append(r['logical_error_rate'])
            fr.append(r['mean_raw_fidelity'])
            fc.append(r['mean_fidelity'])
        else:
            E_corr_ops = [_embed_single_qudit(E, q, n, d_val)
                          for q in range(n)
                          for E in qudit_hardware_error_basis(d_val)]
            r = simulate_ler_with_correction(cs, noise, E_corr_ops, N_SHOTS, seed=42)
            lers.append(r['logical_error_rate'])
            fr.append(r['mean_raw_fidelity'])
            fc.append(r['mean_fidelity'])

    data = {'p_rates': p_rates, 'lers': np.array(lers),
            'fids_raw': np.array(fr), 'fids_corr': np.array(fc)}
    np.savez(sim_path, **data)
    return data

ler_results = {}
for s in summary:
    if s['abstract']:
        continue
    try:
        print(f"  {s['name']}...")
        ler_results[s['name']] = benchmark_code(s)
    except Exception as e:
        print(f"  FAILED: {e}")

# %% [markdown]
# ## 5. Money plots

# %% [markdown]
# ### Figure 1: ((n,3,3))$_3$ family — training loss scaling (left) and logical
# error rate vs the [[5,1,3]] stabilizer benchmark (right)
#
# Merged 2-panel (PDF Fig 4 + Fig 5): left = training loss $\propto n^{-2.54}$;
# right = LER vs $p$ for the qutrit family with the [[5,1,3]]$_{Z_3}$ stabilizer
# and the no-coding baseline overlaid. Replaces the former standalone
# `varqec_vs_stabilizer.png`.

# %%
# Stabilizer benchmark [[5,1,3]]_{Z_3}, computed once for the right panel here
# (and reused by the cross-code figure further down).
bench_name = 'five_qudit_d3'
if bench_name not in ler_results:
    bench = load_code(bench_name)
    single = qudit_hardware_error_basis(3)
    bench_lers, bench_fr, bench_fc = [], [], []
    for p in p_rates:
        noise = make_hardware_noise_fn(3, 5, p)
        r = simulate_ler_with_correction_factored(
            bench['code_states'], noise, single, 5, 3, n_shots=N_SHOTS, seed=42)
        bench_lers.append(r['logical_error_rate'])
        bench_fr.append(r['mean_raw_fidelity'])
        bench_fc.append(r['mean_fidelity'])
    ler_results[bench_name] = {'p_rates': p_rates, 'lers': np.array(bench_lers),
                                'fids_raw': np.array(bench_fr), 'fids_corr': np.array(bench_fc)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Loss vs n
n_d3, loss_d3 = [], []
for s in summary:
    if s['d'] == 3 and s['distance'] == 3 and not s['abstract'] and not s['analytical']:
        n_d3.append(s['n'])
        loss_d3.append(s['loss'])
order = np.argsort(n_d3)
n_d3 = np.array(n_d3)[order]
loss_d3 = np.array(loss_d3)[order]

# Color each scatter point by its own n so cross-panel consistency holds:
# the dot at n=k here matches the n=k curve on the right panel exactly.
for n_val, loss_val in zip(n_d3, loss_d3):
    axes[0].plot(n_val, loss_val, 'o',
                 color=code_color(3, 3, int(n_val)), markersize=12,
                 markeredgecolor='black', markeredgewidth=0.5)
if len(n_d3) >= 2:
    slope, intercept = np.polyfit(np.log(n_d3), np.log(loss_d3), 1)
    n_fit = np.linspace(n_d3.min(), n_d3.max(), 50)
    axes[0].loglog(n_fit, np.exp(intercept) * n_fit**slope, '--',
                   color=NO_CODING_COLOR, alpha=0.7,
                   label=f'loss $\\propto n^{{{slope:.1f}}}$')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
for ni, li in zip(n_d3, loss_d3):
    axes[0].annotate(f'{li:.3f}', (ni, li), textcoords='offset points',
                     xytext=(8, 5), fontsize=10)
axes[0].set_xlabel('Number of qutrits $n$')
axes[0].set_ylabel('Training loss')
axes[0].set_title('n-scaling of ((n,3,3))$_3$ training loss')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# RIGHT: LER vs p — ((n,3,3))_3 family + [[5,1,3]] benchmark + no-coding baseline
p_ref = np.linspace(0.001, 0.25, 100)
axes[1].plot(p_ref, p_ref, ':', color=NO_CODING_COLOR, alpha=0.7,
             label='No coding')
for ni in n_d3:
    names = [s['name'] for s in summary
             if s['d'] == 3 and s['n'] == ni and s['distance'] == 3 and not s['abstract']]
    if not names or names[0] not in ler_results:
        continue
    r = ler_results[names[0]]
    axes[1].semilogy(r['p_rates'], np.maximum(r['lers'], 1e-5), 'o-',
                     color=code_color(3, 3, int(ni)), markersize=8,
                     label=f'VarQEC (({ni},3,3))$_3$')

# [[5,1,3]]_{Z_3} stabilizer benchmark overlay
if bench_name in ler_results:
    r = ler_results[bench_name]
    axes[1].semilogy(r['p_rates'], np.maximum(r['lers'], 1e-5), 'x--',
                     color=STABILIZER_COLOR, label='[[5,1,3]]$_{Z_3}$ stabilizer',
                     markersize=10, linewidth=1.8)

axes[1].set_xlabel('Physical error rate $p$')
axes[1].set_ylabel('Logical error rate')
axes[1].set_title('((n,3,3))$_3$ LER vs [[5,1,3]] benchmark')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].set_ylim(1e-5, 1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/n_scaling_d3_dist3.png', dpi=150)
print("Saved n_scaling_d3_dist3.png (2-panel: loss scaling + LER vs [[5,1,3]])")

# %% [markdown]
# ## 6. Cross-code distance-3 LER
#
# LER vs p across every distance-3 code, with Wilson 95% confidence intervals
# on the 3000-shot estimates (the difference between two near-zero LERs is
# dominated by statistical noise without explicit uncertainty).

# %% [markdown]
# ### Figure 2: LER vs p across all distance-3 codes (Wilson 95% CI)

# %%
fig, ax = plt.subplots(figsize=(10, 6.5))

_dist3_rows = sorted(
    (s for s in summary
     if not s['analytical'] and not s['abstract'] and s['distance'] == 3),
    key=lambda s: (s['d'], s['n']))
# dedupe by (d, n, distance), keeping lowest-loss representative
_seen_cells = {}
for s in _dist3_rows:
    key = (s['d'], s['n'], s['distance'])
    cur = _seen_cells.get(key)
    if cur is None or s['loss'] < cur['loss']:
        _seen_cells[key] = s
_dist3_unique = sorted(_seen_cells.values(), key=lambda s: (s['d'], s['n']))

for s in _dist3_unique:
    name = s['name']
    if name not in ler_results:
        continue
    r = ler_results[name]
    ps = np.asarray(r['p_rates'])
    rates = np.asarray(r['lers'])
    color = code_color(s['d'], s['distance'], s['n'])
    label = f"(({s['n']},{s['d']},{s['distance']}))$_{s['d']}$"

    nz = rates > 0
    if not nz.any():
        # All zero across the tested p range — annotate in the legend instead
        # of stacking 8 occluding upper-limit arrows at the Wilson floor ~1e-3.
        ax.plot([], [], 'o-', color=color, markersize=10, linewidth=2.0,
                label=f"{label}  [LER < $10^{{-3}}$ at all $p$]")
        continue

    ci = np.array([wilson_ci(int(round(rate * N_SHOTS)), N_SHOTS)
                   for rate in rates[nz]])
    los, his = ci[:, 0], ci[:, 1]
    ax.loglog(ps[nz], rates[nz], 'o-', color=color, label=label,
              markersize=10, linewidth=2.0)
    ax.fill_between(ps[nz], np.maximum(los, 1e-5), np.maximum(his, 1e-5),
                    alpha=0.20, color=color, linewidth=0)

# Stabilizer benchmark + no-coding baseline
if 'five_qudit_d3' in ler_results:
    r = ler_results['five_qudit_d3']
    ps = np.asarray(r['p_rates'])
    rates = np.asarray(r['lers'])
    ax.loglog(ps, np.maximum(rates, 1e-5), 'x--',
              color=STABILIZER_COLOR, label='[[5,1,3]]$_{Z_3}$ stabilizer',
              markersize=9, linewidth=1.5)

p_ref = np.linspace(0.001, 0.25, 100)
ax.plot(p_ref, p_ref, ':', color=NO_CODING_COLOR, alpha=0.7, label='No coding')

ax.set_xlabel('Physical error rate $p$', fontsize=13)
ax.set_ylabel('Logical error rate', fontsize=13)
ax.set_xlim(8e-4, 0.3)
ax.set_ylim(1e-5, 1)
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=9, loc='lower right', ncol=1, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ler_vs_p_all_dist3.png', dpi=300)
print("Saved ler_vs_p_all_dist3.png")

# %% [markdown]
# ## 7. Summary table for thesis

# %%
with open('../results/code_summary.md', 'w') as f:
    f.write('# VarQEC Trained Codes Summary\n\n')
    f.write('| Code | d | n | K | dist | L | Loss | LER@p=0.05 | LER@p=0.1 | LER@p=0.2 |\n')
    f.write('|------|---|---|---|------|---|------|------------|----------|----------|\n')
    for s in summary:
        if s['abstract']:
            continue
        name = s['name']
        ler05 = ler10 = ler20 = '-'
        if name in ler_results:
            r = ler_results[name]
            p = r['p_rates']
            lers = r['lers']
            for target, box in [(0.05, 'ler05'), (0.1, 'ler10'), (0.2, 'ler20')]:
                idx = np.argmin(np.abs(p - target))
                if abs(p[idx] - target) < 0.01:
                    val = f'{lers[idx]:.4f}'
                    if box == 'ler05': ler05 = val
                    elif box == 'ler10': ler10 = val
                    elif box == 'ler20': ler20 = val
        loss_str = f'{s["loss"]:.2e}' if s['loss'] > 0 else '0'
        typ = 'stab' if s['analytical'] else ''
        f.write(f'| {name} | {s["d"]} | {s["n"]} | {s["K"]} | '
                f'{s["distance"]} | {s["layers"]} | {loss_str} | '
                f'{ler05} | {ler10} | {ler20} |\n')

print("Saved code_summary.md")
print("\nDone.")
