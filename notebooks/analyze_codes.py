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
        summary.append({
            'name': name, 'd': m['d'], 'n': m['n_qudit'], 'K': m['K'],
            'distance': m['distance'], 'layers': m.get('n_layers', 0),
            'loss': m.get('final_loss', 0),
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
color_map = {3: '#E53935', 4: '#1E88E5', 5: '#43A047'}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for s in summary:
    if s['abstract'] or s['analytical'] or s['losses'] is None:
        continue
    ax = axes[0] if s['distance'] == 2 else axes[1]
    label = f"(({s['n']},{s['d']},{s['distance']}))_{s['d']}"
    ax.semilogy(s['losses'], color=color_map.get(s['d'], 'gray'),
                alpha=0.7, linewidth=1.3, label=label)

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
# ### Figure 1: n-scaling for d=3 distance-3

# %%
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

axes[0].loglog(n_d3, loss_d3, 'o-', markersize=12, linewidth=2, color='#E53935')
if len(n_d3) >= 2:
    slope, intercept = np.polyfit(np.log(n_d3), np.log(loss_d3), 1)
    n_fit = np.linspace(n_d3.min(), n_d3.max(), 50)
    axes[0].loglog(n_fit, np.exp(intercept) * n_fit**slope, '--', color='gray',
                   alpha=0.7, label=f'loss $\\propto n^{{{slope:.1f}}}$')
for ni, li in zip(n_d3, loss_d3):
    axes[0].annotate(f'{li:.3f}', (ni, li), textcoords='offset points',
                     xytext=(8, 5), fontsize=10)
axes[0].set_xlabel('Number of qutrits $n$')
axes[0].set_ylabel('Training loss')
axes[0].set_title('n-scaling of ((n,3,3))$_3$ training loss')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# RIGHT: LER vs p
p_ref = np.linspace(0.001, 0.25, 100)
axes[1].plot(p_ref, p_ref, 'k--', alpha=0.5, label='No coding')
colors_n = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(n_d3)))
for ni, col in zip(n_d3, colors_n):
    names = [s['name'] for s in summary
             if s['d'] == 3 and s['n'] == ni and s['distance'] == 3 and not s['abstract']]
    if not names or names[0] not in ler_results:
        continue
    r = ler_results[names[0]]
    axes[1].semilogy(r['p_rates'], np.maximum(r['lers'], 1e-5), 'o-',
                     color=col, markersize=8, label=f'n={ni}')

axes[1].set_xlabel('Physical error rate $p$')
axes[1].set_ylabel('Logical error rate')
axes[1].set_title('((n,3,3))$_3$ logical error rates')
axes[1].legend(loc='upper left')
axes[1].set_ylim(1e-5, 1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/n_scaling_d3_dist3.png', dpi=150)
print("Saved n_scaling_d3_dist3.png")

# %% [markdown]
# ### Figure 2: VarQEC vs stabilizer benchmark

# %%
fig, ax = plt.subplots(figsize=(9, 6))

# Benchmark stabilizer code
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

varqec_codes = [
    ('qutrit_d3', 'VarQEC ((5,3,3))$_3$', '#EF5350'),
    ('d3_n6_dist3', 'VarQEC ((6,3,3))$_3$', '#FB8C00'),
    ('qutrit_n7_d3', 'VarQEC ((7,3,3))$_3$', '#66BB6A'),
    ('qutrit_n8_d3', 'VarQEC ((8,3,3))$_3$', '#42A5F5'),
]
for key, label, color in varqec_codes:
    if key in ler_results:
        r = ler_results[key]
        ax.semilogy(r['p_rates'], np.maximum(r['lers'], 1e-5),
                    'o-', color=color, label=label, markersize=9, linewidth=2)

if bench_name in ler_results:
    r = ler_results[bench_name]
    ax.semilogy(r['p_rates'], np.maximum(r['lers'], 1e-5),
                'x--', color='black', label='[[5,1,3]]$_{Z_3}$ stabilizer',
                markersize=10, linewidth=1.8)

ax.plot(p_ref, p_ref, ':', color='gray', alpha=0.7, label='No coding')
ax.set_xlabel('Physical error rate $p$', fontsize=13)
ax.set_ylabel('Logical error rate', fontsize=13)
ax.set_title('Native-gate VarQEC vs stabilizer benchmark', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(1e-5, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/varqec_vs_stabilizer.png', dpi=150)
print("Saved varqec_vs_stabilizer.png")

# %% [markdown]
# ## 6. Summary table for thesis

# %%
with open('../results/code_summary.md', 'w') as f:
    f.write('# VarQEC Trained Codes Summary\n\n')
    f.write('| Code | d | n | K | dist | L | Loss | LER@p=0.05 | LER@p=0.1 |\n')
    f.write('|------|---|---|---|------|---|------|------------|----------|\n')
    for s in summary:
        if s['abstract']:
            continue
        name = s['name']
        ler05 = ler10 = '-'
        if name in ler_results:
            r = ler_results[name]
            p = r['p_rates']
            lers = r['lers']
            idx05 = np.argmin(np.abs(p - 0.05))
            idx10 = np.argmin(np.abs(p - 0.1))
            ler05 = f'{lers[idx05]:.4f}' if abs(p[idx05] - 0.05) < 0.01 else '-'
            ler10 = f'{lers[idx10]:.4f}' if abs(p[idx10] - 0.1) < 0.01 else '-'
        loss_str = f'{s["loss"]:.2e}' if s['loss'] > 0 else '0'
        typ = 'stab' if s['analytical'] else ''
        f.write(f'| {name} | {s["d"]} | {s["n"]} | {s["K"]} | '
                f'{s["distance"]} | {s["layers"]} | {loss_str} | {ler05} | {ler10} |\n')

print("Saved code_summary.md")
print("\nDone.")
