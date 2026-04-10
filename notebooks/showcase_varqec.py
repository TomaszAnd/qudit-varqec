# %% [markdown]
# # VarQEC: Analysis of Trained Quantum Error-Correcting Codes
#
# This notebook loads pre-trained codes from `results/params/` and produces
# all analysis plots. No training — run scripts in `scripts/` first.
#
# **Structure**: Load codes → Characterize → Convergence → Noise models → d=2 analysis → d=3 analysis
#
# **References**: Cao et al. arXiv:2204.03560 (VarQEC), Meth et al. arXiv:2310.12110v3 (trapped-ion noise)

# %% [markdown]
# ## 1. Setup and Error Sets

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys, itertools, glob
sys.path.insert(0, '..')

from src.legacy.ququart_pipeline import single_qudit_paulis, single_qudit_dephasing_paulis
from src.legacy.ququart_pipeline import build_error_sets, build_dephasing_error_sets
from src.legacy.ququart_pipeline import create_encoder
from src.loss import load_varqec_result, estimate_memory_mb
from src.correlated_noise import (
    control_qudit_kraus, control_qudit_kraus_simplified,
    target_qudit_kraus, spectator_qudit_kraus,
)
from src.correlated_noise import build_correlated_error_set
from src.simulation import (
    simulate_ler_with_detection, simulate_decoder_comparison,
    make_pauli_dephasing_noise_fn, make_pauli_depolarizing_noise_fn,
    make_correlated_dephasing_noise_fn,
)
from nb_utils import (
    load_trained_code, get_code_states, kl_residuals, kl_residuals_factored,
    weight_enumerators, COLORS, MARKERS, LABELS,
)
from src.legacy.ququart_pipeline import build_error_sets_factored, factored_to_dense
import pennylane as qml
from pennylane import numpy as pnp

# %% [markdown]
# ### Error set sizes
#
# | Model | Distance | E\_det | E\_corr | Representation |
# |-------|----------|--------|---------|----------------|
# | Dephasing | 2 | 16 | 1 | Dense 1024×1024 |
# | Dephasing | 3 | 106 | 16 | Dense |
# | Depolarizing | 2 | 76 | 1 | Dense |
# | Depolarizing | 3 | 2326 | 76 | **Factored** (37 GB dense) |
# | Correlated | 2 | ~25k | ~25k | Diagonal vectors |

# %%
from math import comb
print("Error set sizes for [[5,1,d]]_4:")
for dist in [2, 3]:
    Ed_deph, Ec_deph = build_dephasing_error_sets(5, dist)
    if dist <= 2:
        Ed_depol, Ec_depol = build_error_sets(5, dist)
        depol_str = f"E_det={len(Ed_depol)}, E_corr={len(Ec_depol)}"
    else:
        # Don't materialize depolarizing d=3 (37 GB dense) — compute count analytically
        n_depol_det = sum(comb(5, w) * 15**w for w in range(dist))
        n_depol_corr = sum(comb(5, w) * 15**w for w in range((dist - 1) // 2 + 1))
        depol_str = f"E_det={n_depol_det} (factored), E_corr={n_depol_corr} (factored)"
    print(f"  d={dist}: dephasing E_det={len(Ed_deph)}, E_corr={len(Ec_deph)} | "
          f"depolarizing {depol_str}")

gate_pairs = [(i, i + 1, 0, 1) for i in range(4)]
E_c, _ = build_correlated_error_set(n_qudits=5, d=4, gate_pairs=gate_pairs,
                                     eta=0.95, n_max=2, noise_model="simplified")
dim_full = 4 ** 5
print(f"  Correlated (simplified): {len(E_c)} diagonal ops, "
      f"{len(E_c)*dim_full*16/(1024**2):.0f} MB diagonal vs "
      f"{estimate_memory_mb(len(E_c), dim_full):.0f} MB dense")

# %% [markdown]
# ## 2. Load Codes and Extract Code States

# %%
CODE_CONFIGS = [
    ("dephasing_d2",        "dephasing",             2, 1, 42),
    ("dephasing_d3",        "dephasing",             3, 2, 2),
    ("depolarizing_d2",     "depolarizing",          2, 2, 42),
    ("depolarizing_d3",     "depolarizing",          3, 6, 42),
    ("corr_simplified_d2",  "correlated_simplified", 2, 2, 42),
    ("corr_simplified_d3",  "correlated_simplified", 3, 3, 42),
]

trained = {}
for label, noise_type, dist, layers, seed in CODE_CONFIGS:
    losses, theta = load_trained_code(noise_type, dist, layers, seed)
    if losses is not None:
        trained[label] = {'losses': losses, 'theta': theta}

codes = {}
for label, data in trained.items():
    codes[label] = get_code_states(data['theta'])
    print(f"  {label}: {codes[label].shape}")

# %% [markdown]
# ## 3. Code Characterization
#
# VarQEC codes are found by variational optimization of the Knill-Laflamme (KL) conditions.
# They are NOT guaranteed to be stabilizer codes. Our codes have parameters ((n, K, d))\_q
# with n=5 ququarts, K=4 codewords, q=4. We characterize them via KL residuals and
# Shor-Laflamme weight enumerators.

# %%
# --- KL condition residuals ---
print("KL condition residuals:")
print(f"  {'Code':<25} {'max|off-diag|':>14} {'max Var(diag)':>14} {'KL loss':>12}")
print("  " + "-" * 67)

for label in codes:
    if 'deph' in label:
        dist = 3 if 'd3' in label else 2
        E_det, E_corr = build_dephasing_error_sets(5, dist)
        max_off, max_var = kl_residuals(codes[label], E_det, E_corr, dist)
    elif 'depol' in label:
        dist = 3 if 'd3' in label else 2
        if dist <= 2:
            E_det, E_corr = build_error_sets(5, dist)
            max_off, max_var = kl_residuals(codes[label], E_det, E_corr, dist)
        else:
            # Use factored errors for depolarizing d=3 (avoids 37 GB dense)
            E_det_f, E_corr_f = build_error_sets_factored(5, dist)
            max_off, max_var = kl_residuals_factored(codes[label], E_det_f, E_corr_f, dist)
    else:
        E_det, E_corr = build_dephasing_error_sets(5, 2)
        dist = 2
        max_off, max_var = kl_residuals(codes[label], E_det, E_corr, dist)

    loss = trained[label]['losses'][-1]
    print(f"  {label:<25} {max_off:>14.2e} {max_var:>14.2e} {loss:>12.2e}")

# %%
# --- Quantum weight enumerators (Shor-Laflamme, weight 0-1) ---
print("\nWeight enumerators (Shor-Laflamme):")
print(f"  {'Code':<25} {'A_0':>8} {'A_1':>8}  {'B_0':>8} {'B_1':>8}")
print("  " + "-" * 57)
t0 = time.time()
for label in codes:
    A, B = weight_enumerators(codes[label], max_weight=1)
    print(f"  {label:<25} {A[0]:>8.4f} {A[1]:>8.4f}  {B[0]:>8.4f} {B[1]:>8.4f}")
print(f"  ({time.time()-t0:.0f}s, weight-2 omitted: ~9 min/code)")

# %% [markdown]
# **Interpretation**: A\_0=1 and B\_0=1 for all codes (normalization). For a perfect
# distance-2+ code protecting against ALL weight-1 Paulis: A\_1≈0, B\_1≈1. The depolarizing
# code comes closest (A\_1≈0.09, B\_1≈1.3). Dephasing codes have large A\_1 and B\_1 because
# they only protect against Z-type Paulis, not all 15.

# %% [markdown]
# ## 4. Training Convergence

# %%
available_runs = []
run_configs = [
    ("Dephasing d=2\n(1 layer)",   "dephasing_d2",       '#2196F3'),
    ("Dephasing d=3\n(2 layers)",  "dephasing_d3",       '#1565C0'),
    ("Depolarizing d=2\n(2 layers)", "depolarizing_d2",  '#F44336'),
    ("Depolarizing d=3\n(6 layers)", "depolarizing_d3",  '#D32F2F'),
    ("Corr. simpl. d=2\n(2 layers)", "corr_simplified_d2", '#FF9800'),
    ("Corr. simpl. d=3\n(3 layers)",  "corr_simplified_d3", '#E65100'),
]
for title, key, color in run_configs:
    if key in trained:
        available_runs.append((title, trained[key]['losses'], color, key))

if available_runs:
    n_plots = len(available_runs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.2 * n_plots, 4.5))
    if n_plots == 1: axes = [axes]
    for ax, (title, losses, color, key) in zip(axes, available_runs):
        ax.semilogy(losses, linewidth=1.5, color=color, alpha=0.7)
        if 'corr_simplified' in key and len(losses) > 20:
            w = 20
            rolling = np.convolve(losses, np.ones(w)/w, mode='valid')
            ax.semilogy(np.arange(w-1, len(losses)), rolling,
                        linewidth=2.5, color='black', alpha=0.8, label=f'Avg (w={w})')
            ax.annotate('minibatch noise', xy=(0.5, 0.02), xycoords='axes fraction',
                        fontsize=7, fontstyle='italic', color='gray')
        ax.set_xlabel("Step"); ax.set_ylabel("KL loss")
        ax.grid(True, alpha=0.3)
        ax.axhline(1e-6, color='red', linestyle='--', alpha=0.5, label='1e-6')
        final = losses[-1]
        ax.annotate(f'{final:.1e}', xy=(len(losses)-1, final), fontsize=8, ha='right', va='bottom')
        status = "CONV" if final < 1e-5 else f"{len(losses)} steps"
        ax.set_title(f"{title}\n{status}", fontsize=9)
        if final < 1e-5: ax.set_facecolor('#f0fff0')
        ax.legend(fontsize=7)
    plt.suptitle("VarQEC Training Convergence", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("../results/plots/convergence_all.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Kraus Structure: Simplified vs Physical Noise Model

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
eta, d_kr = 0.95, 5
x = np.arange(d_kr)
level_labels = [f"|{k}>" for k in range(d_kr)]

ax = axes[0]
dp = control_qudit_kraus(d=d_kr, control_level=0, eta=eta, n_max=4)
ds = control_qudit_kraus_simplified(d=d_kr, control_level=0, eta=eta, n_max=4)
w = 0.35
ax.bar(x-w/2, np.abs(np.real(dp[0])), w, label='Physical', color='#2196F3', alpha=0.85)
ax.bar(x+w/2, np.abs(np.real(ds[0])), w, label='Simplified', color='#FF9800', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(level_labels)
ax.set_xlabel("Level"); ax.set_ylabel("|E_0[k,k]|")
ax.set_title("Control qudit E_0\n(models differ)"); ax.legend(fontsize=8); ax.set_ylim(0, 1.1)

ckr = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
for panel, title, diags in [
    (1, "Target qudit\n(identical)", target_qudit_kraus(d=d_kr, control_level=0, target_level=1, eta=eta, n_max=4)),
    (2, "Spectator qudit\n(identical)", spectator_qudit_kraus(d=d_kr, eta=eta, n_max=4)),
]:
    ax = axes[panel]
    for n, (diag, c) in enumerate(zip(diags, ckr)):
        off = (n - len(diags)/2 + 0.5) * 0.15
        ax.bar(x + off, np.abs(diag), 0.15, label=f'n={n}', color=c, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(level_labels)
    ax.set_xlabel("Level"); ax.set_ylabel("|E_n[k,k]|")
    ax.set_title(title); ax.legend(fontsize=8, ncol=2); ax.set_ylim(0, 1.1)

plt.suptitle(f"Trapped-ion Kraus operators (d={d_kr}, eta={eta})", fontsize=13)
plt.tight_layout()
plt.savefig("../results/plots/kraus_dual_models.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. d=2 Analysis: Detection + Cross-Code Comparison
#
# Distance-2 codes detect errors via post-selection. The depolarizing and correlated codes
# detect more errors (~90% at p=0.15) due to tighter code spaces. The dephasing code only
# detects ~30% because Z-errors don't push far from its code space.

# %%
eta_values = [0.995, 0.99, 0.98, 0.95, 0.90, 0.85]
gate_pairs_ler = [(i, i + 1, 0, 1) for i in range(4)]
n_shots = 300
d2_codes = {k: v for k, v in codes.items() if 'd2' in k}

single_deph = [np.array(e) for e in single_qudit_dephasing_paulis()]
single_depol = [np.array(e) for e in single_qudit_paulis()]

t_start = time.time()
detection_results = {}
for code_label, code_states in d2_codes.items():
    ps_fids, det_fracs = [], []
    for eta_val in eta_values:
        noise_fn = make_correlated_dephasing_noise_fn(
            n_qudits=5, d=4, gate_pairs=gate_pairs_ler,
            eta=eta_val, n_max=5, noise_model="simplified")
        r = simulate_ler_with_detection(code_states, noise_fn, n_shots=n_shots,
                                         detection_threshold=0.1, seed=42)
        ps_fids.append(r['post_selected_fidelity'])
        det_fracs.append(r['detected_fraction'])
    detection_results[code_label] = {'ps_fid': ps_fids, 'det_frac': det_fracs}
print(f"Detection sweep: {time.time() - t_start:.0f}s")

# Cross-code under Pauli noise
p_sweep = [0.005, 0.01, 0.05, 0.1, 0.15]
t_start = time.time()
cross_deph, cross_depol = {}, {}
for code_label, cs in d2_codes.items():
    cd, cdp = [], []
    for p in p_sweep:
        cd.append(simulate_ler_with_detection(cs, make_pauli_dephasing_noise_fn(single_deph, 5, 4, p),
                                               n_shots=n_shots, detection_threshold=0.1, seed=42)['post_selected_fidelity'])
        cdp.append(simulate_ler_with_detection(cs, make_pauli_depolarizing_noise_fn(single_depol, 5, 4, p),
                                                n_shots=n_shots, detection_threshold=0.1, seed=42)['post_selected_fidelity'])
    cross_deph[code_label] = cd
    cross_depol[code_label] = cdp
print(f"Cross-code: {time.time() - t_start:.0f}s")

# %%
p_phys = [1 - e for e in eta_values]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for code_label, data in detection_results.items():
    axes[0].plot(p_phys, data['ps_fid'], marker=MARKERS.get(code_label,'o'),
                 color=COLORS.get(code_label,'gray'), label=LABELS.get(code_label,code_label),
                 linewidth=1.5, markersize=6)
axes[0].set_xlabel("Error rate (1-eta)"); axes[0].set_ylabel("Post-selected fidelity")
axes[0].set_title("Post-Selected Fidelity"); axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

for code_label, data in detection_results.items():
    axes[1].plot(p_phys, data['det_frac'], marker=MARKERS.get(code_label,'o'),
                 color=COLORS.get(code_label,'gray'), label=LABELS.get(code_label,code_label),
                 linewidth=1.5, markersize=6)
axes[1].set_xlabel("Error rate (1-eta)"); axes[1].set_ylabel("Detection rate")
axes[1].set_title("Error Detection Rate"); axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.05)
plt.suptitle("d=2 Detection Under Correlated Dephasing Noise", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("../results/plots/detection_d2.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, results, xv, xl) in zip(axes, [
    ("Pauli dephasing", cross_deph, p_sweep, "Per-qudit error prob"),
    ("Pauli depolarizing", cross_depol, p_sweep, "Per-qudit error prob"),
    ("Correlated dephasing", {k: v['ps_fid'] for k, v in detection_results.items()},
     p_phys, "Error rate (1-eta)"),
]):
    for cl, fids in results.items():
        ax.plot(xv, fids, marker=MARKERS.get(cl,'o'), color=COLORS.get(cl,'gray'),
                label=LABELS.get(cl,cl), linewidth=1.5, markersize=6)
    ax.set_xlabel(xl); ax.set_ylabel("Post-selected fidelity"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
plt.suptitle("Cross-Code Comparison: d=2 Codes Under 3 Noise Models", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("../results/plots/cross_code_d2.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. d=3 Analysis: Decoder Comparison
#
# We compare two decoders on each d=3 code under Pauli dephasing and correlated dephasing noise.
#
# **Decoders:**
# - **Projection**: Project onto code space, renormalize. Simple, no error knowledge.
# - **Lookup table (ML)**: For each correctable error E in E\_corr, compute overlap
#   with {E|psi\_k>}, apply E† for the best match. Maximum-likelihood over the correctable set.
#
# **Error sets for the lookup table decoder:**
# - *Dephasing d=3*: uses its own 16 dephasing E\_corr operators (exact match).
# - *Depolarizing d=3*: uses 76 depolarizing E\_corr operators (built from factored representation).
# - *Correlated d=3*: uses dephasing E\_corr as proxy. The correlated error set has 25k
#   diagonal operators which cannot serve as lookup table correction operators directly.
#   This means the lookup table is mismatched for correlated noise — expect limited benefit.
#
# **Convergence caveat:** The correlated d=3 code (loss ~7e-4) has not fully converged —
# the KL residuals are non-zero, limiting correction performance. The depolarizing d=3
# code (loss ~6) is far from convergence and shows minimal correction capability.

# %%
d3_codes = {k: v for k, v in codes.items() if 'd3' in k}
if d3_codes:
    E_det_deph3, E_corr_deph3 = build_dephasing_error_sets(5, 3)

    # Build matching error sets for each d=3 code's decoder
    d3_error_sets = {}
    for label in d3_codes:
        if 'deph' in label:
            d3_error_sets[label] = E_corr_deph3
        elif 'depol' in label:
            # Build depolarizing E_corr (76 dense ops = 1.2 GB, fits in memory)
            _, E_corr_depol3_f = build_error_sets_factored(5, 3)
            d3_error_sets[label] = [factored_to_dense(f, 5, 4) for f in E_corr_depol3_f]
        else:
            # Correlated d=3: use dephasing E_corr as proxy
            d3_error_sets[label] = E_corr_deph3

    # --- Pauli dephasing noise sweep ---
    p_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    t_start = time.time()
    pauli_ler_all = {}
    for label, cs in d3_codes.items():
        res_by_p = {}
        for p in p_values:
            noise_fn = make_pauli_dephasing_noise_fn(single_deph, n_qudits=5, dim_qudit=4, p=p)
            res_by_p[p] = simulate_decoder_comparison(cs, noise_fn, d3_error_sets[label],
                                                       n_shots=n_shots, seed=42)
        pauli_ler_all[label] = {
            dec: [res_by_p[p][dec]['ler'] for p in p_values]
            for dec in ['projection', 'lookup_table']
        }
    print(f"Pauli dephasing d=3 sweep: {time.time() - t_start:.0f}s")

    # --- Correlated dephasing noise sweep ---
    eta_d3 = [0.999, 0.995, 0.99, 0.95, 0.90, 0.85, 0.70]
    p_corr = [1 - e for e in eta_d3]
    t_start = time.time()
    corr_ler_all = {}
    for label, cs in d3_codes.items():
        res_by_e = {}
        for eta_val in eta_d3:
            noise_fn = make_correlated_dephasing_noise_fn(
                n_qudits=5, d=4, gate_pairs=gate_pairs_ler,
                eta=eta_val, n_max=5, noise_model="simplified")
            res_by_e[eta_val] = simulate_decoder_comparison(cs, noise_fn, d3_error_sets[label],
                                                             n_shots=n_shots, seed=42)
        corr_ler_all[label] = {
            dec: [res_by_e[e][dec]['ler'] for e in eta_d3]
            for dec in ['projection', 'lookup_table']
        }
    print(f"Correlated dephasing d=3 sweep: {time.time() - t_start:.0f}s")

    # Print summary
    for label in d3_codes:
        print(f"\n  {label}:")
        for dec in ['projection', 'lookup_table']:
            print(f"    {dec} (Pauli):  {[f'{v:.3f}' for v in pauli_ler_all[label][dec]]}")
            print(f"    {dec} (corr.):  {[f'{v:.3f}' for v in corr_ler_all[label][dec]]}")

    # --- Plot ---
    dec_styles = {
        'projection':   {'color': '#4CAF50', 'marker': 's', 'ls': '-',  'label': 'Projection'},
        'lookup_table': {'color': '#1565C0', 'marker': 'o', 'ls': '-',  'label': 'Lookup table (ML)'},
    }

    n_d3 = len(d3_codes)
    fig, axes = plt.subplots(n_d3, 2, figsize=(14, 5 * n_d3), squeeze=False)
    floor = 1.0 / (2 * n_shots)

    for row, label in enumerate(d3_codes):
        # Left: Pauli dephasing
        ax = axes[row, 0]
        for dec, vals in pauli_ler_all[label].items():
            s = dec_styles[dec]
            ax.semilogy(p_values, [max(v, floor) for v in vals],
                        marker=s['marker'], color=s['color'], linestyle=s['ls'],
                        label=s['label'], linewidth=1.5, markersize=6)
        ax.axhline(floor, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel("Per-qudit error prob p"); ax.set_ylabel("LER")
        ax.set_title(f"{LABELS.get(label, label)}\nPauli dephasing noise")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(floor*0.5, 1.1)

        # Right: Correlated dephasing
        ax = axes[row, 1]
        for dec, vals in corr_ler_all[label].items():
            s = dec_styles[dec]
            ax.semilogy(p_corr, [max(v, floor) for v in vals],
                        marker=s['marker'], color=s['color'], linestyle=s['ls'],
                        label=s['label'], linewidth=1.5, markersize=6)
        ax.axhline(floor, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel("Error rate (1-eta)"); ax.set_ylabel("LER")
        ax.set_title(f"{LABELS.get(label, label)}\nCorrelated dephasing noise")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(floor*0.5, 1.1)

    plt.suptitle("d=3 Codes: Decoder Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("../results/plots/decoder_comparison_d3.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No d=3 codes found — skipping decoder comparison")

# %% [markdown]
# ## 8. Summary

# %%
print(f"\n{'Model':<30} {'Steps':>6} {'Final Loss':>12} {'Status':<12}")
print("-" * 65)
for label, noise_type, dist, layers, seed in CODE_CONFIGS:
    key = f"{noise_type.replace('correlated_', 'corr_')}_d{dist}"
    if key in trained:
        losses = trained[key]['losses']
        final = losses[-1]
        status = "CONVERGED" if final < 1e-5 else "training"
        print(f"{LABELS.get(key, key):<30} {len(losses):>6} {final:>12.2e} {status:<12}")
    else:
        print(f"{LABELS.get(key, key):<30} {'—':>6} {'—':>12} {'not trained':<12}")

print(f"\nSaved codes:")
for f in sorted(glob.glob("../results/params/*.npz")):
    d = np.load(f, allow_pickle=True)
    fl = d.get('final_loss', None)
    print(f"  {os.path.basename(f)}: loss={float(fl):.2e}" if fl is not None else f"  {os.path.basename(f)}")

# %% [markdown]
# ## Key Findings
#
# - **Detection-style loss** (paper Eq. 16) eliminates quadratic E\_a†E\_b products,
#   enabling efficient d=3 training: correlated d=3 at 1.1s/step, depolarizing d=3 at 3.5s/step
# - **Factored errors** make depolarizing d=3 tractable: 600 KB vs 37 GB
# - **d=2 detection**: channel-adapted code (corr. simplified) achieves highest post-selected fidelity
# - **Cross-code d=2**: each code performs best under its training noise model
# - **d=3 correction**: lookup table (ML) decoder outperforms projection under Pauli dephasing;
#   under correlated noise, projection alone is effective (errors not in Pauli correctable set)
# - **Weight enumerators** confirm code quality: depolarizing d=2 has A\_1~0, B\_1~1.3 (near-ideal)
# - **Depolarizing d=3** (loss ~6) has not converged — finding a ((5,4,3))\_4 code protecting
#   against all Paulis is genuinely hard with the current ansatz. Further work needed (more layers,
#   JAX-based optimizer, or different circuit architecture)
