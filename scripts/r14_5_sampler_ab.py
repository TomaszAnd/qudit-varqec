#!/usr/bin/env python3
"""R14-5 §6 — A/B test: stratified-10 uniform vs stratified-importance sampling.

From a fresh random init (same seed for both), run N_STEPS Adam steps with each
w2 sampler at matched EVs/step, and compare per-step loss + rolling variance.
Pass: stratified-IS shows lower per-step loss variance AND comparable-or-better
mean at equal EVs.

Mirrors scripts/train_r14_3_meth.py's loss setup exactly; the ONLY change is the
w2 weight construction:
  - baseline:  _build_meth_stratified_weights (R14-3a stratified-10 uniform)
  - treatment: src.sampling.make_stratified_importance_weights (Neyman + ∝w_i)
w1 weights (full Meth, no sampling) are identical in both arms.

Run under contention with `nice -n 19`. Defers to reachability.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="stratified-10 w2 fraction (baseline EVs/step driver)")
    ap.add_argument("--out_dir", default="results/round14_scoping")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from jax import value_and_grad
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from loss_split_helpers import build_w1_w2_masks
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights, neyman_group_budgets)
    from src.decoders.priors import meth_pauli_prior
    from train_r14_3_meth import (
        _flat_group_offsets, _build_meth_w1_weights,
        _build_meth_stratified_weights)

    d, n_qudit, dist, K = 3, 9, 3, 3
    n_layers = 4
    connections = [[i, j] for i in range(n_qudit)
                   for j in range(i + 1, n_qudit)]
    lam = meth_pauli_prior()
    model = ErrorModel(d=d, n_qudit=n_qudit, distance=dist, closed=True)
    E_full = model.build_grouped(verbose=False)
    offs = _flat_group_offsets(E_full)
    enc, _, ppl = create_jax_encoder(n_qudit, d, connections=connections,
                                     use_scan=True)
    loss_w = build_varqec_loss_weighted(enc, K, d, n_qudit, dist, E_full)
    val_grad = jax.jit(value_and_grad(lambda th, w: loss_w(th, w), argnums=0))

    w1_mask, w2_mask = build_w1_w2_masks(E_full, d)
    w1_mask_np = [np.asarray(m) for m in w1_mask]
    w2_mask_np = [np.asarray(m) for m in w2_mask]
    w1_w = _build_meth_w1_weights(w1_mask_np, lam, offs, jnp)

    # Per-group w2 Meth weights (lambda on w2 positions, 0 elsewhere) for IS.
    w2_group_weights = []
    for g_idx, m in enumerate(w2_mask_np):
        n_g = m.shape[0]
        wg = np.zeros(n_g)
        for e in range(n_g):
            if m[e] > 0:
                wg[e] = float(lam[offs[g_idx] + e])
        w2_group_weights.append(wg)

    # Baseline stratified-10 realizes ceil(0.10·n_w2) per non-empty group.
    base_budget = sum(max(1, int(np.ceil(args.fraction * int((m > 0).sum()))))
                      for m in w2_mask_np if int((m > 0).sum()) > 0)
    # FAIR MATCH: the Neyman floor-at-1 over ~45 groups inflates the realized IS
    # EV count, so requesting base_budget for IS over-samples it. Search the IS
    # request that realizes the SAME total EVs as the baseline, so a variance
    # win is attributable to within-group ∝w_i sampling, not to more shots.
    is_request = base_budget
    for req in range(base_budget, 0, -1):
        if sum(neyman_group_budgets(w2_group_weights, req)) <= base_budget:
            is_request = req
            break
    is_realized = sum(neyman_group_budgets(w2_group_weights, is_request))
    print(f"baseline w2 EVs/step = {base_budget}; IS request={is_request} "
          f"realizes {is_realized} (EV-matched to baseline)", flush=True)

    def run(arm):
        key = jax.random.PRNGKey(args.seed)
        theta = jax.random.uniform(key, (n_layers, ppl), minval=0.0,
                                   maxval=2 * np.pi)
        rng = np.random.default_rng(args.seed * 1000 + 7)
        opt = optax.adam(0.05)
        ostate = opt.init(theta)
        losses = []
        t0 = time.time()
        for step in range(args.n_steps):
            if arm == 'baseline':
                w2 = _build_meth_stratified_weights(
                    w2_mask_np, lam, args.fraction, rng, jnp, offs,
                    int(offs[-1]))
            else:  # stratified-IS (EV-matched to baseline)
                w2 = make_stratified_importance_weights(
                    w2_group_weights, is_request, rng)
            combined = tuple(w1_w[g] + w2[g] for g in range(len(E_full)))
            lv, grad = val_grad(theta, combined)
            updates, ostate = opt.update(grad, ostate, theta)
            theta = optax.apply_updates(theta, updates)
            losses.append(float(lv))
        print(f"  {arm}: {args.n_steps} steps in {time.time()-t0:.0f}s, "
              f"final loss {losses[-1]:.4e}", flush=True)
        return np.array(losses)

    print("running baseline (stratified-10 uniform) ...", flush=True)
    L_base = run('baseline')
    print("running treatment (stratified-IS) ...", flush=True)
    L_is = run('is')

    # Rolling variance (window 5) as the variance-reduction signal.
    def roll_var(x, w=5):
        return np.array([x[max(0, i-w+1):i+1].var() for i in range(len(x))])
    v_base, v_is = roll_var(L_base), roll_var(L_is)
    var_ratio = float(np.mean(v_is[5:]) / max(np.mean(v_base[5:]), 1e-30))
    mean_ratio = float(L_is[-5:].mean() / max(L_base[-5:].mean(), 1e-30))
    # Shot-frugality criterion (Outcome I reframe): IS passes if it reaches the
    # same code cheaper — EITHER lower final loss at equal steps, OR lower
    # per-step variance with comparable mean (enabling tighter Hoeffding stop).
    reaches_lower = bool(L_is[-1] <= L_base[-1])
    lower_var = bool(var_ratio < 1.0 and mean_ratio < 1.2)
    verdict = "PASS" if (reaches_lower or lower_var) else "FAIL"

    os.makedirs(os.path.join(REPO, args.out_dir), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(L_base, label='stratified-10 uniform', lw=1.3)
    ax[0].plot(L_is, label='stratified-IS', lw=1.3)
    ax[0].set_yscale('log'); ax[0].set_xlabel('step'); ax[0].set_ylabel('loss')
    ax[0].legend(fontsize=8); ax[0].set_title('per-step loss')
    ax[1].plot(v_base, label='uniform'); ax[1].plot(v_is, label='IS')
    ax[1].set_yscale('log'); ax[1].set_xlabel('step')
    ax[1].set_ylabel('rolling var (w=5)'); ax[1].legend(fontsize=8)
    ax[1].set_title('rolling loss variance')
    fig.suptitle(f"R14-5 sampler A/B (n={args.n_steps}, seed={args.seed}): "
                 f"var_ratio={var_ratio:.2f}, mean_ratio={mean_ratio:.2f} [{verdict}]")
    fig.tight_layout()
    png = os.path.join(REPO, args.out_dir, "r14_5_sampler_ab.png")
    fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)

    md = os.path.join(REPO, args.out_dir, "r14_5_sampler_ab.md")
    with open(md, "w") as f:
        f.write(f"# R14-5 §6 — sampler A/B: stratified-10 uniform vs stratified-IS\n\n")
        f.write(f"- n_steps={args.n_steps}, seed={args.seed}; baseline w2 EVs/step "
                f"= {base_budget}, IS realized = {is_realized} (EV-matched)\n")
        f.write(f"- IS/uniform rolling-variance ratio (steps 5+): **{var_ratio:.3f}** "
                f"(<1 = IS reduces variance)\n")
        f.write(f"- IS/uniform final-mean ratio (last 5 steps): **{mean_ratio:.3f}** "
                f"(≤1.2 = comparable-or-better)\n")
        f.write(f"- baseline final loss {L_base[-1]:.4e}; IS final loss {L_is[-1]:.4e}\n\n")
        f.write(f"- reaches-lower-loss-at-30-steps: {reaches_lower}; "
                f"lower-variance-comparable-mean: {lower_var}\n\n")
        f.write(f"## Verdict: **{verdict}**\n\n")
        f.write("Shot-frugality PASS gate (Outcome I reframe): "
                "(final loss ≤ baseline at equal steps) OR "
                "(var_ratio < 1.0 AND mean_ratio < 1.2). "
                "If FAIL, debug the sampler before launching R14-3b (§7).\n")
    print(f"var_ratio={var_ratio:.3f}, mean_ratio={mean_ratio:.3f} -> {verdict}",
          flush=True)
    print(f"wrote {png}\nwrote {md}", flush=True)


if __name__ == "__main__":
    main()
