#!/usr/bin/env python3
"""Round-14 R14-3-launch — train ((9,3,3))_3 a2a 4L with Meth-weighted KL
+ stratified-10 sampling on w2.

Headline arm (R14-3a): per-EV importance weights from
`results/round14_scoping/meth_pauli_weights.npz` (Pauli-twirled Meth
physical channel onto the R12/13/14 closure basis). Sampling on w2:
stratified-10 (R13-4 winner). Both factors are passed to
`src.jax_backend.build_varqec_loss_weighted` as combined per-EV
weights: w_combined = λ_i^Meth × stratified_mask_i, where stratified_mask
is N_g/n_g on sampled positions and 0 elsewhere.

Adam two-stage LR matches R13 (0.05 → 0.01 at loss < 0.1). 1500 steps,
3 seeds. No kill switch (R13-6 policy).

Sensitivity arm (R14-3b): same configuration but loads weights from
`meth_pauli_weights_simplified.npz` (Meth literal Eq. J3 simplified
mode). Single seed; sanity check that the two readings produce
operationally distinct trained codes.

Output:
  results/round14_scoping/r14_3a_meth_physical/  (or r14_3b_meth_simplified)
    d3_n9_dist3_4L_meth_<mode>_seed{0,1,2}.npz, _best.npz
    SUMMARY.md, training_loss_traces.npz
  figures/r14_3a_meth_physical_training.png
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _save_npz(path, theta, d, n_qudit, K, dist, n_layers, connections,
              losses, wall, training_basis, connectivity, weights_meta):
    np.savez(
        path,
        params=np.asarray(theta),
        d=d, n_qudit=n_qudit, K=K, distance=dist, n_layers=n_layers,
        connectivity=connectivity,
        connections=np.asarray(connections, dtype=int),
        final_loss=float(losses[-1]),
        best_loss=float(np.min(losses)),
        loss_trace=losses,
        training_basis=training_basis,
        weights_meta=weights_meta,
        wall_s=float(wall),
    )


def _first_step_below(losses, thr):
    bsf = np.minimum.accumulate(losses)
    idx = np.where(bsf < thr)[0]
    return int(idx[0]) + 1 if len(idx) else None


def _build_meth_stratified_weights(w2_mask_np, lambda_meth_flat,
                                     fraction, rng, jnp,
                                     group_sizes_flat_start, n_total_ops):
    """Per-step weights: λ_i^Meth × stratified_mask_i on w2 positions
    (where w2_mask_np[g][e] > 0 for op g/e), 0 on w1 positions.

    lambda_meth_flat is the FLAT weight vector aligned with E_grouped
    (one entry per (group, e_idx) pair, in the same order as
    E_grouped iteration). group_sizes_flat_start gives the flat-index
    offset of each group.

    Stratified-10 within w2: per group, sample
    n_sampled = max(1, ceil(fraction × n_w2_in_group)) entries
    uniformly from the w2 positions.
    """
    weights_list = []
    flat_offset = 0
    for g_idx, m in enumerate(w2_mask_np):
        n_g = m.shape[0]
        positions = np.where(m > 0)[0]
        n_w2 = len(positions)
        w = np.zeros(n_g, dtype=float)
        if n_w2 > 0:
            n_keep = max(1, int(np.ceil(fraction * n_w2)))
            n_keep = min(n_keep, n_w2)
            chosen_local = rng.choice(n_w2, n_keep, replace=False)
            stratified_scale = float(n_w2) / float(n_keep)
            for k in chosen_local:
                e_idx = positions[k]
                lambda_i = float(
                    lambda_meth_flat[group_sizes_flat_start[g_idx] + e_idx])
                w[e_idx] = stratified_scale * lambda_i
        weights_list.append(jnp.asarray(w, dtype=jnp.float64))
        flat_offset += n_g
    return tuple(weights_list)


def _build_meth_w1_weights(w1_mask_np, lambda_meth_flat,
                            group_sizes_flat_start, jnp):
    """w1 weights: full Meth importance weight λ_i^Meth on w1 positions
    (no sampling on w1)."""
    weights_list = []
    for g_idx, m in enumerate(w1_mask_np):
        n_g = m.shape[0]
        w = np.zeros(n_g, dtype=float)
        for e_idx in range(n_g):
            if m[e_idx] > 0:
                lambda_i = float(
                    lambda_meth_flat[group_sizes_flat_start[g_idx] + e_idx])
                w[e_idx] = lambda_i
        weights_list.append(jnp.asarray(w, dtype=jnp.float64))
    return tuple(weights_list)


def _flat_group_offsets(E_grouped):
    """Return list of flat-index offsets per group, so flat_idx[g] + e
    gives the linear index into the flat weight vector."""
    offsets = [0]
    for g in E_grouped:
        offsets.append(offsets[-1] + int(g['matrices'].shape[0]))
    return offsets[:-1]


def train_r14_3(mode, weights_npz, out_dir, fig_dir, n_layers, n_steps,
                seeds, lr, lr_switch, lr_switch_threshold, fraction):
    """Train ((9,3,3))_3 a2a 4L with Meth-weighted KL + stratified-10."""
    import jax
    import jax.numpy as jnp
    import optax
    from jax import value_and_grad
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder
    from loss_split_helpers import (build_w1_w2_masks,
                                     build_w1w2_split_loss)

    d, n_qudit, dist = 3, 9, 3
    K = d
    connectivity = "a2a"
    connections = [[i, j] for i in range(n_qudit)
                    for j in range(i + 1, n_qudit)]

    print(f"\n=== R14-3 {mode}: (({n_qudit},{K},{dist}))_{d} {connectivity} "
          f"{n_layers}L ===")
    print(f"  edges: {len(connections)}; n_steps={n_steps}, seeds={seeds}")

    # Load Meth Pauli weights
    weights_data = np.load(weights_npz, allow_pickle=True)
    lambda_meth_flat = np.asarray(weights_data["weights"])
    weights_meta = {
        "weights_npz": str(weights_npz),
        "noise_model": str(weights_data.get("noise_model", "?")),
        "eta": float(weights_data.get("eta", 0.0)),
        "role_approximation": str(weights_data.get("role_approximation", "?")),
    }
    print(f"  Loaded {len(lambda_meth_flat)} Meth weights from "
          f"{weights_npz}")
    print(f"    noise_model={weights_meta['noise_model']}, "
          f"η={weights_meta['eta']}, role≈{weights_meta['role_approximation']}")
    print(f"    Σ λ = {lambda_meth_flat.sum():.6f}, "
          f"std/mean = {lambda_meth_flat.std()/max(lambda_meth_flat.mean(),1e-30):.3f}")

    # Build the structural basis
    model = ErrorModel(d=d, n_qudit=n_qudit, distance=dist, closed=True)
    E_full = model.build_grouped(verbose=False)
    n_full = sum(int(g['matrices'].shape[0]) for g in E_full)
    print(f"  |E_full| = {n_full}")
    assert n_full == len(lambda_meth_flat), \
        f"Weight vector size {len(lambda_meth_flat)} ≠ |E_full| {n_full}"

    group_offsets = _flat_group_offsets(E_full)

    # Encoder + loss-split
    enc, _, ppl = create_jax_encoder(n_qudit, d, connections=connections,
                                     use_scan=True)
    loss_w1_fn, loss_w2_fn = build_w1w2_split_loss(
        enc, K, d, n_qudit, dist, E_full)
    w1_mask, w2_mask = build_w1_w2_masks(E_full, d)
    w1_mask_np = [np.asarray(m) for m in w1_mask]
    w2_mask_np = [np.asarray(m) for m in w2_mask]

    # Pre-compute Meth-weighted w1 weights (no sampling on w1)
    w1_meth_weights = _build_meth_w1_weights(
        w1_mask_np, lambda_meth_flat, group_offsets, jnp)

    def _total_loss(theta, w2_weights):
        return loss_w1_fn(theta) + loss_w2_fn(theta, w2_weights)
    # Custom: loss_w1_fn doesn't take weights; we need a weighted w1
    # contribution. Build a weighted full loss path that takes per-EV
    # Meth weights for both w1 (constant) and w2 (per-step).
    from src.jax_backend import build_varqec_loss_weighted
    loss_weighted_full = build_varqec_loss_weighted(
        enc, K, d, n_qudit, dist, E_full)

    def _total_meth_loss(theta, combined_weights):
        return loss_weighted_full(theta, combined_weights)
    val_grad = jax.jit(value_and_grad(_total_meth_loss, argnums=0))

    seed_traces = {}
    seed_finals = {}
    seed_param_paths = {}
    for seed in seeds:
        print(f"\n  -- seed {seed} --")
        key = jax.random.PRNGKey(seed)
        theta = jax.random.uniform(key, (n_layers, ppl),
                                   minval=0.0, maxval=2 * np.pi)
        rng = np.random.default_rng(seed * 1000 + 42)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(theta)
        switched = False
        losses = np.zeros(n_steps, dtype=np.float64)
        t0 = time.time()
        for step in range(n_steps):
            # Combined weights: w1 = constant Meth × (full eval), w2 =
            # stratified-10 × Meth λ_i. Build per-group tuple.
            w2_step = _build_meth_stratified_weights(
                w2_mask_np, lambda_meth_flat, fraction, rng, jnp,
                group_offsets, n_full)
            # Add w1 weights (independent of step)
            combined = tuple(
                w1_meth_weights[g] + w2_step[g]
                for g in range(len(E_full)))
            lv, g_grad = val_grad(theta, combined)
            updates, opt_state = optimizer.update(g_grad, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
            flv = float(lv)
            losses[step] = flv
            if not switched and flv < lr_switch_threshold:
                optimizer = optax.adam(lr_switch)
                opt_state = optimizer.init(theta)
                switched = True
            if step % 200 == 0 or step == n_steps - 1:
                print(f"    [r14-3 {mode}] step {step:4d} | "
                      f"loss = {flv:.4e}")
        wall = time.time() - t0
        seed_traces[seed] = losses
        seed_finals[seed] = float(losses[-1])
        param_path = os.path.join(
            out_dir, f"d{d}_n{n_qudit}_dist{dist}_{n_layers}L_meth_"
                      f"{mode}_seed{seed}.npz")
        _save_npz(param_path, theta, d, n_qudit, K, dist, n_layers,
                  connections, losses, wall,
                  training_basis=f"Meth ({weights_meta['noise_model']}) "
                                  f"per-Pauli weights + stratified-{int(fraction*100)} w2",
                  connectivity=connectivity,
                  weights_meta=str(weights_meta))
        seed_param_paths[seed] = param_path
        print(f"    seed {seed} done in {wall:.0f}s; "
              f"final = {seed_finals[seed]:.4e}, "
              f"min = {float(np.min(losses)):.4e}")

    best_seed = min(seeds, key=lambda s: seed_finals[s])
    best_path = os.path.join(
        out_dir, f"d{d}_n{n_qudit}_dist{dist}_{n_layers}L_meth_"
                  f"{mode}_best.npz")
    shutil.copy2(seed_param_paths[best_seed], best_path)
    print(f"  best seed: {best_seed} ({seed_finals[best_seed]:.4e})")

    # Training-curves plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, n_steps + 1)
    for s in seeds:
        bsf = np.minimum.accumulate(seed_traces[s])
        ax.plot(x, bsf, lw=1.2, alpha=0.7, label=f"seed {s}")
    ax.axhline(0.05, color="grey", linestyle=":", lw=0.7,
               label="0.05 working-code")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("Meth-weighted KL loss (best-so-far)")
    ax.set_title(f"R14-3 {mode}: Meth-weighted training, "
                 f"(({n_qudit},{K},{dist}))_{d} a2a {n_layers}L")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    train_fig = os.path.join(fig_dir,
                              f"r14_3_meth_{mode}_training.png")
    fig.savefig(train_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {train_fig}")

    # Per-seed traces .npz
    traces_path = os.path.join(out_dir, "training_loss_traces.npz")
    np.savez(traces_path,
             **{f"seed_{s}": seed_traces[s] for s in seeds})
    print(f"  wrote {traces_path}")

    # SUMMARY.md
    fs05 = {s: _first_step_below(seed_traces[s], 0.05) for s in seeds}
    summary_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(f"# R14-3 {mode} — SUMMARY\n\n")
        f.write(f"((9,3,3))_3 a2a {n_layers}-layer with Meth ({mode}) "
                f"per-Pauli importance weights + stratified-{int(fraction*100)} "
                f"w2 sampling.\n\n")
        f.write(f"- Weights source: `{weights_npz}`\n")
        f.write(f"- noise_model = {weights_meta['noise_model']}\n")
        f.write(f"- η = {weights_meta['eta']}\n")
        f.write(f"- role_approximation = {weights_meta['role_approximation']}\n")
        f.write(f"- Σ λ = {lambda_meth_flat.sum():.6f}, "
                f"std/mean = {lambda_meth_flat.std()/max(lambda_meth_flat.mean(),1e-30):.3f}\n\n")
        f.write("| seed | final | min | first-step-<0.05 |\n")
        f.write("|---:|---:|---:|---:|\n")
        for s in seeds:
            fs = fs05[s]
            fs_str = str(fs) if fs is not None else "—"
            f.write(f"| {s} | {seed_finals[s]:.4e} | "
                    f"{float(np.min(seed_traces[s])):.4e} | {fs_str} |\n")
        f.write(f"\nbest seed: **{best_seed}** "
                f"({seed_finals[best_seed]:.4e}).\n")
    print(f"  wrote {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["physical", "simplified"],
                   default="physical",
                   help="R14-3a (physical, headline) or R14-3b (simplified, sensitivity)")
    p.add_argument("--weights_npz",
                   default="results/round14_scoping/meth_pauli_weights.npz",
                   help="Path to Meth weights npz from meth_pauli_weights.py")
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=1500)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_switch", type=float, default=0.01)
    p.add_argument("--lr_switch_threshold", type=float, default=0.1)
    p.add_argument("--fraction", type=float, default=0.10,
                   help="w2 stratified sampling fraction (R13-4 winner)")
    p.add_argument("--out_dir_root", default="results/round14_scoping")
    p.add_argument("--fig_dir", default="figures")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    fig_dir = os.path.join(REPO, args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    if args.mode == "physical":
        out_dir = os.path.join(REPO, args.out_dir_root,
                                "r14_3a_meth_physical")
    else:
        out_dir = os.path.join(REPO, args.out_dir_root,
                                "r14_3b_meth_simplified")
    os.makedirs(out_dir, exist_ok=True)

    weights_npz = os.path.join(REPO, args.weights_npz)
    train_r14_3(args.mode, weights_npz, out_dir, fig_dir,
                args.n_layers, args.n_steps, seeds, args.lr,
                args.lr_switch, args.lr_switch_threshold, args.fraction)


if __name__ == "__main__":
    main()
