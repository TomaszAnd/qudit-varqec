#!/usr/bin/env python3
"""R14-7 §C — qudit-dimension scaling: VarQEC at d>3, n≤10, distance 3, a2a.

Demonstrates the VarQEC framework + stratified-IS training extend beyond
((9,3,3))_3 to higher qudit dimension. Default target ((7,1,3))_4 (7 ququarts,
K=d=4, distance 3, basis 925, dim 16384 — green per §A feasibility).

**Uniform-prior, not channel-aware.** No empirical Meth calibration exists for
d=4, so weights are uniform over the weight-≤2 closure basis (standard VarQEC
loss). This demonstrates *feasibility at d=4*; channel-aware d=4 training waits
for Innsbruck d=4 calibration. With uniform weights the stratified-IS sampler
reduces to stratified-uniform within group (sampling ∝ uniform).

Mirrors train_r14_3b_meth_is.py; saves full metadata for §H weight-enumerator
analysis. nice -19. Run --n_steps 5 first for a JIT/timing feasibility check.
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
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n_qudit", type=int, default=7)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr_switch", type=float, default=0.01)
    ap.add_argument("--lr_switch_threshold", type=float, default=0.1)
    ap.add_argument("--w2_budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir",
                    default="results/round14_scoping/r14_7c_dim_scaling")
    ap.add_argument("--ckpt_every", type=int, default=200,
                    help="checkpoint every N steps (0=off; default 200 for R14-7c)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from existing checkpoint if present")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from jax import value_and_grad
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from loss_split_helpers import build_w1_w2_masks
    from src.sampling.stratified_importance import make_stratified_importance_weights
    from train_r14_3_meth import _flat_group_offsets, _build_meth_w1_weights, _save_npz

    d, n_qudit, dist = args.d, args.n_qudit, 3
    K = d  # one logical qudit
    connectivity = "a2a"
    connections = [[i, j] for i in range(n_qudit)
                   for j in range(i + 1, n_qudit)]
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = ErrorModel(d=d, n_qudit=n_qudit, distance=dist, closed=True)
    E_full = model.build_grouped(verbose=False)
    n_total = sum(int(g['matrices'].shape[0]) for g in E_full)
    offs = _flat_group_offsets(E_full)
    # Uniform prior over the basis (sum-to-1).
    lam = np.ones(n_total) / n_total

    enc, _, ppl = create_jax_encoder(n_qudit, d, connections=connections,
                                     use_scan=True)
    loss_w = build_varqec_loss_weighted(enc, K, d, n_qudit, dist, E_full)
    val_grad = jax.jit(value_and_grad(lambda th, w: loss_w(th, w), argnums=0))

    w1_mask, w2_mask = build_w1_w2_masks(E_full, d)
    w1_mask_np = [np.asarray(m) for m in w1_mask]
    w2_mask_np = [np.asarray(m) for m in w2_mask]
    w1_w = _build_meth_w1_weights(w1_mask_np, lam, offs, jnp)
    w2_group_weights = []
    for gi, m in enumerate(w2_mask_np):
        wg = np.zeros(m.shape[0])
        for e in range(m.shape[0]):
            if m[e] > 0:
                wg[e] = float(lam[offs[gi] + e])
        w2_group_weights.append(wg)

    seed = args.seed
    print(f"=== R14-7c dim-scaling seed {seed}: (({n_qudit},1,{dist}))_{d} a2a "
          f"{args.n_layers}L, basis {n_total}, dim {d**n_qudit}, uniform prior, "
          f"w2_budget {args.w2_budget} ===", flush=True)
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (args.n_layers, ppl), minval=0.0,
                               maxval=2 * np.pi)
    rng = np.random.default_rng(seed * 1000 + 42)
    opt = optax.adam(args.lr)
    ostate = opt.init(theta)
    switched = False
    losses = np.zeros(args.n_steps)
    ckpt_path = os.path.join(out_dir, f"seed{seed}_checkpoint.npz")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        theta = jnp.asarray(ck['theta'])
        start_step = int(ck['step']) + 1
        losses[:start_step] = ck['loss_trace'][:start_step]
        switched = bool(ck['switched'])
        cur_lr = args.lr_switch if switched else args.lr
        opt = optax.adam(cur_lr); ostate = opt.init(theta)
        print(f"  [resume] seed {seed} from step {start_step} "
              f"(loss={losses[start_step-1]:.4e}, lr={cur_lr})", flush=True)
    t0 = time.time()
    for step in range(start_step, args.n_steps):
        w2 = make_stratified_importance_weights(w2_group_weights, args.w2_budget, rng)
        combined = tuple(w1_w[g] + w2[g] for g in range(len(E_full)))
        lv, grad = val_grad(theta, combined)
        updates, ostate = opt.update(grad, ostate, theta)
        theta = optax.apply_updates(theta, updates)
        losses[step] = float(lv)
        if not switched and losses[step] < args.lr_switch_threshold:
            opt = optax.adam(args.lr_switch); ostate = opt.init(theta); switched = True
        if step % 200 == 0 or step < 5 or step == args.n_steps - 1:
            print(f"  [r14-7c d{d}n{n_qudit}] step {step:4d} | loss = "
                  f"{losses[step]:.4e} | {(time.time()-t0)/(step+1):.1f}s/step avg",
                  flush=True)
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            np.savez(ckpt_path, theta=np.asarray(theta), step=step,
                     loss_trace=losses[:step + 1], switched=switched,
                     d=d, n_qudit=n_qudit, K=K, distance=dist,
                     n_layers=args.n_layers,
                     connections=np.array(connections))
    wall = time.time() - t0
    path = os.path.join(out_dir,
                        f"d{d}_n{n_qudit}_dist{dist}_{args.n_layers}L_uniform_seed{seed}.npz")
    _save_npz(path, theta, d, n_qudit, K, dist, args.n_layers, connections,
              losses, wall,
              training_basis="uniform prior (d>3 feasibility) + stratified-IS w2",
              connectivity=connectivity,
              weights_meta=str({"sampler": "stratified_importance", "prior": "uniform",
                                "w2_budget": args.w2_budget}))
    print(f"  seed {seed} done in {wall:.0f}s; final={losses[-1]:.4e}, "
          f"min={float(np.min(losses)):.4e}; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
