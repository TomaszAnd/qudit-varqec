#!/usr/bin/env python3
"""R14-7 §B.2 — full-basis stratified-IS training (samples w1 AND w2).

Sub-100k path B: unlike R14-3b (w1 full-batch 36 + w2 sampled 54 = 90/step),
this samples BOTH sectors via make_stratified_importance_weights_full_basis:
  w1 sampled (budget 12) + w2 sampled (budget 54) = ~66 op-EVs/step
  → 66 × 1500 = 99k op-EVs/seed (sub-100k at full step count).

Mirrors train_r14_3b_meth_is.py; only the per-step weight construction changes.
nice -19. LER analysis (vs R14-3a/3b) is R14-8.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr_switch", type=float, default=0.01)
    ap.add_argument("--lr_switch_threshold", type=float, default=0.1)
    ap.add_argument("--w1_budget", type=int, default=12)
    ap.add_argument("--w2_budget", type=int, default=54)
    ap.add_argument("--out_dir",
                    default="results/round14_scoping/r14_7b_full_is")
    ap.add_argument("--ckpt_every", type=int, default=200,
                    help="checkpoint every N steps (0=off; default 200 for R14-7b)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from existing checkpoint if present")
    args = ap.parse_args()

    import jax, jax.numpy as jnp, optax
    from jax import value_and_grad
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from loss_split_helpers import build_w1_w2_masks
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights_full_basis, neyman_group_budgets)
    from src.decoders.priors import meth_pauli_prior
    from train_r14_3_meth import _flat_group_offsets, _save_npz

    d, n_qudit, dist, K = 3, 9, 3, 3
    connectivity = "a2a"
    connections = [[i, j] for i in range(n_qudit) for j in range(i + 1, n_qudit)]
    out_dir = os.path.join(REPO, args.out_dir); os.makedirs(out_dir, exist_ok=True)

    lam = meth_pauli_prior()
    E_full = ErrorModel(d=d, n_qudit=n_qudit, distance=dist, closed=True).build_grouped(verbose=False)
    offs = _flat_group_offsets(E_full)
    enc, _, ppl = create_jax_encoder(n_qudit, d, connections=connections, use_scan=True)
    loss_w = build_varqec_loss_weighted(enc, K, d, n_qudit, dist, E_full)
    val_grad = jax.jit(value_and_grad(lambda th, w: loss_w(th, w), argnums=0))

    w1_mask, w2_mask = build_w1_w2_masks(E_full, d)
    w1_mask_np = [np.asarray(m) for m in w1_mask]
    w2_mask_np = [np.asarray(m) for m in w2_mask]
    # Full Meth weight per group (both sectors); the sampler splits by mask.
    group_w = [lam[offs[g]:offs[g] + int(E_full[g]['matrices'].shape[0])]
               for g in range(len(E_full))]

    # report realized EVs/step
    w1gw = [np.where(m > 0, gw, 0.0) for gw, m in zip(group_w, w1_mask_np)]
    w2gw = [np.where(m > 0, gw, 0.0) for gw, m in zip(group_w, w2_mask_np)]
    realized = sum(neyman_group_budgets(w1gw, args.w1_budget)) + \
        sum(neyman_group_budgets(w2gw, args.w2_budget))
    print(f"=== R14-7b full-IS seed {args.seed}: ((9,3,3))_3 a2a {args.n_layers}L, "
          f"w1_budget {args.w1_budget} + w2_budget {args.w2_budget} → realized "
          f"{realized} op-EVs/step ({realized*args.n_steps/1000:.0f}k/seed) ===",
          flush=True)

    key = jax.random.PRNGKey(args.seed)
    theta = jax.random.uniform(key, (args.n_layers, ppl), minval=0.0, maxval=2 * np.pi)
    rng = np.random.default_rng(args.seed * 1000 + 42)
    opt = optax.adam(args.lr); ostate = opt.init(theta); switched = False
    losses = np.zeros(args.n_steps)
    ckpt_path = os.path.join(out_dir, f"seed{args.seed}_checkpoint.npz")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        theta = jnp.asarray(ck['theta'])
        start_step = int(ck['step']) + 1
        losses[:start_step] = ck['loss_trace'][:start_step]
        switched = bool(ck['switched'])
        cur_lr = args.lr_switch if switched else args.lr
        opt = optax.adam(cur_lr); ostate = opt.init(theta)
        print(f"  [resume] seed {args.seed} from step {start_step} "
              f"(loss={losses[start_step-1]:.4e}, lr={cur_lr})", flush=True)
    t0 = time.time()
    for step in range(start_step, args.n_steps):
        w = make_stratified_importance_weights_full_basis(
            group_w, w1_mask_np, w2_mask_np, args.w1_budget, args.w2_budget, rng)
        lv, grad = val_grad(theta, w)
        updates, ostate = opt.update(grad, ostate, theta)
        theta = optax.apply_updates(theta, updates)
        losses[step] = float(lv)
        if not switched and losses[step] < args.lr_switch_threshold:
            opt = optax.adam(args.lr_switch); ostate = opt.init(theta); switched = True
        if step % 200 == 0 or step == args.n_steps - 1:
            print(f"  [r14-7b full-IS] step {step:4d} | loss = {losses[step]:.4e}", flush=True)
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            np.savez(ckpt_path, theta=np.asarray(theta), step=step,
                     loss_trace=losses[:step + 1], switched=switched,
                     d=d, n_qudit=n_qudit, K=K, distance=dist,
                     n_layers=args.n_layers,
                     connections=np.array(connections))
    wall = time.time() - t0
    path = os.path.join(out_dir,
                        f"d{d}_n{n_qudit}_dist{dist}_{args.n_layers}L_meth_physical_seed{args.seed}.npz")
    _save_npz(path, theta, d, n_qudit, K, dist, args.n_layers, connections, losses, wall,
              training_basis=f"Meth (physical) full-basis stratified-IS (w1+w2, {realized} EV/step)",
              connectivity=connectivity,
              weights_meta=str({"sampler": "full_basis_IS", "w1_budget": args.w1_budget,
                                "w2_budget": args.w2_budget, "realized_ev_per_step": realized}))
    print(f"  seed {args.seed} done in {wall:.0f}s; final={losses[-1]:.4e}, "
          f"min={float(np.min(losses)):.4e}; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
