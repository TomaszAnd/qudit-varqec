#!/usr/bin/env python3
"""
Unified VarQEC training script.

Usage:
    python3 scripts/train.py --d 3 --n 5 --distance 3 --layers 10 --backend jax
    python3 scripts/train.py --d 5 --n 5 --distance 2 --layers 2 --steps 1000
    python3 scripts/train.py --d 3 --n 7 --distance 3 --layers 8 --backend jax --seeds 0,1,2
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Train a VarQEC code")
    p.add_argument("--d", type=int, required=True, help="Qudit dimension")
    p.add_argument("--n", type=int, required=True, help="Number of qudits")
    p.add_argument("--distance", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_switch", type=float, default=0.01)
    p.add_argument("--batch_fraction", type=float, default=0.3)
    p.add_argument("--backend", choices=["pennylane", "jax"], default="jax")
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated seeds. Best across seeds is kept.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    d, n, dist = args.d, args.n, args.distance
    K = d
    dim = d ** n

    closed = dist >= 3
    use_factored = dim > 500 or d >= 5 or dist >= 3
    force_manual = n >= 4

    print(f"=== VarQEC Training: (({n},{K},{dist}))_{d} ===")
    print(f"dim={dim}, layers={args.layers}, backend={args.backend}")
    print(f"closed={closed}, factored={use_factored}")

    from src.errors import ErrorModel
    model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=closed)
    print(f"Error model: {model}")

    if use_factored:
        E_det_f, E_corr_f = model.build_factored()
        E_det_dense = None
        print(f"E_det (factored): {len(E_det_f)} errors")
    else:
        E_det_dense, E_corr = model.build_dense()
        E_det_f = None
        print(f"E_det (dense): {len(E_det_dense)} errors")

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "params")
    os.makedirs(save_dir, exist_ok=True)

    best_loss = 1e10
    best_params = None
    best_losses = None
    best_seed = None

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()

        if args.backend == "jax":
            params, losses = _train_jax(
                d, n, dist, K, E_det_dense, E_det_f, use_factored,
                args.layers, args.steps, args.lr, args.lr_switch, seed)
        else:
            params, losses = _train_pennylane(
                d, n, dist, K, E_det_dense, E_det_f, use_factored,
                args.layers, args.steps, args.lr, args.batch_fraction,
                force_manual, seed)

        elapsed = time.time() - t0
        final = losses[-1] if losses else 1e10
        best_in_run = min(losses) if losses else 1e10
        print(f"  Seed {seed}: final={final:.4e}, best={best_in_run:.4e}, "
              f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")

        if best_in_run < best_loss:
            best_loss = best_in_run
            best_params = params
            best_losses = losses
            best_seed = seed

    tag = f"d{d}_n{n}_dist{dist}_{args.layers}L"
    if len(seeds) > 1:
        tag += f"_best{len(seeds)}s"
    save_path = os.path.join(save_dir, f"{tag}_seed{best_seed}.npz")

    if not args.force and os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True)
        if float(existing.get('final_loss', 1e10)) < best_loss:
            print(f"\nKeeping existing result (lower loss)")
            return

    from src.loss import save_varqec_result
    save_varqec_result(save_path, best_params, best_losses,
                       f"native_d{d}", dist, args.layers,
                       metadata={"K": K, "n_qudit": n, "d": d,
                                 "seed": best_seed, "backend": args.backend,
                                 "closed": closed, "factored": use_factored})

    print(f"\nBest: seed={best_seed}, loss={best_loss:.4e}")


def _train_jax(d, n, dist, K, E_det_dense, E_det_f, use_factored,
               n_layers, n_steps, lr, lr_switch, seed):
    """Train with JAX backend."""
    from src.jax_backend import (
        create_jax_encoder, create_jax_loss_scan,
        create_jax_loss_factored, train_jax)

    use_scan = n_layers >= 3
    enc, conns, ppl = create_jax_encoder(n, d, use_scan=use_scan)

    if use_factored:
        # Loop-based factored loss: slower per-step but compiles in minutes not hours
        loss_fn = create_jax_loss_factored(enc, E_det_f, n, d, K, dist)
    else:
        loss_fn = create_jax_loss_scan(enc, E_det_dense, K, dist)

    print(f"  JAX (scan_enc={use_scan}, factored={use_factored}), "
          f"{ppl} params/layer, total={n_layers * ppl}")

    params, losses = train_jax(loss_fn, n_layers, ppl, n_steps=n_steps,
                               lr=lr, lr_switch=lr_switch, seed=seed)
    return params, losses


def _train_pennylane(d, n, dist, K, E_det_dense, E_det_f, use_factored,
                     n_layers, n_steps, lr, batch_fraction, force_manual, seed):
    """Train with PennyLane backend."""
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer
    from src.encoder import create_native_encoder
    from src.loss import (kl_loss_detection_minibatch, kl_loss_fast,
                          kl_loss_detection_factored_minibatch)

    enc, conns, ppl = create_native_encoder(n, d, force_manual=force_manual)
    rng = np.random.default_rng(seed + 1000)

    if use_factored and dist >= 3:
        def loss_fn(params):
            return kl_loss_detection_factored_minibatch(
                params, enc, E_det_f, K, dist, n, d,
                batch_fraction=batch_fraction, rng=rng)
    elif dist >= 3:
        def loss_fn(params):
            return kl_loss_detection_minibatch(
                params, enc, E_det_dense, K, dist,
                batch_fraction=batch_fraction, rng=rng)
    else:
        if E_det_dense is None:
            from src.errors import build_native_error_set
            E_det_dense, _ = build_native_error_set(n, d, dist)
        def loss_fn(params):
            return kl_loss_fast(params, enc, E_det_dense, [], K, dist)

    np.random.seed(seed)
    theta = pnp.array(
        np.random.uniform(0, 2 * np.pi, (n_layers, ppl)), requires_grad=True)

    opt = AdamOptimizer(lr)
    lr_switched = False
    losses = []

    print(f"  PennyLane, {ppl} params/layer, total={n_layers * ppl}")

    for step in range(n_steps):
        theta, loss = opt.step_and_cost(loss_fn, theta)
        lv = float(loss)
        losses.append(lv)

        if not lr_switched and lv < 0.1:
            opt.stepsize = lr / 5
            lr_switched = True

        if step % 50 == 0:
            print(f"    step {step:4d} | loss={lv:.4e}")

        if lv < 1e-6:
            print(f"    CONVERGED at step {step}")
            break

        if step >= 500:
            w_old = np.mean(losses[-500:-250])
            w_new = np.mean(losses[-250:])
            if (w_old - w_new) / max(abs(w_old), 1e-15) < 0.02:
                print(f"    PLATEAU at step {step}, loss={lv:.2e}")
                break

    return np.array(theta), losses


if __name__ == "__main__":
    main()
