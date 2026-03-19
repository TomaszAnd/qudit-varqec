#!/usr/bin/env python3
"""Train VarQEC dephasing code, distance 3 (2 layers). ~5 hrs with 500 steps."""
import argparse
import warnings
warnings.filterwarnings("ignore")

import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
import numpy as np

from src.error_sets import build_dephasing_error_sets
from src.encoder import create_encoder
from src.kl_loss_fast import (
    kl_loss_minibatch, precompute_error_products_dedup, estimate_memory_mb,
    save_varqec_result
)


def main():
    parser = argparse.ArgumentParser(description="Train dephasing d=3 code")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing result even if it has more steps")
    args = parser.parse_args()

    N_QUDITS, DIM_QUDIT, DISTANCE = 5, 4, 3
    K = DIM_QUDIT
    N_LAYERS = 2
    LR, LR_SWITCH = 0.1, 0.02
    BATCH_FRACTION = 0.3

    E_det, E_corr = build_dephasing_error_sets(N_QUDITS, DISTANCE)
    print(f"E_det={len(E_det)}, E_corr={len(E_corr)}")

    encoder, connections = create_encoder(N_QUDITS, DIM_QUDIT)
    PARAMS_PER_LAYER = N_QUDITS * 15 + len(connections)

    M_products = precompute_error_products_dedup(E_corr)
    mem_mb = estimate_memory_mb(len(M_products), E_corr[0].shape[0])
    print(f"Precomputed {len(M_products)} unique error products ({mem_mb:.1f} MB)")

    rng = np.random.default_rng(args.seed + 1000)

    def loss_fn(params):
        return kl_loss_minibatch(params, encoder, E_det, M_products, K, DISTANCE,
                                 batch_fraction=BATCH_FRACTION, rng=rng)

    np.random.seed(args.seed)
    theta = pnp.array(
        np.random.uniform(0, 2 * np.pi, (N_LAYERS, PARAMS_PER_LAYER)),
        requires_grad=True
    )

    opt = AdamOptimizer(LR)
    lr_switched = False
    losses = []

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "params")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"dephasing_d{DISTANCE}_{N_LAYERS}layer_seed{args.seed}.npz")

    print(f"Training: dephasing d={DISTANCE}, {N_LAYERS} layers, seed={args.seed}, "
          f"{args.n_steps} steps, minibatch={BATCH_FRACTION}")
    t0 = time.time()

    for step in range(args.n_steps):
        t_step = time.time()
        theta, loss = opt.step_and_cost(loss_fn, theta)
        loss_val = float(loss)
        losses.append(loss_val)
        dt = time.time() - t_step

        if (not lr_switched) and loss_val < 0.1:
            opt.stepsize = LR_SWITCH
            lr_switched = True

        if step % 10 == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t0
            print(f"  step {step:3d} | loss={loss_val:.4e} | {dt:.1f}s/step | elapsed {elapsed:.0f}s")

        if loss_val < 1e-6:
            print(f"  CONVERGED at step {step}")
            break

        # Plateau detection: compare rolling averages over 100-step window
        if step >= 100:
            window_old = np.mean(losses[-100:-50])
            window_new = np.mean(losses[-50:])
            improvement = (window_old - window_new) / max(window_old, 1e-15)
            if improvement < 0.05:
                print(f"  PLATEAU at step {step}, loss={loss_val:.2e}")
                break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Achievable loss: {losses[-1]:.2e}")

    # Don't overwrite better results unless --force
    if not args.force and os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True)
        existing_steps = len(existing.get('losses', []))
        if existing_steps > len(losses):
            print(f"  Keeping existing result ({existing_steps} steps > {len(losses)} steps)")
            return

    save_varqec_result(save_path, theta, losses, "dephasing", DISTANCE, N_LAYERS,
                       metadata={"K": K, "n_qudits": N_QUDITS, "dim_qudit": DIM_QUDIT,
                                 "seed": args.seed})


if __name__ == "__main__":
    main()
