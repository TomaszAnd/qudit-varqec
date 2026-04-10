#!/usr/bin/env python3
# Legacy: use scripts/train.py instead
"""Train VarQEC ((5,3,3))_3 with native trapped-ion gates, distance 3."""
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

from src.errors import build_native_error_set
from src.encoder import create_native_encoder
from src.loss import kl_loss_detection_minibatch, save_varqec_result


def main():
    parser = argparse.ArgumentParser(description="Train native ((5,3,3))_3 code")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_fraction", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    N_QUDIT, D, DISTANCE = 5, 3, 3
    K = D
    N_LAYERS = args.n_layers

    E_det, E_corr = build_native_error_set(N_QUDIT, D, DISTANCE, closed=True)
    print(f"E_det={len(E_det)}, E_corr={len(E_corr)}, dim={D**N_QUDIT}")

    encoder, connections, params_per_layer = create_native_encoder(
        N_QUDIT, D, force_manual=True)
    print(f"Params per layer: {params_per_layer}, total: {N_LAYERS * params_per_layer}")
    print(f"Connections: {connections}")

    rng = np.random.default_rng(args.seed + 1000)

    def loss_fn(params):
        return kl_loss_detection_minibatch(
            params, encoder, E_det, K, DISTANCE,
            batch_fraction=args.batch_fraction, rng=rng
        )

    np.random.seed(args.seed)
    theta = pnp.array(
        np.random.uniform(0, 2 * np.pi, (N_LAYERS, params_per_layer)),
        requires_grad=True
    )

    LR_SWITCH = args.lr / 5
    opt = AdamOptimizer(args.lr)
    lr_switched = False
    losses = []

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "params")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f"native_d3_dist{DISTANCE}_{N_LAYERS}layer_seed{args.seed}.npz")

    print(f"Training: native (({N_QUDIT},{K},{DISTANCE}))_3, {N_LAYERS} layer(s), "
          f"seed={args.seed}, {args.n_steps} steps, batch={args.batch_fraction}")
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
            print(f"  LR -> {LR_SWITCH}")

        if step % 10 == 0 or step == args.n_steps - 1:
            print(f"  step {step:4d} | loss={loss_val:.4e} | {dt:.1f}s/step")

        if loss_val < 1e-6:
            print(f"  CONVERGED at step {step}")
            break

        # Save periodically
        if step > 0 and step % 200 == 0:
            save_varqec_result(save_path, theta, losses, "native_hardware_d3", DISTANCE, N_LAYERS,
                               metadata={"K": K, "n_qudit": N_QUDIT, "d": D,
                                         "seed": args.seed, "connections": connections,
                                         "params_per_layer": params_per_layer,
                                         "batch_fraction": args.batch_fraction})

        # Plateau detection (lenient for noisy minibatch)
        if step >= 500:
            window_old = np.mean(losses[-500:-250])
            window_new = np.mean(losses[-250:])
            improvement = (window_old - window_new) / max(abs(window_old), 1e-15)
            if improvement < 0.02:
                print(f"  PLATEAU at step {step}, loss={loss_val:.2e}")
                break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final loss: {losses[-1]:.2e}")

    if not args.force and os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True)
        if float(existing.get('final_loss', 1e10)) < losses[-1]:
            print(f"  Keeping existing result (lower loss)")
            return

    save_varqec_result(save_path, theta, losses, "native_hardware_d3", DISTANCE, N_LAYERS,
                       metadata={"K": K, "n_qudit": N_QUDIT, "d": D,
                                 "seed": args.seed, "connections": connections,
                                 "params_per_layer": params_per_layer,
                                 "batch_fraction": args.batch_fraction})


if __name__ == "__main__":
    main()
