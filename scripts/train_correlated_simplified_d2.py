#!/usr/bin/env python3
"""Train VarQEC correlated-simplified code, distance 2 (2 layers). ~10 min."""
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

from src.encoder import create_encoder
from src.correlated_error_sets import build_correlated_error_set
from src.kl_loss_fast import kl_loss_diagonal_minibatch, save_varqec_result


def main():
    parser = argparse.ArgumentParser(description="Train correlated-simplified d=2 code")
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing result even if it has more steps")
    args = parser.parse_args()

    N_QUDITS, DIM_QUDIT, DISTANCE = 5, 4, 2
    K = DIM_QUDIT
    N_LAYERS = 2
    LR = 0.05
    LR_SWITCH_THRESHOLD = 0.01
    LR_SWITCH = 0.01
    BATCH_FRACTION = 0.01  # cap at ~250 sampled ops from ~25k
    ETA = 0.95
    N_MAX = 2

    gate_pairs = [(i, i + 1, 0, 1) for i in range(4)]  # chain topology

    E_det, E_corr = build_correlated_error_set(
        n_qudits=N_QUDITS, d=DIM_QUDIT, gate_pairs=gate_pairs,
        eta=ETA, n_max=N_MAX, noise_model="simplified"
    )
    print(f"E_det={len(E_det)} diagonal operators")

    # Effective batch fraction: cap at ~250 sampled ops
    eff_bf = min(BATCH_FRACTION, 250 / max(len(E_det), 1))
    eff_bf = max(eff_bf, 0.005)
    n_sampled = max(1, int(len(E_det) * eff_bf))
    print(f"Sampling {n_sampled}/{len(E_det)} ops per step ({eff_bf:.1%})")

    encoder, connections = create_encoder(N_QUDITS, DIM_QUDIT)
    PARAMS_PER_LAYER = N_QUDITS * 15 + len(connections)

    rng = np.random.default_rng(args.seed + 1000)

    def loss_fn(params):
        return kl_loss_diagonal_minibatch(
            params, encoder, E_det, K, DISTANCE,
            batch_fraction=eff_bf, rng=rng
        )

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
    save_path = os.path.join(save_dir,
                             f"correlated_simplified_d{DISTANCE}_{N_LAYERS}layer_seed{args.seed}.npz")

    print(f"Training: correlated-simplified d={DISTANCE}, {N_LAYERS} layers, "
          f"seed={args.seed}, {args.n_steps} steps, lr={LR}")
    t0 = time.time()

    for step in range(args.n_steps):
        t_step = time.time()
        theta, loss = opt.step_and_cost(loss_fn, theta)
        loss_val = float(loss)
        losses.append(loss_val)
        dt = time.time() - t_step

        if (not lr_switched) and loss_val < LR_SWITCH_THRESHOLD:
            opt.stepsize = LR_SWITCH
            lr_switched = True
            print(f"  [step {step}] loss={loss_val:.3e} -> switch lr -> {LR_SWITCH}")

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

    save_varqec_result(save_path, theta, losses, "correlated_simplified", DISTANCE, N_LAYERS,
                       metadata={"K": K, "n_qudits": N_QUDITS, "dim_qudit": DIM_QUDIT,
                                 "seed": args.seed, "eta": ETA, "n_max": N_MAX,
                                 "batch_fraction": eff_bf})


if __name__ == "__main__":
    main()
