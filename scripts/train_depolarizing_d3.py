#!/usr/bin/env python3
"""
Train VarQEC depolarizing code, distance 3.

Uses FACTORED error representation to avoid 37 GB dense E_det storage.
Uses detection-style loss (VarQEC paper Eq. 16) — single sum over E_det,
no quadratic E_a†E_b products.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import gc
import time
import os
import sys
import platform
import resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
import numpy as np

from src.error_sets_factored import build_error_sets_factored
from src.encoder import create_encoder
from src.kl_loss_fast import kl_loss_detection_factored_minibatch, save_varqec_result


MEM_LIMIT_GB = 12

def mem_mb():
    """Current RSS in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        return rss / (1024 * 1024)  # bytes on macOS
    return rss / 1024  # KB on Linux


def main():
    parser = argparse.ArgumentParser(description="Train depolarizing d=3 code (factored, detection loss)")
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--batch_det", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    N_QUDITS, DIM_QUDIT, DISTANCE = 5, 4, 3
    K = DIM_QUDIT
    N_LAYERS = args.n_layers
    LR = args.lr
    LR_SWITCH = LR / 5

    E_det_f, _ = build_error_sets_factored(N_QUDITS, DISTANCE)
    print(f"E_det={len(E_det_f)} (factored)")
    print(f"  Memory for factored errors: ~{len(E_det_f) * 2 * 4 * 4 * 16 / 1024:.0f} KB "
          f"(vs ~37 GB dense)")

    encoder, connections = create_encoder(N_QUDITS, DIM_QUDIT)
    PARAMS_PER_LAYER = N_QUDITS * 15 + len(connections)

    n_sample = max(1, int(len(E_det_f) * args.batch_det))
    print(f"Detection loss: sampling {n_sample}/{len(E_det_f)} E_det per step (no E_a†E_b pairs)")

    rng = np.random.default_rng(args.seed + 1000)

    def loss_fn(params):
        return kl_loss_detection_factored_minibatch(
            params, encoder, E_det_f, K, DISTANCE,
            N_QUDITS, DIM_QUDIT,
            batch_fraction=args.batch_det, rng=rng)

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
                             f"depolarizing_d{DISTANCE}_{N_LAYERS}layer_seed{args.seed}.npz")

    print(f"Training: depolarizing d={DISTANCE}, {N_LAYERS} layers, seed={args.seed}, "
          f"{args.n_steps} steps, lr={LR}")
    print(f"  Total params: {N_LAYERS * PARAMS_PER_LAYER}")
    print(f"  Initial memory: {mem_mb():.0f} MB")
    t0 = time.time()

    for step in range(args.n_steps):
        t_step = time.time()
        theta, loss = opt.step_and_cost(loss_fn, theta)
        loss_val = float(loss)
        del loss
        gc.collect()
        losses.append(loss_val)
        dt = time.time() - t_step

        if (not lr_switched) and loss_val < 0.1:
            opt.stepsize = LR_SWITCH
            lr_switched = True
            print(f"  [step {step}] loss={loss_val:.3e} -> lr={LR_SWITCH}")

        if step % 10 == 0 or step < 5 or step == args.n_steps - 1:
            elapsed = time.time() - t0
            remaining = dt * (args.n_steps - step - 1)
            cur_mem = mem_mb()
            print(f"  step {step:3d} | loss={loss_val:.4e} | {dt:.1f}s/step | "
                  f"elapsed {elapsed:.0f}s | ~{remaining/60:.0f}min left | "
                  f"mem {cur_mem:.0f}MB")
            if cur_mem > MEM_LIMIT_GB * 1024:
                print(f"  MEMORY LIMIT ({MEM_LIMIT_GB} GB) — saving checkpoint and exiting")
                save_varqec_result(save_path, theta, losses, "depolarizing", DISTANCE, N_LAYERS,
                                   metadata={"K": K, "n_qudits": N_QUDITS, "dim_qudit": DIM_QUDIT,
                                             "seed": args.seed, "batch_det": args.batch_det,
                                             "detection_loss": True, "checkpoint": True})
                return

        # Periodic checkpoint every 50 steps
        if step > 0 and step % 50 == 0:
            save_varqec_result(save_path, theta, losses, "depolarizing", DISTANCE, N_LAYERS,
                               metadata={"K": K, "n_qudits": N_QUDITS, "dim_qudit": DIM_QUDIT,
                                         "seed": args.seed, "batch_det": args.batch_det,
                                         "detection_loss": True, "checkpoint": True})

        if loss_val < 1e-6:
            print(f"  CONVERGED at step {step}")
            break

        if step >= 300 and step % 50 == 0:
            window_old = np.percentile(losses[-200:-100], 25)
            window_new = np.percentile(losses[-100:], 25)
            improvement = (window_old - window_new) / max(window_old, 1e-15)
            if improvement < 0.01:
                print(f"  PLATEAU at step {step}, loss={loss_val:.2e} "
                      f"(p25 old={window_old:.2e}, new={window_new:.2e})")
                break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Achievable loss: {losses[-1]:.2e}")
    print(f"  Final memory: {mem_mb():.0f} MB")

    if not args.force and os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True)
        existing_steps = len(existing.get('losses', []))
        if existing_steps > len(losses):
            print(f"  Keeping existing result ({existing_steps} steps > {len(losses)} steps)")
            return

    save_varqec_result(save_path, theta, losses, "depolarizing", DISTANCE, N_LAYERS,
                       metadata={"K": K, "n_qudits": N_QUDITS, "dim_qudit": DIM_QUDIT,
                                 "seed": args.seed, "batch_det": args.batch_det,
                                 "detection_loss": True})


if __name__ == "__main__":
    main()
