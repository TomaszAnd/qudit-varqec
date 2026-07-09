#!/usr/bin/env python3
"""Stage H go/no-go: full-batch racing vs importance-sampled racing on ((5,3,3))_3.

ALWAYS full Meth noise (--noise full: correlated dephasing (Gaussian-fixed) +
subspace depolarizing + amplitude damping as the ErrorModel full basis provides),
all-to-all, XX/YY MS. Never dephasing-only. If a run will not converge, the fix
is capacity/target (more layers/steps, target from a CONVERGED reference), never
a weaker noise model.

Protocol (docs/SAMPLING_BUDGET.md):
  1. Reference: a full-batch race (2 seeds) at the given layers/steps -> the best
     full-batch loss is the converged floor. target_loss = floor * 1.1.
     If the floor is not driven well down (> --converged-below), STOP and report:
     raise --layers (capacity), do NOT weaken the objective.
  2. Race (a) full-batch and (b) importance-sampled + full-batch certification to
     that SAME target, SAME seeds, SAME step budget.
  3. Equal-quality check: winner LER at p=0.05 (Wilson-comparable), plus certified
     full-batch loss.

--probe just runs step 1 (single seed) and reports the floor, for cheaply finding
a converging layer count. Writes params + results JSON under
results/best_practice_runs/stage_h/ only. Run nice -19, backgrounded.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from src.seed_race import run_race

D, N, DIST = 3, 5, 3
LER_P = 0.05
LER_SHOTS = 5000
OUTDIR = os.path.join(REPO, "results", "best_practice_runs", "stage_h")


def ler_at_p(npz_path, layers, p=LER_P, n_shots=LER_SHOTS):
    import jax
    import jax.numpy as jnp
    from src.jax_backend import create_jax_encoder
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import simulate_ler_with_correction_factored

    data = np.load(npz_path, allow_pickle=True)
    params = jnp.asarray(data["params"])
    connections = [[i, j] for i in range(N) for j in range(i + 1, N)]
    enc, _, _ = create_jax_encoder(N, D, connections=connections,
                                    use_scan=layers >= 3)
    enc_vm = jax.jit(jax.vmap(enc, in_axes=(None, 0)))
    code_states = np.asarray(enc_vm(params, jnp.arange(D)))
    single = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    noise = make_hardware_noise_fn(D, N, p)
    r = simulate_ler_with_correction_factored(
        code_states, noise, single, N, D, n_shots=n_shots, seed=42)
    return float(r["logical_error_rate"])


def summarize(tag, res, wall, ler):
    r = res["result"]
    return {
        "tag": tag, "sampler": res["sampler"], "certified": res["certified"],
        "certified_full_loss": (r["full_loss"] if r is not None else None),
        "op_evs_at_first_certified": res["op_evs_at_first_certified"],
        "total_race_op_evs": res["race_op_evs"],
        "total_ops_full_batch": res["total_ops"],
        "wall_s": round(wall, 1),
        "winner_seed": (r["seed"] if r is not None else None),
        "ler_at_p0.05": ler,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--trigger-margin", type=float, default=0.25)
    ap.add_argument("--trigger-patience", type=int, default=5)
    ap.add_argument("--converged-below", type=float, default=0.05,
                    help="reference floor must be below this to proceed")
    ap.add_argument("--probe", action="store_true",
                    help="only run the single-seed reference probe and report floor")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    common = dict(d=D, n=N, distance=DIST, layers=args.layers,
                  connectivity="all-to-all", entangler="ms", noise="full",
                  verbose=True)
    log = {"config": vars(args) | {"d": D, "n": N, "distance": DIST,
                                   "noise": "full", "ler_p": LER_P,
                                   "ler_shots": LER_SHOTS}}

    # ---- Phase 1: converged full-batch reference floor ----
    print(f"[phase 1] full-batch reference (L={args.layers}, {args.steps} steps) ...",
          flush=True)
    t0 = time.time()
    ref = run_race(num_seeds=(1 if args.probe else 2), steps=args.steps,
                   target_loss=1e-9, sample_frac=1.0, sampler="uniform",
                   out=None, first_seed=0, **common)
    floor = float(ref["result"]["full_loss"])
    log["phase1"] = {"floor_full_loss": floor, "wall_s": round(time.time() - t0, 1)}
    print(f"[phase 1] floor={floor:.4e}  ({log['phase1']['wall_s']}s)", flush=True)

    if args.probe:
        print(json.dumps(log, indent=2))
        print(f"\nPROBE floor at L={args.layers}: {floor:.4e} "
              f"({'CONVERGED' if floor < args.converged_below else 'NOT converged -- raise --layers'})")
        return

    if floor >= args.converged_below:
        log["decision"] = "STOP-NOT-CONVERGED"
        with open(os.path.join(OUTDIR, "stage_h_results.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nSTOP: reference floor {floor:.4e} >= {args.converged_below} at "
              f"L={args.layers}. Full-Meth ((5,3,3))_3 did not converge in budget; "
              f"raise --layers (capacity), do NOT weaken the noise model.")
        return

    target = floor * 1.10
    log["phase1"]["target_loss"] = target
    print(f"[phase 1] target={target:.4e}", flush=True)

    # ---- Phase 2a: full-batch racing ----
    print("[phase 2a] full-batch racing to target ...", flush=True)
    out_a = os.path.join(OUTDIR, "race_fullbatch_d3_n5_dist3.npz")
    ta = time.time()
    res_a = run_race(num_seeds=args.seeds, steps=args.steps, target_loss=target,
                     sample_frac=1.0, sampler="uniform", out=out_a,
                     first_seed=0, **common)
    wall_a = time.time() - ta

    # ---- Phase 2b: importance-sampled racing + full-batch certification ----
    print("[phase 2b] importance-sampled racing to target ...", flush=True)
    out_b = os.path.join(OUTDIR, "race_importance_d3_n5_dist3.npz")
    tb = time.time()
    res_b = run_race(num_seeds=args.seeds, steps=args.steps, target_loss=target,
                     sample_frac=args.sample_frac, sampler="importance",
                     trigger_margin=args.trigger_margin,
                     trigger_patience=args.trigger_patience, out=out_b,
                     first_seed=0, **common)
    wall_b = time.time() - tb

    # ---- Phase 3: equal-quality LER check ----
    print("[phase 3] LER of both winners ...", flush=True)
    ler_a = ler_at_p(out_a, args.layers) if res_a["result"] is not None else None
    ler_b = ler_at_p(out_b, args.layers) if res_b["result"] is not None else None
    log["full_batch"] = summarize("full_batch", res_a, wall_a, ler_a)
    log["importance"] = summarize("importance", res_b, wall_b, ler_b)

    a, b = log["full_batch"], log["importance"]
    go = (a["certified"] and b["certified"]
          and a["op_evs_at_first_certified"] is not None
          and b["op_evs_at_first_certified"] is not None
          and b["op_evs_at_first_certified"] < a["op_evs_at_first_certified"])
    log["decision"] = "GO" if go else "NO-GO"
    if (a["op_evs_at_first_certified"] and b["op_evs_at_first_certified"]):
        log["op_ev_speedup_to_first_certified"] = round(
            a["op_evs_at_first_certified"] / b["op_evs_at_first_certified"], 2)

    with open(os.path.join(OUTDIR, "stage_h_results.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\n===== STAGE H RESULT =====")
    print(json.dumps(log, indent=2))
    print(f"\nDECISION: {log['decision']}")


if __name__ == "__main__":
    main()
