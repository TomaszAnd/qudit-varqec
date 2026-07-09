#!/usr/bin/env python3
"""STAGE 3 (NO-GO branch) — corrected-noise Fig 8: sampler loss-vs-op-EV, STRAIGHT.

Racing was found moot on the weighted objective (low seed->basin variance, 2.7%/4.1%
at n=5/n=9; see docs/SAMPLING_BUDGET.md), so Fig 8 is reproduced with the samplers
straight (no racing): one seed per sampler on the Meth-CHANNEL-WEIGHTED KL under the
CORRECTED Gaussian channel, logging the FULL-BATCH weighted loss vs cumulative op-EVs.

Samplers (subset of Fig 8; the two full-basis-IS variants need the w1/w2-mask path,
deferred + noted): full-batch, stratified@0.2, importance@0.2, importance@0.35.
The Fig-8 message = importance reaches the same weighted-loss basin at fewer op-EVs.

Writes traces + status to results/best_practice_runs/fig8_weighted/. nice -19, bg.
Supersedes the pre-fix committed Fig 8 (keep old CSVs, flagged pre-fix; never
overwrite published params).
"""
from __future__ import annotations
import os, sys, time, json, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np

OUTDIR = os.path.join(REPO, "results", "best_practice_runs", "fig8_weighted")
STATUS = os.path.join(OUTDIR, "status.json")


def write_status(d):
    json.dump(d, open(STATUS, "w"), indent=2)


def train_one(sampler, frac, args, log_every=25):
    """Single-seed weighted training; returns (steps, loss_trace, opev_trace)."""
    import jax, jax.numpy as jnp, optax
    from src.seed_race import build_race
    from src.jax_backend import stratified_weights
    from src.sampling.stratified_importance import make_stratified_importance_weights

    (val_grad_fn, full_loss_fn, group_sizes, ppl, _,
     full_weights, sgw) = build_race(args.d, args.n, 3, args.layers,
                                     weighted=True, connectivity="all-to-all")
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(frac * total_ops)))
    theta = jax.random.uniform(jax.random.PRNGKey(args.seed), (args.layers, ppl),
                               minval=0.0, maxval=2 * np.pi)
    rng = np.random.default_rng(args.seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=args.lr)
    st = opt.init(theta); stage = 0
    steps_l, loss_l, opev_l = [], [], []
    cum_ev = 0
    for step in range(args.steps):
        if sampler == "full_batch":
            w = full_weights; cum_ev += total_ops
        elif sampler == "stratified":
            w = stratified_weights(group_sizes, frac, rng)
            cum_ev += int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        else:  # importance
            w = make_stratified_importance_weights(sgw, budget, rng)
            cum_ev += int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        lv, g = val_grad_fn(theta, w)
        if step % log_every == 0 or step == args.steps - 1:
            fl = float(full_loss_fn(theta)); cum_ev += total_ops  # full-batch log eval
            steps_l.append(step); loss_l.append(fl); opev_l.append(cum_ev)
        upd, st = opt.update(g, st, theta); theta = optax.apply_updates(theta, upd)
        lvf = float(lv)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = args.lr / 5; stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = args.lr / 50; stage = 2
    return steps_l, loss_l, opev_l, total_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    # (name, sampler, frac)
    CONFIGS = [("full_batch", "full_batch", 1.0),
               ("stratified_0.2", "stratified", 0.2),
               ("importance_0.2", "importance", 0.2),
               ("importance_0.35", "importance", 0.35)]
    status = {"config": vars(args), "objective": "weighted (Meth, corrected)",
              "samplers": {}, "started": True}
    write_status(status)
    for name, sampler, frac in CONFIGS:
        t = time.time()
        print(f"[fig8] training {name} (sampler={sampler} frac={frac}) ...", flush=True)
        status["samplers"][name] = {"status": "RUNNING"}; write_status(status)
        try:
            steps_l, loss_l, opev_l, tops = train_one(sampler, frac, args)
            np.savez(os.path.join(OUTDIR, f"{name}.npz"),
                     steps=np.array(steps_l), loss=np.array(loss_l),
                     op_evs=np.array(opev_l), sampler=sampler, frac=frac,
                     total_ops=tops, n=args.n, d=args.d)
            status["samplers"][name] = {"status": "DONE", "wall_s": round(time.time()-t, 1),
                                        "min_loss": float(min(loss_l)),
                                        "final_op_evs": int(opev_l[-1])}
            print(f"[fig8] {name} DONE min_loss={min(loss_l):.3e} "
                  f"final_opEV={opev_l[-1]} ({round(time.time()-t,1)}s)", flush=True)
        except Exception as e:
            status["samplers"][name] = {"status": "FAILED", "error": str(e)}
            print(f"[fig8] {name} FAILED: {e}", flush=True)
        write_status(status)
    status["done"] = True; write_status(status)
    print("[fig8] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
