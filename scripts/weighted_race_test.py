#!/usr/bin/env python3
"""STAGE 2 — is RACING useful on the WEIGHTED (Meth-channel) objective?

Isolate racing: fix sampler=importance at a Fig-8-validated fraction; toggle ONLY
racing.
  Arm A (no racing): N seeds each trained to COMPLETION (= Fig 8 protocol), best-of-N.
  Arm B (racing):    same N seeds, same sampler/fraction, mercy-prune + trigger-cert.
Metric = total op-EVs to an EQUAL channel-LER (weighted-MAP, Wilson/bootstrap CI)
certified code. Also report the SEED->BASIN VARIANCE on the weighted objective (the
mechanism: low variance => racing moot => NO-GO "1 seed + sampler"; high => composes).

FULL corrected Meth noise always: the loss weights, the certification loss, and the
MAP-LER channel + prior are ALL computed live from the Gaussian-fixed
src.correlated_noise (via src.meth_weights). Writes -> results/best_practice_runs/
weighted_race_test/. nice -19, background for n=9.
"""
from __future__ import annotations
import os, sys, time, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
from src.seed_race import run_race

OUTDIR = os.path.join(REPO, "results", "best_practice_runs", "weighted_race_test")


def map_channel_ler(npz_path, d, n, eta=0.9296, n_shots=2000, seed=0):
    """Weighted-MAP channel LER under the CORRECTED Meth channel with the CORRECTED
    per-op prior (both from src.meth_weights / src.correlated_noise)."""
    from src.errors import ErrorModel
    from src.decoders._common import encoder_forward
    from src.decoders.weighted_map import (build_weighted_correction_set,
                                            simulate_ler_with_weighted_map)
    from src.simulation import make_correlated_dephasing_noise_fn
    from src.meth_weights import compute_meth_group_weights
    from benchmark_ler_meth_kraus import build_gate_pairs
    from benchmark_ler_meth_pauli import bootstrap_ler_ci

    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    cs, *_ = encoder_forward(npz_path, conns)
    E_full = ErrorModel(d=d, n_qudit=n, distance=3, closed=True).build_grouped(verbose=False)
    _, prior = compute_meth_group_weights(d, n, distance=3, eta=eta, n_max=2)  # corrected
    corr = build_weighted_correction_set(E_full, prior, n, d, max_weight=2)
    gp = build_gate_pairs(n, conns)
    nf = make_correlated_dephasing_noise_fn(n_qudits=n, d=d, gate_pairs=gp, eta=eta,
                                            n_max=5, noise_model='physical')
    out = simulate_ler_with_weighted_map(cs, nf, corr, n, d, n_shots=n_shots, seed=seed)
    ler, lo, hi = bootstrap_ler_ci(out, rng=np.random.default_rng(seed))
    return float(ler), (float(lo), float(hi))


def race(**kw):
    t = time.time()
    r = run_race(weighted=True, sampler="importance", noise="full",
                 connectivity="all-to-all", entangler="ms", verbose=False, **kw)
    r["wall_s"] = round(time.time() - t, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--frac", type=float, default=0.2, help="Fig-8-validated importance fraction")
    ap.add_argument("--n-shots", type=int, default=2000)
    ap.add_argument("--tag", default="n5")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    log = {"config": vars(args), "objective": "weighted (Meth channel, corrected)"}

    # ---- Arm A: N seeds to completion (best-of-N), no racing ----
    print(f"[Arm A] {args.seeds} seeds to completion, weighted importance@{args.frac} ...", flush=True)
    floorsA, opA, wallA, bestA = [], 0, 0.0, None
    for s in range(args.seeds):
        out = os.path.join(OUTDIR, f"{args.tag}_armA_s{s}.npz")
        r = race(d=args.d, n=args.n, distance=3, layers=args.layers, num_seeds=1,
                 steps=args.steps, target_loss=1e-9, sample_frac=args.frac,
                 first_seed=s, out=out)
        fl = r["result"]["full_loss"]; floorsA.append(fl); opA += r["race_op_evs"]; wallA += r["wall_s"]
        if bestA is None or fl < bestA[0]:
            bestA = (fl, out, s)
        print(f"  seed {s}: weighted floor={fl:.4e} op_evs={r['race_op_evs']} ({r['wall_s']}s)", flush=True)
    floorsA = np.array(floorsA)
    seed_var = float(floorsA.std() / floorsA.mean())
    target = float(bestA[0]) * 1.10
    log["arm_A"] = {"floors": floorsA.tolist(), "seed_var_std_over_mean": seed_var,
                    "best_floor": float(bestA[0]), "best_seed": bestA[2],
                    "total_op_evs": opA, "wall_s": round(wallA, 1), "target_for_B": target}
    print(f"[Arm A] seed floor variance std/mean={seed_var:.3f}; best={bestA[0]:.4e}; "
          f"total op-EVs={opA}; target_B={target:.4e}", flush=True)

    # ---- Arm B: race the same N seeds to target (mercy + trigger-cert) ----
    print(f"[Arm B] racing {args.seeds} seeds to target {target:.4e} ...", flush=True)
    outB = os.path.join(OUTDIR, f"{args.tag}_armB_winner.npz")
    rB = race(d=args.d, n=args.n, distance=3, layers=args.layers, num_seeds=args.seeds,
              steps=args.steps, target_loss=target, sample_frac=args.frac,
              trigger_margin=0.25, trigger_patience=5, first_seed=0, out=outB)
    log["arm_B"] = {"certified": rB["certified"],
                    "op_evs_at_first_certified": rB["op_evs_at_first_certified"],
                    "total_op_evs": rB["race_op_evs"], "wall_s": rB["wall_s"],
                    "winner_seed": (rB["result"]["seed"] if rB["result"] else None),
                    "winner_full_loss": (rB["result"]["full_loss"] if rB["result"] else None)}
    print(f"[Arm B] certified={rB['certified']} op_evs_1st={rB['op_evs_at_first_certified']} "
          f"total={rB['race_op_evs']} ({rB['wall_s']}s)", flush=True)

    # ---- confirmatory MAP channel-LER: Arm A best vs Arm B winner ----
    print("[LER] weighted-MAP channel LER (corrected) ...", flush=True)
    lerA, ciA = map_channel_ler(bestA[1], args.d, args.n, n_shots=args.n_shots)
    lerB, ciB = (map_channel_ler(outB, args.d, args.n, n_shots=args.n_shots)
                 if rB["result"] else (None, None))
    log["ler"] = {"armA_best": {"ler": lerA, "ci": ciA},
                  "armB_winner": {"ler": lerB, "ci": ciB}}
    print(f"[LER] Arm A best={lerA:.4e} {ciA} | Arm B winner={lerB} {ciB}", flush=True)

    # ---- verdict (VARIANCE-PRIMARY) ----
    # The mechanism: racing (a MULTI-seed technique) can only help if you genuinely
    # need multiple seeds -- i.e. if the seed->basin variance is high enough that a
    # single seed risks a bad basin. If the variance is low, ONE seed + importance
    # sampling already lands the good basin, so neither best-of-N nor racing is
    # warranted; Arm B beating best-of-N then merely reflects best-of-N wasting the
    # N-1 redundant seeds (racing recovers ~1-seed cost), which is NOT a racing win.
    SEED_VAR_THRESH = 0.15
    ler_equal = (lerB is not None and ciA[0] <= ciB[1] and ciB[0] <= ciA[1])
    b_first = rB["op_evs_at_first_certified"]
    cheaper_than_bestofN = (b_first is not None and b_first < opA)
    if seed_var < SEED_VAR_THRESH:
        go = False
        explanation = (f"NO-GO (racing moot): weighted-objective seed floors are "
                       f"near-identical (std/mean={seed_var:.3f} < {SEED_VAR_THRESH}) "
                       f"-> ONE seed + importance sampling suffices; you would not run "
                       f"N seeds. Arm B's op-EV win over best-of-N "
                       f"({round(opA/b_first,2) if b_first else 'n/a'}x) only reflects "
                       f"best-of-N wasting N-1 redundant seeds (Arm B op_evs_first "
                       f"~ 1-seed cost). Recommendation: run 1 seed + weighted "
                       f"importance sampling, no racing.")
    else:
        go = bool(rB["certified"] and ler_equal and cheaper_than_bestofN)
        explanation = ("GO: high seed variance (multiple seeds genuinely needed) AND "
                       "Arm B reaches an equal-LER certified code at fewer op-EVs than "
                       "best-of-N -- racing prunes bad-basin seeds early."
                       if go else
                       "NO-GO: seeds vary but Arm B did not reach an equal-LER "
                       "certified code more cheaply than best-of-N.")
    log["verdict"] = {
        "seed_var_std_over_mean": seed_var,
        "seed_var_threshold": SEED_VAR_THRESH,
        "variance_regime": ("low -> racing moot" if seed_var < SEED_VAR_THRESH
                            else "high -> racing may compose"),
        "arm_B_reaches_equal_LER_certified": bool(rB["certified"] and ler_equal),
        "op_evs_B_first_cert": b_first, "op_evs_A_total": opA,
        "op_evs_A_per_seed_approx": round(opA / max(len(floorsA), 1)),
        "op_ev_ratio_A_over_Bfirst": (round(opA / b_first, 2) if b_first else None),
        "GO": go, "explanation": explanation,
    }
    with open(os.path.join(OUTDIR, f"weighted_race_{args.tag}.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\n===== WEIGHTED RACE GATE =====")
    print(json.dumps(log["verdict"], indent=2))


if __name__ == "__main__":
    main()
