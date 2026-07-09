#!/usr/bin/env python3
"""Stage H' — sampling-budget knee sweep (find f*), per-seed PAIRED design.

FULL Meth noise always (--noise full). The full-basis KL loss has large seed-to-
basin variance (some inits reach ~0.05, others plateau ~0.37), so an absolute
target set from one seed set is not comparable across another. This sweep is
therefore PAIRED per seed: for each seed s we first measure its own full-batch
floor_s (f=1.0, unreachable target -> runs to completion), set target_s =
floor_s * 1.1, and then run every fraction f from the SAME init with target_s.
Reducing the budget is thus tested against that seed's OWN basin — the knob
(src/seed_race.py `sample_frac`) drives both gradient and trigger; certification
is always full-batch.

Primary knee finder = the basin-matched degradation ratio best_loss(f,s)/floor_s
(1.0 = no degradation), averaged over seeds (mean +/- 95% CI). Certified rate,
op-EVs to first-cert, trigger-fire & false-trigger rates tracked per fraction.
LER@p=0.05 (Wilson 95%) computed ONLY at the knee f* and the full-batch anchor.

Sub-knee failure class: gradient-corruption (ratio >> 1, worse basin) vs
trigger-starvation (ratio ~ 1 but low cert / low fire / high false-trigger).

Writes params/JSON to results/best_practice_runs/stage_hprime/ only. nice -19, bg.
"""
from __future__ import annotations
import os
import sys
import time
import json
import math
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from src.seed_race import run_race

OUTDIR = os.path.join(REPO, "results", "best_practice_runs", "stage_hprime")
KNEE_TOL = 0.10     # degradation ratio within 10% of 1.0 = "full-batch-equivalent"
LER_P = 0.05
LER_SHOTS = 5000


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def mean_ci(xs, z=1.96):
    xs = np.asarray(xs, float)
    m = float(xs.mean())
    if len(xs) < 2:
        return m, 0.0
    return m, z * float(xs.std(ddof=1)) / math.sqrt(len(xs))


def ler_at_p(npz_path, n, layers, d=3, p=LER_P, n_shots=LER_SHOTS):
    import jax
    import jax.numpy as jnp
    from src.jax_backend import create_jax_encoder
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import simulate_ler_with_correction_factored
    data = np.load(npz_path, allow_pickle=True)
    params = jnp.asarray(data["params"])
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    enc, _, _ = create_jax_encoder(n, d, connections=conns, use_scan=layers >= 3)
    cs = np.asarray(jax.jit(jax.vmap(enc, in_axes=(None, 0)))(params, jnp.arange(d)))
    single = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    r = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(d, n, p), single, n, d,
        n_shots=n_shots, seed=42)
    k = int(round(r["logical_error_rate"] * n_shots))
    return float(r["logical_error_rate"]), wilson(k, n_shots)


def one_run(f, seed, args, target, save=None):
    sampler = "uniform" if f >= 1.0 else "importance"
    t = time.time()
    res = run_race(d=args.d, n=args.n, distance=3, layers=args.layers, num_seeds=1,
                   steps=args.steps, target_loss=target, sample_frac=f,
                   sampler=sampler, trigger_margin=args.trigger_margin,
                   trigger_patience=args.trigger_patience, out=save,
                   first_seed=seed, verbose=False, connectivity="all-to-all",
                   entangler="ms", noise="full")
    r = res["result"]
    return {
        "f": f, "seed": seed, "sampler": sampler,
        "best_full_loss": float(r["full_loss"]),
        "certified": bool(res["certified"]),
        "race_op_evs": int(res["race_op_evs"]),
        "op_evs_first_cert": res["op_evs_at_first_certified"],
        "n_trigger_attempts": int(res["n_trigger_attempts"]),
        "n_false_triggers": int(res["n_false_triggers"]),
        "wall_s": round(time.time() - t, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--trigger-margin", type=float, default=0.25)
    ap.add_argument("--trigger-patience", type=int, default=5)
    ap.add_argument("--tag", default="n5")
    ap.add_argument("--outdir", default=None,
                    help="override output dir (default results/best_practice_runs/stage_hprime/)")
    ap.add_argument("--write-f-campaign", action="store_true",
                    help="after the sweep, compute f9* (smallest fraction with seed "
                         "yield >= the f=1.0 yield) and write results/best_practice_runs/"
                         "f_campaign.json with f_campaign = max(--f5-star, f9*).")
    ap.add_argument("--f5-star", type=float, default=0.40,
                    help="the n=5 knee, used for f_campaign = max(f5*, f9*).")
    args = ap.parse_args()
    fractions = [float(x) for x in args.fractions.split(",")]
    outdir = args.outdir or OUTDIR
    os.makedirs(outdir, exist_ok=True)
    log = {"config": vars(args) | {"fractions": fractions, "d": args.d, "distance": 3,
                                   "noise": "full", "connectivity": "all-to-all",
                                   "entangler": "ms", "knee_tol": KNEE_TOL,
                                   "design": "per-seed paired (target_s = floor_s*1.1)"}}

    # ---- per-seed paired sweep ----
    runs = []
    floors = {}
    best_npz = {}  # f -> (best_full_loss, npz)
    for s in range(args.seeds):
        # 1) this seed's own full-batch floor (unreachable target -> runs full)
        fr = one_run(1.0, s, args, target=1e-9, save=None)
        floor_s = fr["best_full_loss"]
        target_s = floor_s * 1.10
        floors[s] = floor_s
        print(f"[seed {s}] floor={floor_s:.4e} target={target_s:.4e}", flush=True)
        # 2) every fraction from the same init, certified against target_s
        for f in fractions:
            npz = os.path.join(outdir, f"{args.tag}_f{f}_s{s}.npz")
            r = one_run(f, s, args, target=target_s, save=npz)
            r["floor_s"] = floor_s
            r["degradation"] = r["best_full_loss"] / floor_s
            runs.append(r)
            if f not in best_npz or r["best_full_loss"] < best_npz[f][0]:
                best_npz[f] = (r["best_full_loss"], npz)
            print(f"  f={f} s={s}: best={r['best_full_loss']:.4f} "
                  f"deg={r['degradation']:.2f} cert={r['certified']} "
                  f"first_cert={r['op_evs_first_cert']} "
                  f"trig={r['n_trigger_attempts']} false={r['n_false_triggers']} "
                  f"({r['wall_s']}s)", flush=True)
    log["floors"] = floors
    log["runs"] = runs

    # ---- aggregate per fraction (basin-matched) ----
    agg = {}
    for f in fractions:
        rf = [r for r in runs if r["f"] == f]
        degs = [r["degradation"] for r in rf]
        m, ci = mean_ci(degs)
        certs = [r["certified"] for r in rf]
        fired = [r["n_trigger_attempts"] > 0 for r in rf]
        false_rate = [r["n_false_triggers"] / max(1, r["n_trigger_attempts"]) for r in rf]
        fc = [r["op_evs_first_cert"] for r in rf if r["op_evs_first_cert"] is not None]
        agg[f] = {
            "mean_degradation": m, "deg_ci95": ci,
            "mean_best_loss": float(np.mean([r["best_full_loss"] for r in rf])),
            "cert_rate": float(np.mean(certs)),
            "mean_total_op_evs": float(np.mean([r["race_op_evs"] for r in rf])),
            "mean_op_evs_first_cert": (float(np.mean(fc)) if fc else None),
            "n_certified": len(fc),
            "trigger_fire_rate": float(np.mean(fired)),
            "false_trigger_rate": float(np.mean(false_rate)),
        }
    log["aggregate"] = agg

    # ---- knee: smallest f with degradation ~1 AND full cert ----
    knee = 1.0
    for f in sorted(fractions):
        a = agg[f]
        if a["cert_rate"] >= 0.999 and a["mean_degradation"] <= 1.0 + KNEE_TOL:
            knee = f
            break
    log["knee_f_star"] = knee

    cls = {}
    for f in sorted(fractions):
        a = agg[f]
        if f >= knee:
            cls[f] = "at-or-above-knee"
        elif a["mean_degradation"] > 1.5:
            cls[f] = "gradient-corruption"
        elif a["cert_rate"] < 0.999 and (a["trigger_fire_rate"] < 0.999
                                          or a["false_trigger_rate"] > 0.5):
            cls[f] = "trigger-starvation"
        else:
            cls[f] = "borderline"
    log["failure_class"] = cls

    # ---- LER only at f* and full-batch anchor ----
    ler = {}
    for f in sorted({knee, 1.0}):
        _, npz = best_npz[f]
        val, ci = ler_at_p(npz, args.n, args.layers, d=args.d)
        ler[f] = {"ler_p05": val, "wilson95": list(ci), "npz": os.path.basename(npz)}
        print(f"[LER] f={f}: {val:.4e} Wilson95={ci}", flush=True)
    log["ler"] = ler
    if agg[1.0]["mean_op_evs_first_cert"] and agg[knee]["mean_op_evs_first_cert"]:
        log["op_ev_saving_at_fstar_vs_fullbatch"] = round(
            agg[1.0]["mean_op_evs_first_cert"] / agg[knee]["mean_op_evs_first_cert"], 2)

    # ---- f9* + f_campaign (JOB A handoff to the campaign) ----
    if args.write_f_campaign:
        yield_10 = agg[1.0]["cert_rate"]           # the full-batch seed yield
        f9 = 1.0
        for f in sorted(fractions):
            if agg[f]["cert_rate"] >= yield_10 - 1e-9:  # holds yield >= full-batch
                f9 = f
                break
        f_campaign = max(args.f5_star, f9)
        log["f9_star"] = f9
        log["f_campaign"] = f_campaign
        f_path = os.path.join(REPO, "results", "best_practice_runs", "f_campaign.json")
        with open(f_path, "w") as fh:
            json.dump({"f5_star": args.f5_star, "f9_star": f9,
                       "f_campaign": f_campaign,
                       "full_batch_yield": yield_10,
                       "yield_by_fraction": {str(f): agg[f]["cert_rate"] for f in fractions},
                       "k_scaling_note": ("f9* > f5*: optimal fraction rises with the "
                                          "closure" if f9 > args.f5_star + 1e-9 else
                                          "f9* <= f5*: fraction does not rise with n"),
                       "source": f"stage_hprime_{args.tag}.json", "n": args.n,
                       "sampler": "importance", "noise": "full"}, fh, indent=2)
        print(f"\n[f_campaign] f5*={args.f5_star} f9*={f9} -> f_campaign={f_campaign} "
              f"(wrote {f_path})", flush=True)

    with open(os.path.join(outdir, f"stage_hprime_{args.tag}.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print("\n===== STAGE H' SWEEP =====")
    print(json.dumps({"knee_f_star": knee,
                      "mean_degradation": {f: round(agg[f]["mean_degradation"], 3) for f in fractions},
                      "cert_rate": {f: agg[f]["cert_rate"] for f in fractions},
                      "failure_class": cls, "ler": ler,
                      "op_ev_saving": log.get("op_ev_saving_at_fstar_vs_fullbatch")},
                     indent=2))


if __name__ == "__main__":
    main()
