#!/usr/bin/env python3
"""JOB B — Fig 6 cross-code campaign training (corrected full-Meth noise).

Re-train the 8 distance-3 codes with the seed race at the campaign sampler budget
f_campaign (read from results/best_practice_runs/f_campaign.json; FALL BACK to
full-batch f=1.0 if missing/ambiguous — never a guessed fraction). Always
--noise full (correlated Gaussian dephasing + subspace depol + damping error set),
all-to-all, XX/YY MS, importance-sampled racing + full-batch evaluation. Best code
over the seeds (lowest full-batch loss) is saved per code.

Codes ((n,K,3))_q with K=q=d (single logical qudit); L=16 = the full-noise depth
proven on ((5,3,3))_3 in Stage H'. Morning LER validation determines whether any
larger code needs more capacity.

Writes params/JSON to results/best_practice_runs/campaign/ only.
Usage: campaign_train.py [--seeds 3] [--steps 2000] [--only d,n]
"""
from __future__ import annotations
import os, sys, time, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from src.seed_race import run_race

OUTDIR = os.path.join(REPO, "results", "best_practice_runs", "campaign")
# (d, n, distance, layers) — d==K (single logical qudit); Fig-6 distance-3 set.
CODES = [
    (3, 5, 3, 16), (3, 6, 3, 16), (3, 7, 3, 16), (3, 8, 3, 16), (3, 9, 3, 16),
    (4, 5, 3, 16), (4, 6, 3, 16), (5, 5, 3, 16),
]


def read_f_campaign():
    p = os.path.join(REPO, "results", "best_practice_runs", "f_campaign.json")
    try:
        d = json.load(open(p))
        f = float(d["f_campaign"])
        if 0.3 <= f <= 1.0:
            return f, "importance", f"f_campaign.json ({f})"
    except Exception as e:
        print(f"  f_campaign.json unusable ({e})", flush=True)
    return 1.0, "uniform", "FALLBACK full-batch (f=1.0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--only", default=None, help="comma d,n to run one code")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    frac, sampler, src = read_f_campaign()
    print(f"[campaign] budget f={frac} sampler={sampler} ({src}); seeds={args.seeds} steps={args.steps}", flush=True)

    codes = CODES
    if args.only:
        d0, n0 = (int(x) for x in args.only.split(","))
        codes = [c for c in CODES if c[0] == d0 and c[1] == n0]

    summary = {}
    for (d, n, dist, L) in codes:
        tag = f"d{d}_n{n}_dist{dist}_{L}L"
        out = os.path.join(OUTDIR, tag + ".npz")
        t = time.time()
        print(f"\n[code {tag}] ((n={n},K={d},dist={dist}))_{d} training ...", flush=True)
        try:
            res = run_race(d=d, n=n, distance=dist, layers=L, num_seeds=args.seeds,
                           steps=args.steps, target_loss=1e-9, sample_frac=frac,
                           sampler=sampler, trigger_margin=0.25, trigger_patience=5,
                           out=out, first_seed=0, verbose=False,
                           connectivity="all-to-all", entangler="ms", noise="full")
            r = res["result"]
            summary[tag] = {"best_full_loss": float(r["full_loss"]),
                            "seed": int(r["seed"]), "wall_s": round(time.time() - t, 1),
                            "frac": frac, "sampler": sampler, "npz": tag + ".npz"}
            print(f"[code {tag}] best_full_loss={r['full_loss']:.4e} "
                  f"({summary[tag]['wall_s']}s)", flush=True)
        except Exception as e:
            summary[tag] = {"FAILED": str(e), "wall_s": round(time.time() - t, 1)}
            print(f"[code {tag}] FAILED: {e}", flush=True)
        json.dump(summary, open(os.path.join(OUTDIR, "campaign_summary.json"), "w"), indent=2)
    print("\n[campaign] DONE", flush=True)


if __name__ == "__main__":
    main()
