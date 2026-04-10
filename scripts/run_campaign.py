#!/usr/bin/env python3
"""
Run the full VarQEC code discovery campaign.

Usage:
    python3 scripts/run_campaign.py --wave 1             # wave 1 only
    python3 scripts/run_campaign.py --wave 2 --seeds 3   # wave 2, 3 seeds each
    python3 scripts/run_campaign.py --dry-run             # print configs only
"""
import subprocess
import os
import sys
import time
import json
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (d, n, dist, layers, steps, seeds, note)
CAMPAIGN = {
    1: [
        (3, 4, 2, 1, 500, 3, "qutrit 4-qudit"),
        (3, 5, 2, 1, 500, 3, "qutrit 5-qudit det baseline"),
        (3, 6, 2, 1, 1000, 3, "qutrit 6-qudit"),
        (3, 7, 2, 2, 1500, 3, "qutrit 7-qudit det"),
        (3, 8, 2, 2, 2000, 2, "qutrit 8-qudit det"),
        (4, 4, 2, 1, 500, 3, "ququart 4-qudit"),
        (4, 5, 2, 2, 1000, 3, "ququart 5-qudit"),
        (5, 3, 2, 1, 500, 3, "ququint 3-qudit"),
        (5, 4, 2, 1, 500, 3, "ququint 4-qudit"),
    ],
    2: [
        (3, 6, 3, 8, 5000, 3, "qutrit 6-qudit dist-3"),
        (3, 8, 3, 8, 5000, 2, "qutrit 8-qudit dist-3"),
        (4, 5, 3, 6, 5000, 3, "ququart 5-qudit dist-3"),
        (5, 5, 3, 6, 5000, 2, "ququint 5-qudit dist-3"),
    ],
    3: [
        (3, 7, 4, 10, 5000, 2, "distance-4 attempt"),
        (3, 9, 3, 10, 3000, 1, "9-qutrit stretch"),
    ],
}


def run_config(d, n, dist, layers, steps, n_seeds, note, dry_run=False):
    seeds = ",".join(str(i) for i in range(n_seeds))
    cmd = [
        sys.executable, "scripts/train.py",
        "--d", str(d), "--n", str(n), "--distance", str(dist),
        "--layers", str(layers), "--steps", str(steps),
        "--backend", "jax", "--seeds", seeds, "--force",
    ]

    tag = f"d{d}_n{n}_dist{dist}_{layers}L"
    print(f"\n{'='*60}")
    print(f"  {tag}: (({n},{d},{dist}))_{d}  [{note}]")
    print(f"  layers={layers}, steps={steps}, seeds={n_seeds}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  CMD: {' '.join(cmd)}")
        return {"config": tag, "status": "dry-run", "note": note}

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600 * 12)
        elapsed = time.time() - t0
        output = result.stdout

        best_loss = None
        for line in output.split("\n"):
            matches = re.findall(r'loss[=:](\d+\.?\d*(?:e[+-]?\d+)?)', line.lower())
            for m in matches:
                try:
                    val = float(m)
                    if best_loss is None or val < best_loss:
                        best_loss = val
                except ValueError:
                    pass

        status = ("converged" if best_loss and best_loss < 1e-4 else
                  "good" if best_loss and best_loss < 0.5 else
                  "partial" if best_loss and best_loss < 5 else "failed")

        print(f"  Result: {status}, loss={best_loss:.4e if best_loss else '?'}, "
              f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-200:]}")

        return {"config": tag, "d": d, "n": n, "dist": dist,
                "layers": layers, "best_loss": best_loss,
                "elapsed": elapsed, "status": status, "note": note}
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        return {"config": tag, "status": "timeout", "note": note}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"config": tag, "status": "error", "note": note}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    waves = [args.wave] if args.wave else [1, 2, 3]
    all_results = []

    for wave in waves:
        print(f"\n{'#'*60}")
        print(f"  WAVE {wave}")
        print(f"{'#'*60}")

        for d, n, dist, layers, steps, n_seeds, note in CAMPAIGN[wave]:
            if args.seeds is not None:
                n_seeds = args.seeds
            result = run_config(d, n, dist, layers, steps, n_seeds, note, args.dry_run)
            all_results.append(result)

    print(f"\n{'='*80}")
    print("CAMPAIGN SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':25} {'Status':>10} {'Loss':>12} {'Time':>8} Note")
    print("-" * 80)
    for r in all_results:
        loss_str = f"{r['best_loss']:.4e}" if isinstance(r.get('best_loss'), float) else "?"
        time_str = f"{r.get('elapsed', 0) / 60:.0f}m" if r.get('elapsed') else "?"
        print(f"{r['config']:25} {r.get('status', '?'):>10} {loss_str:>12} {time_str:>8} "
              f"{r.get('note', '')}")

    save_path = os.path.join("results", "campaign_results.json")
    os.makedirs("results", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
