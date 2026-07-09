#!/usr/bin/env python3
"""LER benchmark for all-to-all-connectivity VarQEC codes.

The frozen `src/catalog.py::load_code` reconstructs code states from a
ring-connectivity encoder regardless of what the training used. This
script mirrors `scripts/benchmark_ler.py` but reads `connections` from
the saved npz metadata and rebuilds the encoder with the correct
connectivity. Output schema matches `results/benchmarks_30k/*.csv`.

Usage:
    python3 scripts/benchmark_ler_alltoall.py \
        --params results/saved_params/all_to_all/d3_n9_dist3_4L_best3s_seed0.npz \
        --n_shots 30000 \
        --out_dir results/benchmarks_30k_alltoall
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_P_RATES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--n_shots", type=int, default=30000)
    p.add_argument("--p_range", type=str,
                   default=",".join(str(p) for p in DEFAULT_P_RATES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {args.params}")
    data = np.load(args.params, allow_pickle=True)
    d = int(data["d"])
    n_qudit = int(data["n_qudit"])
    K = int(data.get("K", d))
    dist = int(data.get("distance", 3))
    n_layers = int(data.get("n_layers", 4))
    params = np.asarray(data["params"])
    connectivity = str(data.get("connectivity", "ring"))
    if "connections" in data.files:
        connections = [list(c) for c in np.asarray(data["connections"])]
    else:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]

    final_loss = float(data.get("final_loss", -1.0))
    print(f"  d={d}, n={n_qudit}, K={K}, dist={dist}, layers={n_layers}, "
          f"connectivity={connectivity}, |E|={len(connections)}, "
          f"final_loss={final_loss:.3e}")

    from src.encoder import create_native_encoder
    encoder, _, _ = create_native_encoder(n_qudit, d, connections=connections,
                                          force_manual=True)
    print("  reconstructing code states...")
    t0 = time.time()
    from pennylane import numpy as pnp
    params_pl = pnp.array(params, requires_grad=False)
    code_states = np.zeros((K, d ** n_qudit), dtype=complex)
    for k in range(K):
        code_states[k] = np.asarray(encoder(params_pl, k))
    print(f"  code states built ({time.time()-t0:.1f}s)")

    from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
    from src.simulation import simulate_ler_with_correction_factored

    single_errors = qudit_hardware_error_basis(d)
    p_rates = [float(x) for x in args.p_range.split(",")]

    base = os.path.basename(args.params).replace(".npz", "")
    csv_path = os.path.join(args.out_dir, f"ler_{base}_{args.n_shots}.csv")
    rows = [["p", "logical_error_rate", "mean_fidelity", "mean_raw_fidelity",
             "ler_low_wilson", "ler_high_wilson"]]

    print(f"  LER sweep ({args.n_shots} shots):")
    print(f"  {'p':>7} | {'LER':>9} | {'F_corr':>7} | {'F_raw':>7}")
    print("  " + "-" * 42)
    for p_val in p_rates:
        noise = make_hardware_noise_fn(d, n_qudit, p_val)
        r = simulate_ler_with_correction_factored(
            code_states, noise, single_errors, n_qudit, d,
            args.n_shots, seed=args.seed)
        ler = float(r["logical_error_rate"])
        fc = float(r["mean_fidelity"])
        fr = float(r["mean_raw_fidelity"])
        # Wilson 95% CI
        n = args.n_shots
        k = int(round(ler * n))
        z = 1.96
        denom = 1 + z * z / n
        center = (k / n + z * z / (2 * n)) / denom
        rad = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / denom
        lo, hi = max(0.0, center - rad), min(1.0, center + rad)
        rows.append([p_val, ler, fc, fr, lo, hi])
        print(f"  {p_val:7.3f} | {ler:9.4e} | {fc:7.4f} | {fr:7.4f}")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
