#!/usr/bin/env python3
"""
Round-12 Commit 12: Sec VII rerun with the genuine Hrmo et al. 2023 native
qudit LS gate (`ls_global_gate`, formerly `light_shift_gate_hrmo`) substituted
for the level-restricted `zz_subspace_gate` (formerly `light_shift_gate`) used by
the original §VII ablations.

Variants (mirroring the original §VII table):

  pure_ls_ring   : XY (1 round) + Hrmo-LS (ring) + Z         per layer
  csum_star_ls   : XY (1 round) + CSUM (star)   + Hrmo-LS (ring) + Z

This script builds its own JAX encoder + uses the project's existing
wire-grouped loss (`src.jax_backend.create_jax_loss_vmap`) and error model.
We do not modify the frozen `src/jax_backend.py::create_jax_encoder`.

The original §VII training-loss plot ran ~230 steps; the Round-12 spec asks
for the full campaign protocol (1500 steps, 3 seeds, Adam two-stage LR) for
a fair head-to-head against the campaign codes. Outputs go to
`results/sec7_lsgate_hrmo/` and figures to `figures/sec7_hrmo_*`.

Usage:
    python3 scripts/sec7_lsgate_hrmo_rerun.py \
        --variant pure_ls_ring --n 5 --layers 4 --seeds 0,1,2 --steps 1500
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_ring(n):
    return [(i, (i + 1) % n) for i in range(n)]


def make_star(n, hub=0):
    return [(hub, j) for j in range(n) if j != hub]


def build_hrmo_encoder(n_qudit, d, n_layers, ring_edges, star_edges,
                       use_csum_on_star, params_per_layer):
    """JAX encoder for the Sec VII variants. Returns encoder_fn(params, code_ind).

    Layer structure (matches the §V.A campaign ansatz with MS replaced
    by Hrmo LS and an optional CSUM-on-star prelude):
      1. Single-qudit XY — TWO rounds — 2 params per (qudit, transition)
         per round = 4 params per (qudit, transition)
      2. (optional) CSUM on star edges — no parameters
      3. Hrmo LS gate on ring edges — 1 parameter per ring edge
      4. Z corrections — 1 param per (qudit, transition)
    """
    # Import src.jax_backend FIRST so it sets jax_enable_x64 before we
    # construct any complex128 arrays.
    from src.jax_backend import (
        _xy_gate_jax, _z_gate_jax,
        _apply_single_gate_jax, _apply_two_gate_jax,
    )
    import jax
    import jax.numpy as jnp

    from src.gates import CSUM_gate

    n_transitions = d - 1
    dim = d ** n_qudit

    csum_mat = jnp.asarray(CSUM_gate(d=d), dtype=jnp.complex128) if (
        use_csum_on_star and star_edges) else None

    def _hrmo_ls(theta, d):
        diag = jnp.ones(d * d, dtype=jnp.complex128)
        phase = jnp.exp(1j * theta)
        for a in range(d):
            for b in range(d):
                if a != b:
                    diag = diag.at[a * d + b].set(phase)
        return jnp.diag(diag)

    def _apply_layer(state, layer_p):
        pi = 0
        # 1. Two-round XY (matches the §V.A campaign ansatz)
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi + 1], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi + 1], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
        # 2. Optional CSUM on star edges (no params)
        if csum_mat is not None:
            for (q1, q2) in star_edges:
                state = _apply_two_gate_jax(state, csum_mat, q1, q2, n_qudit, d)
        # 3. Hrmo LS on ring edges, 1 param each
        for (q1, q2) in ring_edges:
            U = _hrmo_ls(layer_p[pi], d)
            state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
            pi += 1
        # 4. Z corrections
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _z_gate_jax(layer_p[pi], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 1
        return state

    @jax.checkpoint
    def _ckp_layer(state, layer_p):
        return _apply_layer(state, layer_p)

    def _init_state(code_ind):
        state = jnp.zeros(dim, dtype=jnp.complex128)
        return state.at[code_ind * d ** (n_qudit - 1)].set(1.0)

    def encoder(params, code_ind):
        state = _init_state(code_ind)
        state, _ = jax.lax.scan(
            lambda s, p: (_ckp_layer(s, p), None), state, params)
        return state

    return encoder


def variant_params_per_layer(variant, n_qudit, d, ring_edges):
    """Per-layer parameter count for each variant.

    XY two-round : 4 * n_qudit * (d-1)
    Hrmo LS      : 1 per ring edge
    Z corrections:     n_qudit * (d-1)
    Total        : 5 * n_qudit * (d-1) + |ring_edges|
    """
    return 5 * n_qudit * (d - 1) + len(ring_edges)


def train_one_seed(d, n, dist, n_layers, seed, n_steps, lr, lr_switch,
                   encoder_fn, ppl):
    import jax
    import jax.numpy as jnp
    import optax
    from jax import value_and_grad

    from src.errors import ErrorModel
    from src.jax_backend import create_jax_loss_vmap

    model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
    E_det_grouped = model.build_grouped(verbose=False)
    core_loss = create_jax_loss_vmap(E_det_grouped, d, d, n, dist)

    enc_vmapped = jax.vmap(encoder_fn, in_axes=(None, 0))

    @jax.jit
    def loss_fn(params):
        code_states = enc_vmapped(params, jnp.arange(d))
        return core_loss(code_states)

    val_grad = jax.jit(value_and_grad(loss_fn))
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (n_layers, ppl),
                               minval=0.0, maxval=2 * np.pi)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)
    switched = False
    losses = []
    best_loss = 1e10
    best_theta = theta

    for step in range(n_steps):
        lv, g = val_grad(theta)
        updates, opt_state = optimizer.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        flv = float(lv)
        losses.append(flv)
        if flv < best_loss:
            best_loss = flv
            best_theta = theta
        if not switched and flv < 0.1:
            optimizer = optax.adam(lr_switch)
            opt_state = optimizer.init(theta)
            switched = True
        if step % 100 == 0:
            print(f"    step {step:4d} | loss={flv:.4e}")
        if flv < 1e-6:
            print(f"    CONVERGED at step {step}")
            break

    return np.asarray(best_theta), np.asarray(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["pure_ls_ring", "csum_star_ls"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_switch", type=float, default=0.01)
    p.add_argument("--star_hub", type=int, default=0)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    d, n, dist, L = args.d, args.n, args.distance, args.layers
    ring_edges = make_ring(n)
    star_edges = make_star(n, hub=args.star_hub) if args.variant == "csum_star_ls" else []

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "sec7_lsgate_hrmo")
    os.makedirs(out_dir, exist_ok=True)

    ppl = variant_params_per_layer(args.variant, n, d, ring_edges)
    print(f"=== {args.variant} on (({n},{d},{dist}))_{d} ===")
    print(f"layers={L}, params/layer={ppl}, total params={L * ppl}")
    print(f"ring_edges={ring_edges}")
    if star_edges:
        print(f"star_edges (hub={args.star_hub})={star_edges}")

    encoder_fn = build_hrmo_encoder(
        n_qudit=n, d=d, n_layers=L,
        ring_edges=ring_edges, star_edges=star_edges,
        use_csum_on_star=(args.variant == "csum_star_ls"),
        params_per_layer=ppl)

    best_loss = 1e10
    best_theta = None
    best_losses = None
    best_seed = None
    all_traces = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        theta, losses = train_one_seed(
            d, n, dist, L, seed, args.steps, args.lr, args.lr_switch,
            encoder_fn, ppl)
        elapsed = time.time() - t0
        final = float(losses[-1])
        bir = float(np.min(losses))
        print(f"  Seed {seed}: final={final:.4e}, best={bir:.4e}, "
              f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")
        all_traces[f"seed{seed}_losses"] = losses
        if bir < best_loss:
            best_loss = bir
            best_theta = theta
            best_losses = losses
            best_seed = seed

    tag = f"{args.variant}_d{d}_n{n}_dist{dist}_{L}L_best{len(seeds)}s"
    np.savez(
        os.path.join(out_dir, f"{tag}_seed{best_seed}.npz"),
        params=best_theta, losses=best_losses,
        variant=args.variant, d=d, n=n, distance=dist, layers=L,
        params_per_layer=ppl, best_seed=best_seed, best_loss=best_loss,
        **all_traces,
    )
    print(f"\nBest: seed={best_seed}, loss={best_loss:.4e}")
    print(f"Saved {tag}_seed{best_seed}.npz")


if __name__ == "__main__":
    main()
