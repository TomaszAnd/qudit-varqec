#!/usr/bin/env python3
# seed-racing protocol and mercy-rule early stopping adapted from U. Holzer,
# Tensorform_Wire-grouped-Batching_Random_Seeding.py
"""Multi-start "seed race" for VarQEC code discovery (core library).

Races many random initializations of the same ansatz against each other and
keeps the seed that reaches the target loss in the fewest steps. Built entirely
on repo machinery: ErrorModel (src/errors.py), create_jax_encoder
(src/jax_backend.py), the weighted vmap loss (build_varqec_loss_weighted), and
stratified_weights for optional per-step subsampling.

Layering: the seed race is multi-start ORCHESTRATION that sits ABOVE the
differentiable backend. src/jax_backend.py stays the core (encoder, loss,
gradient, sampling weights); this module composes those primitives into the
named seed-race protocol. Keeping it as its own module (not folded into
jax_backend, not under a src/training/ package) keeps the orchestration->core
layering clean and makes the protocol discoverable. The thin CLI wrapper lives
in scripts/seed_race.py, which re-exports these functions.

Deviations from the source protocol, each fixing an audit/04 defect:

- H1: victory REQUIRES the full-batch loss to be below target_loss. When
  sample_frac < 1 the sampled loss acts only as a cheap trigger; every trigger
  is re-evaluated full-batch before acceptance, and both values are logged.
  Contract pinned by tests/test_full_vs_sampled_convergence.py and
  tests/test_seed_race.py.
- H2: the mercy rule activates only after the first CERTIFIED record exists
  (mercy_baseline optionally seeds it, with explicit provenance in the
  metadata). Until then every seed gets the full steps budget.
- H3: the K/4 variance coefficient (paper Eq. 10) is inherited from
  src.jax_backend.create_jax_loss_vmap_weighted -- nothing to fix here.
- H5: no identity padding -- ErrorModel builds E_det from the hardware/full
  basis without inserting the identity (src/errors.py), so the 2.6% wasted ops
  of the source script do not arise.
"""
import os
import sys

import numpy as np

# Best-practice campaign sampler budget (Stage H', docs/SAMPLING_BUDGET.md).
# SINGLE SOURCE OF TRUTH for the campaign sampling fraction (no other hardcoded
# value; scripts/seed_race.py CLI defaults reference this).
#
# The knob is a pure SEARCH-COST knob: certification is always full-batch, so any
# fraction that certifies yields a full-batch-quality code (n=5 LER CIs across
# fractions overlap). It only changes op-EVs and per-seed yield. So the cheaper
# fraction wins UNLESS it lowers seed yield on the LARGEST campaign code.
#
# n=5 ((5,3,3))_3 sweep (--noise full): sharp gradient-corruption cliff -- f<=0.3
# diverge, 0.32-0.35 stochastic ~75% yield, robust 4/4-seed knee f5* = 0.40 (5.41x
# fewer op-EVs than full-batch). n=9 ((9,3,3))_3 transfer (bigger weight-2 closure)
# is being measured to get f9* = smallest fraction holding full-batch seed yield.
#
# INTERIM = 0.50 (knee + one grid-step margin) until f9* lands. Rule to apply then:
#   f9* <= 0.40 -> set 0.40 (bank ~35% vs 0.50);  f9* > 0.40 -> set f9* (largest-code
#   knee) and record the K-scaling.  Do NOT drop below 0.40 without n=9 confirmation.
CAMPAIGN_SAMPLE_FRAC = 0.5   # INTERIM; n=5 knee f5*=0.40, pending n=9 f9*
CAMPAIGN_SAMPLER = "importance"


def evaluate_candidate(full_loss_fn, theta, target_loss):
    """Full-batch certification gate (audit/04 H1 fix).

    A candidate theta -- no matter how its sampled loss looks -- is certified
    as converged if and only if its FULL-batch loss is below target_loss.

    Returns (certified: bool, full_loss: float).
    """
    full_val = float(full_loss_fn(theta))
    return bool(full_val < target_loss), full_val


def build_race(d, n, distance, layers, noise="full", entangler="ms",
               connectivity="all-to-all", weighted=False, eta=0.9296, n_max=2):
    """Assemble encoder + weighted loss from repo machinery.

    Returns (val_grad_fn, full_loss_fn, group_sizes, params_per_layer,
    connections, full_weights, sampler_group_weights).
    - val_grad_fn(theta, weights) differentiates w.r.t. theta only.
    - full_loss_fn(theta) is the FULL-BATCH loss: uniform (unweighted) by default,
      or the Meth-CHANNEL-weighted KL when weighted=True.
    - full_weights: the per-group weight tuple for the full batch (ones, or the
      corrected Meth per-op channel weights).
    - sampler_group_weights: per-op weights the importance sampler draws by (ones
      per group when unweighted; the Meth weights when weighted, so it concentrates
      samples on high-probability channel errors).

    When weighted=True the model uses the DEFAULT hardware closure basis (basis=None)
    to align with the Meth weights (computed live from the corrected channel via
    src.meth_weights), NOT the "full" basis.
    """
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import (create_jax_encoder,
                                 build_varqec_loss_weighted)

    if connectivity == "all-to-all":
        connections = [[i, j] for i in range(n) for j in range(i + 1, n)]
    else:
        connections = [[i, (i + 1) % n] for i in range(n)]

    if entangler == "ls":
        # Same comparison encoder train.py uses for --entangler ls. The HRMO
        # builder is a sibling *script* (scripts/sec7_lsgate_hrmo_rerun.py);
        # from this src/ module the repo scripts dir must be added explicitly.
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(repo, "scripts"))
        from sec7_lsgate_hrmo_rerun import build_hrmo_encoder
        ppl = 5 * n * (d - 1) + len(connections)
        enc = build_hrmo_encoder(n, d, layers, ring_edges=connections,
                                 star_edges=[], use_csum_on_star=False,
                                 params_per_layer=ppl)
    else:
        enc, _, ppl = create_jax_encoder(n, d, connections=connections,
                                         use_scan=layers >= 3)

    closed = distance >= 3
    # Weighted (Meth) arm uses the default hardware closure basis to align with the
    # channel weights; unweighted arm keeps the "full" basis for noise=="full".
    basis = None if weighted else ("full" if noise == "full" else None)
    model = ErrorModel(d=d, n_qudit=n, distance=distance, closed=closed,
                       basis=basis)
    grouped = model.build_grouped(verbose=False)
    group_sizes = [g['matrices'].shape[0] for g in grouped]

    # K/4 (H3) lives inside create_jax_loss_vmap_weighted; the error set has
    # no identity padding (H5) because ErrorModel's bases exclude it.
    loss_w = build_varqec_loss_weighted(
        encoder_fn=enc, K=d, d=d, n_qudit=n, distance=distance,
        E_det_grouped=grouped)

    if weighted:
        from src.meth_weights import compute_meth_group_weights
        sampler_group_weights, _ = compute_meth_group_weights(
            d, n, distance=distance, eta=eta, n_max=n_max)
        if sum(len(w) for w in sampler_group_weights) != sum(group_sizes):
            raise ValueError("Meth weight / basis misalignment")
        full_weights = tuple(jnp.asarray(w, dtype=jnp.float64)
                             for w in sampler_group_weights)
    else:
        sampler_group_weights = [np.ones(m, dtype=float) for m in group_sizes]
        full_weights = tuple(jnp.ones((m,), dtype=jnp.float64) for m in group_sizes)

    # loss_w is internally @jit'd, but value_and_grad over it still re-traces
    # the grad transform every call (measured 38.7 -> 0.59 ms/step on the
    # n=3/d=3/dist=2 smoke config); the outer jit caches it.
    val_grad_fn = jax.jit(jax.value_and_grad(loss_w))  # grads w.r.t. theta only

    def full_loss_fn(theta):
        return loss_w(theta, full_weights)

    return (val_grad_fn, full_loss_fn, group_sizes, ppl, connections,
            full_weights, sampler_group_weights)


def run_race(d, n, distance, layers, num_seeds, steps, target_loss=1e-6,
             sample_frac=1.0, noise="full", entangler="ms",
             connectivity="all-to-all", mercy_baseline=None, out=None,
             lr=0.05, first_seed=0, verbose=True,
             sampler="uniform", trigger_margin=0.0, trigger_patience=1,
             weighted=False, eta=0.9296, n_max=2):
    """Race num_seeds initializations; return a result dict and save an npz.

    The winner is the seed reaching a CERTIFIED (full-batch) loss below
    target_loss in the fewest steps. If no seed certifies, the best
    uncertified theta (lowest full-batch loss at its best sampled step) is
    saved instead, with certified=False in the metadata.

    Sampling budget (Stage H). When sample_frac < 1 the per-step loss is a cheap
    subsampled estimate that only TRIGGERS a full-batch certification (H1); it
    never grants victory. Two sampler modes:
      - "uniform":     per-group uniform subsampling (jax_backend.stratified_weights).
      - "importance":  Neyman + within-group importance sampling
                       (src.sampling.stratified_importance) over the per-op
                       weights of the loss (uniform weights for the unweighted
                       detection loss -> Neyman-by-sqrt-size stratification; the
                       same mechanism carries the Meth-weighted objective when a
                       weighted loss is supplied).
    False triggers are suppressed by requiring the sampled loss below
    target_loss*(1+trigger_margin) for trigger_patience CONSECUTIVE steps before
    the (expensive) full-batch certification is attempted.

    The returned dict includes op-EV accounting (race_op_evs = total operator-
    expectation-value evaluations across all seeds, counting both sampled steps
    and full-batch certification/eval calls; op_evs_at_first_certified = the
    running total at the moment the winning seed first certified) so full-batch
    racing and importance-sampled racing can be compared at equal certified loss.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from src.jax_backend import stratified_weights
    from src.loss import save_varqec_result

    (val_grad_fn, full_loss_fn, group_sizes, ppl, connections,
     full_weights, sampler_group_weights) = build_race(
        d, n, distance, layers, noise=noise, entangler=entangler,
        connectivity=connectivity, weighted=weighted, eta=eta, n_max=n_max)

    ones = full_weights  # full-batch weights: uniform, or Meth channel weights
    subsampling = sample_frac < 1.0
    total_ops = int(sum(group_sizes))

    # Sampler setup. group_weights are the per-op loss weights the importance
    # sampler draws by: uniform ones (unweighted -> Neyman-by-sqrt(group size)), or
    # the Meth channel weights (weighted -> concentrate on high-P(error) ops).
    if sampler == "importance":
        from src.sampling.stratified_importance import (
            make_stratified_importance_weights)
        group_weights = sampler_group_weights
        is_budget = max(1, int(round(sample_frac * total_ops)))
    elif sampler != "uniform":
        raise ValueError(f"sampler must be 'uniform' or 'importance', got '{sampler}'")

    def _sample_weights(rng):
        if sampler == "importance":
            return make_stratified_importance_weights(group_weights, is_budget, rng)
        return stratified_weights(group_sizes, sample_frac, rng)

    def _op_evs(weights):
        # hardware cost of a step = number of distinct ops actually evaluated
        return int(sum(int(np.count_nonzero(np.asarray(w))) for w in weights))

    # op-EV accounting (H) across the whole race.
    race_op_evs = 0
    op_evs_at_first_certified = None
    # Trigger accounting (H'): a "trigger" is a full-batch certification attempt
    # armed by the patience gate; a "false trigger" is one the full batch rejects.
    n_trigger_attempts = 0
    n_false_triggers = 0

    # H2 fix: the record starts empty unless the caller supplies a baseline
    # with explicit provenance; no seed is killed before a record exists.
    record_steps = mercy_baseline
    record_from_baseline = mercy_baseline is not None
    winner = None  # dict(seed, steps, theta, losses, full_loss)
    best_uncert = None  # dict(seed, steps, theta, losses, full_loss)

    for seed in range(first_seed, first_seed + num_seeds):
        base_key = jax.random.PRNGKey(seed)
        init_key, _ = jax.random.split(base_key)
        theta = jax.random.uniform(init_key, (layers, ppl),
                                   minval=0.0, maxval=2 * np.pi)
        rng = np.random.default_rng(seed)

        # Ulrich's three-stage LR schedule (0.05 / 0.01 below 0.5 / 0.001
        # below 1e-3), but the Adam first/second moments are CARRIED across
        # switches via optax.inject_hyperparams instead of re-initializing
        # the optimizer (his re-init zeroes the moments and restarts bias
        # correction -- audit/04 H8; house style, but there is no reason to
        # inherit the transient here).
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
        opt_state = optimizer.init(theta)
        lr_stage = 0

        seed_losses = []
        seed_best_sampled = np.inf
        seed_best_theta = theta
        certified_here = False
        consec_below = 0  # consecutive steps under the trigger margin (H)

        for step in range(steps):
            # Mercy rule (H2): only once a certified record exists.
            if record_steps is not None and step >= record_steps:
                if verbose:
                    print(f"seed {seed}: eliminated at step {step} "
                          f"(record {record_steps})")
                break

            if subsampling:
                weights = _sample_weights(rng)
            else:
                weights = ones
            race_op_evs += _op_evs(weights)
            loss_val, grads = val_grad_fn(theta, weights)
            lv = float(loss_val)
            seed_losses.append(lv)
            if lv < seed_best_sampled:
                seed_best_sampled = lv
                seed_best_theta = theta

            # Trigger gating (H): the sampled loss must sit under the margin for
            # trigger_patience CONSECUTIVE steps before the expensive full-batch
            # certification is attempted. Suppresses false triggers from a noisy
            # single-step sampled estimate.
            if lv < target_loss * (1.0 + trigger_margin):
                consec_below += 1
            else:
                consec_below = 0

            # Victory check on the PRE-update theta the loss belongs to.
            if consec_below >= trigger_patience:
                certified, full_val = evaluate_candidate(
                    full_loss_fn, theta, target_loss)
                race_op_evs += total_ops  # a full-batch certification eval
                n_trigger_attempts += 1
                if not certified:
                    n_false_triggers += 1
                if verbose:
                    print(f"seed {seed} step {step}: sampled={lv:.3e}, "
                          f"full-batch={full_val:.3e} -> "
                          f"{'CERTIFIED' if certified else 'REJECTED'}")
                if certified:
                    certified_here = True
                    if record_steps is None or step < record_steps:
                        record_steps = step
                        record_from_baseline = False
                        if op_evs_at_first_certified is None:
                            op_evs_at_first_certified = race_op_evs
                        winner = {"seed": seed, "steps": step,
                                  "theta": np.array(theta),
                                  "losses": list(seed_losses),
                                  "full_loss": full_val}
                    break
                # Not certified: the sampled trigger was a false alarm --
                # keep training this seed (the H1 failure mode).
                consec_below = 0

            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

            # In-place hyperparams mutation is only sound in this EAGER
            # loop: opt_state is a concrete pytree here. If this update
            # step is ever moved inside jax.jit / lax.scan, the mutation
            # would be silently dropped (traced state is immutable) -- use
            # optax.tree_utils/functional replacement instead there.
            if lr_stage == 0 and lv < 0.5:
                opt_state.hyperparams['learning_rate'] = lr / 5
                lr_stage = 1
            if lr_stage <= 1 and lv < 1e-3:
                opt_state.hyperparams['learning_rate'] = lr / 50
                lr_stage = 2

            if verbose and step % 200 == 0:
                print(f"seed {seed} step {step:5d} | loss={lv:.4e}")

        if not certified_here:
            _, full_val = evaluate_candidate(full_loss_fn, seed_best_theta,
                                             target_loss)
            race_op_evs += total_ops  # end-of-seed best-uncertified full eval
            if best_uncert is None or full_val < best_uncert["full_loss"]:
                best_uncert = {"seed": seed, "steps": len(seed_losses),
                               "theta": np.array(seed_best_theta),
                               "losses": list(seed_losses),
                               "full_loss": full_val}

    result = winner if winner is not None else best_uncert
    certified = winner is not None
    if verbose:
        if certified:
            print(f"\nWINNER: seed {result['seed']} certified "
                  f"full-batch loss {result['full_loss']:.3e} "
                  f"at step {result['steps']}")
        elif result is not None:
            print(f"\nNo seed certified; best uncertified full-batch loss "
                  f"{result['full_loss']:.3e} (seed {result['seed']})")

    if result is not None and out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        save_varqec_result(
            out, result["theta"], result["losses"],
            f"native_d{d}", distance, layers,
            metadata={"K": d, "n_qudit": n, "d": d,
                      "seed": result["seed"],
                      "steps_to_target": result["steps"],
                      "certified": certified,
                      "certified_full_loss": result["full_loss"],
                      "target_loss": target_loss,
                      "sample_frac": sample_frac,
                      "noise": noise, "entangler": entangler,
                      "connectivity": connectivity,
                      "num_seeds": num_seeds,
                      "mercy_baseline": (mercy_baseline
                                         if mercy_baseline is not None
                                         else -1),
                      "mercy_baseline_used": record_from_baseline,
                      "backend": "jax",
                      "sampler": sampler,
                      "trigger_margin": trigger_margin,
                      "trigger_patience": trigger_patience,
                      "race_op_evs": race_op_evs,
                      "connections": np.asarray(connections, dtype=int)})

    return {"certified": certified, "result": result,
            "record_steps": record_steps,
            "sampler": sampler,
            "total_ops": total_ops,
            "race_op_evs": race_op_evs,
            "op_evs_at_first_certified": op_evs_at_first_certified,
            "n_trigger_attempts": n_trigger_attempts,
            "n_false_triggers": n_false_triggers}
