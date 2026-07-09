#!/usr/bin/env python3
"""Stratified-importance sampling for the importance-weighted VarQEC loss
(R14-5 §6 / Phase B1).

R14-3a sampled the weight-2 orthogonality terms UNIFORMLY within each wire
group (stratified-10) and applied the Meth weights only in the loss WEIGHT.
This sampler instead draws WITHIN each group with probability ∝ w_i (Rosalin,
Arrasmith et al. arXiv:2004.06252) and allocates the per-group budget by
Neyman allocation (∝ √Σ w_i², the stratum spread). The Meth weighting is baked
into the returned per-op weights, so the estimator is unbiased for the full
Meth-weighted loss Σ_i w_i f_i.

Drop-in for the per-group weight tuple consumed by
`src.jax_backend.create_jax_loss_vmap_weighted` (the same interface as
`src.jax_backend.stratified_weights`), so no change to the frozen jax_backend.

Unbiasedness. Within group g (weights W_g = Σ_{e∈g} w_e, within-group dist
p_e = w_e/W_g), drawing B_g ops with replacement and assigning each drawn op
e the weight (W_g / B_g) per draw gives
    E[ Σ_{drawn} (W_g/B_g) f_e ] = (W_g) Σ_e p_e f_e = Σ_e w_e f_e,
the exact group contribution. Summed over groups → the full weighted loss.
"""
from __future__ import annotations
from typing import List

import numpy as np


def neyman_group_budgets(group_weights: List[np.ndarray], total_budget: int):
    """Per-group sample budgets by Neyman allocation B_g ∝ √Σ_{e∈g} w_e².

    Empty/zero-weight groups get budget 0; non-empty groups get ≥1 so every
    stratum with mass is represented each step.
    """
    spreads = np.array([float(np.sqrt(np.sum(np.asarray(Wg) ** 2)))
                        for Wg in group_weights])
    Z = spreads.sum()
    budgets = []
    for g, Wg in enumerate(group_weights):
        if len(Wg) == 0 or spreads[g] <= 0.0:
            budgets.append(0)
        else:
            budgets.append(max(1, int(round(total_budget * spreads[g] / Z))))
    return budgets


def make_stratified_importance_weights(group_weights: List[np.ndarray],
                                       total_budget: int,
                                       rng: np.random.Generator,
                                       dtype=None):
    """Return a per-group weight tuple (one array per wire group) for an
    unbiased single-step estimate of the Meth-weighted loss, via stratified
    importance sampling.

    Args:
      group_weights: list aligned with E_det_grouped; group_weights[g] is the
        (n_g,) array of per-op Meth weights w_e for that group (w may be 0 on
        ops the channel doesn't touch — those get sampling prob 0).
      total_budget: target total EVs/step (B). Distributed across groups by
        Neyman allocation.
      rng: numpy Generator.
      dtype: jnp dtype for the returned arrays (default jnp.float64).

    Returns:
      tuple of per-group weight arrays. In group g, op e drawn n_e times gets
      weight (W_g / B_g)·n_e (W_g = Σ_{e∈g} w_e, B_g = group budget); ops not
      drawn get 0. Setting these as `weights_per_group` makes
      create_jax_loss_vmap_weighted compute the unbiased single-step estimate.
    """
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float64

    budgets = neyman_group_budgets(group_weights, total_budget)
    out = []
    for g, Wg in enumerate(group_weights):
        Wg = np.asarray(Wg, dtype=float)
        n_g = len(Wg)
        w = np.zeros(n_g, dtype=float)
        Bg = budgets[g]
        total_w = Wg.sum()
        if n_g == 0 or Bg == 0 or total_w <= 0.0:
            out.append(jnp.asarray(w, dtype=dtype))
            continue
        p = Wg / total_w                      # within-group sampling ∝ w_e
        draws = rng.choice(n_g, size=Bg, replace=True, p=p)
        counts = np.bincount(draws, minlength=n_g)
        # Each draw of op e contributes (W_g / B_g)·f_e; n_e draws → ×count.
        w = (total_w / Bg) * counts
        out.append(jnp.asarray(w, dtype=dtype))
    return tuple(out)


def make_stratified_importance_weights_full_basis(group_weights: List[np.ndarray],
                                                  w1_mask: List[np.ndarray],
                                                  w2_mask: List[np.ndarray],
                                                  w1_budget: int,
                                                  w2_budget: int,
                                                  rng: np.random.Generator,
                                                  dtype=None):
    """R14-7 §B.2 — stratified-IS over BOTH the w1 and w2 sectors.

    R14-3a/R14-3b evaluate the 36 w1 ops full-batch (deterministic) and sample
    only w2. This samples w1 too (treating the sectors symmetrically), so a
    full-batch sub-100k budget becomes reachable: w1_budget + w2_budget EVs/step.

    Splits each group's Meth weights into its w1-mask and w2-mask portions,
    runs the per-sector stratified-IS sampler independently (each unbiased for
    its sector's contribution), and sums — the two supports are disjoint within
    a group, so the combined estimator is unbiased for the full weighted loss.
    """
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float64
    w1_gw = [np.where(np.asarray(m) > 0, np.asarray(gw, dtype=float), 0.0)
             for gw, m in zip(group_weights, w1_mask)]
    w2_gw = [np.where(np.asarray(m) > 0, np.asarray(gw, dtype=float), 0.0)
             for gw, m in zip(group_weights, w2_mask)]
    w1_w = make_stratified_importance_weights(w1_gw, w1_budget, rng, dtype=dtype)
    w2_w = make_stratified_importance_weights(w2_gw, w2_budget, rng, dtype=dtype)
    return tuple(jnp.asarray(np.asarray(w1_w[g]) + np.asarray(w2_w[g]), dtype=dtype)
                 for g in range(len(group_weights)))
