"""Sampling strategies for the importance-weighted VarQEC loss (R14-5 §6).

stratified_importance: Neyman per-group allocation + Rosalin within-group
importance sampling (∝ Meth weights). Returns a per-group weight tuple that is
a drop-in for `src.jax_backend.create_jax_loss_vmap_weighted`'s
weights_per_group argument (same interface as jax_backend.stratified_weights),
giving an unbiased single-step estimate of the Meth-weighted loss.
"""
from src.sampling.stratified_importance import (
    make_stratified_importance_weights,
    make_stratified_importance_weights_full_basis,
    neyman_group_budgets,
)

__all__ = [
    "make_stratified_importance_weights",
    "make_stratified_importance_weights_full_basis",
    "neyman_group_budgets",
]
