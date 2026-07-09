"""Channel-prior builders for the weighted MAP / Petz decoders.

A prior is a flat per-op weight vector aligned with the weight-≤2 closure
basis `ErrorModel(d,n_qudit,distance=3,closed=True).build_grouped()` — the
same flat ordering as `meth_pauli_weights.npz['weights']`.
"""
from __future__ import annotations
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def meth_pauli_prior(weights_npz_path: str = None) -> np.ndarray:
    """Load the twirled Meth-physical per-Pauli weights (the R14-3a training
    weights). Returns the flat (685,) vector, Σ = 1."""
    if weights_npz_path is None:
        weights_npz_path = os.path.join(
            REPO, "results/round14_scoping/meth_pauli_weights.npz")
    return np.asarray(np.load(weights_npz_path, allow_pickle=True)['weights'])


def uniform_depolarizing_prior(E_full_grouped: list, n_qudit: int, d: int,
                                p: float, n_single: int) -> np.ndarray:
    """Flat per-op priors for uniform per-qudit Pauli depolarizing at strength p
    (matches src.simulation.make_pauli_depolarizing_noise_fn).

    Identity:                      (1-p)^n
    Single-qudit (e < n_single):   (1-p)^(n-1)·p·(1/n_single)
    Single-qudit closure (e ≥ n_single): 0  (not produced by single-Pauli channel)
    Two-qudit:                     (1-p)^(n-2)·(p/n_single)²
    Weight-≥3 mass omitted (outside the weight-≤2 basis).
    """
    n_total = sum(int(g['matrices'].shape[0]) for g in E_full_grouped)
    w = np.zeros(n_total)
    flat = 0
    for g in E_full_grouped:
        wires = g['wires']
        n_g = int(g['matrices'].shape[0])
        if len(wires) == 0:
            w[flat] = (1 - p) ** n_qudit
        elif len(wires) == 1:
            for e in range(min(n_single, n_g)):
                w[flat + e] = (1 - p) ** (n_qudit - 1) * p * (1.0 / n_single)
        else:
            for e in range(n_g):
                w[flat + e] = (1 - p) ** (n_qudit - 2) * (p / n_single) ** 2
        flat += n_g
    return w
