#!/usr/bin/env python3
"""Weighted MAP decoder with weight-≤2 corrections + channel prior (R14-5 §1).

Extends R14-4's unweighted lookup decoder (src/simulation.py:729) with
  (a) correction set extended from {I, weight-1} to {I, weight-1, weight-2},
      from the SAME weight-≤2 closure basis R14-3a trained against, and
  (b) a per-correction prior weight from the channel distribution,
so the decoder picks  argmax_C  w_C · Σ_k |⟨C·ψ_k | noisy⟩|²  (MAP rather than
the unweighted lookup's uniform-prior MLE).

Analytical note (R14-4 §5): a single weight-≤2 error on a distance-3 code is
recovered by projection alone (KL: P_C·E|ψ_L⟩ ∝ |ψ_L⟩), which the unweighted
lookup already realizes via its identity-correction + projection fallback. So
the weight-2 extension does not beat lookup on single-error channels; its value
(if any) is on multi-error channels where total weight exceeds the distance.
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np

from src.decoders._common import apply_local_np


def build_weighted_correction_set(E_full_grouped: list,
                                   prior_weights: np.ndarray,
                                   n_qudit: int, d: int,
                                   max_weight: int = 2,
                                   drop_zero: bool = True) -> List[Tuple]:
    """Return a list of (wires, C_dag_matrix, inverse_perm, prior_w) tuples.

    Iterates E_full_grouped in the same flat order as
    meth_pauli_weights.npz['weights'] so prior_weights[flat_idx] lines up.
    Correction C† = M† applied on the op's wires; prior = prior_weights[flat].
    """
    corrections = []
    flat = 0
    for g in E_full_grouped:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        for e in range(mats.shape[0]):
            w = float(prior_weights[flat])
            flat += 1
            if len(wires) > max_weight:
                continue
            if drop_zero and w <= 0.0:
                continue
            corrections.append((wires, mats[e].conj().T, inv, w))
    return corrections


def simulate_ler_with_weighted_map(code_states: np.ndarray,
                                   noise_fn: Callable,
                                   correction_set: List[Tuple],
                                   n_qudit: int, d: int,
                                   n_shots: int, seed: int,
                                   batch: int = 200) -> np.ndarray:
    """Per-shot bernoulli outcomes (1 = logical error) under weighted MAP.

    The MAP score for correction C_i (= E_i, applied as E_i† to the noisy state)
    is  w_i · ‖P_C E_i† noisy‖² = w_i · Σ_k |⟨E_i ψ_k | noisy⟩|².  Precomputing
    CES[i,k] = E_i ψ_k once and batching the ⟨CES|noisy⟩ contraction makes the
    per-shot cost one matmul over CES (read once per batch) instead of
    |corrections| separate tensordots — ~16× faster, the same trick as the Petz
    K_psi eval. Only the single best correction is then applied + projected per
    shot. Algebraically identical to the per-correction argmax loop.
    """
    K, dim = code_states.shape
    # Precompute CES[i,k] = E_i ψ_k  (E_i = C_dag†) and the weight vector.
    CES = np.stack([
        np.stack([apply_local_np(code_states[k], C_dag.conj().T, wires, inv,
                                 d, n_qudit) for k in range(K)])
        for (wires, C_dag, inv, w) in correction_set])      # (n_corr, K, dim)
    wvec = np.array([w for (_, _, _, w) in correction_set])  # (n_corr,)
    corr_meta = [(wires, C_dag, inv) for (wires, C_dag, inv, _) in correction_set]

    rng = np.random.default_rng(seed)
    outcomes = np.zeros(n_shots, dtype=np.int8)
    for start in range(0, n_shots, batch):
        b = min(batch, n_shots - start)
        alphas = np.empty((b, K), dtype=complex)
        NOISY = np.empty((dim, b), dtype=complex)
        for j in range(b):
            a = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            a /= np.linalg.norm(a)
            alphas[j] = a
            NOISY[:, j] = noise_fn(a @ code_states, rng)
        # inner[i,k,j] = ⟨CES[i,k] | NOISY[:,j]⟩ = conj(CES · NOISY*)
        inner = np.tensordot(CES, NOISY.conj(), axes=([2], [0])).conj()  # (n_corr,K,b)
        score = wvec[:, None] * np.sum(np.abs(inner) ** 2, axis=1)        # (n_corr, b)
        best = np.argmax(score, axis=0)                                   # (b,)
        for j in range(b):
            wires, C_dag, inv = corr_meta[int(best[j])]
            corrected = apply_local_np(NOISY[:, j], C_dag, wires, inv, d, n_qudit)
            coeffs = code_states.conj() @ corrected            # (K,)
            projected = coeffs @ code_states
            nrm = np.linalg.norm(projected)
            if nrm > 1e-10:
                projected = projected / nrm
            logical = alphas[j] @ code_states
            fid = float(np.abs(np.vdot(logical, projected)) ** 2)
            if fid < 0.5:
                outcomes[start + j] = 1
    return outcomes
