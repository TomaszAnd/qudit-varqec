#!/usr/bin/env python3
"""Petz transpose recovery for VarQEC codes (R14-5 §2 / §C).

The recovery channel whose ε-correctability is bounded by Cao Prop 2-4
(Bényi-Oreshkov 2010, Lemma 1) — i.e. the recovery the VarQEC ℓ_2 loss is
actually training against. For code projector P_C = Σ_k |ψ_k⟩⟨ψ_k| and a
weighted-Pauli channel N(ρ) = Σ_i w_i E_i ρ E_i† (Kraus K_i = √w_i E_i over
the weight-≤2 closure basis):

    Petz Kraus   R_i = P_C K_i† N(P_C)^{-1/2}
    recovery     R(ω) = Σ_i R_i ω R_i†
    fidelity     F = Σ_i |⟨ψ_L| R_i |noisy⟩|²  (pure-state input + noise outcome)

N(P_C)^{-1/2} is the Moore-Penrose inverse-square-root on supp(N(P_C)) (SVD,
threshold tol, invert non-zero singular values, zero the rest — NOT Tikhonov).

Low-rank construction (never materialize the dense d^n × d^n N(P_C)):
    M ∈ C^{d^n × (n_kept·K)},  M[:, i·K+k] = √w_i E_i ψ_k.
    Then N(P_C) = M M† exactly. SVD M = U diag(s) V† gives
    N(P_C) = U diag(s²) U†,  N(P_C)^{-1/2} = U diag(1/s · [s>tol]) U†.
The same per-(op,codeword) tensor K_psi[i,k] = √w_i E_i ψ_k serves both the
M build and the per-shot fidelity.

**Channel-(b) note (R14-5 §A.2 / Option 1).** The composed Meth Kraus channel
has no tractable Kraus list at n=9 (6^324). So Petz for the Meth channel is
built against the TWIRLED Meth Pauli channel (the 685-op weights R14-3a trained
against) and EVALUATED on real Meth Kraus noise via a separate `noise_fn`. The
Petz operators for the twirled-Pauli (channel a) and the real Kraus (channel b)
evaluations are IDENTICAL — only the noise sampled differs. Build once per code,
evaluate against both channels. This is "recovery designed against your channel
model, tested on reality," and the twirled channel is exactly what Prop 3 bounds.

src/decoders/ is the one src/ subtree the R14-5 directive unfroze; all other
src/ files are untouched.
"""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np

from src.decoders._common import apply_local_np


def build_petz_recovery(code_states: np.ndarray,
                        E_full_grouped: list,
                        weights: np.ndarray,
                        d: int, n_qudit: int,
                        rcond: float = 1e-10,
                        weight_floor: float = 1e-15) -> dict:
    """Build the low-rank Petz factors against the weighted-Pauli channel
    N(ρ) = Σ_i weights[i] E_i ρ E_i† on the weight-≤2 closure basis.

    Args:
      code_states: (K, d^n) orthonormal codewords.
      E_full_grouped: ErrorModel(...).build_grouped() (flat order matches `weights`).
      weights: (n_total,) per-op channel weights (the twirled Meth prior, the
        depolarizing prior, or a sparse single-op channel for tests).
      rcond: RELATIVE singular-value cutoff for the Moore-Penrose
        inverse-sqrt — keep σ > rcond·σ_max, drop (regularize to zero) the
        rest. Relative (not absolute) because the Gram-matrix build squares
        the conditioning: an absolute σ-threshold would sit in the eigh-noise
        floor of M†M and keep spurious near-null modes. rcond·σ_max is the
        standard pinv regularization and matches np.linalg.pinv semantics.
      weight_floor: drop ops with weight ≤ this (zero-weight Kraus contribute
        nothing to N(P_C) or to the fidelity).

    Returns dict with:
      U: (d^n, rank) left singular vectors of M for kept modes (s > tol).
      s_inv: (rank,) 1/s on the kept modes — for N(P_C)^{-1/2}.
      s: (rank,) kept singular values of M (= sqrt of N(P_C) eigenvalues).
      K_psi: (n_kept, K, d^n) tensor √w_i E_i ψ_k (reused for evaluation).
      rank: # singular values > tol (= # columns of U).
      n_kept: # ops with weight > weight_floor.

    Build uses the Gram-matrix economy SVD: instead of np.linalg.svd on the
    tall (d^n × r) M (which swaps and takes ~90 min at d^n=19683 because the
    LAPACK gesdd workspace + the 647 MB U exceed RAM), form G = M†M
    (r × r, tiny), eigendecompose it, and recover U = M V / s on the kept
    modes. Everything except M itself stays small → ~1-2 min, no swap.
    """
    K, dim = code_states.shape
    K_psi_list = []
    flat = 0
    for g in E_full_grouped:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        for e in range(mats.shape[0]):
            w = float(weights[flat])
            flat += 1
            if w <= weight_floor:
                continue
            sw = np.sqrt(w)
            cols = np.stack([
                sw * apply_local_np(code_states[k], mats[e], wires, inv,
                                    d, n_qudit)
                for k in range(K)])  # (K, d^n)
            K_psi_list.append(cols)

    if not K_psi_list:
        raise ValueError("no Kraus ops above weight_floor")
    K_psi = np.stack(K_psi_list)               # (n_kept, K, d^n)
    M = K_psi.reshape(-1, dim).T               # (d^n, r), r = n_kept*K

    # Gram-matrix economy SVD: G = M†M = V diag(s²) V†.
    G = M.conj().T @ M                          # (r, r) Hermitian PSD
    evals, V = np.linalg.eigh(G)                # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    V = V[:, order]
    s_all = np.sqrt(np.clip(evals, 0.0, None))
    keep = s_all > rcond * s_all.max()
    s = s_all[keep]
    # U_kept = M V_kept / s_kept  (candidate left singular vectors).
    U = (M @ V[:, keep]) / s[np.newaxis, :]
    # Genuine-mode filter. Forming G = M†M squares the conditioning, so eigh
    # produces spurious near-null eigenvalues whose √ can exceed rcond·σ_max
    # and overlap the genuine small-σ range (no rcond cleanly separates them).
    # A genuine left singular vector satisfies M V_i = s_i U_i with ‖U_i‖ = 1;
    # a spurious mode has M V_i ≈ 0, so ‖U_i‖ = ‖M V_i‖/s_i ≈ 0. Keep unit-norm
    # columns. (Spurious modes are ⊥ range(M) = supp(N(P_C)), so they never
    # affected the LER — verified identical to full-SVD — but this keeps U
    # a clean orthonormal basis of the support.)
    col_norms = np.linalg.norm(U, axis=0)
    genuine = col_norms > 0.5
    U = np.ascontiguousarray(U[:, genuine])
    s = s[genuine]
    s_inv = 1.0 / s
    Uh = np.ascontiguousarray(U.conj().T)   # precompute once (reused per batch)
    return {'U': U, 'Uh': Uh, 's_inv': s_inv, 's': s, 'K_psi': K_psi,
            'rank': int(genuine.sum()), 'n_kept': len(K_psi_list)}


def n_pc_inv_sqrt_apply(petz: dict, state: np.ndarray) -> np.ndarray:
    """Apply N(P_C)^{-1/2} to a statevector (or (dim, B) batch) via the
    low-rank factors. U holds only the kept modes, so s_inv has no zeros.
    Uses the precomputed Uh = U† to avoid a per-call conjugate copy."""
    U, Uh, s_inv = petz['U'], petz['Uh'], petz['s_inv']
    y = Uh @ state                       # (rank,) or (rank, B)
    if y.ndim == 1:
        return U @ (s_inv * y)
    return U @ (s_inv[:, None] * y)


def support_projector_apply(petz: dict, state: np.ndarray) -> np.ndarray:
    """Apply Π_supp(N(P_C)) = U U† to a statevector (U is kept-only, s>tol)."""
    U = petz['U']
    return U @ (U.conj().T @ state)


def fidelity_under_petz(petz: dict, code_states: np.ndarray,
                        alpha: np.ndarray, noisy_state: np.ndarray) -> float:
    """Post-recovery fidelity with the original logical state ψ_L = alpha @ code_states.

    F = Σ_i |⟨ψ_L| R_i |noisy⟩|² = Σ_i |⟨ (√w_i E_i ψ_L) | v ⟩|²,
    where v = N(P_C)^{-1/2} |noisy⟩ and √w_i E_i ψ_L = alpha @ K_psi[i].

    Computed as C[i,k] = ⟨K_psi[i,k]|v⟩ = conj(K_psi[i,k] · v*), then
    inner[i] = Σ_k conj(alpha_k) C[i,k]. The single np.dot reads K_psi once
    (no transpose, no large temporaries) — ~20× faster than tensordot-ing
    alpha into the (n_kept, K, d^n) tensor each shot.
    """
    v = n_pc_inv_sqrt_apply(petz, noisy_state)
    C = np.dot(petz['K_psi'], v.conj()).conj()   # (n_kept, K)
    inner = C @ alpha.conj()                      # (n_kept,)
    return float(np.sum(np.abs(inner) ** 2))


def simulate_ler_with_petz(code_states: np.ndarray, petz: dict,
                           noise_fn: Callable, n_qudit: int, d: int,
                           n_shots: int, seed: int,
                           fidelity_threshold: float = 0.5,
                           batch: int = 200) -> np.ndarray:
    """Per-shot bernoulli outcomes (1 = logical error) under Petz recovery.

    `noise_fn` may sample from a DIFFERENT channel than the one `petz` was built
    against (Option 1: twirl-built Petz evaluated on real Kraus noise).

    Batched: the noisy states still come from per-shot `noise_fn` calls, but the
    Petz linear algebra (U/Uh and K_psi contractions) is applied to `batch`
    shots at once so the ~1.2 GB of Petz factors are read once per batch, not
    once per shot — ~100× fewer memory passes than the naïve per-shot loop.
    """
    K, dim = code_states.shape
    rng = np.random.default_rng(seed)
    K_psi = petz['K_psi']
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
        V = n_pc_inv_sqrt_apply(petz, NOISY)             # (dim, b)
        # C[i,k,j] = ⟨K_psi[i,k] | V[:,j]⟩ = conj(K_psi[i,k] · V[:,j]*)
        C = np.tensordot(K_psi, V.conj(), axes=([2], [0])).conj()  # (n_kept, K, b)
        inner = np.einsum('ikj,jk->ij', C, alphas.conj())          # (n_kept, b)
        F = np.sum(np.abs(inner) ** 2, axis=0)                     # (b,)
        outcomes[start:start + b] = (F < fidelity_threshold).astype(np.int8)
    return outcomes
