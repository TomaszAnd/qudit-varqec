"""Pauli-twirled Meth-channel importance weights, computed LIVE from the
corrected `src.correlated_noise` Kraus channel (Gaussian-fixed).

For each operator E_i in `ErrorModel(d, n_qudit, distance, closed=True)`
(default hardware closure basis, the R14 weight-≤2 closure), the weight is

    lambda_i = Prod_q  p_spec(E_i^q)

where E_i^q is the q-th single-qudit tensor factor of E_i and p_spec is the
single-qudit Pauli-twirl probability under the SPECTATOR channel (the pragmatic
R14-3 role approximation; see the recovered generator scripts/meth_pauli_weights.py).
Because it calls `control/target/spectator_qudit_kraus` directly, the noise fix
(base=f*sqrt(sigma_p^2), decay=eta^{f^2/2}) flows through automatically — these
weights are for the CORRECTED channel.

Ported from the (dropped) scripts/meth_pauli_weights.py so both the CLI and the
seed race compute the same corrected weights.
"""
from __future__ import annotations
import numpy as np


def pauli_twirl_single_qudit(kraus_diags_or_mats, pauli_ops, d):
    """lambda_P = (1/d^2) Sum_a |Tr(P^dag K_a)|^2 for each Pauli P."""
    weights = np.zeros(len(pauli_ops), dtype=float)
    for i, P in enumerate(pauli_ops):
        s = 0.0
        for K in kraus_diags_or_mats:
            if K.ndim == 1:
                t = np.sum(P.conj().diagonal() * K)
            else:
                t = np.trace(P.conj().T @ K)
            s += abs(t) ** 2
        weights[i] = s / (d ** 2)
    return weights


def build_single_qudit_pauli_basis(d):
    """[I] + hardware errors + same-qudit closure (13 ops for d=3)."""
    from src.errors import qudit_hardware_error_basis, close_error_basis
    Id = np.eye(d, dtype=complex)
    base = qudit_hardware_error_basis(d)
    cross = close_error_basis(base)
    ops = ([Id] + [np.asarray(o, dtype=complex) for o in base]
           + [np.asarray(o, dtype=complex) for o in cross])
    labels = (["I"] + [f"Z_{k}" for k in range(1, d)]
              + [f"X_{k}{k+1}" for k in range(d - 1)]
              + [f"closure_{i}" for i in range(len(cross))])
    return list(zip(labels, ops))


def build_per_role_weights(d, eta, n_max=2):
    """Single-qudit Pauli-twirl probabilities per role (control/target/spectator),
    from the CORRECTED physical Kraus operators."""
    from src.correlated_noise import (control_qudit_kraus, target_qudit_kraus,
                                       spectator_qudit_kraus)
    pl = build_single_qudit_pauli_basis(d)
    pauli_ops = [op for _, op in pl]
    k_ctrl = control_qudit_kraus(d, control_level=0, eta=eta, n_max=n_max)
    k_tgt = target_qudit_kraus(d, control_level=0, target_level=1, eta=eta, n_max=n_max)
    k_spec = spectator_qudit_kraus(d, eta=eta, n_max=n_max)
    return {"labels": [lbl for lbl, _ in pl], "ops": pauli_ops,
            "control": pauli_twirl_single_qudit(k_ctrl, pauli_ops, d),
            "target": pauli_twirl_single_qudit(k_tgt, pauli_ops, d),
            "spectator": pauli_twirl_single_qudit(k_spec, pauli_ops, d)}


def _factor_per_qudit(grouped_op_matrix, wires, n_qudit, d):
    """n_qudit single-qudit d×d factors (identity off support); SVD-split for
    weight-2 tensor-product ops."""
    Id = np.eye(d, dtype=complex)
    factors = [Id.copy() for _ in range(n_qudit)]
    if len(wires) == 0:
        return factors
    M = np.asarray(grouped_op_matrix)
    if len(wires) == 1:
        factors[wires[0]] = M.astype(complex)
        return factors
    M_re = M.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    U, S, Vh = np.linalg.svd(M_re)
    A = (np.sqrt(S[0]) * U[:, 0]).reshape(d, d)
    B = (np.sqrt(S[0]) * Vh[0, :]).reshape(d, d)
    if not np.allclose(M, np.kron(A, B), atol=1e-8) and \
       np.allclose(M, np.kron(B, A), atol=1e-8):
        A, B = B, A
    factors[wires[0]] = A
    factors[wires[1]] = B
    return factors


def _pauli_proj(M, pauli_ops, d):
    return np.array([abs(np.trace(P.conj().T @ M)) ** 2 / (d ** 2)
                     for P in pauli_ops], dtype=float)


def lift_to_flat_weights(E_grouped, d, n_qudit, per_role):
    """lambda_i = Prod_q Sum_k |<P_k, E_i^q>|^2 * w_spec[k], normalised Sum=1."""
    pauli_ops = per_role["ops"]
    w_spec = per_role["spectator"]
    n_total = sum(int(g['matrices'].shape[0]) for g in E_grouped)
    weights = np.zeros(n_total, dtype=float)
    flat = 0
    for g in E_grouped:
        wires = g['wires']
        for e in range(g['matrices'].shape[0]):
            factors = _factor_per_qudit(g['matrices'][e], wires, n_qudit, d)
            logw, valid = 0.0, True
            for q in range(n_qudit):
                p_q = float(np.sum(_pauli_proj(factors[q], pauli_ops, d) * w_spec))
                if p_q <= 0:
                    valid = False
                    break
                logw += np.log(p_q)
            weights[flat] = float(np.exp(logw)) if valid else 0.0
            flat += 1
    s = weights.sum()
    return weights / s if s > 0 else weights


def compute_meth_group_weights(d, n_qudit, distance=3, eta=0.9296, n_max=2):
    """Return (group_weights list aligned with ErrorModel default-basis grouped
    E_det, flat_weights). Computed live from the corrected channel."""
    from src.errors import ErrorModel
    grouped = ErrorModel(d=d, n_qudit=n_qudit, distance=distance,
                         closed=(distance >= 3)).build_grouped(verbose=False)
    per_role = build_per_role_weights(d, eta, n_max)
    flat = lift_to_flat_weights(grouped, d, n_qudit, per_role)
    sizes = [g['matrices'].shape[0] for g in grouped]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    group_weights = [np.asarray(flat[offs[i]:offs[i + 1]], dtype=float)
                     for i in range(len(sizes))]
    return group_weights, flat
