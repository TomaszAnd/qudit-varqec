"""Shared primitives for the VarQEC decoder package.

Lifted verbatim from scripts/benchmark_ler_meth_pauli.py (R14-4) so all
decoder code shares one canonical wire-grouped-matrix application primitive,
group-offset helper, and encoder-forward routine.

`apply_local_np(state, M, wires, inverse_perm, d, n_qudit) -> state` is the
numpy port of src.jax_backend._apply_local_matrix.
"""
from __future__ import annotations
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def apply_local_np(state: np.ndarray, matrix: np.ndarray,
                   wires: tuple, inverse_perm: tuple,
                   d: int, n_qudit: int) -> np.ndarray:
    """Apply a (d^w, d^w) matrix to `wires` of an n-qudit statevector.

    Identity sentinel (wires=(), matrix=[[1.0]]) is a scalar multiply.
    """
    if len(wires) == 0:
        return matrix[0, 0] * state
    shape = (d,) * n_qudit
    s = state.reshape(shape)
    w = len(wires)
    M = matrix.reshape((d,) * (2 * w))
    axes = list(range(w, 2 * w))
    contracted = np.tensordot(M, s, axes=(axes, list(wires)))
    s = np.transpose(contracted, inverse_perm)
    return s.reshape(-1)


def flat_group_offsets(E_grouped: list) -> np.ndarray:
    """Cumulative offsets per group: offsets[g] = sum of sizes of groups 0..g-1.

    offsets has length len(E_grouped)+1; offsets[g] + e is the flat index of
    op e in group g (the same flat ordering as meth_pauli_weights.npz['weights']).
    """
    sizes = [int(g['matrices'].shape[0]) for g in E_grouped]
    return np.concatenate([[0], np.cumsum(sizes)])


def encoder_forward(checkpoint_npz: str, connections: list):
    """Load a .npz checkpoint, rebuild the JAX encoder, return code_states (K, d^n)
    as numpy plus (d, n_qudit, K)."""
    import jax
    import jax.numpy as jnp
    from src.jax_backend import create_jax_encoder

    data = np.load(checkpoint_npz, allow_pickle=True)
    params = jnp.asarray(data['params'])
    d = int(data['d'])
    n_qudit = int(data['n_qudit'])
    K = int(data['K'])

    enc, _, _ = create_jax_encoder(n_qudit, d, connections=connections,
                                    use_scan=True)
    enc_vmapped = jax.jit(jax.vmap(enc, in_axes=(None, 0)))
    code_states = np.asarray(enc_vmapped(params, jnp.arange(K)))
    return code_states, d, n_qudit, K
