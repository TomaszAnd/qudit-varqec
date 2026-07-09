#!/usr/bin/env python3
"""Reproduce Table III (Shor-Laflamme weight enumerators) and Table IV
(analytic per-code encoder resources) from committed trained parameters.

Table III enumerators are computed from the codeword projector over the
*hardware* single-qudit error basis (qudit_hardware_error_basis) -- this is
the basis the paper uses (it reproduces the [[5,1,3]] benchmark row exactly;
the generalized-Pauli basis instead gives A_j=0 for the pure code).

Usage:  PYTHONPATH=. python3 scripts/reproduce_tables.py
No retraining: reads results/params (via src.catalog) only.
"""
import itertools
import numpy as np
from src.catalog import load_code, five_qudit_code_states
from src.errors import qudit_hardware_error_basis
from src.decoders._common import apply_local_np


def _enumerators_grouped(code_states, n, d, weight, ops):
    """Wire-grouped Shor-Laflamme A_w, B_w via LOCAL error application only
    (no d^n x d^n projector), so this scales to n=9 in well under a second --
    the same method the paper uses. For each weight-w wire subset and each
    tensor product of single-qudit `ops`, the error is applied to each codeword
    on its <=w wires and the K x K matrix M_kl = <psi_k|E|psi_l> is formed:
        A_w = (1/K^2) sum_E |Tr M|^2 ,   B_w = (1/K) sum_E sum_kl |M_kl|^2 .
    """
    K = code_states.shape[0]
    A = B = 0.0
    for sub in itertools.combinations(range(n), weight):
        left = [i for i in range(n) if i not in sub]
        inv = tuple((list(sub) + left).index(i) for i in range(n))
        for choice in itertools.product(ops, repeat=weight):
            E = choice[0]
            for e in choice[1:]:
                E = np.kron(E, e)
            E_psi = np.stack([apply_local_np(code_states[k], E, sub, inv,
                                             d=d, n_qudit=n) for k in range(K)])
            M = np.conj(code_states) @ E_psi.T
            A += abs(np.trace(M)) ** 2
            B += float(np.sum(np.abs(M) ** 2))
    return A / K ** 2, B / K


def enumerators(code_states, n, d):
    ops = [np.asarray(m) for m in qudit_hardware_error_basis(d)]
    A = np.ones(3)
    B = np.ones(3)  # A_0 = B_0 = 1 (identity)
    for w in (1, 2):
        A[w], B[w] = _enumerators_grouped(code_states, n, d, w, ops)
    return A, B


# ---- Table III -------------------------------------------------------------
# (label, catalog name or None for the analytic benchmark, n, d)
TABLE3 = [
    ("[[5,1,3]]_Z3", None,          5, 3),
    ("((9,3,3))_3",  "d3_n9_dist3", 9, 3),
    ("((6,4,3))_4",  "d4_n6_dist3", 6, 4),
    ("((5,5,3))_5",  "d5_n5_dist3", 5, 5),
]


def table_iii():
    print("TABLE III  Shor-Laflamme weight enumerators (hardware basis, j<=2)")
    print(f"{'Code':14} {'A0':>6} {'A1':>8} {'A2':>9} {'B0':>6} {'B1':>8} "
          f"{'B2':>9} {'B1-A1':>8} {'B2-A2':>8}")
    for label, name, n, d in TABLE3:
        cs = five_qudit_code_states(d) if name is None else load_code(name)['code_states']
        A, B = enumerators(cs, n, d)
        print(f"{label:14} {A[0]:6.3f} {A[1]:8.3f} {A[2]:9.3f} "
              f"{B[0]:6.3f} {B[1]:8.3f} {B[2]:9.3f} "
              f"{B[1]-A[1]:8.3f} {B[2]-A[2]:8.3f}")


# ---- Table IV --------------------------------------------------------------
# Analytic per-code encoder resources at ring connectivity (|E| = n), L = 4.
# Formulas from the paper caption:
#   P_l   = (5n + 2|E|)(q-1)     per-layer parameter count
#   XY/l  = 2n(q-1) ,  MS/l = |E|(q-1) ,  Z/l = n(q-1)
TABLE4 = [("((9,3,3))_3", 3, 9), ("((6,4,3))_4", 4, 6), ("((5,5,3))_5", 5, 5)]
L = 4


def table_iv():
    print("\nTABLE IV  Per-code encoder resources (ring, |E|=n, L=4)")
    print(f"{'Code':14} {'q':>2} {'n':>2} {'K':>2} {'|E|':>3} {'Pl':>5} "
          f"{'L*Pl':>6} {'XY/l':>5} {'MS/l':>5} {'Z/l':>4}")
    for label, q, n in TABLE4:
        E = n  # ring connectivity
        Pl = (5 * n + 2 * E) * (q - 1)
        xy, ms, z = 2 * n * (q - 1), E * (q - 1), n * (q - 1)
        print(f"{label:14} {q:2d} {n:2d} {q:2d} {E:3d} {Pl:5d} "
              f"{L*Pl:6d} {xy:5d} {ms:5d} {z:4d}")


if __name__ == "__main__":
    table_iii()
    table_iv()
