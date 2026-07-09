"""Decoders for VarQEC qudit codes.

Implementations:
  - lookup (src/simulation.py): unweighted lookup table; R14-4 baseline.
  - weighted_map: MAP with channel prior over weight-≤2 corrections.
  - petz (added in R14-5 §C): Petz transpose recovery (Bényi-Oreshkov 2010).

All produce per-shot bernoulli outcome arrays compatible with the
bootstrap CI helper in scripts/benchmark_ler_meth_pauli.py.
"""
from src.decoders.weighted_map import (
    build_weighted_correction_set,
    simulate_ler_with_weighted_map,
)
from src.decoders.priors import (
    uniform_depolarizing_prior,
    meth_pauli_prior,
)
from src.decoders.petz import (
    build_petz_recovery,
    simulate_ler_with_petz,
    fidelity_under_petz,
)

__all__ = [
    "build_weighted_correction_set",
    "simulate_ler_with_weighted_map",
    "uniform_depolarizing_prior",
    "meth_pauli_prior",
    "build_petz_recovery",
    "simulate_ler_with_petz",
    "fidelity_under_petz",
]
