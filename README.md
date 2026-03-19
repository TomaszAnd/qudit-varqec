# VarQEC: Variational Quantum Error-Correcting Codes for Trapped-Ion Qudits

Finding noise-adapted quantum error-correcting codes for ququart (d=4) systems
using the VarQEC algorithm (Cao et al., arXiv:2204.03560).

## Overview

This project implements the VarQEC algorithm to discover ((5, 4, d))\_4 quantum
error-correcting codes tailored to trapped-ion noise models from Meth et al.
(arXiv:2310.12110v3, Appendix J).

Five ququarts are encoded as 10 qubits (1024-dim Hilbert space). The variational
circuit optimizes the Knill-Laflamme conditions to find codes that detect or
correct errors from specific noise channels.

### Trained codes

| Code | Noise model | Distance | Layers | Final loss | Status |
|------|------------|----------|--------|------------|--------|
| Dephasing d=2 | Pauli Z-type | 2 | 1 | ~1e-7 | Converged |
| Dephasing d=3 | Pauli Z-type | 3 | 2 | ~6e-5 | Converged |
| Depolarizing d=2 | Full Pauli | 2 | 2 | ~6e-5 | Converged |
| Depolarizing d=3 | Full Pauli | 3 | 6 | ~6 | Partial |
| Correlated d=2 | Trapped-ion (simplified) | 2 | 2 | ~2e-4 | Converged |
| Correlated d=3 | Trapped-ion (simplified) | 3 | 3 | ~7e-4 | Converged |

## Quick start

```bash
cd qudit-varqec/
pip install -r requirements.txt

# Train all d=2 codes (~25 min total)
bash scripts/run_all.sh

# Train specific d=3 codes
python3 scripts/train_dephasing_d3.py --n_steps 500
python3 scripts/train_correlated_simplified_d3.py --n_steps 1000
python3 scripts/train_depolarizing_d3.py --n_steps 1000 --n_layers 6
```

### Analysis notebook

```bash
cd qudit-varqec/notebooks
jupyter notebook showcase_varqec.ipynb
```

### Tests

```bash
python3 -m pytest tests/ -v
```

## Repository structure

```
├── scripts/
│   ├── train_dephasing_d2.py           # 1 layer, ~3 min
│   ├── train_dephasing_d3.py           # 2 layers, ~5 hrs
│   ├── train_depolarizing_d2.py        # 2 layers, ~10 min
│   ├── train_depolarizing_d3.py        # 6 layers, detection loss, ~3.5s/step
│   ├── train_correlated_simplified_d2.py  # 2 layers, ~10 min
│   ├── train_correlated_simplified_d3.py  # 3 layers, detection loss, ~1.1s/step
│   └── run_all.sh                      # Run all d=2 scripts
├── src/
│   ├── pauli_ops.py                    # Pauli matrices for ququarts (15 non-identity)
│   ├── error_sets.py                   # Dense Pauli error sets for detection/correction
│   ├── error_sets_factored.py          # Memory-efficient factored errors (600 KB vs 37 GB)
│   ├── gates.py                        # Parameterized gates (G_theta, entangling layers)
│   ├── encoder.py                      # Variational quantum circuit encoder
│   ├── kl_loss_fast.py                 # KL loss functions (detection + correction variants)
│   ├── trapped_ion_noise.py            # Trapped-ion Kraus operators (Meth et al.)
│   ├── correlated_error_sets.py        # Correlated noise error sets (diagonal)
│   ├── logical_error_rate.py           # Monte Carlo LER simulation
│   └── decoders.py                     # Projection, detection, lookup table decoders
├── notebooks/
│   ├── showcase_varqec.ipynb           # Main analysis notebook
│   └── nb_utils.py                     # Notebook helper functions
├── tests/                              # pytest suite (124 tests)
├── results/                            # Trained parameters and plots
├── .gitignore
├── README.md
└── requirements.txt
```

## Key concepts

### Noise models

- **Dephasing**: Only Z-type errors (3 diagonal Paulis per qudit)
- **Depolarizing**: All 15 non-identity single-qudit Paulis
- **Correlated (simplified)**: Trapped-ion noise from nearest-neighbor gate
  interactions (Meth et al. Eq. J3), producing ~25k diagonal error operators

### Loss functions

The KL loss enforces the Knill-Laflamme conditions variationally:

- **Detection-style loss** (Paper Eq. 16): Single sum over E\_det with both
  off-diagonal and diagonal variance terms. O(|E\_det|). Used for d>=3.
- **Correction-style loss**: Separate off-diagonal (E\_det) and diagonal variance
  (E\_a†E\_b products) terms. O(|E\_corr|^2) for Term 2. Used for d=2 and dephasing d=3.

### Memory optimization

Depolarizing d=3 has 2326 detection errors. Dense 1024x1024 matrices would
require 37 GB. The factored representation stores each error as single-qudit
operator pairs, reducing memory to ~600 KB.

## References

- Cao et al., "Quantum variational learning for quantum error-correcting codes",
  [arXiv:2204.03560](https://arxiv.org/abs/2204.03560)
- Meth et al., trapped-ion noise model,
  [arXiv:2310.12110v3](https://arxiv.org/abs/2310.12110v3), Appendix J
