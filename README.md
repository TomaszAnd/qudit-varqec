# VarQEC: Variational Quantum Error-Correcting Codes for Trapped-Ion Qudits

Noise-adapted QEC codes for trapped-ion qudit hardware using the VarQEC
algorithm ([Cao et al., arXiv:2204.03560](https://arxiv.org/abs/2204.03560))
with native gates (XY, Z, Molmer-Sorensen). Trained codes for d=3,4,5
and n=3..8 with distance 2 or 3.

## Headline result

The ((n,3,3))\_3 qutrit family shows monotonic loss scaling: **loss proportional to n^(-2.7)**
across n=5,6,7,8. At p=0.2 the ((8,3,3))\_3 code achieves LER = 0.009,
which is **4.2x better** than the [[5,1,3]]\_{Z\_3} stabilizer benchmark
(LER = 0.038) under hardware-specific noise.

### Trained codes

| Code | d | n | dist | Loss | LER@p=0.1 |
|------|---|---|------|------|----------|
| ((5,3,3))\_3 | 3 | 5 | 3 | 0.136 | 0.004 |
| ((6,3,3))\_3 | 3 | 6 | 3 | 0.129 | 0.001 |
| ((7,3,3))\_3 | 3 | 7 | 3 | 0.069 | 0.001 |
| ((8,3,3))\_3 | 3 | 8 | 3 | 0.041 | **0.000** |
| [[5,1,3]]\_{Z\_3} | 3 | 5 | 3 | 0 | 0.005 |

Plus 10 distance-2 detection codes (d=3,4,5; n=3..8) and the first
native-gate ququart distance-3 code ((5,4,3))\_4.

See `results/code_summary.md` for the full table.

## Quick start

```bash
pip install -r requirements.txt

# Reproduce the flagship ((8,3,3))_3 code (~80 min, PennyLane factored)
python3 scripts/train.py --d 3 --n 8 --distance 3 --layers 8 --steps 2000 \
    --backend pennylane --seeds 0

# Benchmark LER
python3 scripts/benchmark_ler.py --code qutrit_n8_d3 --n_shots 3000

# Run tests
python3 -m pytest tests/ -v
```

## Repository structure

```
src/                          Native trapped-ion pipeline
  gates.py                    XY, Z, MS, CSUM, CSUB, light-shift (any d)
  encoder.py                  PennyLane encoder (QNode d=3, manual d>3)
  errors.py                   Hardware error basis, closure, ErrorModel
  loss.py                     KL detection loss (Eq. 16)
  simulation.py               Monte Carlo LER + decoders (dense + factored)
  correlated_noise.py         Meth et al. Kraus operators
  jax_backend.py              JAX scan encoder + scan loss + training
  catalog.py                  Code loading, [[5,1,3]]_Z_q benchmark
  analysis.py                 KL residuals, weight enumerators, entropy
  legacy/
    ququart_pipeline.py       Old abstract-gate pipeline (6 codes)
scripts/
  train.py                    Unified training (JAX or PennyLane)
  run_campaign.py             Batch campaign runner
  benchmark_ler.py            LER benchmark CLI
notebooks/
  analyze_codes.py            Jupytext source for analysis notebook
results/
  params/                     Trained .npz files
  simulations/                Cached LER sweeps
  plots/                      Generated figures
  code_summary.md             Summary table
tests/                        242 tests
```

## The loss function

Training optimizes the Knill-Laflamme detection loss (Eq. 16):

    L = sum_E [ sum_{i<j} |<psi_i|E|psi_j>|^2 + (K/4) Var_i <psi_i|E|psi_i> ]

The first term enforces off-diagonal KL conditions. The second (active
for distance >= 3) enforces diagonal uniformity. Both terms use squared
overlaps for smooth gradients with Adam.

### Error basis closure

Hardware errors {Z\_k, X\_{k,k+1}} do not form a group. Products like
Z\_1 * X\_{01} produce operators outside the basis. For distance >= 3,
`closed=True` augments E\_det with these cross-products (e.g. 181 -> 221
for d=3, n=5). Without closure, training silently under-constrains the
KL conditions.

## LER benchmarking

Per-qudit i.i.d. noise with probability p. Lookup-table decoder tries
identity + each single-qudit correction and picks the best code-space
overlap. Two implementations:

- **Dense** (n <= 6): precomputed d^n x d^n correction operators
- **Factored** (n >= 7): single-qudit tensordot, avoids dense matrices

Each data point uses 3000 shots. Distance-2 codes use detection + post-selection.

## Training backends

| Backend | Speed (n=5, 10L) | Compile | Use case |
|---------|------------------|---------|----------|
| PennyLane autograd | 1.89 s/step | 0 | Small codes, analysis |
| JAX scan+checkpoint | 0.21 s/step | ~3 min | Large codes (n <= 7) |
| PennyLane factored | ~14 s/step | 0 | n >= 8 (JAX compile fails) |

## Roadmap

### Scientific
- Distance-3 codes for d=4,5 (partially trained, needs more layers/seeds)
- Distance-4 attempt: ((7,3,4))\_3
- ((9,3,3))\_3 to extend n-scaling to 5 points
- Correlated noise training (Meth et al. channel)

### Engineering
- Fix JAX factored scan compile for n >= 8
- GPU training support
- Unified minibatch for JAX backend

## References

- Cao et al., [arXiv:2204.03560](https://arxiv.org/abs/2204.03560) -- VarQEC algorithm
- Meth et al., [arXiv:2310.12110v3](https://arxiv.org/abs/2310.12110) -- Trapped-ion noise model (App. J)
- Chau 1997, Rains 1997 -- [[5,1,3]]\_{Z\_q} benchmark code
