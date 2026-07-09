# VarQEC — Reproduction Package

Minimal, self-contained reproduction of the data figures and tables in
*"Native-gate variational error-correcting codes for trapped-ion qudits."*
Codes are found with the VarQEC algorithm
([Cao et al., arXiv:2204.03560](https://arxiv.org/abs/2204.03560)) using native
trapped-ion gates (XY, Z, Mølmer–Sørensen) and benchmarked under the
trapped-ion noise model of
[Meth et al., arXiv:2310.12110v3](https://arxiv.org/abs/2310.12110) (App. J).

This repository is a **pruned copy** of the working research repo: it keeps only
what is on the reproduction path of a paper data figure/table (plus the `src/`
modules, tests, and trained parameters those paths need). **Figures are
regenerated from committed trained parameters — no retraining is required.**

## What reproduces

See **[REPRODUCE.md](REPRODUCE.md)** for the exact command → params → expected
value for each figure/table. (REPRODUCE numbers figures 8–15, a fixed **+4
offset** from the PDF's Fig 4–11.)

| Figure / Table | Generator |
|------|--------|
| PDF Fig 4+5 — 2-panel (loss + LER vs [[5,1,3]]) | `notebooks/analyze_codes.py` → `n_scaling_d3_dist3.png` |
| PDF Fig 6 — cross-code distance-3 | `notebooks/analyze_codes.py` → `ler_vs_p_all_dist3.png` |
| PDF Fig 8 — 5-sampler loss / op-EV | `scripts/r14_10_training_curves.py` |
| PDF Fig 9 — sampler-equivalence MAP LER | `scripts/r14_10_sampler_equivalence.py` |
| PDF Fig 10 — LER vs p_phys (MAP) | `scripts/r14_10_ler_vs_pphys.py` |
| Table II / III / IV | `scripts/reproduce_tables.py` (+ committed CSVs / params) |

## Install

```bash
pip install -r requirements.txt          # Python 3.12; jax 0.4.35, pennylane 0.42, optax 0.2.4
export PYTHONPATH=.                       # so `import src...` and notebooks resolve
```

## Quick start

```bash
# PDF Fig 4+5 (merged 2-panel) + Fig 6 + Table III (reads committed params + cached LER sims)
cd notebooks && python3 analyze_codes.py && cd ..

# PDF Fig 8–10 (Sec IX sampler study; reads committed training traces + MAP-LER CSVs)
python3 scripts/r14_10_training_curves.py
python3 scripts/r14_10_sampler_equivalence.py
python3 scripts/r14_10_ler_vs_pphys.py

# Tests guarding the reproduction modules
python3 -m pytest tests/ -q
```

## Repository structure

```
src/               Native trapped-ion pipeline (all modules on a figure path)
scripts/           Figure generators (r14_10_*), Sec IX trainers, seed_race CLI, benchmarks
notebooks/         analyze_codes.py — generator for the analyze-codes figures + Table III
results/
  params/          Trained .npz backing the figures / Table III (committed on purpose)
  saved_params/    All-to-all campaign params for Table II
  simulations/     Cached hardware-noise LER sweeps
  round14_scoping/ Sec IX training traces + MAP-LER CSVs
  benchmarks_*/    LER benchmark CSVs (Table II)
  best_practice_runs/  New training output (gitignored; never mixes with published params)
tests/             Tests guarding the kept src/ modules
archive/           Preserved Sec VII pixels/machinery, launchers
docs/              NOISE_MODEL, PROVENANCE, GATE_AUDIT, RESULTS_best_practice, FOLLOWUP
REPRODUCE.md       Figure/table → command → expected value
```

## `src/` module reference

**Top level**

- **`gates.py`** — native trapped-ion gate set for any dimension `d`: the
  single-qudit `XY_gate` (XY-plane rotation in a chosen 2-level subspace) and
  `Z_gate` (relative phase), the two-qudit `MS_gate` (Mølmer–Sørensen, the
  default entangler), and the fixed comparison gates `CSUM_gate`/`CSUB_gate`
  (genuine-qudit controlled sum/subtract), `CEX_gate` (embedded-qubit
  controlled exchange), `zz_subspace_gate` (paper Eq. 4), and `ls_global_gate`
  (Hrmo Eq. 3). See *Entangling gates* below.
- **`encoder.py`** — native encoder for any `d`: a PennyLane qutrit device for
  `d=3` (fast QNode path) and a manual state-vector simulator for `d>3`. Builds
  the L-layer ansatz `(single-qudit XY/Z layer) × (MS entangling layer)` over a
  given connectivity; the entangler is always `MS_gate`.
- **`errors.py`** — hardware error basis for trapped-ion qudits. `ErrorModel`
  builds the discrete detection/correction sets `E_det` from the native
  `{Z_k, X_{k-1,k}}` transitions (with `Y, L, L†` closure for `basis="full"`),
  grouped by acting wires and **without** identity padding. See *Error model*.
- **`loss.py`** — Knill-Laflamme loss functions (Cao et al. Eq. 16 = paper
  Eq. 10). Two families: correction-based (off-diagonal coherence + `K/4`
  diagonal variance over `E_a†E_b`, `O(|E_corr|²)`) and detection-based (single
  sum over `E_det`, `O(|E_det|)`, used for depolarizing/correlated `d=3`). Also
  `save/load_varqec_result`.
- **`simulation.py`** — Monte-Carlo logical-error-rate simulation: encode a
  random logical state → apply a noise channel → project/decode → count logical
  errors. Provides the noise-fn factories (dephasing, depolarizing, per-qudit
  Pauli, and `make_correlated_dephasing_noise_fn` for the Meth channel) and the
  R14-4 unweighted lookup decoder. See *Decoders + LER*.
- **`correlated_noise.py`** — correlated trapped-ion noise model (Meth App. J):
  single-qudit dephasing Kraus operators (control/target/spectator roles),
  subspace depolarizing, amplitude damping, and the correlated/combined error-set
  builders. The core `_kraus_diagonal` uses the Gaussian-averaged form (see
  *Error model*).
- **`jax_backend.py`** — the differentiable core. JIT general-`d` encoder,
  wire-grouped `vmap` KL losses (weighted v1/v2, GEMM-batched fast path), the
  sampling weight builders `stratified_weights` and `hoeffding_weights`, and an
  Adam training loop. Everything above (seed race, samplers) composes these.
- **`seed_race.py`** — multi-start "seed race" orchestration: races many random
  inits of the same ansatz and keeps the seed that certifies (full-batch) the
  target loss soonest. Sits **above** `jax_backend` (its own module, not folded
  into the backend); the CLI wrapper is `scripts/seed_race.py`. See *Seed race*.
- **`analysis.py`** — code characterization: weight enumerators, KL residuals,
  distance verification, entanglement entropy for any trained/analytic code.
- **`catalog.py`** — catalog + unified loader (`load_code`, `list_codes`) for
  trained VarQEC codes and analytic benchmark codes (e.g. `[[5,1,3]]_{Z_3}`).

**`decoders/`**

- **`_common.py`** — shared decoder primitives (canonical wire-grouped matrix
  application, group-offset helpers, encoder-forward), lifted from R14-4.
- **`weighted_map.py`** — weighted MAP decoder over the weight-≤2 closure basis
  with a channel prior: picks `argmax_C  w_C · Σ_k |⟨C·ψ_k|noisy⟩|²` (paper
  Sec IX). Reduces to the unweighted lookup on single-error channels.
- **`priors.py`** — channel-prior builders (per-op weight vectors on the
  weight-≤2 closure basis) consumed by weighted-MAP / Petz.
- **`petz.py`** — Petz transpose recovery decoder (the recovery the ℓ2 loss
  trains against; appendix).
- **`ler_driver.py`** — shared MAP-LER sweep driver for the R14-6/7d/7e/9/10
  channel-(b) η-sweep experiments (each figure script is a thin config wrapper).

**`sampling/`**

- **`stratified_importance.py`** — the Meth-weighted samplers: Neyman per-group
  budget allocation + within-group importance sampling (Rosalin), plus the
  full-basis variant. Unbiased single-step estimates of the weighted loss. See
  *Sampling techniques*.

**`legacy/`**

- **`ququart_pipeline.py`** — abstract ququart (2-qubit-per-ququart) pipeline
  for loading/evaluating the six original abstract-gate Pauli-basis VarQEC codes
  (backward-compat; re-exported from `src/__init__.py`).

## Error model

Two distinct surfaces, both traceable to Meth App. J:

1. **Discrete training basis** (`src/errors.py::ErrorModel`, `train.py --noise
   full`, the default). The KL loss is enforced over the native single-qudit
   transitions `{Z_k, X_{k-1,k}}` (plus `Y, L, L†` and the weight-≤2 closure for
   `basis="full"`), grouped by wires, with **no identity padding**. These are the
   phase-flip / level-swap unitaries the hardware actually produces; `σ_p²` does
   not enter here directly (Meth-weighted training reweights these ops by a prior
   λ derived from the Kraus channel).

2. **Continuous correlated Kraus channel** (`src/correlated_noise.py`, used for
   the Sec IX LER benchmark). For a random motional-mode phase `Φ ~ N(0, σ_p²)`,
   reparametrized as `η = e^{−σ_p²}`, the diagonal Kraus operators are

   ```
   E_n[k] = (f_k √(σ_p²))^n · η^{f_k²/2} / √(n!)
   ```

   with level-coupling `f_k` (control/target/spectator roles). This is the
   **Gaussian-averaged** channel: coherence between levels differing in coupling
   by `Δf` decays as `E[e^{iΔfΦ}] = e^{−½σ_p²Δf²}`, the standard Gaussian
   characteristic function — i.e. it **agrees with Meth** (verified: coherence
   multipliers `1.0/0.9642/0.8642` at `η=0.9296`, `Σ_n E_n†E_n = I`).

   > The paper's Appendix C prints an extra factor 2 (`base=f√(2σ²)`,
   > `decay=η^{f²}`, i.e. `e^{−σ²Δf²}`). That is a **manuscript-side**
   > transcription error to correct paper-side, not in the code. Because the
   > corrected channel is weaker at a fixed `η`, the operating point shifts
   > (`p_phys`: 0.56 → 0.376 at `η=0.9296`; the paper's `p_phys≈0.56` now maps to
   > `η=0.8642`). Pre-fix Sec IX / `η→p_phys` artifacts are flagged for
   > regeneration in [`docs/FOLLOWUP.md`](docs/FOLLOWUP.md).

See [`docs/NOISE_MODEL.md`](docs/NOISE_MODEL.md) for the full element-by-element
conformance check.

## Entangling gates

The trained ansatz uses one entangler; the rest are library-only comparison
gates. A key distinction:

- **Embedded-qubit entanglers** act inside a 2-level target subspace.
  `CEX_gate(d, c, t1, t2)` swaps `|c,t1⟩ ↔ |c,t2⟩` (identity elsewhere); on
  `(|00⟩+|10⟩)/√2` it produces the embedded-qubit Bell state `|00⟩+|11⟩`, and it
  reduces to the qubit CNOT at `d=2`. Fixed / parameter-free; library-only.
- **Genuine-qudit entanglers** act across all levels. `CSUM_gate`
  (`|i,j⟩→|i,(j+i) mod d⟩`) makes the full-dimensional GHZ-like state
  `|00⟩+|11⟩+…+|(d-1)(d-1)⟩`; `CSUB_gate = CSUM†`.

| Gate | What it is | When used |
|---|---|---|
| **`MS_gate`** | physical XX+YY Mølmer–Sørensen, `σ_φ = cosφ·X + sinφ·Y` (Ringbauer Eq. 2) | **default** ansatz/training entangler |
| `zz_subspace_gate` | diagonal ZZ on a 2-level subspace (paper Eq. 4) | Sec VII ablations (non-default) |
| `ls_global_gate` | global light-shift phase on all `j≠k` pairs (Hrmo Eq. 3) | Sec VII comparison, `--entangler ls` (non-default) |
| `CSUM_gate` / `CSUB_gate` | genuine-qudit controlled sum / subtract | comparison; makes `|00⟩+|11⟩+|22⟩` |
| `CEX_gate` | embedded-qubit controlled exchange | comparison; makes `|00⟩+|11⟩` |

`scripts/train.py --entangler` defaults to `ms`; `ls` is opt-in (jax backend
only). There is no `--entangler cex` unless the CLI dispatches it — CEX stays a
library primitive.

## Sampling techniques

The per-step loss can be estimated from a subset of `E_det` to cut hardware cost
(operator-expectation-value, "op-EV", evaluations per step):

- **Full-batch** — all of `E_det` every step (`684` op-EVs/step for the Sec IX
  `((9,3,3))₃` config). Exact gradient, most expensive.
- **Stratified (uniform)** — `jax_backend.stratified_weights`: uniform
  subsampling within each wire group at a fixed fraction (`≈117`/step).
- **Hoeffding (flat-uniform)** — `jax_backend.hoeffding_weights`: flat sampling
  across all of `E_det`.
- **Importance-weighted (Neyman/Rosalin)** —
  `sampling/stratified_importance.py`: Neyman per-group budgets `∝ √Σ w_e²` plus
  within-group draws `∝ w_e`. **This is the optimal sampler**: it reaches the
  **same loss basin and the same LER as full-batch at ≈7.6× lower op-EV cost**
  (`684 → 90`/step), because Neyman allocation minimizes the single-step
  estimator variance for a fixed budget.
- **Importance-truncated / full-basis importance** — variants that split the
  weight-1 / weight-2 sectors and sample each independently (unbiased for the
  full loss over disjoint supports).

All samplers feed the same weighted `vmap` loss; passing a ones-weight tuple
recovers full-batch exactly. The earlier reverted "gather" sampled-loss path was
removed upstream and is not present here.

## Seed race

`src/seed_race.py` runs a multi-start search: each seed trains the same ansatz
from `θ ~ U[0, 2π)` and races the others.

- **H1 — full-batch certification.** A seed wins only when its **full-batch**
  loss drops below the target; a cheap sampled loss can *trigger* a check but
  never grants victory (false triggers are re-evaluated full-batch).
- **H2 — mercy rule.** The step budget is honored in full until the first
  certified record exists; only then are slower seeds pruned at the record step.

The ansatz is the L-layer XY/MS/Z native circuit (paper Sec V A); `L` is chosen
per code. The CLI wrapper is `scripts/seed_race.py` (re-exports the core).

**Sampling budget (campaign default).** Each racing seed trains on an
importance-sampled loss (cheap steps) and certifies on the full batch — so the
fraction is a pure *search-cost* knob (any certifying fraction gives a
full-batch-quality code; n=5 LER CIs overlap across fractions). A budget sweep on
((5,3,3))₃ under `--noise full` (Stage H′) found a sharp gradient-corruption cliff
with robust knee **f5\* = 0.4** (5.4× fewer op-EVs than full-batch); below ~0.4 the
noisy sampled gradient corrupts training. The campaign default is the **interim**
**`sample_frac = 0.5`, `sampler = importance`**
(`src/seed_race.py::CAMPAIGN_SAMPLE_FRAC`, single source of truth) pending the
((9,3,3))₃ transfer knee f9\* (larger closure → likely f9\* ≤ 0.4 → default drops to
0.4). See [`docs/SAMPLING_BUDGET.md`](docs/SAMPLING_BUDGET.md).

## Default training parameters

- Optimizer **Adam**; init `θ ~ U[0, 2π)`, shape `(L, params_per_layer)`, `L`
  per code.
- Learning rate `0.05`, dropped to `0.01` once the loss falls below `0.1`
  (`train.py`); the seed-race campaign uses the three-stage schedule
  `0.05 → lr/5 (<0.5) → lr/50 (<1e-3)` carrying Adam moments across switches.
- ~`1500` training steps, typically `3` seeds; the **best-step** parameters are
  retained (not the last step).

`scripts/train.py` also defaults to `--noise full --entangler ms --connectivity
all-to-all` (the best-practice forward direction). The committed campaign params
were trained ring/dephasing — this only matters if you retrain, since figures
read the committed params directly.

## Decoders + LER

- **Lookup decoder** (Sec VI, `src/simulation.py`) — unweighted MLE over `{I,
  weight-1}` corrections + projection fallback; exact for a single weight-≤2
  error on a distance-3 code.
- **Weighted MAP decoder** (Sec IX, `decoders/weighted_map.py`) — extends the
  correction set to weight-≤2 with a channel prior; `argmax_C w_C Σ_k
  |⟨C·ψ_k|noisy⟩|²`.
- **Petz recovery** (`decoders/petz.py`) — the transpose-channel recovery the
  ℓ2 loss trains against (appendix).

LER is a Monte-Carlo estimate over `n_shots` (3k for the analyze-codes figures,
30k/100k for the benchmark CSVs); point estimates carry **Wilson 95% CIs**. The
simulation runs a **factored** wire-grouped path by default (equivalent to the
dense path, verified by tests) and falls back to dense for small systems.
Distance-2 codes use **detect-and-postselect** (`simulate_ler_with_detection`)
rather than correction.

## References

- Cao et al., [arXiv:2204.03560](https://arxiv.org/abs/2204.03560) — VarQEC algorithm
- Meth et al., [arXiv:2310.12110v3](https://arxiv.org/abs/2310.12110) — trapped-ion noise model (App. J)
- Ringbauer et al., Nat. Phys. 18, 1053 (2022) — Mølmer–Sørensen qudit gate (Eq. 2)
- Hrmo et al., Nat. Commun. 14, 2242 (2023) — light-shift qudit gate (Eq. 3)
- Chau 1997, Rains 1997 — [[5,1,3]]\_{Z\_q} benchmark code
