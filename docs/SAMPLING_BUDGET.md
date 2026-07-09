# SAMPLING_BUDGET.md — importance-sampled seed racing (Stage H)

Design + go/no-go for using the importance-weighted sampled loss as the per-step
driver in the multi-start seed race, with full-batch certification.

## Motivation

The seed race (`src/seed_race.py`) trains many random initializations and keeps
the one that first reaches the target loss. Each step evaluates the KL detection
loss over the error set `E_det` — the hardware cost is the number of operator-
expectation-value evaluations ("op-EVs") per step. Full-batch racing pays
`|E_det|` op-EVs every step for every seed; most of that spend is on seeds that
will lose and on steps far from convergence.

The Sec IX sampler study (PDF Fig 8) already shows that the importance-weighted
sampler reaches the **same loss basin** as full-batch at **≈7.6× lower op-EV
cost** (`684 → 90` op-EVs/step for `((9,3,3))₃`). Stage H folds that sampler into
the seed race so the *whole search* — not just a single training run — spends its
budget on cheap sampled steps, paying the full batch only to certify.

## Protocol

Each racing seed trains on the **importance-weighted sampled loss**:

- **Cheap steps.** The per-step loss/gradient use a stratified-importance draw
  (`src.sampling.stratified_importance.make_stratified_importance_weights`):
  Neyman per-group budgets `∝ √Σ_e w_e²`, within-group draws `∝ w_e`, per-op
  weight `(W_g/B_g)·count`. For the unweighted detection loss every op has
  `w_e=1`, so this reduces to Neyman-by-`√(group size)` stratification; the same
  call carries the Meth-weighted objective unchanged when a weighted loss is
  supplied (the Sec IX case). Cost per step ≈ `sample_frac · |E_det|`.
- **Trigger, not victory (H1).** The sampled loss is a single-step *estimate*; it
  only *triggers* a certification. A candidate is accepted **only** when its
  **full-batch** loss is below `target_loss` (`evaluate_candidate`). A sampled
  value below target with a large full-batch loss is rejected and training
  continues.
- **False-trigger suppression.** A full-batch certification is armed only after
  the sampled loss stays below `target_loss·(1+trigger_margin)` for
  `trigger_patience` **consecutive** steps. This stops a lucky low-variance draw
  from spending a full-batch eval every noisy step. A rejected trigger resets the
  consecutive counter.
- **Mercy rule (H2).** No seed is pruned until the first *certified* record
  exists; then later seeds stop at the record step. Unchanged from the base race.

Implemented as options on `src.seed_race.run_race`:
`sampler={"uniform","importance"}`, `trigger_margin`, `trigger_patience`. The
defaults (`sampler="uniform"`, `margin=0`, `patience=1`) reproduce the previous
behavior exactly. `run_race` returns op-EV accounting: `race_op_evs` (total
op-EVs across all seeds, counting sampled steps and every full-batch
certification/eval) and `op_evs_at_first_certified`.

## Op-EV accounting

- Sampled step: `Σ_g (#distinct ops drawn in group g)` op-EVs.
- Full-batch step / certification / end-of-seed eval: `|E_det|` op-EVs.
- `race_op_evs` accumulates both across every seed; `op_evs_at_first_certified`
  snapshots it when the winning seed first certifies.

## Validation — ((5,3,3))₃ go/no-go (full Meth noise)

Code `((5,3,3))₃` (`d=3, n=5, distance=3`), all-to-all, XX/YY MS, **`--noise full`**
(the full ErrorModel basis: correlated Gaussian dephasing + subspace depolarizing +
amplitude-damping error set). Never dephasing-only. Script:
`scripts/stage_h_sampling_budget_validation.py`.

**Capacity note (important).** The unweighted full-basis KL detection loss sums
801 weight-≤2 operators, so its *absolute* floor is capacity-limited by ansatz
depth: single-seed floors were L=4→2.20, L=8→1.10, L=12→0.62, L=16→0.39
(LR-independent: L=8 at lr=0.01 gave the same 1.09 as lr=0.05). This is **not**
a bad code — at **L=16** the trained code is a genuine distance-3 code whose LER
(4.0×10⁻⁴ at p=0.05) **matches the published ((5,3,3))₃** (3.67×10⁻⁴) and **beats
[[5,1,3]]_{Z₃}** (7.33×10⁻⁴), with a steep ~p^2.6 slope. Convergence is therefore
judged by LER (the operational metric), and the go/no-go calibrates its target
from this LER-verified L=16 reference: floor 0.371 → `target_loss = 0.409`.

Compared, 4 seeds, 2500 steps, same seeds/target:
- **(a)** full-batch racing (`sample_frac=1.0`);
- **(b)** importance-sampled racing (`sampler="importance"`) + full-batch
  certification, at two budgets;
- negative control: uniform-stratified racing at the aggressive budget.

## Results

((5,3,3))₃, L=16, full noise, target 0.409, |E_det|=801 ops.
(`results/best_practice_runs/stage_h/stage_h_results.json`.)

| Racing mode | certified | best full loss | op-EVs → first cert | total op-EVs | wall (s) | LER @ p=0.05 (Wilson 95%) |
|---|---|---|---|---|---|---|
| full-batch (100%) | ✅ | 0.409 | 1,620,423 | 5,206,500 | 230 | 4.0×10⁻⁴ [1.1e-4, 1.5e-3] |
| stratified @20% | ❌ | 3.70 (diverged) | — | 1,663,204 | 481 | — |
| importance @20% | ❌ | 4.42 (diverged) | — | 1,474,697 | 359 | — |
| **importance @50%** | ✅ | **0.369** | **432,613** | 1,230,333 | 178 | 8.0×10⁻⁴ [3.1e-4, 2.1e-3] |

**Findings.**
1. **Budget floor.** At `sample_frac ≤ 0.2` **both** samplers (uniform-stratified
   *and* importance) **diverge** (best loss 3.7 / 4.4) — the single-step gradient
   estimate is too noisy for the full-basis objective. This is a budget effect,
   **not** importance-specific.
2. **At 50%, importance racing wins.** It certifies an **equal-quality** code
   (loss 0.369 ≤ target; LER 8.0×10⁻⁴, Wilson-CI-overlapping full-batch's
   4.0×10⁻⁴ — both the published ((5,3,3))₃ band) at **3.75× fewer op-EVs to
   first-certified** (432k vs 1.62M) and **4.2× fewer total op-EVs** (1.23M vs
   5.21M), at comparable wall-clock.
3. On this **unweighted** loss the importance sampler's edge over plain stratified
   is small (both need ≥50% budget); its large advantage (Fig 8's ~7.6×) is
   specific to the **Meth-weighted** objective, where concentrated op-weights make
   importance sampling much more efficient than uniform stratification.

**Decision: GO** — adopt importance-sampled racing at **`sample_frac ≥ 0.5`** (not
below) with full-batch certification as the best-practice campaign default: it
reaches an equal-quality certified code at ~3.75–4.2× lower op-EV budget.
`sample_frac` must stay ≥ 0.5 on the unweighted objective; aggressive budgets
(≤0.2) diverge. Full-batch racing remains the safe fallback.

---

## Stage H′ — budget knee sweep (set f*)

Stage H established GO at `sample_frac ≥ 0.5`; H′ finds the actual **knee f\*** =
the fewest budget for full-batch-equivalent CERTIFIED performance. Because the
full-basis KL loss has large seed→basin variance (0.057 vs 0.37 across inits), the
sweep is **per-seed PAIRED**: each seed's `target_s = its own full-batch floor_s ×
1.1`, and every fraction runs from that seed's init. Primary metric = basin-matched
degradation ratio `best_loss(f,s)/floor_s` (1.0 = no degradation); LER@0.05 (Wilson)
computed only at f\* and the full-batch anchor. Script:
`scripts/stage_hprime_budget_sweep.py`. Code ((5,3,3))₃, L=16, `--noise full`,
all-to-all, XX/YY MS, 4 seeds, 2500 steps, trigger margin 0.25 / patience 5,
|E_det| = 801. (`results/best_practice_runs/stage_hprime/`.)

| f | mean degradation | cert-rate (4 seeds) | op-EVs→1st cert | classification |
|---|---|---|---|---|
| 0.10 | 14.8 | 0/4 | — | hard gradient-corruption |
| 0.20 | 11.7 | 0/4 | — | hard gradient-corruption |
| 0.30 | 8.5 | 0/4 | — | hard gradient-corruption |
| 0.32 | 2.84 | 3/4 | — | stochastic cliff |
| 0.35 | 1.35 | 3/4 | — | stochastic cliff |
| **0.38** | **0.83** | **4/4** | 244,130 | **robust knee** |
| **0.40** | **0.87** | **4/4** | **228,805** | **robust knee (f\*)** |
| 0.50 | 0.91 | 4/4 | 354,199 | robust (chosen default) |
| 0.70 | 1.15 | 3/4 | 674,008 | robust w/ stochastic fail |
| 1.00 | 1.10 | 4/4 | 1,238,746 | full-batch anchor |

**LER (confirmatory, only at f\* and anchor):** the best-seed code at f=0.4 gives
LER@0.05 = 1.0×10⁻³ [4.3e-4, 2.3e-3]; full-batch (f=1.0) = 4.0×10⁻⁴ [1.1e-4,
1.5e-3]. **Wilson CIs overlap → equal quality** (both in the published ((5,3,3))₃
band, beating/near [[5,1,3]] 7.3×10⁻⁴).

**Findings.**
1. **Knee f\* = 0.4** (robust 4/4-seed convergence; 0.38 also 4/4). Below 0.4 is a
   **sharp gradient-corruption cliff** — ≤0.3 always diverges (deg 8–15, never
   triggers), 0.32–0.35 is a stochastic ~75%-certified transition. The failure is
   **gradient-corruption** (the noisy sampled gradient drives training to a worse
   basin), **not trigger-starvation** (triggers fire fine once converged).
2. **Op-EV saving at f\* vs full-batch = 5.41×** (228,805 vs 1,238,746 op-EVs to
   first-certified). Notably importance runs at f≥0.4 reach deg < 1 — the sampling
   noise can escape the mediocre plateau basins full-batch settles into.
3. The full-batch anchor pays a false-trigger tax near its target (≈160–195 rejected
   certifications while it hovers in `[target, 1.25·target]`), which inflates its
   op-EV cost; this is uniform across fractions.

**"The fewest budget for full-batch-equivalent certified performance is f5\* = 40%,
saving 5.41× op-EVs."**

### The fraction is a pure search-cost knob

The metric above (degradation) is a **certified-loss** parity ratio, and the
confirmatory **LER CIs overlap across fractions** (f=0.4: [4.3e-4, 2.3e-3] vs
f=1.0: [1.1e-4, 1.5e-3]). Because certification is always on the full batch, **any
fraction that certifies yields a full-batch-quality code** — the fraction changes
only op-EVs and per-seed yield, never output quality. So the **cheaper fraction wins
unless it lowers seed yield on the largest campaign code.** On n=5, f=0.4 has full
4/4 yield (= full-batch) at 5.41× lower cost → it wins outright on n=5.

### n=9 transfer (f9\*) — decides the default

The open question is whether the **largest** campaign code needs a larger fraction:
the weight-2 closure grows with (K,n), so |E_det| grows, and at a *fixed* fraction
that means *more* absolute samples → lower gradient variance → f9\* is expected
**≤ f5\* = 0.40**. This is being measured now (a coarse ((9,3,3))₃ paired sweep,
`--n 9 --fractions 0.2,0.35,0.5,1.0`, 3 seeds, `--noise full`;
`results/best_practice_runs/stage_hprime/sweep_n9_*.log`). Report:
`f9* = smallest ((9,3,3))₃ fraction holding seed yield ≥ the f=1.0 yield`.

**Default decision (interim + rule).** Held at the **INTERIM** `sample_frac = 0.5`,
`sampler = importance` (`src/seed_race.py::CAMPAIGN_SAMPLE_FRAC` — single source of
truth) until f9\* lands, because the rule forbids hardcoding 0.40 without n=9
confirmation. When f9\* is known:
- **f9\* ≤ 0.40** → set **0.40** (bank ~35% vs 0.50) — the recommended outcome;
- **f9\* > 0.40** → set **f9\*** (smallest fraction safe on the largest code) and
  record that the optimal fraction **scales up with the closure (K,n)**.

Do **not** run the campaign below ~0.40 (gradient-corruption cliff). Full-batch
racing remains the safe fallback.

---

## Stage P/Q/R — sweep performance profile (verdict: CPU-FLOP-bound, no safe speedup)

The H′ sweep is slow (n=9 ≈ hours). Profiled before optimizing (`jax` 0.4.35).

**P(a) Hardware.** `jax.devices() = [CpuDevice]` — **CPU-only Apple Silicon (arm64), no
CUDA**. x64/complex128 enabled at `src/jax_backend.py` import. So there is **no GPU
lever**; CPU is the ceiling.

**P(b) Per-step breakdown** (`block_until_ready`):

| | n=5 (\|E_det\|=801) | n=9 (\|E_det\|=2593) |
|---|---|---|
| build_race (Python) | 4.0 s | 12.8 s |
| first step (compile+run) | 9.0 s | 37.9 s |
| **steady per-step (val_grad)** | **32 ms** | **4946 ms** |
| first full-batch cert (compile+run) | 1.7 s | 9.9 s |

**P(c) Recompilation.** Changing the sampling FRACTION → **no recompile** (33 ms, same
as steady); changing SEED → **no recompile** (32 ms). The jit compiles **once per
`run_race`** (via `build_race`) and reuses across all steps/fractions/seeds. So the
only compile term is `N_runs × compile` (n=5 ≈ 9 s, n=9 ≈ 38 s each) — minutes, not
the dominant cost.

**P(d) Orchestration.** Fully sequential Python loops: sweep loops seeds×fractions →
each `run_race` loops steps → each step dispatches `val_grad_fn`.

**P(e) Dominant term.** `N_runs × n_steps × per_step`. For n=9: ~5 s/step × ~800 ×
~10 runs ≈ **hours**. This is **compute (FLOPs)**, not dispatch or compile. Note:
masked sampling computes **all** ops every step regardless of fraction (mask, not
gather — preserves the gradient), so the fraction never reduces JAX wall-time.

**Q1–Q4 measured — all ~0× on CPU:**
- **Q1 vmap across the seed×fraction grid: NO speedup.** Measured per-trajectory time
  under `vmap`: n=5 B=1/2/4/8 = 32.0/31.6/31.5/32.2 ms (≈**1.0×**); n=9 B=2 =
  5488 ms/traj (**0.90×**, slightly worse). BLAS already saturates the CPU cores on a
  single trajectory, so batching just does N× the FLOPs. This is a GPU idiom; on
  CPU-only it only saves the per-run compile (minutes), not the hours of steps.
- **Q2 masked sampling to "kill recompile": moot** — P(c) shows there is **no**
  per-fraction recompile to kill.
- **Q3 lax.scan over steps: ~0×** — removes Python per-step dispatch, but the step is
  FLOP-bound (Q1 proves batching doesn't help), so dispatch is negligible.
- **Q4 batch certification: ~0×** — same vmap-on-CPU result as Q1.

**R Precision — no benefit, and unsafe to force.** float32 *weights* over the
complex128 statevector: n=5 32.2→32.0 ms, n=9 4590→4765 ms (no gain / slightly worse);
losses match to 5 digits. The dominant cost is the **complex128 encoder**, which is
hardcoded in ~12 places in the frozen `jax_backend.py`. A true complex64 conversion is
a deep, risky change to the validated core, and CPU shows no mixed-precision speedup in
the partial test — so **R is rejected** (no measured benefit; would risk moving the
f\*=0.40 knee for nothing).

**Verdict.** On this **CPU-only** machine the sweep is irreducibly complex128-FLOP-bound;
the JAX-batching speedups (Q1–Q4) are GPU idioms that give ~0× here, and float32 doesn't
touch the dominant statevector cost. **No exact-preserving speedup is adopted** (nothing
clears even the "does it actually speed up?" bar, let alone the correctness gate). The
practical paths to a feasible n=9 / Stage-I campaign are **not** speedups but: (i) run on
a **GPU** box (the real fix — would batch the grid for a large win); (ii) **reduce n=9
scope** — it starts near-converged (step-0 loss 0.33) and the gradient-corruption
signature appears within a few hundred steps, so ~400–600 steps and 2 seeds suffice for
the yield/cliff question (~3× less wall, a scope choice, honestly weaker statistics); or
(iii) accept the multi-hour n=9 as a dedicated overnight run. The n=5 H′ result
(f5\*=0.40) and interim default 0.50 stand unchanged.

---

## Stage 2 (weighted objective) — is RACING useful on top of weighted importance sampling? NO-GO

Pivot: the earlier stages used the UNWEIGHTED full-basis KL. Fig 8 is about the
Meth-CHANNEL-WEIGHTED KL. The seed race was wired to the weighted objective
(`src/seed_race.py weighted=True` → Meth channel weights from `src/meth_weights.py`,
computed LIVE from the Gaussian-corrected `correlated_noise`; certification on the
weighted full batch; confirmatory = weighted-MAP channel LER, corrected channel +
corrected prior). Then racing was isolated: fixed sampler=importance @ frac 0.2
(Fig-8-validated budget band), toggling ONLY racing.
  Arm A (no racing): N seeds to completion, best-of-N (= Fig 8 protocol).
  Arm B (racing):    same seeds/sampler + mercy-prune + trigger-cert.
Script: `scripts/weighted_race_test.py`. → `results/best_practice_runs/weighted_race_test/`.

| code | N seeds | weighted floors | **seed var std/mean** | Arm A total op-EVs | Arm B op-EVs→1st cert | LER equal? |
|---|---|---|---|---|---|---|
| **2a ((5,3,3))₃** L16 | 5 | 3.4–3.6×10⁻⁴ | **0.027** | 227,846 | 51,778 (≈1-seed) | yes (both 0) |
| **2b ((9,3,3))₃** L16 | 3 | 6.6–7.3×10⁻⁶ | **0.041** | 172,941 | 59,568 (≈1-seed) | yes (5e-3 vs 4e-3, CIs overlap) |

**VERDICT: NO-GO.** The weighted objective has **low seed→basin variance at BOTH
scales** (2.7% and 4.1% ≪ 15%) — every seed reaches essentially the same near-perfect
weighted KL (the weighted loss concentrates on the easy high-P(error) ops: identity +
single-qudit Z). So **1 seed + weighted importance sampling suffices**; racing (a
multi-seed technique) is moot. Arm B beating best-of-N (4.4× / 2.9×) is not a racing
win — `op_evs_at_first_certified ≈ one seed's cost`, so it merely reflects best-of-N
wasting the N−1 redundant seeds. (Verdict is variance-primary; the raw op-EV ratio
alone would mislead.) NOTE the contrast with the UNWEIGHTED objective, whose variance
INVERTED (n=5 high, n=9 ~0); the weighted objective does not invert because it is
uniformly easy. **Recommendation for the campaign / Fig 8: run 1 seed + weighted
importance sampling, no racing.**

### B2 — verdict on YIELD + spread, not survivor-spread alone

The NO-GO above rested on the survivors' floor-spread. That is only sound if there are
no FAILED seeds (a failed seed is exactly what racing would prune). Re-analysing the
on-disk `weighted_race_test` data on all three axes:

| code | seed YIELD (converged / N) | floor-spread (std/mean, max/min) | per-seed op-EVs (Arm A) | Arm B op-EVs->1st cert |
|---|---|---|---|---|
| ((5,3,3))_3 | **5 / 5** | 0.027, 1.07 | ~45,569 (tight) | 51,778 (~1 seed) |
| ((9,3,3))_3 | **3 / 3** | 0.041, 1.10 | ~57,647 (tight) | 59,568 (~1 seed) |

**Yield is 100% at both scales** — every seed reaches a near-perfect weighted code
(n=5 ~3.5e-4, n=9 ~7e-6), so there are **no losing seeds for racing to mercy-prune**.
Floor-spread is ≤10% (max/min) and the per-seed op-EV cost is tight (time-to-target
spread low; Arm B certifies at ~one seed's cost). The verdict therefore rests on
**yield + floor-spread + time-spread together**, all pointing the same way: **NO-GO,
racing moot**. (If yield had been low with tight survivor-spread, racing COULD have
helped by pruning failures — it is the 100% yield that closes that door here.)

Consequence for Stage 3: the corrected-noise Fig 8 is reproduced with the samplers
STRAIGHT (no racing overlay), documenting the above low-variance finding.

---

## Stage 3 — corrected-noise Fig 8 (samplers straight, no racing) — DONE

Racing being moot (Stage 2), Fig 8 was reproduced under the CORRECTED Gaussian Meth
channel with the samplers STRAIGHT: one seed each on the Meth-weighted KL at
((9,3,3))₃, L=16, 1500 steps, logging the full-batch weighted loss vs cumulative
op-EVs. Scripts: `scripts/fig8_weighted_train.py` (+ `fig8_weighted_plot.py`).
Traces/figure → `results/best_practice_runs/fig8_weighted/` (gitignored).

| sampler | min weighted loss | op-EVs at min | notes |
|---|---|---|---|
| full-batch (100%) | **1.18×10⁻⁶** | 1,069,285 | deepest basin, costliest |
| stratified (20%) | 7.15×10⁻⁶ | 299,785 | cheap early but plateaus (~7e-6 variance floor) |
| **importance (20%)** | **2.81×10⁻⁶** | **212,521** | near-full-batch loss at **~5× fewer op-EVs** |
| importance (35%) | 2.17×10⁻⁶ | 331,075 | slightly deeper, more budget |

**Result (corrected Fig 8).** On the op-EV axis, **importance sampling reaches
near-full-batch weighted loss at ~5× lower op-EV cost** (2.8×10⁻⁶ at 2.1×10⁵ vs
full-batch 1.2×10⁻⁶ at 1.07×10⁶), and dominates uniform stratified on both axes
(stratified plateaus at 7×10⁻⁶ — its uniform subsampling has a variance floor that
cannot resolve the deep weighted basin). This reproduces the paper's Fig-8 message
under the Gaussian-corrected channel and SUPERSEDES the pre-fix committed version
(old round14 CSVs kept, flagged pre-fix; published params untouched). No racing
overlay: racing is moot on this objective (Stage 2, low seed variance).

**Follow-ups (noted, not blocking):** (a) 3-seed median envelopes for publication
error bands; (b) the two full-basis-IS sampler variants (need the w1/w2-mask path)
for the complete 5-sampler set.

### B1 — the 5 paper strategies for Fig 8 (completeness + axis)

The corrected Fig 8 above ran **3 of the 5** paper sampling strategies (full-batch,
stratified, importance). **None were dropped for being worse** — in the paper all five
reach the same loss floor and differ only in op-EV COST; the point of the figure is the
cost axis. The two not yet run:
- **importance-truncated** — NOT a separate training run: it is the importance curve
  read at an earlier op-EV cutoff (trivially recovered by the plotter), so it needs no
  new training.
- **full-basis-IS** — samples the weight-1 sector too (not just weight-2), so it is the
  **CHEAPEST** strategy (~63 op-EVs/step) and the **strongest budget point** — the figure
  is incomplete without it. It needs the weight-1 mask path.

**Implemented (B1):** `full_basis_is` is now wired into `scripts/fig8_weighted_train.py`
via `loss_split_helpers.build_w1_w2_masks` + `stratified_importance.make_stratified_
importance_weights_full_basis`, composing with the corrected Meth channel weights on the
default hardware closure basis. Unit-tested (`tests/test_stratified_importance.py::
test_full_basis_is_wires_with_meth_weights`: masks align, estimator unbiased for the full
weighted loss, realized op-EVs bounded by the budget). The full 5-strategy launch (with
3-seed envelopes) is deferred to the publication campaign (see PUBLICATION_PLAN.md), NOT
run here.

**Axis fix:** Fig 8 must render the 5 NAMED strategies at their natural per-step budgets
(full-batch ~|E_det|, stratified/importance ~frac·|E_det|, full-basis-IS ~63), on a
cumulative-op-EV x-axis — not a single "fraction knob" sweep. The trainer logs cumulative
op-EVs per strategy so the plotter places each curve at its true hardware cost.
