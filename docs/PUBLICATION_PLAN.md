# PUBLICATION_PLAN.md — pressing experiments/plots left for the paper

Honest, prioritized outline of the runs still needed, informed by B1–B3
(SAMPLING_BUDGET.md, WEIGHTED_CODE_CHARACTERIZATION.md). **No training is launched by
this doc** — it is the plan for a later attended / GPU go. All runs must use the
**corrected Gaussian Meth noise** (the committed Fig 9/10/etc. artifacts are pre-fix,
at the 2× operating point) and write ONLY to `results/best_practice_runs/**`.

## Measured cost basis (this machine)
- **CPU-only Apple Silicon**, no CUDA. jax 0.4.35, complex128. `vmap`-batching gives
  **~0× speedup on CPU** (BLAS already saturates cores; measured, see SAMPLING_BUDGET
  P/Q/R) — so multi-seed / multi-strategy fan-out does NOT parallelize here; it is a GPU
  lever. float32 also gives ~0× (dominant cost is the complex128 encoder).
- Per-step `val_grad`: **n=5 ~32 ms**, **n=9 ~5 s** (≈150×). Compile ~9 s (n=5) / ~38 s
  (n=9), once per run. LER Monte-Carlo (factored / weighted-MAP): n=5 ~5 s / 2k shots,
  **n=9 ~30 s / 2k shots** (scales ~linearly in shots).
- 1500-step n=9 training ≈ 1.7–2.4 h/seed. n=5 training ≈ minutes.

## Tier 1 — cheap on CPU, high value (do first, attended)
1. **Resolve ((5,3,3))₃ vs [[5,1,3]]_{Z₃} at p=0.05.** The draft flags the ~2× advantage
   (3.67×10⁻⁴ vs 7.33×10⁻⁴) as UNRESOLVED at 30k shots (CIs overlap the factor-2). Re-run
   the hardware-noise LER for **both codes at ≥100k shots** (ideally 300k–1M) to separate
   the two with non-overlapping Wilson CIs. *Proves:* the headline coding-advantage claim.
   *Cost:* LER is cheap per shot; 1M shots × 2 codes (n=5) ≈ a few hours CPU. *GPU:* not
   needed. **Highest value/cost.**
2. **Weighted-code characterization at publication CIs.** B3 used 1 winner @ 2k MAP shots.
   Re-run weight enumerators (exact) + weighted-MAP channel LER at **≥20k shots**, averaged
   over the trained seeds, for n=5 and n=9. *Proves:* the Sec-IX codes are genuine (B3
   verdict) with tight CIs. *Cost:* n=9 weighted-MAP @ 20k ≈ 5 min/code; hours total. *GPU:*
   not needed.

## Tier 2 — moderate CPU / overnight
3. **Corrected-noise Fig 9 & Fig 10 re-run.** The committed `r14_10_sampler_equivalence`
   (Fig 9) and `r14_10_ler_vs_pphys` (Fig 10) are **pre-fix** (η=0.9296 = effective 2σ²).
   Re-run the MAP-LER η-sweep under the corrected channel and **remap the operating point**
   (corrected: p_phys 0.56→0.376 at η=0.9296; the paper's p_phys≈0.56 is now η=0.8642).
   *Proves:* the sampler-equivalence + LER-vs-p_phys results survive the noise fix.
   *Cost:* MAP-LER over an η grid × codes at ≥5k shots — several hours CPU overnight
   (`src/decoders/ler_driver.py`, adapted off the n=9 hardwire). *GPU:* not needed but helps.
4. **Complete Fig 8 (5 strategies, 3-seed envelopes), corrected noise.** The 4-config
   1-seed run is done (`fig8_weighted/`); B1 wired the 5th (`full_basis_is`, the cheapest
   strategy). Need: run all **5 strategies × 3 seeds** for median±band envelopes.
   *Proves:* importance / full-basis-IS reach the full-batch weighted-loss floor at
   ~5–17× lower op-EV cost (the corrected Fig 8). *Cost:* 5 × 3 × 1500 steps × ~5 s (n=9)
   ≈ **~31 h CPU** (or ~10 h at 1 seed). **GPU-recommended** (batches the 15 runs).

## Tier 3 — expensive, GPU strongly recommended
5. **Fig 6 cross-code campaign (UNWEIGHTED, generic distance-3, full-batch).** n5/n6/n7
   already trained under corrected noise (`campaign/`, losses 0.39/0.17/0.08); **need
   n8, n9, ((5,4,3))₄, ((6,4,3))₄, ((5,5,3))₅** (`scripts/campaign_train.py`, remaining
   codes; f_campaign=1.0 full-batch since the unweighted knee scales to full-batch at n=9,
   K-scaling result). *Proves:* the 8-code cross-code LER panel (Fig 6). *Cost:* n=9
   full-batch ≈ 2.4 h × 3 seeds; d=4 (dim 4ⁿ) and d=5 (dim 5⁵=3125) are larger/slower —
   the full set is **~15–25 h CPU**. **GPU strongly recommended.** Then regenerate the
   cross-code LER figure from the new codes.

## Notes / dependencies
- **GPU tier:** items 4 and 5 are where the CPU `vmap`-is-free ceiling bites — on a CUDA
  box the seed×strategy / code fan-out vectorizes into one program for a large speedup
  (the Q1 lever that is inert on this CPU). Everything in Tier 1–2 is CPU-feasible.
- **Racing:** none of these use racing — Stage 2 showed it is moot on the weighted
  objective (100% seed yield, low spread) and the unweighted knee scales to full-batch at
  n=9. The recipe is **1 seed + weighted importance sampling** (weighted line) or
  **full-batch** (unweighted large codes).
- **Invariants:** corrected Meth noise always; write only to `results/best_practice_runs/`;
  never overwrite `results/params|saved_params`; keep the pre-fix committed CSVs (flag as
  pre-fix) rather than deleting.
