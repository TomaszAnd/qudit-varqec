# VarQEC Provenance Map (Stage 1 audit + revised KEEP/DROP)

Resume anchor for the minimal-reproduction-package build. Destined for
`TARGET/docs/PROVENANCE.md`. SOURCE is read-only:
`/Users/tomas/PycharmProjects/PythonProject8/qudit` @ branch
`integrate/ulrich-seed-racing`, HEAD `a4d2e1e`.

Paper (source of truth): `qudit/latex/main.pdf` — *"Native-gate variational
error-correcting codes for trapped-ion qudits"* (28pp, 2026-06-12).
`papers/2204.03560v3.pdf` = Cao et al. reference paper, NOT this work.

TARGET: `/Users/tomas/PycharmProjects/qudit-varqec` (fresh `git init`, clean
history, commits as `TomaszAnd <tomasz.andrzejewski225@gmail.com>`).

## Approved decisions (post Stage-1 gate)
1. Fig 3–7: generators exist somewhere (latex/figures PNGs copied from another
   location). DO NOT declare a gap — resolve via Stage D forensic hunt first.
   `*.sh` launchers and `sec7_*` scripts move DROP→Stage-D-search-scope; do not
   drop until D resolves.
2. Untracked in-scope deps: copy into TARGET AND commit; flag provenance.
3. q=4,5 seed mismatch: verify both seed sets vs printed PDF numbers, ship the
   match, report the diff.

## TABLE A — Figure/Table → reproduction path
| Fig/Tab | Sec | Generator | Params | Data | Status |
|---|---|---|---|---|---|
| Fig 3,4,5,6,7 | VII–VIII | UNRESOLVED — no script found in tracked tree (Stage D hunt) | — | — | Stage D |
| Fig 8 | VIII | notebooks/analyze_codes.py:286 | d3 n5–n8 (tracked) + n9 UNTRACKED | results/simulations/ler_*_hardware.npz | untracked dep |
| Fig 9 | VIII | analyze_codes.py:350 | qutrit family + [[5,1,3]] | results/simulations/ler_* | untracked dep |
| Fig 10 | VIII D | analyze_codes.py:415 | + d4_n6, d5_n5 UNTRACKED | + untracked sims + benchmarks_100k_alltoall/*.csv | untracked deps |
| Fig 11 | VIII E | analyze_codes.py:482 | qutrit family (incl untracked n9) | results/simulations/* | untracked dep |
| Fig 12 | IX | scripts/r14_10_training_curves.py:113 | round14_scoping/r14_3*,r14_7* (UNTRACKED dirs) | npz _losses | untracked dirs |
| Fig 13 | IX | scripts/r14_10_sampler_equivalence.py:109 | — | 6 MAP-LER CSVs (tracked) | OK |
| Fig 14 | IX | scripts/r14_10_ler_vs_pphys.py:242 | — | MAP CSVs + r14_7g_eta_to_pphys.csv | OK |
| Fig 15 | IX E | scripts/r14_10_training_curves.py:150 | r14_3a + r14_7c_dim_scaling (UNTRACKED) | npz _losses | untracked dir |
| Table II | VIII | no committed emitter; hand-assembled from saved_params/all_to_all/{COMPARISON.md,run.log} + benchmarks_1M/100k_alltoall/*.csv | tracked seed2 | tracked CSVs | seed note |
| Table III | X B | analyze_codes.py → src/analysis.compute_weight_enumerators (live) | [[5,1,3]] + UNTRACKED best3s params | — | untracked deps |
| Table IV | App D | analytic (encoder.py::_ansatz_layer_manual gate counts); no emitter | — | — | needs tiny helper |

Out of scope: Fig 1, Fig 2 (TikZ/PNG schematics), Table I.

## TABLE B — KEEP/DROP (revised)
KEEP:
- Root scaffolding: README.md, LICENSE, requirements.txt, .gitignore
- src/ — ALL modules (analysis, catalog, correlated_noise, decoders/*, encoder,
  errors, gates, jax_backend, loss, sampling/*, simulation, legacy/*). No dead module.
- notebooks/analyze_codes.py (Fig 8–11 + Table III)
- scripts/: r14_10_* (Fig 12/13/14/15); CSV producers r14_5_sampler_ab,
  r14_6_r14_3b_ler, r14_7d_r14_3a_map_eta_sweep, r14_7e_fullbatch_ler,
  r14_8_ler_scaling, r14_7g_eta_to_pphys, r14_9_pphys_extend, r14_10_r14_7ab_ler;
  Sec-IX trainers train_r14_3_meth, train_r14_3b_meth_is, train_r14_3_fullbatch_meth,
  train_r14_7a_hoeffding, train_r14_7b_full_is, train_r14_7c_dimscaling;
  helpers loss_split_helpers, benchmark_ler_meth_pauli/kraus; seed_race.py;
  benchmark_ler.py, benchmark_ler_alltoall.py; train.py; run_campaign.py
- tests/: ~28 guarding kept modules
- .npz KEEP: d3 n5–n8 tracked; UNTRACKED d3_n9…seed1, d4_n6…seed0, d5_n5…seed0;
  saved_params/all_to_all/{d3_n9 seed1, d4_n6 seed2, d5_n5 seed2}; untracked
  round14_scoping/{r14_3a,r14_3b,r14_3_fullbatch,r14_7a,r14_7b,r14_7c}; untracked
  results/simulations/ler_{d3_n9,d4_n6,d5_n5}_dist3_hardware.npz; tracked MAP-LER
  CSVs + benchmarks_1M/100k_alltoall

STAGE-D SEARCH SCOPE (do not drop until resolved): *.sh campaign/overnight
launchers; sec7_lsgate_hrmo_plot.py, sec7_lsgate_hrmo_rerun.py, sec7_lsgate_hrmo_all.sh

DROP: audit/, docs/, archive/, papers/, latex/(oos), MERGE_LOG.md, *_handoff.md,
texput.log, .DS_Store, __pycache__, .pytest_cache; r14_4/r14_6_plot/r14_7f/r14_7i/
r14_8_extended/r14_9_* scoping; legacy trainers (train_native_*, train_qutrit_d2,
train_dephasing/depolarizing/correlated_*); scoping experiments (adaptive_sampling,
sampling_strategy, sampling_technique_bakeoff, warm_start_*, w1inv_w2ortho_sweep,
weight1_restricted, _salvage_s1, _timed_run, measure_*, meth_pauli_weights,
shot_count_analysis, audit_meth_channel); experimental/*; notebooks/showcase_varqec*;
d2/dist2/correlated/dephasing/depolarizing scoping .npz; saved_params/{all_to_all_6L,
8L,ring_6L}; _audit_projectors.npz

REVERTED GATHER PATH — DROP: src/jax_backend.py:601 create_jax_loss_gather +
build_varqec_loss_gather; guard tests/test_sampling_unbiased.py. Default fast=True
GEMM-batched vmap core at jax_backend.py:553–572 is the training path (KEEP).
Stratified/importance samplers in src/sampling/ KEEP. seed_race.py KEEP.

## Blockers
1. Fig 3–7: no generator in tracked tree → Stage D hunt (approved: generators
   exist elsewhere, PNGs were copied in).
2. Untracked in-scope deps → copy + commit + flag (approved).
3. q=4,5 seed mismatch: Table II tracked all_to_all seed2 vs Fig 10/Table III
   untracked ring seed0 → verify vs PDF, ship match (approved).

---

## Stage-F corrections (verified against the PDF during reproduction)

The Stage-1 audit above was refined by actually regenerating everything and
diffing against the printed values. Corrections now reflected in the package:

1. **Table II is the RING campaign, not all-to-all.** The printed Table II losses
   (0.0374/0.304/0.759) come from the ring params in `results/params` (seed1
   d3_n9, seed0 d4/d5). The all-to-all `saved_params` seed2 give different values
   (0.221/0.635) and were **dropped** (not on any in-scope path; the paper's
   all-to-all runs are the Sec IX training in `results/round14_scoping`).
2. **Table II / Fig 9 LER source = `results/benchmarks_30k`** (30k qutrit/[[5,1,3]],
   100k q=4,5) — reproduces the printed cells exactly (((9,3,3))₃ p=0.05 = 3.3×10⁻⁵,
   1 err; Fig 9 (5,3,3) = 3.67×10⁻⁴/11 err, [[5,1,3]] = 7.33×10⁻⁴/22 err). The
   all-to-all `benchmarks_*_alltoall` (different connectivity) were dropped.
3. **Fig 10 has 8 distance-3 codes** — the 8th, ((5,4,3))₄ = `ququart_n5_d3`
   (`d4_n5_dist3_6L_seed0.npz` + sim), was missing from the first copy and was added.
4. **`results/simulations` sims rebuilt at the paper's shot budget** (30k/100k)
   from the benchmark CSVs via `scripts/build_sims_from_benchmarks.py`, so Fig 8–11
   plot the paper's statistics (the original committed sims were 5k-shot).
5. **Table III has a runnable generator** (`scripts/reproduce_tables.py`) using the
   wire-grouped local-application enumerator (the dense `compute_weight_enumerators`
   is 6.2 GB at n=9 and does not finish); it also emits Table IV.
