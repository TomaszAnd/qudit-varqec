# ULRICH_INTEGRATION.md — how Ulrich Holzer's reference file was integrated into VarQEC

READ-ONLY forensic verification, 2026-07-06. Source repo (untouched):
`/Users/tomas/PycharmProjects/PythonProject8/qudit`.

## Sources cross-checked

1. **GitHub**: `github.com/TomaszAnd/qudit-varqec`, branch `Qutrit_NativeGates`,
   file `Tensorform_Wire-grouped-Batching_Random_Seeding.py`. The raw URL fetched
   successfully. The fetch confirmed the SAME construct inventory as the local
   mirror (hardware_error_basis, build_error_set, group_by_wires,
   apply_local_matrix, precompute_error_products_dedup, XY_gate, Z_gate, MS_gate,
   CSUM_gate, entangling_MS_gate, entangling_LS_gate, encoder QNode,
   build_loss_func, get_params_per_layer, seed-race loop). No construct present on
   GitHub is absent from the local mirror. The local mirror `audit/ulrich_incoming.py`
   is therefore treated as the line-accurate reference below (642 lines).
2. **Local mirror**: `audit/ulrich_incoming.py` — read in full.
3. **Prior analysis**: `audit/06_integration.md` and `audit/04_ulrich_audit.md` —
   read and RE-VERIFIED against live source below (not merely copied).

Ulrich file line refs = `ui:<line>` (audit/ulrich_incoming.py, byte-identical to
the GitHub file's constructs). Repo refs = `path:line`.

---

## MERGED / REJECTED table

| Ulrich construct | ui: lines | Disposition | Landing site (repo) | Deviation from Ulrich |
|---|---|---|---|---|
| seed-race + mercy rule | 546–618 | **MERGED (rebuilt)** | `scripts/seed_race.py` (whole file) | Rebuilt on repo machinery; 4 defects fixed (H1/H2/H3/H5) + LR-reinit replaced (H8). See below. |
| per-group sampler | 460–495 | **REJECTED as module** | credit note only, `src/jax_backend.py:715–723` | Mechanically equivalent to existing `stratified_weights`; second copy would drift. |
| `group_by_wires` / `apply_local_matrix` (loss kernels) | 117–160 | **MERGED (generalized)** | `src/jax_backend.py` `_apply_local_matrix` + `create_jax_loss_vmap` (:143) | Ported then generalized qutrit-only → arbitrary d/n; module globals → explicit shape args. Header credit `jax_backend.py:9–14`. |
| `build_loss_func` (batched KL tensorform) | 433–505 | **MERGED (generalized)** | `src/jax_backend.py:143 / :414 / :571` | Same port; **K/3 → K/4** (see below). |
| `MS_gate` (θ,φ swapped) | 226–250 | **REJECTED** | repo `src/gates.py:130` (own impl) | His `MS_gate(theta, phi, …)` transposes the repo's `MS_gate(phi, theta, …)`; adopting it = silent-transposition hazard. Repo gate already = Ringbauer Eq. 2 to <1e-15. |
| `entangling_MS_gate` (= ZZ, not MS) | 289–307 | **REJECTED** | identical to repo `src/gates.py:205 zz_subspace_gate` (paper Eq. 4) | Not an MS gate; his own docstring says "using ZZ instead of XX and YY". |
| `entangling_LS_gate` (= global Hrmo) | 310–326 | **REJECTED** | identical to repo `src/gates.py:253 ls_global_gate` (Hrmo Eq. 3) | Numerically 0.0 vs repo gate. |
| `XY_gate` | 202–216 | **REJECTED** | repo `src/gates.py:16` | Repo has generalized, tested equivalent. |
| `Z_gate` | 219–223 | **REJECTED** | repo `src/gates.py:53` | Repo has generalized, tested equivalent. |
| `hardware_error_basis` (8 Gell-Mann) | 55–83 | **NOT MERGED into src/** | `scripts/experimental/ulrich_repro/repro_common.py:44` only | Different span than paper Sec. V C hardware basis (audit/04 H4); would break comparability. Repro-only, never imported by src/. |
| `build_error_set` | 86–110 | **REJECTED** | repo `src/errors.py` `ErrorModel` | Repo has closure-aware, factored, tested equivalent. Ulrich's pads identity (H5). |
| `CSUM_gate` (θ-ZZ conj. by subspace Hadamard) | 253–286 | **DEFERRED** | none in src/; faithful copy in repro `repro_common.py` | Only used in commented-out ansatz blocks (ui:399–405). If wanted, add as `csum_zz_gate`, must not shadow repo CSUM. |
| K/3 variance coefficient | ui:501 | **REJECTED (→ K/4)** | `src/jax_backend.py:210, 329, 372, 466, 565, 670`; `src/loss.py` | Repo uses K/4 (Cao Eq. 16 / paper Eq. 10). Note at `jax_backend.py:198–203`. |
| identity padding in E_det/E_corr | ui:91–92 | **REJECTED** | `ErrorModel` bases exclude it (`src/errors.py`) | Fixes 2.6% wasted ops (H5); documented `seed_race.py:24–26`. |
| LR-reinit at thresholds (0.5, 1e-3) | ui:579–587 | **MERGED (thresholds) / REJECTED (reinit)** | `scripts/seed_race.py:149–216` | Thresholds kept (0.05 / 0.01 / 0.001); Adam **moments carried** via `optax.inject_hyperparams` instead of re-init (H8). |
| "311-step" / "200-step" convergence claim | ui narrative | **UNCERTIFIED, not reproduced** | `audit/06_integration.md §3` | See status below. |

---

## Seed race — line-level deviations (all in `scripts/seed_race.py`)

- **H1 full-batch certification** (`:43–52`, `:184–200`): a sampled trigger
  (`lv < target_loss`) is only accepted after `evaluate_candidate` re-checks the
  full-batch loss. Ulrich accepts on the sampled loss alone (ui:593). Both values
  logged.
- **H2 empty mercy baseline** (`:137–170`): `record_steps = mercy_baseline`
  (default `None`); the kill branch (`:166`) fires only once a certified record
  exists. Ulrich hard-codes `global_fastest_steps = 400` (ui:541), which — per
  audit/06 §3(b) — is BELOW his own fastest certified 500 steps, so his shipped
  race would eliminate every certified solution.
- **H3 K/4** (`:93–97`): inherited from `create_jax_loss_vmap_weighted`; nothing
  to fix in the script.
- **H5 no identity padding** (`:24–26`, `:88–91`): `ErrorModel` builds E_det from
  hardware/full basis with no identity insert.
- **H8 no optimizer re-init** (`:149–216`): `optax.inject_hyperparams(optax.adam)`;
  LR mutated in place (`:212–216`) with an explicit eager-loop-only caveat
  (`:206–211`). Ulrich re-inits the optimizer at each switch (ui:580, 585),
  zeroing Adam moments and restarting bias correction.

---

## Provenance / commits (git-verified)

- `a4b3974` — add multi-start seed race with full-batch certification and mercy rule
  (creates `scripts/seed_race.py`).
- `6a207f4` — jit the seed-race value_and_grad path and guard eager lr mutation.
- `20d98ca` — seed race smoke + certification-gate tests.
- `7efe43f` — record Holzer sampler equivalence on `stratified_weights` docstring.
- `5198ddf` — add faithful Holzer repro harness (`scripts/experimental/ulrich_repro/`).
- `9cecd31` — verify repro encoder state-for-state against Holzer QNode.
- `d993c29` — gate regressions vs Ringbauer / Hrmo / paper Eq. 4.
- `e24c5b8` — rename `light_shift_gate`→`zz_subspace_gate`,
  `light_shift_gate_hrmo`→`ls_global_gate`.
- `af980e3` — document MS gate provenance vs Ringbauer Eq. 2.
- `6c9741e` — meta: add Ulrich Holzer as second author.
- The wire-grouped vmap loss port predates this integration (in `src/jax_backend.py`
  from the campaign release lineage; header credit at `:9–14`).

Note on **CSUB_gate** (`src/gates.py:187`, "Written by Ulrich"): this is MERGED
but comes from a *different* Ulrich file — it does NOT appear in
`Tensorform_Wire-grouped-Batching_Random_Seeding.py`, which instead defines the
parameterized `CSUM_gate` (deferred, above). No conflation.

---

## "311-step" convergence claim — current status

**UNCERTIFIED; not reproduced within budget.** Per audit/06 §3 (re-verified — the
harness `scripts/experimental/ulrich_repro/` and `results_a.json`/`results_b.json`
exist on disk):

- His numbers are first-passage times of a *sampled* (noisy) estimator, not
  certified convergence.
- Run (a), his own rule (sampled < 1e-6, 400-step mercy): **0 triggers in 1,290
  seeds** (1 h). 95% rule-of-three bound: P(trigger < 400 steps) ≤ 3/1290 ≈
  0.23%/seed → up to ~230 events over his 100k seeds remain possible; not refuted.
- Run (b), certified rule (full-batch < 1e-6): **6/200 seeds certified, minimum
  500 steps** (seed 38). 95% bound P(certified < 400) ≤ 3/200 = 1.5%/seed.
- On real trajectories **9 of 15 (60%) sampled triggers FAILED full-batch
  certification** — his first-trigger acceptance rule certifies false convergence
  more often than not (H1 quantified).
- His 400-step mercy cap sits below the 500-step fastest certified run (H2
  quantified) — as shipped, his race could only "win" via the false-trigger
  mechanism.

**Bottom line for the paper / conversation:** the "311"/"200" step counts are
uncertified and should not be cited as convergence results. The only certified
number is **500 steps (6/200 seeds within 2,000 steps)**. The bounds constrain but
do not exclude a rare genuine sub-400 run somewhere in his 100k-seed race. H1 and
H2 are proven code defects independent of that.

---

## Discrepancies vs Ulrich's source (summary)

1. **`MS_gate` argument order is transposed** between the two codebases: Ulrich
   `MS_gate(theta, phi, …)` (ui:226) vs repo `MS_gate(phi, theta, …)`
   (`src/gates.py:130`). Confirmed. This is why his gate was rejected rather than
   swapped in.
2. **`entangling_MS_gate` is NOT a Mølmer–Sørensen gate** — it is the diagonal
   subspace-ZZ (paper Eq. 4), confirmed identical to repo `zz_subspace_gate`.
   His name is misleading; repo renamed to avoid it (`e24c5b8`).
3. **`entangling_LS_gate` = global Hrmo light-shift** (Eq. 3), confirmed identical
   to repo `ls_global_gate`.
4. **K/3 vs K/4**: Ulrich's loss uses `(K/3)` (ui:501); repo uses `(K/4)` per Cao
   Eq. 16 / paper Eq. 10. Both vanish at the KL fixed point, but they are not the
   same objective off the fixed point. His K/3 is preserved only inside the
   repro harness.
5. **Identity padding**: Ulrich seeds E_det/E_corr with `(I3, (0,))` (ui:91–92);
   repo omits it (~2.6% wasted ops avoided).
