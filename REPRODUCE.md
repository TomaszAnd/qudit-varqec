# REPRODUCE.md

One row per in-scope paper data figure/table → exact command → params/data
consumed → expected value (from the PDF) → regenerated value. **No retraining:**
every figure/table is regenerated from committed trained parameters.

Setup once:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

All in-scope figures/tables reproduce the printed values (loss, LER, enumerators,
fit exponent, MAP slope, gate counts) to the quoted digits. The one exception is
Sec VII (Fig 3–7), whose generator was lost — see the bottom row.

---

## Figures

| Fig | Command | Consumes | Expected (PDF) | Regenerated | ✔ |
|-----|---------|----------|----------------|-------------|---|
| **8+9** ((n,3,3))₃ 2-panel (loss + LER vs [[5,1,3]]) | `cd notebooks && python3 analyze_codes.py` → `n_scaling_d3_dist3.png` | `results/params` d3 n5–9 (via `src.catalog`) + `results/simulations/ler_*_hardware.npz` (30k), incl. `five_qudit_d3` | LEFT loss ∝ **n^−2.54** (losses n5..9 0.136/0.129/0.069/0.041/0.037); RIGHT p=0.05: (5,3,3)₃ **3.67×10⁻⁴**, [[5,1,3]] **7.33×10⁻⁴** (~2×) | fit **n^−2.52**, losses **0.1363/0.1290/0.0687/0.0412/0.0374**; (5,3,3) **3.667×10⁻⁴**, [[5,1,3]] **7.333×10⁻⁴** | ✅ |
| **10** cross-q distance-3 (8 codes) | same run → `ler_vs_p_all_dist3.png` | qutrit family + d4_n6, d5_n5, **ququart_n5_d3** (`results/params`) + 100k sims | all q=4,5 below 100k floor (≈4×10⁻⁵) at p=0.05; [[5,1,3]]@0.2 = 0.0418 | q4/q5 below floor; [[5,1,3]]@0.2 **0.0418** | ✅ |
| **12** loss/op-EV, 5 samplers | `python3 scripts/r14_10_training_curves.py` | `results/round14_scoping/r14_3{a,b},r14_3_fullbatch,r14_7{a,b}` traces | 5 samplers converge to ~2.4×10⁻⁵ | 2.39/2.54/1.97/2.63/2.63 ×10⁻⁵ | ✅ |
| **13** sampler-equivalence LER | `python3 scripts/r14_10_sampler_equivalence.py` | MAP-LER CSVs in `results/round14_scoping` | samplers agree in median LER per η | η=0.95 median 0.075/0.065/0.072 | ✅ |
| **14** LER vs p_phys (MAP) | `python3 scripts/r14_10_ler_vs_pphys.py` | MAP-LER CSVs + `r14_7g_eta_to_pphys.csv` | log–log slope **7.4 ± 0.3** on p∈[0.36,0.50] | **7.35 ± 0.29** | ✅ |
| **3,4,5,6,7** (Sec VII) | — | — | — | **NOT regenerable** — generator lost; pixels in `archive/sec7_figures/`. See `docs/PROVENANCE.md`, `docs/GATE_AUDIT.md` | ✗ |

> **Figure-set changes (this follow-up):** PDF Fig 4 (n-scaling loss) and PDF Fig 5
> (VarQEC vs [[5,1,3]]) — REPRODUCE rows **8/9** — are merged into the single 2-panel
> `n_scaling_d3_dist3.png` (left loss, right LER with the [[5,1,3]] + no-coding overlay);
> the standalone `varqec_vs_stabilizer.png` is no longer generated. PDF Fig 7 (fidelity
> vs n, `fidelity_vs_n_d3_dist3.png`, former row 11) and PDF Fig 11 (dim-scaling d3 vs d4,
> `r14_10_dim_scaling_curves.png`, former row 15) are **dropped** (generator blocks removed).
> Row numbers 8–15 keep the historical +4 offset vs the PDF figure numbers.

## Tables

| Table | Command | Consumes | Expected (PDF) | Regenerated | ✔ |
|-------|---------|----------|----------------|-------------|---|
| **II** loss + MS | `cd notebooks && python3 analyze_codes.py` (summary) + `scripts/reproduce_tables.py` (MS) | `results/params` ring codes (seed1 d3_n9, **seed0** d4_n6/d5_n5) | Loss 0.0374/0.304/0.759; MS 72/72/80 | Loss **0.0374/0.304/0.759** (analyze_codes); MS **72/72/80** (Table IV = L·MS/l) | ✅ |
| **II** LER cells | `results/benchmarks_30k/*.csv` (raw) — same data feeds Fig 8–10 | committed 30k/100k ring benchmark CSVs | ((9,3,3))₃ p=0.05 **3.3×10⁻⁵** [5.9e-6,1.9e-4] (1 err); ((6,4,3))₄ p=0.20 **2×10⁻⁵** (2 err); ((5,5,3))₅ below floor | CSVs match to the digit + error count | ✅ |
| **III** weight enumerators | `python3 scripts/reproduce_tables.py` | `results/params` (via `src.catalog`) + hardware error basis | A/B at j≤2 (table below) | matches to 3 d.p. | ✅ |
| **IV** encoder resources | `python3 scripts/reproduce_tables.py` | analytic formulas | Pℓ 126/126/140; L·Pℓ 504/504/560; XY 36/36/40; MS 18/18/20; Z 18/18/20 | exact | ✅ |

### Table III expected (PDF) vs regenerated

| Code | A₀ | A₁ | A₂ | B₀ | B₁ | B₂ | (regen A₁/A₂, B₁/B₂) |
|------|----|----|----|----|----|----|----|
| [[5,1,3]]_Z3 | 1.000 | 2.222 | 1.975 | 1.000 | 2.222 | 1.975 | 2.222/1.975, 2.222/1.975 ✅ |
| ((9,3,3))₃ | 1.000 | 4.052 | 7.317 | 1.000 | 4.053 | 7.343 | 4.052/7.317, 4.052/7.343 ✅ |
| ((6,4,3))₄ | 1.000 | 9.111 | 34.665 | 1.000 | 9.113 | 34.818 | 9.111/34.665, 9.113/34.818 ✅ |
| ((5,5,3))₅ | 1.000 | 14.634 | 85.624 | 1.000 | 14.638 | 85.918 | 14.634/85.624, 14.638/85.918 ✅ |

`reproduce_tables.py` uses the **hardware** single-qudit error basis
(`qudit_hardware_error_basis`) — the basis that reproduces the [[5,1,3]] row —
and the wire-grouped local-application method (no dense d^n projector), so all
four rows compute in seconds even at n=9.

## Out of scope

Fig 1, Fig 2 (TikZ/PNG circuit schematics), Table I (gate-set table) — LaTeX,
not code.

## Provenance notes

- **Seeds (q=4,5).** Table II and Fig 8–10 use the **ring** campaign params in
  `results/params` (seed0 for d4/d5, seed1 for d3_n9). These reproduce the
  printed Table II losses (0.304/0.759). The all-to-all seed2 params give
  *different* values (0.221/0.635) and are **not** part of this package — the
  paper's all-to-all runs are the Sec IX correlated-channel training (carried as
  the `results/round14_scoping` traces, Fig 12), not Table II.
- **LER shot budget.** `results/simulations/ler_*_hardware.npz` are rebuilt by
  `scripts/build_sims_from_benchmarks.py` from the committed 30k/100k benchmark
  CSVs (`results/benchmarks_30k`), so Fig 8–10 plot the paper's shot budget and
  reproduce the printed LER values exactly. LER point estimates carry Wilson 95%
  CIs; re-running the Monte-Carlo benchmark from scratch would land within those
  intervals but not on the identical RNG sample.
- **Gate labels (Sec VII).** The Sec VII "MS Ring"/"LS Ring" architecture labels
  are mislabeled (the "MS" ablation applies subspace-ZZ, not a physical MS). This
  is documentation-only and does not affect the campaign codes (Sec VIII/IX),
  which use the verified physical MS. See `docs/GATE_AUDIT.md`.
