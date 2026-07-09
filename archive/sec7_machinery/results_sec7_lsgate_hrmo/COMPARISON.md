# Sec VII rerun — true Hrmo LS gate vs published level-restricted LS

Round-12 Commit 12. Four variants × three seeds × 1500 steps, Adam with
two-stage LR (0.05 → 0.01 at loss < 0.1). The genuine Hrmo et al. 2023
LS gate (`arXiv:2206.04104`, Eq. 3) was substituted for the
level-restricted `light_shift_gate` of `src/gates.py` (the gate the
original §VII trained against). Per-layer parameter count is
5n(d−1) + |ring_edges|, matching the §V.A campaign formula with the
LS-edge replacing the MS-edge count.

The level-restricted-LS baseline below is sourced from the
content-frozen §VII assets `figures/training_curves.png` and
`figures/varqec_vs_stabilizer.png`. The source CSVs for those plots
are not in `results/`; the audit (Issue F) notes the original caption
claim "below 10⁻⁶" is overstated — the curves "graze" 10⁻⁶ rather
than dropping below. We adopt the visually-inspected interpretation
"reaches ~10⁻⁶" below.

## Per-seed final-loss table

| variant | n | seed | final loss | best loss along traj. | last-50 step max | step → loss < 0.05 | step → loss < 10⁻⁶ |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_ls_ring  | 5 | 0 | 2.70 × 10⁻¹ | 2.69 × 10⁻¹ | 2.74 × 10⁻¹ | — | — |
| pure_ls_ring  | 5 | 1 | 4.06 × 10⁻¹ | 4.06 × 10⁻¹ | 4.11 × 10⁻¹ | — | — |
| pure_ls_ring  | 5 | 2 | 2.97 × 10⁻² | 2.97 × 10⁻² | 3.07 × 10⁻² | 764 | — |
| csum_star_ls  | 5 | 0 | 7.74 × 10⁻¹ | 7.72 × 10⁻¹ | 7.75 × 10⁻¹ | — | — |
| csum_star_ls  | 5 | 1 | 8.23 × 10⁻¹ | 8.20 × 10⁻¹ | 8.30 × 10⁻¹ | — | — |
| csum_star_ls  | 5 | 2 | 7.74 × 10⁻¹ | 7.73 × 10⁻¹ | 7.74 × 10⁻¹ | — | — |
| pure_ls_ring  | 7 | 0 | 7.55 × 10⁻⁷ | 7.55 × 10⁻⁷ | **7.06 × 10⁻⁵** | 298 | 540 |
| pure_ls_ring  | 7 | 1 | 2.20 × 10⁻⁶ | 2.18 × 10⁻⁶ | 2.75 × 10⁻⁶ | 179 | — |
| pure_ls_ring  | 7 | 2 | 1.64 × 10⁻⁵ | 1.70 × 10⁻⁶ | **1.32 × 10⁻⁴** | 245 | — |
| csum_star_ls  | 7 | 0 | 1.96 × 10⁻¹ | 1.96 × 10⁻¹ | 1.96 × 10⁻¹ | — | — |
| csum_star_ls  | 7 | 1 | 1.99 × 10⁻¹ | 1.99 × 10⁻¹ | 1.99 × 10⁻¹ | — | — |
| csum_star_ls  | 7 | 2 | 1.99 × 10⁻¹ | 1.99 × 10⁻¹ | 2.00 × 10⁻¹ | — | — |

**Last-50 step max** is the largest loss observed in the final 50
steps of the training trajectory — a coarse stability indicator. Where
it sits more than ~2× above the final loss, the optimizer was kicked
out of a near-minimum at least once during the final stage.

## Convergence success rate

| variant | system | n_seeds reaching loss < 0.05 | n_seeds reaching loss < 10⁻⁶ | training stable in last 50 steps? |
|---|---|:-:|:-:|:-:|
| pure_ls_ring | (5,1,3)_3 | 1 / 3 | 0 / 3 | yes (the converged seed) |
| csum_star_ls | (5,1,3)_3 | 0 / 3 | 0 / 3 | yes (but at 0.77 plateau) |
| pure_ls_ring | (7,1,3)_3 | 3 / 3 | 1 / 3 | **no** (seed 0 last-50 max = 7 × 10⁻⁵; seed 2 last-50 max = 1.3 × 10⁻⁴) |
| csum_star_ls | (7,1,3)_3 | 0 / 3 | 0 / 3 | yes (but at 0.20 plateau) |

## Head-to-head against the level-restricted-LS §VII baseline

Baseline values are the visually-inspected §VII figure traces
(`figures/training_curves.png` — bare-named Sec VII asset, content
frozen). See note above on the audit's "grazes 10⁻⁶" caveat.

| architecture | system | level-restricted LS (§VII fig) | genuine Hrmo LS (this rerun) | verdict |
|---|---|---|---|---|
| pure-LS-Ring  | (5,1,3)_3 | reaches ~10⁻⁶, three seeds | best-of-3 = 3 × 10⁻²; 1/3 seeds converge | **Hrmo materially WORSE** by ~4 orders of magnitude |
| CSUM-star + LS-Ring | (5,1,3)_3 | reaches ~10⁻⁶, three seeds | best-of-3 = 7.7 × 10⁻¹; 0/3 seeds converge | **Hrmo fails to converge** at this architecture |
| pure-LS-Ring  | (7,1,3)_3 | reaches ~10⁻⁶ (per §VII reported result; audit visual reading less certain at n=7) | best-of-3 = 7.6 × 10⁻⁷; 1/3 seeds below 10⁻⁶; unstable | at best **matches**, but training is unstable (spikes of 10×–100× in last 50 steps) |
| CSUM-star + LS-Ring | (7,1,3)_3 | not separately reported in §VII | best-of-3 = 2.0 × 10⁻¹; 0/3 seeds converge | **Hrmo fails to converge** at this architecture |

## Plain verdict

**The genuine Hrmo et al. 2023 LS gate does NOT uniformly outperform
the level-restricted ZZ approximation that the published §VII trained
against.** Specifically:

1. At ((5,1,3))_3 the Hrmo gate is materially worse than the
   level-restricted form on the pure-LS-Ring architecture (3 × 10⁻² vs
   ~10⁻⁶ — roughly four orders of magnitude). Only one of three random
   seeds converges to the 0.05 threshold; the other two plateau at
   0.27–0.41.

2. At ((5,1,3))_3 the CSUM-star + LS-Ring hybrid fails to converge
   under the Hrmo substitution (0/3 seeds reach 0.05; all plateau at
   0.77–0.82). The published level-restricted version of the same
   architecture converges to ~10⁻⁶ per the §VII figure.

3. At ((7,1,3))_3 the pure-LS-Ring matches the published
   level-restricted in best-of-3 final loss (~10⁻⁶), but the training
   is unstable: seed 0's last-50-step maximum is 7 × 10⁻⁵ (a 100× spike
   from the minimum it reaches); seed 2 spikes to 1.3 × 10⁻⁴. Only one
   of three seeds (seed 0) genuinely settles below 10⁻⁶, and even that
   seed is being kicked out repeatedly. The level-restricted §VII
   curve, by contrast, descends monotonically.

4. At ((7,1,3))_3 the CSUM-star + LS-Ring hybrid fails to converge
   (0/3 seeds reach 0.05; all plateau at ~0.20).

Net: Martin's "switch to true LS to reduce training-step count"
hypothesis is **not supported** by this rerun. The simulation evidence
points the other way — at the small-n end the level-restricted
approximation is structurally easier to optimize, and at the large-n
end the genuine LS at best matches the level-restricted while
introducing training instability that the level-restricted form does
not exhibit.

## Candidate explanations (hypotheses for Martin and Peter)

These are interpretive proposals for the experimentalists to weigh in
on, not conclusions from the simulation alone:

- **Hypothesis 1: optimizer-level-pair flexibility.** The level-restricted
  `light_shift_gate(theta, j, k, d)` is parameterised by (θ, j, k) on
  each instance; the level pair (j, k) is a *structural* degree of
  freedom that the architecture commits to (the §VII ansatz uses one
  fixed (j, k) per layer) but that the published code clearly trained
  successfully for. The Hrmo gate has no level-pair argument — it is
  a fixed-form unitary parameterised by θ alone. The optimizer
  therefore has fewer per-instance degrees of freedom for the
  entangling gate. At small n this matters; the Hilbert space is small
  enough that the level-pair commitment plus the per-pair angle is
  more flexible than a global all-pair-equal phase.

- **Hypothesis 2: sharpness of the Hrmo minimum.** The Hrmo gate's
  all-pair-equal action puts strong simultaneous constraints on every
  off-diagonal-product basis state. This may create a sharper (narrower
  basin of attraction) minimum that Adam reaches but cannot stay in —
  the spikes in seeds 0 and 2 at n = 7 are consistent with a minimum
  whose curvature exceeds the implicit Adam step length even after the
  LR-switch to 0.01. The level-restricted form, by acting only on a
  chosen 2-level subspace, may produce a wider basin that's easier to
  stabilize in.

- **Hypothesis 3: hardware vs simulation mismatch.** If hardware
  actually implements something between these two extremes (e.g. the
  Hrmo gate with controllable global-phase calibration that the
  optimizer cannot easily exploit in simulation), neither simulation
  result is the right predictor for hardware. **Open question for
  Martin and Peter: what does the Innsbruck "light-shift gate"
  actually implement, and does the optimizer-level-pair flexibility
  the level-restricted simulation exploits exist in any form on
  hardware?**

## Output files

- `pure_ls_ring_d3_n5_dist3_4L_best3s_seed2.npz` — best of seeds 0, 1, 2;
  all three seed loss arrays saved alongside.
- `csum_star_ls_d3_n5_dist3_4L_best3s_seed0.npz`
- `pure_ls_ring_d3_n7_dist3_4L_best3s_seed0.npz`
- `csum_star_ls_d3_n7_dist3_4L_best3s_seed0.npz`
- `run.log` — full training trace, per-step loss every 100 steps.
- `../../figures/sec7_hrmo_lsgate_training_curves.png` — two-panel
  (n = 5 / n = 7) plot, all three seeds per variant, best-of-3 traces
  emphasized.

The companion `figures/sec7_hrmo_lsgate_varqec_vs_stabilizer.png` is
deferred: it requires LER benchmarking of the four Hrmo-trained codes
(Haar-random codeword sampler + lookup-table decoder per §VI.A, 3000+
shots per p value), which is itself ~30 min × 4 variants of simulation
time. Given the §VII rerun's negative result on three of four
variants, the LER comparison is informative only on pure_ls_ring n = 7.
That single benchmark is queued for Session C.
