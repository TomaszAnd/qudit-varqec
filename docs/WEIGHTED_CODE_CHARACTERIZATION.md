# WEIGHTED_CODE_CHARACTERIZATION.md — are the weighted-IS codes genuine? (B3)

Defensibility check for the Sec IX / Meth-weighted story: are the codes trained
against the **Meth-channel-weighted** KL (importance-sampled, corrected Gaussian
channel) genuine **noise-adapted approximate-distance-3** codes, or artifacts of an
easy objective? Measured two ways on the on-disk weighted-IS winners
(`results/best_practice_runs/weighted_race_test/`), with the unweighted campaign
code as a reference:
1. **Shor–Laflamme weight enumerators** A_j, B_j over the hardware error basis
   (`scripts/reproduce_tables.py::enumerators`, wire-grouped local application — the
   n=9-scalable path; the dense form is 6.2 GB at n=9). Distance-3 ⟺ B_j = A_j for
   j < 3, so the **deficit B_j − A_j ≥ 0** measures the distance shortfall.
2. **Channel LER** under the corrected Meth Kraus channel with the weighted-MAP
   decoder (`weighted_race_test.map_channel_ler`, corrected channel + corrected prior).

| code | B₁−A₁ (weight-1 deficit) | B₂−A₂ (weight-2 deficit) | A₂ (scale) | channel LER (weighted MAP, 2k shots) |
|---|---|---|---|---|
| **weighted ((5,3,3))₃** | 2.1×10⁻³ | 9.7×10⁻² | ≈3.0 | **0.0** [0, 0] |
| unweighted ((5,3,3))₃ (ref) | 2.1×10⁻³ | 1.16×10⁻¹ | ≈6.5 | 1.5×10⁻³ [0, 3.5e-3] |
| **weighted ((9,3,3))₃** | 4.8×10⁻⁵ | **5.2×10⁻³** | ≈7.1 | 5.0×10⁻³ [2e-3, 8.5e-3] |

## Verdict: GENUINE noise-adapted approximate-distance-3 codes

- **Weight-1 errors are fully corrected.** B₁−A₁ ≈ 0 for all (2×10⁻³ at n=5,
  5×10⁻⁵ at n=9) — the codes satisfy the weight-1 Knill–Laflamme condition.
- **The weight-2 condition is nearly satisfied**, i.e. these are genuine
  *approximate*-distance-3 codes: the deficit is small relative to the enumerator
  scale A₂ (n=5: 9.7×10⁻²/3.0 ≈ 3%; **n=9: 5.2×10⁻³/7.1 ≈ 0.07%**, essentially
  distance-3). The weighted n=5 deficit (0.097) is even a touch *smaller* than the
  unweighted reference (0.116) on this basis — training on the channel-weighted KL
  did NOT sacrifice the uniform weight-2 structure.
- **Channel LER is low**: weighted n=5 = 0 (0 errors / 2000, better than the
  unweighted reference's 1.5×10⁻³), weighted n=9 = 5×10⁻³. The codes perform on the
  actual corrected Meth channel.

So the weighted-IS objective is **not** an easy-objective artifact: it produces real
noise-adapted approximate-distance-3 codes (small uniform weight-2 deficit AND low
channel LER). Why the two agree here: the Meth channel is dephasing-dominated, so its
high-probability errors (Z-class) are exactly the weight-1/weight-2 operators that set
the distance — satisfying the *weighted* KL for them largely satisfies the *uniform*
weight-2 condition too. This is the defensible foundation for the whole Sec IX /
weighted-sampling line: the codes it trains are genuine, and the efficient recipe is
**1 seed + weighted importance sampling** (Stage 2 NO-GO on racing; docs/SAMPLING_BUDGET.md).

Caveat: enumerators/LER are for the single Arm-B winner per code at 2k MAP shots
(indicative, not publication-grade CIs); the publication run should tighten shots and
average over seeds (see PUBLICATION_PLAN.md).
