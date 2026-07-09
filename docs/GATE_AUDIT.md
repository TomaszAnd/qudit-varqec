# GATE_AUDIT — MS / LS gate forensic verification (VarQEC, qutrit native gates)

READ-ONLY audit. Date 2026-07-06. No source files modified.
Numerics: `/Users/tomas/qudit_env/bin/python3`, float64, Frobenius norms over a
(θ,φ) grid × level-pairs {(0,1),(1,2),(0,2)} × d ∈ {2,3,5}.

Sources cross-checked:
- Repo: `/Users/tomas/PycharmProjects/PythonProject8/qudit/src/gates.py`
- Ulrich GitHub `Qutrit_NativeGates/Tensorform_Wire-grouped-Batching_Random_Seeding.py`
  (fetched live — byte-identical to local mirror `audit/ulrich_incoming.py`)
- Paper `latex/main.tex` (Sec III gate defs Eq.4 U_LS; Sec VII ablation labels)
- Ringbauer Eq.2 (arXiv:2109.06903) MS = expm(−i(θ/4)(σ_φ⊗I+I⊗σ_φ)²), σ_φ=cosφ X+sinφ Y
- Hrmo (Nat.Commun.14:2242, arXiv:2206.04104) global LS: |jj>→|jj>, |jk>→e^{iθ}|jk> (j≠k)

## Four verdicts (numerical)

| # | Claim | max Fro Δ | Verdict |
|---|---|---|---|
| 1 | repo `MS_gate` == Ringbauer Eq.2 physical MS | **1.97e-15** | **CONFIRMED — equal** |
| 2 | Ulrich `entangling_MS_gate` == paper Eq.4 ZZ (`zz_subspace_gate`) | **0.00e+00** | **CONFIRMED — equal (it is ZZ, NOT MS)** |
| 3 | Ulrich `MS_gate` has swapped (θ,φ) args vs repo/Ringbauer | swap→**1.73e-15**, same-order→**3.80** | **CONFIRMED — args transposed** |
| 4 | Ulrich `entangling_LS_gate` == global Hrmo gate (`ls_global_gate`) | **0.00e+00** | **CONFIRMED — equal** |

Supporting checks:
- Sanity, claim 2: Ulrich `entangling_MS_gate` vs the true physical MS = **1.83** (a different gate — it is diagonal ZZ, no XX/YY mixing).
- Bonus: paper Eq.4 `U_LS` (subspace ZZ) vs Hrmo global gate at d=3 = **2.36** (different gates; coincide only at d=2).
- Repo `MS_gate(phi,theta,...)`; Ulrich `MS_gate(theta,phi,...)` — argument order is opposite. Feeding matched physics with matched order gives 1.7e-15; feeding one order into the other gives 3.8. Integration hazard.

All gates unitary. Repo `MS_gate` is exact to 2e-15 including the single-active
spectator phase e^{−iθ/4} and the |jj>↔|kk> phases e^{∓2iφ}; at d=2 it is the
standard qubit MS up to the documented global phase e^{−iθ/2}.

## Gate-identity map (who is what)

| symbol | true physics | repo name | Ulrich name |
|---|---|---|---|
| physical Mølmer–Sørensen (Ringbauer Eq.2) | XX+YY entangler | `MS_gate` ✅ | `MS_gate` ✅ (args swapped) |
| subspace ZZ = paper Eq.4 `U_LS` = exp(−iθ/2 Z_jk⊗Z_jk) | diagonal, {j,k}-restricted | `zz_subspace_gate` | **`entangling_MS_gate`** (misnamed "MS") |
| Hrmo global light-shift (Eq.3) | diagonal, all off-diagonal |jk> get e^{iθ} | `ls_global_gate` | `entangling_LS_gate` |

Repo names are physically honest. Ulrich's `entangling_MS_gate` is the single
misnomer at the code level: it is a ZZ gate, not an MS gate.

## Where the PAPER mislabels a gate (Sec VII architecture figures)

Root cause: the paper's Eq.4 (main.tex:116–120) DEFINES the "light-shift gate"
`U_LS` to be the **subspace ZZ** `exp(−iθ/2 Z⊗Z)`. But Ulrich's Sec VII ansatz
(ulrich file lines 410, 416–417) builds its "LS Ring" from `entangling_LS_gate`
(= Hrmo **global** gate) and its "MS Ring" from `entangling_MS_gate` (= the
subspace ZZ = **exactly the paper's own Eq.4 U_LS**). So the two Sec VII labels
are crossed against the paper's own definitions:

1. **"MS Ring / MS_01"** — main.tex:266 (Fig. `fig:csum/ms_ansatz` caption),
   :278, :279, :319, :333. The entangler used is subspace ZZ, i.e. **no
   Mølmer–Sørensen gate is applied**, and the gate is *identical to the paper's
   Eq.4 light-shift gate*. Calling it "MS" is wrong twice over. Fig. captions and
   every "CSUM–MS hybrid" / "pure MS Ring" result inherit this.
   (Extra discrepancy: caption says MS restricted to {01} only, but the code
   applies `entangling_MS_gate` on BOTH (0,1) and (1,2) — so even the subspace
   restriction stated in the caption does not match the code.)

2. **"LS Ring"** — main.tex:260 (Fig. `fig:ls_ansatz` caption), :276, :318, :331,
   :332, :339. The entangler is the Hrmo **global** gate, which is **not** the
   paper's Eq.4 `U_LS` (they differ by Fro 2.36 at d=3). So "LS" in Sec VII names
   a different unitary than "U_LS" defined in Sec III. It does match the physics
   literature's Hrmo light-shift gate, but not the paper's own equation.

3. **Conclusion, main.tex:574** — "identified the light-shift gate on a ring …
   most universally capable." The winning gate is the Hrmo global gate, not the
   Eq.4 `U_LS` the reader was shown. Claim is about a gate the paper never defined.

4. **Open problems, main.tex:560** — "MS-only or mixed MS-and-light-shift." The
   "MS" ablation used zero MS gates (used subspace ZZ). The framed MS-vs-LS axis
   is illusory as labeled.

5. **Real ablation axis (correction for the record).** A prior note (audit/03)
   said the two Sec VII entanglers were the *same* gate. That is **not** the case
   in Ulrich's current file: `entangling_MS_gate` (subspace ZZ on {01},{12}) and
   `entangling_LS_gate` (Hrmo global) are genuinely different unitaries (Fro 2.36
   at d=3). But **neither is a physical MS**. The true Sec VII comparison is
   *subspace-restricted ZZ* vs *global all-levels light-shift* (both diagonal
   entanglers), plus the CSUM prelude — not "MS vs LS."

## Scope — what is CLEAN

- **Campaign codes, Sec VIII/IX** (encoder.py:68,132; jax_backend.py MS path):
  use repo `MS_gate` = the verified physical MS (Ringbauer Eq.2, Δ=2e-15).
  Every trained code and every LER number in the main campaign is built on the
  correct MS gate. **No campaign result is affected by these mislabels.**
- Repo `src/gates.py` naming is already corrected (`zz_subspace_gate`,
  `ls_global_gate`, with deprecated aliases) and docstrings warn correctly.
- Only Ulrich's Sec VII ablation figures/prose carry mislabeled gate names.

## Bottom line

The physics implemented in the code is correct. Repo `MS_gate` is the genuine
Ringbauer/Meth Mølmer–Sørensen gate; the campaign (Sec VIII/IX) uses it. The
problem is purely nomenclature in **Section VII**: the "MS Ring" is actually the
paper's own Eq.4 subspace-ZZ gate, and the "LS Ring" is the Hrmo global gate
(not the Eq.4 U_LS). Captions/labels for Fig. `fig:ls_ansatz`,
`fig:csum/ms_ansatz`, and the Sec VII/Conclusion prose need renaming — but no
numerical result is invalidated.

**GATE PHYSICS: CLEAN** (campaign MS = Ringbauer Eq.2 to 2e-15; all four claims
confirmed). Section VII gate *labels* need a nomenclature fix (documentation-only,
does not gate the training run).
