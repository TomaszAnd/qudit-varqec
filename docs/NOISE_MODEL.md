# NOISE_MODEL.md — Default noise path vs Meth et al. App. J

READ-ONLY forensic verification. SOURCE: `/Users/tomas/PycharmProjects/PythonProject8/qudit` (untouched).
Canonical target: `latex/main.tex` App. A `\label{app:kraus}` (lines 593–714), which claims to
reproduce Meth arXiv:2310.12110v3 App. J verbatim. Numeric checks via `/Users/tomas/qudit_env/bin/python3`.

> **STATUS: FIXED — the default channel now AGREES with Meth (Gaussian-averaged dephasing).**
> `src/correlated_noise.py::_kraus_diagonal` was changed to the Gaussian form
> `base = f_k·√(σ_p²)`, `decay = η^{f_k²/2}` so coherence between levels with coupling
> difference Δf decays as `e^{−½σ_p²Δf²} = E[e^{iΔfΦ}]` for `Φ ~ N(0, σ_p²)` — the standard
> Gaussian characteristic function, which is Meth's underlying model. Verified numerically:
> at η=0.9296, target f=(1,2,3,3), the off-diagonal coherence multipliers equal the Gaussian
> matrix below (1.0/0.9642/0.8642) and `Σ_n E_n†E_n = I` (dev 4e-16). The paper's Appendix C
> carries the old factor-2 (`base=f√(2σ²)`, `decay=η^{f²}`); that is a **paper-side transcription
> error to correct in the manuscript**, not a code change. The Bucket (B) "OPEN DECISION" below
> is now RESOLVED in favour of Meth; the Stage-C training HOLD is LIFTED. The forensic analysis
> that follows is retained for the record and describes the PRE-FIX state.

---

## STEP 1 — The DEFAULT noise path(s)

There are **two distinct default noise surfaces**, and the factor-2 question only lives on one of them.

### (T) Training basis — discrete surrogate (campaign + Meth-weighted training)
- Entry: `scripts/train.py` → default `--noise full` (train.py:49–51, 80–81) →
  `ErrorModel(d, n_qudit, distance, closed, basis="full")` (`src/errors.py`).
- Basis is the **discrete unitary set** `{Z_k, X_{k-1,k}} (+ Y, L, L†` for `basis="full"`), NOT the
  continuous Kraus channel. σ_p² never enters here directly — these are phase-flip / swap unitaries.
- Meth-weighted training (`scripts/train_r14_3_meth.py`, `seed_race.py`) keeps the same discrete
  basis but **weights** each operator by a Meth prior λ derived from the Kraus channel at a given η
  (operating point η=0.9296). So σ_p² enters the training objective only through those weights.

### (C) Correlated-channel LER benchmark — continuous Kraus channel (Sec. IX)
- Entry: `src/simulation.py::make_correlated_dephasing_noise_fn` (simulation.py:345–388),
  defaults `n_max=5`, `noise_model="physical"`.
- Consumes `src/correlated_noise.py`:
  - `control_qudit_kraus` (physical, f_k=k) — correlated_noise.py:51–75 [DEFAULT]
  - `target_qudit_kraus` (f=1/2/3) — correlated_noise.py:109–134
  - `spectator_qudit_kraus` (f_k=k) — correlated_noise.py:137–153
  - all built on `_kraus_diagonal` (correlated_noise.py:18–48).
- η enters as a script argument; every operating-point script uses **η=0.9296** (= e^{-0.073});
  e.g. `r14_4_head_to_head.py:149`, `meth_pauli_weights.py:261`, `benchmark_ler_meth_kraus.py:128`,
  `train_correlated_simplified_d{2,3}.py`, `r14_9_pphys_extend.py:25`.
- The MAP decoder prior in Sec. IX is computed from the same Kraus channel.

**The core numerical channel is `_kraus_diagonal` (correlated_noise.py:39–46).**
PRE-FIX (matched paper App. A, the factor-2 form):
```
base     = coupling_strengths * sqrt(2*sigma_p2)   # f_k·√(2σ_p²)
decay    = eta ** (coupling_strengths**2)     # η^{f_k²} = e^{-σ_p² f_k²}
```
FIXED (Gaussian-averaged, matches Meth ground truth):
```
sigma_p2 = -np.log(eta)                       # σ_p² = -ln η
base     = coupling_strengths * sqrt(sigma_p2)     # f_k·√(σ_p²)
decay    = eta ** (0.5*coupling_strengths**2) # η^{f_k²/2} = e^{-½σ_p² f_k²}
E_n[k]   = base**n * decay / sqrt(n!)
```

---

## STEP 2 — Conformance to Meth App. J (canonical = main.tex App. A)

Element-by-element against the "verbatim" appendix (main.tex:604–714):

| Item | Canonical (App. A) | Code | Verdict |
|---|---|---|---|
| Reparam | η := e^{-σ_p²} (599–601) | `sigma_p2=-np.log(eta)` (:39) | ✅ exact |
| Control Eq J3 | E_n^{(c)}=\|i⟩⟨i\| + Σ_{j≠i} [j√(-2lnη)]^n η^{j²}/√n! \|j⟩⟨j\| (618), f_j=j | `control_qudit_kraus` f_k=k, f_i=0 (:73–75) | ✅ matches App. A |
| Target Eq J4 | f(k)=1/2/3 for control/target/spectator (633–641) | `target_qudit_kraus` f=1/2/3 (:130–132) | ✅ exact |
| Spectator D_n | Σ_k [k√(-2lnη)]^n η^{k²}/√n! (653–658), f_k=k | `spectator_qudit_kraus` f_k=k (:152) | ✅ matches App. A |
| Amplitude (base) | (f√(2σ_p²))^n/√n! | `f·√(2σ_p²)` (:40) | ✅ exact |
| Decay | e^{-σ_p² f²} = η^{f²} (613/641/653) | `eta**(f²)` (:41) | ✅ exact |
| Operating point | σ_p²=0.073 → η=0.9296 (main.tex:443) | scripts pass η=0.9296 | ✅ e^{-0.073}=0.929601 (verified) |

**Verdict: the code faithfully implements the paper's App. A Kraus family, term-for-term, including
the operating point η=0.9296.** No stray factor, no wrong index, no sign error relative to App. A.

### Caveat that is NOT a code-vs-paper divergence
- Meth's *actual* App. J text (per `audit/02_noise_model.md`, cross-checked against the v3 HTML) is a
  **Monte-Carlo stochastic-phase model, not a Kraus family**, and its literal control mapping is
  *uniform* f=2 (Eq. 24), not f=k. The App. A "Kraus family, f_k=k" is the paper's own
  re-derivation/reinterpretation. The code matches the **paper**, and the literal-Meth f=2 variant
  (`control_qudit_kraus_simplified`, :78–106) exists but is **unused in production**. So code↔paper are
  consistent; both differ from Meth's literal MC model by a documented modeling fork. This is a
  paper-prose issue ("verbatim" claim is inaccurate), not an implementation typo.

---

## STEP 3 — Discrepancy classification

### Bucket (A) — IMPLEMENTATION BUG (code diverges from the model it implements): **NONE**
Every index, factor, and sign in `_kraus_diagonal` / the three role functions matches App. A exactly
(table above). The default `noise_model="physical"` (f_k=k) is the App. A form, not a mistake. There
is nothing to one-line-fix. **Bucket A is empty.**

### Bucket (B) — the factor-2 σ_p² dephasing — **RESOLVED (fixed to Gaussian)**

> **RESOLUTION (applied):** option (b)/(the Gaussian family) below was taken. The code now
> implements `E_n[k]=(f_k σ_p)^n/√n!·e^{-½σ_p² f_k²}`, reproducing Meth's Gaussian channel at
> `σ_p²=0.073` for `η=0.9296`. This resolves the open decision in favour of Meth (ground truth);
> the paper App. A/App. C factor-2 must be corrected manuscript-side. The rest of this section is
> the PRE-FIX analysis, retained for provenance.

**File:line:** `src/correlated_noise.py:41`
```
decay = eta ** (coupling_strengths ** 2)   # = e^{-σ_p² f_k²}
```
(equivalently the whole `_kraus_diagonal` convention, :39–46; mirrored in the paper App. A Eqs.
613/641/653).

**The two readings.**
- *As-implemented (matches paper App. A):* coherence between levels with phase-coupling difference
  Δf decays as **e^{-σ_p² Δf²}** (= η^{Δf²}). The Kraus family is complete (Σ_n E_n†E_n → I) and is a
  valid channel — it is the channel for phase variance **2σ_p²**.
- *Gaussian average of Meth's stochastic model:* E[e^{iΔf·Φ}] with Φ~N(0,σ_p²) gives
  **e^{-½ σ_p² Δf²}**. This is the standard Gaussian-dephasing result (the missing ½).

**Numeric effect on η (verified independently against the source functions, converged n_max=80,
target f=(1,2,3,3), η=0.9296 ⇒ σ_p²=0.073):**
```
Kraus channel coherence multiplier   Gaussian E[e^{iΔfΦ}]=e^{-σ²Δf²/2}
[[1.0000 0.9296 0.7468 0.7468]       [[1.0000 0.9642 0.8642 0.8642]
 [0.9296 1.0000 0.9296 0.9296]        [0.9642 1.0000 0.9642 0.9642]
 [0.7468 0.9296 1.0000 1.0000]        [0.8642 0.9642 1.0000 1.0000]
 [0.7468 0.9296 1.0000 1.0000]]       [0.8642 0.9642 1.0000 1.0000]]
log-ratio of exponents (Kraus/Gaussian) = 2.0000 for every off-diagonal
```
So the code at "η=0.9296" simulates an **effective variance 2σ_p² = 0.146**, twice Meth's calibrated
0.073. To reproduce Meth's Gaussian channel at σ_p²=0.073 one would pass
**η = e^{-0.073/2} ≈ 0.9642**, not 0.9296 (verified). Alternatively the family with σ²←σ²/2,
i.e. `E_n[k]=(f_k σ_p)^n/√n!·e^{-σ_p² f_k²/2}`, reproduces the Gaussian channel.

**Why this is (B) not (A):** the code does not diverge from the equations it implements — it matches
the paper App. A exactly, and the paper and code are mutually consistent. The disagreement is between
{code+paper} and {Meth's underlying MC model}, i.e. a convention/normalization decision, not a stray
typo. Per the task, do NOT change it. Resolution options (all require a human/paper decision, not a
patch):
  (a) relabel the operating point as σ_p²_eff = 0.146 and cite the family as-is;
  (b) rescale to η = e^{-σ_p²/2} = 0.9642 and re-run Sec. IX + the η→p_phys table;
  (c) justify a two-independent-draws-per-gate reading of Meth (not supported by the source text).
The paper's App. A "reproduced verbatim" wording must change under any option.

**This is the OPEN DECISION with Nicolai.** Already logged as blocks-paper #1 in
`audit/02_noise_model.md:19–31, 267–272`.

**Gating (PRE-FIX, now lifted):** the downstream (Meth-weighted / correlated-channel) training +
Sec. IX benchmark are the paths where σ_p² actually enters. The factor-2 choice shifted the effective
operating point by 2×. → was **HOLD**. **Now that the Gaussian fix has landed, the HOLD is LIFTED.**
Consequence for existing artifacts: results computed at η=0.9296 under the old channel used effective
variance 2σ_p²=0.146; the corrected channel at η=0.9296 uses σ_p²=0.073 (so `p_phys`: 0.56→0.376).
The paper's p_phys≈0.56 operating point corresponds to η=e^{−0.146}=0.8642 under the corrected channel.
Pre-fix η→p_phys / Sec. IX LER artifacts (e.g. `r14_7g_eta_to_pphys.csv`, Fig 10) are scheduled for
regeneration under the corrected channel in Stage J (see `docs/FOLLOWUP.md`).

(Secondary, not gating, already logged: spectator-ION dephasing D_n has no direct Meth-App.-J source
(our extension); Sec. IX drops σ_a² amplitude fluctuations, subspace depolarizing, and amplitude
damping; error-set builds use n_max=2 (15–41% channel weight lost) vs the n_max=5 MC path. See
audit/02 §1a,§1d,§4.)

---

## Summary verdict

- **Default noise path:** training = discrete surrogate basis (`train.py --noise full` →
  `ErrorModel`, `src/errors.py`); correlated benchmark = `make_correlated_dephasing_noise_fn`
  (`src/simulation.py:345`) over `src/correlated_noise.py` Kraus families (physical f_k=k, n_max=5,
  η=0.9296).
- **App. J conformance:** code now reproduces Meth's Gaussian-averaged dephasing exactly
  (`e^{−½σ_p²Δf²}`); pre-fix it matched the paper App. A factor-2 form. The paper App. A/App. C
  must be corrected manuscript-side.
- **Bucket (A):** none.
- **Bucket (B):** factor-2 in `correlated_noise.py` — **RESOLVED**: fixed to the Gaussian family
  `base=f√(σ²)`, `decay=η^{f²/2}`. Verified: coherence multipliers 1.0/0.9642/0.8642, `Σ E_n†E_n=I`.
  Noise tests re-pinned to the Gaussian target.

**NOISE: FIXED — agrees with Meth (Gaussian). HOLD lifted; training unblocked.**
