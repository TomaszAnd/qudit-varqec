# Best-practice training runs (Stage C) — HELD

Status: **not launched.** This is a forward-looking deliverable (new codes
trained under the recommended configuration), distinct from reproducing the
paper's figures. It is gated on the noise-model verification (`docs/NOISE_MODEL.md`)
coming back clean, and it did not.

## Why held

`docs/NOISE_MODEL.md` (Stage B3) confirmed the default correlated channel
reproduces the paper's Appendix C **exactly**, but surfaced an **open modeling
decision**: the channel decays coherences as `e^{−σ²Δf²}`, a factor of 2 in the
exponent versus the Gaussian average of Meth's stochastic model
(`e^{−½σ²Δf²}`). Verified numerically: the log-ratio of exponents is exactly
2.0, so the "η = 0.9296" operating point simulates effective variance 2σ² = 0.146,
twice Meth's 0.073 (matching Meth would need η ≈ 0.9642).

σ_p² enters both the Meth-weighted training prior and the Sec IX channel/decoder.
Training now would bake an unresolved 2× operating-point shift into new codes.
Per the run policy, the correct action is **HOLD, not train**, until the factor-2
σ² question is resolved (open decision with Nicolai).

## Intended configuration (when unblocked)

- Connectivity: all-to-all; entangler: MS (physical Mølmer–Sørensen, verified
  == Ringbauer Eq. 2 — see `docs/GATE_AUDIT.md`).
- `seed_race` multi-start with **full-batch certification** (a win requires the
  full-basis loss below target; sampled loss is only a cheap trigger).
- Sec IX importance/stratified sampling budget (the weighted-vmap masked path,
  `src/sampling/stratified_importance.py` + `jax_backend.stratified_weights`;
  **not** the reverted gather path, which is removed from this package).
- Output to a **separate** `results/best_practice_runs/` — must never overwrite
  the published params in `results/params` / `results/saved_params`.
- Suggested scope: ((5,3,3))₃ as a modest-seed validation run first, then
  ((9,3,3))₃ as the headline if load and budget allow. `nice -n 19`, background.

## Note

This package does not depend on Stage C for figure reproduction — every in-scope
figure/table is regenerated from committed params (see `REPRODUCE.md`).
