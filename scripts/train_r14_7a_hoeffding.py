#!/usr/bin/env python3
"""R14-7 §B.1 — Hoeffding-stopped stratified-IS training.

Sub-100k path A: stratified-IS (R14-3b) at n_steps=1100. At 90 op-EVs/step,
1100 × 90 = 99k op-EVs/seed (sub-100k). 1100 matches the empirical convergence
point — R14-3a/R14-3b reached their loss minimum by step ~1200 across all six
seeds (R14-6 §B), so stopping at 1100 lands at/near the minimum.

"Hoeffding-style stopping" here reduces to a fixed shorter step count (the limit
of a well-tuned tolerance). A principled rolling Hoeffding-tail-bound stop is
deferred to R14-8 if this fixed-1100 result is encouraging.

Thin wrapper over train_r14_3b_meth_is (sampler, Adam schedule, Meth weights,
budget all identical) — only the default n_steps changes to 1100.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_r14_3b_meth_is

if __name__ == "__main__":
    if not any(a == "--n_steps" or a.startswith("--n_steps=") for a in sys.argv):
        sys.argv += ["--n_steps", "1100"]
    if not any(a == "--ckpt_every" or a.startswith("--ckpt_every=") for a in sys.argv):
        sys.argv += ["--ckpt_every", "200"]
    if not any(a == "--out_dir" or a.startswith("--out_dir=") for a in sys.argv):
        sys.argv += ["--out_dir", "results/round14_scoping/r14_7a_hoeffding"]
    train_r14_3b_meth_is.main()
