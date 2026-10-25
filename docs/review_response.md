# Response to the ML4Sys reviews, point by point

Written for the arXiv v2 and any resubmission. The new paper is `Paper/v2/argus_v2.tex` (compiles with Tectonic;
every number comes from a result file through `Paper/v2/build_numbers.py`). Evidence for each point is listed so a
reader can check it. Where a fix changed the conclusion, that is stated.

## Reviewer haku

**1. The predictor is never evaluated as a predictor; the label is a price spike, not a reclaim.**
Evaluated on an untouched later test slice with AUROC, AUPRC, lift, calibration and precision/recall
(paper Section 3.2, `ml/model/wide_results/final_eval.json`): lift 101.4, AUROC 0.980, 0.939 among the hour-end
windows where a spike is possible, seed range 98.5 to 101.4, Brier 0.00136 against 0.00147. Two controls the original
evaluation lacked: a clock-only rule already reaches AUROC 0.876, and the old model scores AUROC 0.699 on the same
October windows (its earlier headline lift did not survive six months). Retraining at five cut-off dates keeps
AUROC between 0.974 and 0.985. The 0.65 deployed threshold never fired against calibrated scores; it was a
configuration error and is now 0.05 (the 10% alarm budget).
On the label: we obtained real reclaim labels (Section 3.4). Of 36 sensor instances, 29 were reclaimed (median 12
minutes). The price model has **no skill** on them (AUROC 0.44, 0.36 in the first 30 minutes of an instance's life),
while AWS's placement score reaches 0.81 (0.89 in the first 30 minutes). The paper says so plainly and limits the
benchmark's conclusion accordingly. This point is answered but not "won": the price-based predictor does not predict
real reclaims.

**2. No non-learned predictive baseline.**
Added a volatility rule (AUROC 0.43), the Spot Advisor interruption band as a prior (AUROC 0.65, a prior across
types with no timing signal), a Young-Daly interval set from the Advisor bands (slowdown 1.60 at 300 s checkpoints,
worse than Young-Daly from observed events), random alarms at the same budget, and the clock rule.

**3. The predictive arm is misconfigured (600 s lead, 221 checkpoints, 1 of 5 runs finishes).**
There is no configured lead any more. Decisions come from the model's causal score; every run completes; the alarm
budget is chosen on a separate calibration slice by running the same simulation there.

**4. Single-run numbers (the 0.0 s wasted compute).**
Hundreds of thousands of simulated runs per sweep; intervals paired against Young-Daly and resampled by whole
calendar days (`benchmark/recompute_ci.py`); the 15-node, 300 s result is a 0.200 slowdown reduction, interval 0.178
to 0.222, better on 100% of the 8 test days. Replayed in seven more regions (Appendix C): the model wins in two of the
four regions with 20 or more events, ties in two, never loses significantly; the other regions had too few events.

**5. The lead guideline assumes a near-zero checkpoint time (3.8 MB).**
Measured in region on the real cluster: about 114 MB/s per node (2 GB in 19 s), independent of concurrency
(`benchmark/results/trace/checkpoint_write_eks.csv`). Table 1 turns that into time per model size: a 1.3B-parameter
model needs 136 s and a 7B model 735 s, so beyond roughly a billion parameters a checkpoint does not fit in the
2-minute notice. The benchmark sweeps checkpoint time from 5 to 600 s (Figure 3). A 2 GB checkpoint was also flushed
and acknowledged in 34.3 s on the real cluster.

**6. If a notice exists, why not checkpoint fast enough to fit inside it?**
Answered with data (Sections 1, 3.1, 3.3): reacting to the notice is near optimal while a checkpoint fits in
120 s (1.02 slowdown at 15 nodes) and collapses to no protection beyond it (4.25). Prediction matters exactly in
the regime where Table 1 says checkpoints stop fitting.

**7. The benchmark cannot test prediction (Poisson kills, configured lead).**
Replaced by trace-driven replay of real Spot price history with the real model's causal scores driving the policy
(`benchmark/trace_sim.py`, 20 known-answer tests). Interruptions in the replay are price spikes, which Section 3.4
shows are a poor stand-in for real reclaims; this is stated as the main limitation.

**8. AI-sounding prose.**
Rewritten from scratch in plain language. Checked against the project's writing rules by `Paper/v2/check_style.py`
(no em or en dashes, `Section~\ref`, no typed section numbers, no colons or semicolons in prose). A human read of
the whole paper by both authors is still needed before submission and cannot be done by a script.

## Reviewer ZNZL

**9. Narrow application field.**
Scoped honestly rather than hidden. The argument depends on a job only through its node count and checkpoint time,
Table 1 spans a 0.3M to 7B parameter range, the price model covers 30 instance types including GPU families in eight
regions, and the benchmark was replayed in all eight regions. Whether the community considers Spot checkpointing
narrow is a judgement we cannot change.

**10. Single node, small model.**
The benchmark covers 1, 5 and 15 nodes and checkpoint costs up to 600 s. On the real cluster two jobs ran together on
three nodes, one carrying a 2 GB checkpoint, through the real operator and the live model. This exposed five defects
in our own design (handshake, cooldown, polling, temp-copy upload, silent load error), all fixed and tested. What is
still not real: a synchronous multi-node training job with the operator coordinating its nodes, and a model large
enough to need a GPU. These remain future work and are listed as limitations.

**11. Figures 1 and 2 hard to read.**
Both redrawn as vector figures with 8 to 9 pt type at column width. The architecture diagram has four rows and large
labels. The old dual-axis lead-time chart was replaced by a single-axis plot of extra time against checkpoint time
(now Figure 3), with the other policies on the same axes.

## What is not closed

- Real reclaim prediction. Neither the price model nor any model we trained is shown to predict real reclaims. The
  placement score (AUROC 0.79) and pool history (0.90) do carry signal, and a model trained on real labels reaches
  0.81, and it now rests on 152 reclaims over two days in four regions, with day-ahead, held-out-pool and
  held-out-region tests (placement + age 0.74 to 0.77). It does not clearly beat the placement score alone, and the
  Rebalance Recommendation baseline has only 7 covered reclaims because AWS paused launches for a payment check.
- A human read-through of the final paper by both authors.
