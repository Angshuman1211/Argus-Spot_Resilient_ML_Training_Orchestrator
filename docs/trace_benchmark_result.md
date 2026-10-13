# Trace-driven benchmark: does prediction pay when it has to predict?

This replaces the Poisson-kill benchmark as the evidence for the paper's central question. The old
benchmark could not test prediction: Poisson kills carry no signal to learn, and its "predictive" arm
fired at (kill time minus a configured lead), so it was an oracle (see `docs/objective2_result.md`).
Every number below comes from `benchmark/results/trace/*.csv`; nothing is copied from an earlier draft.

Reproduce:

```bash
python ml/model/score_trace.py                     # real model scores for calibration + test windows
python benchmark/run_trace_benchmark.py            # ~7 min, 154k simulated runs
python benchmark/run_trace_benchmark.py --arms-prefix synthetic_yd --tag synth   # ~8 min
python benchmark/measure_checkpoint_write.py       # checkpoint write time vs size
python benchmark/make_trace_figures.py             # figures in benchmark/results/figures/
python -m pytest tests/benchmark                   # simulator known-answer tests
```

## What was built

| Piece | File |
|---|---|
| Causal replay trace: the shipped Transformer's calibrated score at every 5-minute step of the calibration and test slices, plus the non-learned signals | `ml/model/score_trace.py`, `benchmark/traces/trace_scores.csv.gz` |
| Event-driven simulator of a synchronous N-node training job under checkpoint policies | `benchmark/trace_sim.py` |
| Sweep over policies, fleet sizes and checkpoint costs, with block-bootstrap CIs | `benchmark/run_trace_benchmark.py` |
| 20 known-answer tests for the simulator | `tests/benchmark/test_trace_sim.py` |

**Interruptions** are price-spike events (a more than 1% jump) on real eu-north-1 Spot price series, the same
proxy the model was trained on. Real reclaim labels are not available (IAM restriction, FIS blocked), so a
"kill" here means "proxy spike event". This is the central limitation of the whole study.

**The held-out slice.** The test slice is the last part of each of 15 (instance type, AZ) series, about 9 days,
containing 28 spike events. The model, scaler and calibrator never saw it. Alarm thresholds are chosen on the
calibration slice (9 events) as an alarm budget, then applied unchanged to the test slice.

**The job.** A synchronous N-node job needing 12 hours of compute: a reclaim of any node kills it. A checkpoint
pauses the job for C seconds and is durable only if it completes. A kill loses all work since the last durable
checkpoint and costs 120 s to restart. A migration is a checkpoint plus a restart on fresh capacity that cannot
be reclaimed for the next 15 minutes (assumption). After a reactive checkpoint the node is draining, so the job
stops until the kill lands. Reactive gets a 120 s notice before every kill.

**Checkpoint cost C** is swept over 10, 60 and 300 s. `benchmark/measure_checkpoint_write.py` measured local
disk write at 230 to 510 MB/s. S3 upload speed was not measured (no AWS access here). Using training state of
12 bytes per parameter (fp32 weights plus Adam moments) and stated upload rates, C = 10 s is a ResNet-50 to
BERT-large class model, C = 60 s is about 1B parameters at a few hundred MB/s, and C = 300 s is a 7B-class model
(84 GB of state, 168 s at 500 MB/s, 840 s at 100 MB/s). This answers the reviewer concern that the lead-time
guideline only holds for a near-instant 3.8 MB checkpoint.

## 1. Is the model a predictor at all?

Window-level, same 38,085 test windows as `docs/objective3_result.md` (`predictor_quality.csv`):

| Signal | Split | Positives | AUPRC | Lift over base rate | AUROC |
|---|---|---:|---:|---:|---:|
| Transformer | test | 84 | 0.0480 | 21.8x | 0.748 |
| Volatility rule (rolling std / mean) | test | 84 | 0.0022 | 1.0x | 0.444 |
| Advisor interruption tier | test | 84 | 0.0017 | 0.8x | 0.271 |
| Transformer | calibration | 27 | 0.0015 | 2.1x | 0.662 |
| Volatility rule | calibration | 27 | 0.0007 | 1.0x | 0.443 |
| Advisor interruption tier | calibration | 27 | 0.0006 | 0.8x | 0.366 |

The model has real signal and the two non-learned predictors have none, so the learned model is worth having
over them. Two cautions. The calibration slice has only 9 spike events and the model's lift there is 2.1x, far
below the test slice, which is consistent with the documented run-to-run variance and with the shipped
checkpoint being a favourable draw (`docs/objective3_result.md`, Round 4). And an AUROC of 0.75 is modest.

At the event level (`detection.csv`, test slice, 26 kills inside the common window, alarm budget 0.2% of
node-steps): the model alarms before 5 of 26 kills (**19% recall**) with **7% precision** (8% at the 0.05%
budget) and about 1.7 alarms per node per day. The volatility rule catches 0 of 26. The thresholds were fitted
on the calibration slice, and the model's scores run higher on the test slice, so the realized alarm rate on
test is about 0.6% of node-steps, three times the nominal 0.2% budget. The "same budget" random and
volatility arms use the nominal budget.

## 2. Policy comparison

Slowdown = makespan / ideal work time (1.0 means no loss). Mean over start times, with a 95% block-bootstrap
CI in brackets, blocks of about one job length. Starts overlap in time, so effective sample size is small
(a 9-day slice holds roughly 14 independent 12-hour windows). Treat differences under about 0.02 as noise.

### 15-node job (about 3.1 kills per day across the fleet)

| Policy | C = 10 s | C = 60 s | C = 300 s |
|---|---|---|---|
| no protection | 3.058 [1.922, 4.306] | 3.058 [1.918, 4.303] | 3.058 [1.962, 4.304] |
| reactive on 2-min notice | **1.008** [1.004, 1.012] | **1.008** [1.004, 1.012] | 3.076 [1.983, 4.444] |
| periodic, every 1 h | 1.079 [1.047, 1.112] | 1.093 [1.060, 1.131] | **1.163** [1.129, 1.201] |
| periodic, Young-Daly interval | 1.033 [1.021, 1.046] | 1.079 [1.051, 1.111] | 1.189 [1.109, 1.266] |
| periodic, Advisor-tier interval | 1.032 [1.015, 1.051] | 1.090 [1.055, 1.127] | 1.220 [1.122, 1.321] |
| random alarms (same budget) | 1.314 [1.143, 1.530] | 1.318 [1.167, 1.534] | 1.338 [1.173, 1.534] |
| volatility rule alarms | 2.811 [1.730, 3.886] | 2.823 [1.737, 3.956] | 2.877 [1.803, 3.988] |
| model alarms, checkpoint only | 1.439 [1.253, 1.676] | 1.447 [1.257, 1.662] | 1.503 [1.308, 1.731] |
| model alarms, checkpoint + migrate | 1.454 [1.247, 1.668] | 1.463 [1.290, 1.693] | 1.524 [1.346, 1.737] |
| Young-Daly + model alarms (ckpt) | 1.035 [1.022, 1.049] | 1.077 [1.051, 1.101] | 1.180 [1.123, 1.255] |
| Young-Daly + model alarms (migrate) | 1.043 [1.031, 1.057] | 1.089 [1.063, 1.122] | 1.196 [1.129, 1.272] |
| oracle (5 min of perfect warning) | 1.004 [1.002, 1.007] | 1.006 [1.003, 1.009] | 1.014 [1.008, 1.022] |

The same sweep for 5-node and single-node jobs is in `summary.csv`; the patterns are the same, with smaller
absolute losses (no protection is 1.887 at 5 nodes and 1.076 at 1 node).

Figures: `benchmark/results/figures/trace_policy_slowdown.png`, `trace_fleet_size.png`.

## 3. What the results say

1. **Reactive checkpointing is near-optimal while a checkpoint fits inside the 2-minute notice, and useless
   once it does not.** At C = 10 s and 60 s it is the best policy at every fleet size (1.008 at 15 nodes). At
   C = 300 s it collapses to no protection (3.076 vs 3.058). This is the real regime where "just use the free
   notice" fails: when the checkpoint takes longer than the notice, which is the large-model case. It answers
   the reviewers' question about checkpointing fast enough to fit inside the notice, and it says when that
   stops being possible.
2. **A well-tuned periodic policy handles every checkpoint size.** Young-Daly stays between 1.03 and 1.19 at 15
   nodes. When the notice is unusable, periodic is the practical baseline to beat, and it is cheap.
3. **Prediction alone does not beat periodic checkpointing here.** The model-driven arms (1.44 to 1.52 at 15
   nodes) are worse than Young-Daly at every C and fleet size. At 15 nodes they are also worse than random
   alarms at the same budget (1.31 to 1.34), because the model's alarms cluster in time (93% of its alarm steps
   follow another alarm within 15 minutes, against 26% for random alarms at a similar overall rate) and leave long
   unprotected stretches, while random alarms spread out and act like periodic checkpoints. At 1 and 5 nodes
   the model beats random alarms (for example 1.19 vs 1.37 at 5 nodes, C = 10 s).
4. **Prediction as an addition to Young-Daly is indistinguishable from Young-Daly.** Differences are at most
   0.013 and every pair of CIs overlaps (15 nodes: 1.035 vs 1.033, 1.077 vs 1.079, 1.180 vs 1.189). The honest
   statement is "no measurable gain on this trace", not "a small gain".
5. **The headroom is real.** The oracle reaches 1.004 to 1.014. A predictor good enough to approach it would
   change the picture, which is what Section 4 quantifies.
6. **The learned model is better than non-learned predictors** (Section 1), but being better than a rule that
   has no skill is not the same as being useful for scheduling.

## 4. How good would a predictor need to be?

Young-Daly checkpointing plus a synthetic predictor with chosen recall and precision, 10 minutes of lead,
migrating on every alarm (`summary_synth.csv`). Slowdown at 15 nodes. The best non-predictive policy is
reactive (1.008) at C = 10 s and 60 s, so no predictor helps there. At C = 300 s the best baseline is periodic
every 1 h (1.163) or Young-Daly (1.189):

| Precision \ Recall | 20% | 50% | 80% | 95% |
|---|---:|---:|---:|---:|
| 90% | 1.156 | 1.107 | 1.071 | 1.058 |
| 30% | 1.150 | 1.109 | 1.078 | 1.064 |
| 10% | 1.160 | 1.118 | 1.094 | 1.087 |
| 5% | 1.165 | 1.123 | 1.123 | 1.124 |

(C = 300 s, 15 nodes. Figure: `trace_predictor_frontier.png`, all three C values.) Recall matters far more than
precision, because a false alarm costs only one checkpoint plus a restart. At 300 s checkpoints, recall of 20%
roughly ties the best non-predictive policy (1.150 to 1.165 against 1.163), 50% recall beats it clearly (1.107 to
1.123) and is worth about 0.08 of slowdown on top of Young-Daly, and 80% or more approaches the oracle. The
shipped model measures 19% recall at 7% precision, in the first column. These synthetic alarms are idealized (independent, exactly 10 minutes ahead), while real model
alarms cluster, so read the table as an optimistic bound.

## 5. Limitations (state all of these in the paper)

- **Kills are a proxy** (price spikes), the same proxy the model was trained on. The model's skill on this
  trace may not transfer to real reclaims.
- **Small sample.** 28 spike events in the test slice, 26 inside the common 15-series window, one region, about
  9 days. Thresholds come from a calibration slice with only 9 events. Bootstrap CIs resample overlapping
  windows and probably understate the uncertainty.
- **Reactive assumes the 2-minute notice always arrives** and that the whole job can checkpoint inside it.
- **Migration assumptions:** migrated nodes cannot be reclaimed for 15 minutes plus the restart, and the new
  capacity is independent of the old. Both favor migrate-on-alarm, and it still did not win.
- **Checkpoint cost is parameterized**, not measured on EKS. S3 upload speed was not measured.
- **The synthetic predictor is idealized** (independent alarms, fixed lead).
- **The shipped checkpoint is a favourable draw** (21.8x vs a 14.7x five-seed mean), and it is weaker on the
  calibration slice.

## 6. What this means for the paper

The defensible claim is a guideline and a negative result, not "prediction beats checkpointing":

- A free reactive notice is enough while checkpoints fit inside it, and fails exactly when they do not.
- Periodic checkpointing at the Young-Daly interval is a strong, cheap fallback in that regime.
- On real Spot price traces, the shipped model-driven policy does not beat it, and adds nothing measurable on
  top of it. The predictor would need roughly 50% recall (not the measured 19%) to matter clearly.
- The gap to the oracle (1.004 to 1.014) shows what a better predictor, probably trained on real reclaim labels,
  could be worth.

This directly answers reviewer haku's points 1, 2, 3, 4 and 7: the predictor is evaluated as a predictor
(Section 1), there is a non-learned baseline (volatility rule, Advisor tier), the predictive arm is no longer
a configured parameter, and the single-run 0.0 s headline is gone.
