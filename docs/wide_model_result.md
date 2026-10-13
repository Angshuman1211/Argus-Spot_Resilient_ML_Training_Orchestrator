# Retraining the risk model on 100x more data (October 2026)

The original model was trained on about 300 positive windows from one region and its quality swung
with the random seed (9.5x to 22.6x lift) and the evaluation slice (21.8x on test, 2.1x on
calibration). This round replaces the data, the model class and the evaluation, and checks every
claim against controls. Every number below comes from `ml/model/wide_results/*.csv|json` and
`benchmark/results/trace_oct_*/`.

Reproduce:

```bash
AWS_PROFILE=argus-demo python ml/data/fetch_spot_prices_wide.py   # 8 regions, 30 types, 90 days
python ml/data/feature_pipeline_wide.py                            # ml/data/features_wide2.npz
python ml/model/wide_xgb_ablation.py                               # feature ablation, 3 seeds each
python ml/model/wide_xgb_tune.py                                   # tuning on the calibration slice only
python ml/model/train_final.py && python ml/model/add_region_thresholds.py
python ml/model/wide_experiments.py --ablation                     # Transformer ablation (GPU)
python ml/model/score_old_on_wide.py                               # old shipped model, same windows
python -m pytest tests/ml                                          # causality + serving tests
```

## 1. Data

| | Before | Now |
|---|---:|---:|
| Regions | 1 (eu-north-1) | 8 |
| Instance types | 5 | 30 |
| Series (type x AZ) | 15 | 768 |
| Price update records | 4,117 | 279,636 |
| Five-minute rows | 387k | 19.9M |
| Spike events (>1% jump) | 102 | 10,204 |
| Train / calibration / test positive windows | ~300 total | 20,819 / 6,787 / 2,893 |

Windows are split by time inside every series (80 / 10 / 10 with a purge gap), so the test slice is
the last days of the history (4 to 13 October) and nothing in training, tuning or calibration saw it.
The scaler is fitted on training rows only (the old one was fitted on all data).

## 2. Two things the old evaluation hid

**AWS now updates Spot prices about once an hour.** Every spike lands on the :00 step, so every
positive window ends at :45, :50 or :55 (`xgb_controls.json`: 2,893 of 2,893 positives sit in 491,273
of 1,965,208 windows). A rule that only knows "the top of the hour is near" gets **lift 4.0x and AUROC
0.876** with no model at all. That is higher than the old model's reported AUROC of 0.75, and about
equal to a retrained Transformer on the base features. Skill must be reported against this clock
baseline and also among the in-hour-end windows, where a spike is possible and the clock tells you
nothing. Both are in the tables below.

**The old model did not survive six months.** Scored on the same October test windows
(`old_model_on_wide.json`), the shipped Transformer (trained Jan to Apr) has:

| Slice | Positives | Lift | AUROC |
|---|---:|---:|---:|
| All series | 2,893 | 2.2x | 0.699 |
| eu-north-1 | 894 | 1.35x | 0.640 |
| eu-north-1, the five types it was trained on | 198 | 1.45x | 0.656 |

Its 21.8x was a favourable draw on a 84-positive test slice, not a repeatable result.

## 3. Feature ablation (which features help, multi-seed)

Gradient boosting (XGBoost) on engineered 2-hour window features, 3 seeds per row, test slice. "In
hour-end" is the AUROC among the windows where a spike is possible (clock baseline there: 0.5).

| Feature set | Test lift | AUROC | In-hour-end AUROC |
|---|---:|---:|---:|
| clock only (control) | 4.0x | 0.876 | 0.50 |
| 13 base features | 14.8x | 0.946 | 0.802 |
| + cross-AZ divergence | 15.3x | 0.945 | 0.802 |
| + Advisor interruption rate | 14.4x | 0.944 | 0.806 |
| + update cadence (n updates 1 h, mins since change) | 26.7x | 0.966 | 0.867 |
| + price extremes (vs 6 h / 24 h high and low) | 48.5x | 0.972 | 0.920 |
| all 19 | 93.2x | 0.978 | 0.936 |
| all 19 without clock features | 92.4x | 0.978 | 0.935 |
| + longer memory (24 h / 72 h range, last change, time since up-move / spike) = all 28 | 99.3x | 0.981 | 0.941 |
| all 28 without clock features | 100.5x | 0.981 | 0.940 |
| all 28 without cross-AZ and Advisor | 97.2x | 0.979 | 0.940 |

Seed standard deviation is 0.1 to 2.0 lift points, against 1.2 on a mean of 14.7 for the old model.

- **The two features rejected earlier stay rejected, now with evidence.** Cross-AZ divergence and the
  Advisor rate move lift by under one point, inside the seed spread, and the in-hour-end AUROC by 0.004
  at most. Drop both.
- **Update cadence and price position carry the signal.** A price that has sat near its recent low and
  has been updating often is the one that jumps. That is mean reversion in AWS's pricing, not
  something known to relate to reclaims (Section 7).
- The clock features are not needed once cadence is known, so the production feature set has none.

The Transformer on the same features (3 seeds; `ablation.csv`):

| Feature set | Test lift (mean, min to max) | AUROC |
|---|---:|---:|
| 13 base features | 9.8x (7.0 to 13.2) | 0.870 |
| + cross-AZ divergence | 10.8x (9.9 to 12.3) | 0.876 |
| + Advisor rate | 9.9x (7.0 to 12.3) | 0.869 |
| + update cadence | 21.6x (20.7 to 22.5) | 0.943 |
| + price extremes | 35.5x (32.6 to 37.1) | 0.949 |
| all 19 | 50.3x (48.1 to 51.7) | 0.960 |
| all 28 | 58.2x (55.7 to 61.7) | 0.967 |

The same conclusion holds for both model classes. Gradient boosting is about twice as good on the full
feature set (about 93x to 101x against 50x to 58x) and far steadier (98.5x to 101.4x across seeds, against
7.0x to 13.2x for the Transformer on base features). It is also
a lighter dependency: the serving image needs neither torch nor scikit-learn.

## 4. The shipped model

Gradient boosting, depth 3, 307 rounds, 22 features (`spot_features.LEAN_FEATURES`), Platt
calibration. Hyperparameters were chosen on the calibration slice only (72 configurations; the best
and worst differ by 6% in calibration AUPRC, so the choice is not delicate). The shipped model is
seed 0, declared before looking at the test slice; seeds 1 to 4 are reported only for variability.

Test slice, natural class balance (`final_eval.json`):

| Slice | Positives | Lift | AUROC |
|---|---:|---:|---:|
| all 768 series | 2,893 | 101.4x | 0.980 |
| in-hour-end windows only | 2,893 | 25.5x | 0.939 |
| eu-north-1 | 894 | 42.2x | 0.964 |
| eu-north-1, the original five types | 198 | 43.5x | 0.967 |

Seeds 0 to 4: 101.4, 100.5, 98.5, 100.0, 99.7 lift. Calibration: Brier 0.00136 against 0.00147 for
always predicting the base rate. At the best-F1 threshold (0.199) precision is 22% and recall 19% at an
alarm rate of 0.13% of windows. Raw lift is large because the base rate is 0.15%; precision at high
recall is modest (Section 6 and `docs/trace_benchmark_result.md`).

## 5. Serving

- **One feature code path.** `ml/model/spot_features.py` is used by training and by the service
  (`ml/api/predictor.py`). Train/serve skew was this project's recurring bug.
- **No look-ahead, tested.** `tests/ml/test_spot_features.py` recomputes features on histories cut at
  random steps and requires identical values at the cut. The predictor also drops any record dated
  after "now". Writing that test exposed two real bugs: an unfiltered future record changed the
  score, and `astype("int64") // 10**9` silently produced nonsense timestamps on pandas 3.
- **Live data, not a frozen CSV.** The service reads recent price updates from EC2
  (`PRICE_SOURCE=ec2`), from the Lambda collector's S3 snapshots (`s3`) or from a file (`csv`). Run
  against live eu-north-1 prices it returned scores in about 5 s uncached and 6 ms cached. The EC2 and S3
  sources need an IAM policy on the pod role (`ec2:DescribeSpotPriceHistory`, or `s3:GetObject` and
  `s3:ListBucket` on the feature store). That was not applied from here; use `csv` until it is.
- Image `argus/predict-service:v2-xgb` (also `latest`) is pushed to ECR.

## 6. Does the better model change the checkpointing result?

Yes, in the regime that motivates the project. Full method and tables are in
`docs/trace_benchmark_result.md`, section 7. In short, on a replay of the last 9 days of eu-north-1
prices (66 spike events, 15 nodes), with the alarm budget chosen on the calibration slice:

- 15 nodes, 300 s checkpoints: model **1.138x [1.124, 1.155]** against Young-Daly 1.339x, periodic 1.466x
  and reactive 4.248x (reactive collapses when a checkpoint outlasts the 2-minute notice).
- 15 nodes, 10 s checkpoints: reactive wins (1.019x); the model ties Young-Daly (1.057x vs 1.058x).
- Single node: the model loses to Young-Daly.
- The old model on the same events catches 10.6% of them at 0.8% precision (new: 74.2% at 9.5%) and is far
  worse than periodic checkpointing (2.479x at 300 s).

## 7. Real reclaims show no price signal (n = 2)

The account's own Spot history contains two **real** reclaims (CloudTrail `BidEvictedEvent`,
`ml/data/real/real_reclaims.csv`), both in eu-north-1b, both within minutes of launch:

| Instance | Launched (UTC) | Reclaimed (UTC) | Minutes alive |
|---|---|---|---:|
| c5.xlarge | 29 Jul 18:36:57 | 29 Jul 18:40:37 | 3.7 |
| m5.large | 30 Jul 08:54:54 | 30 Jul 09:07:21 | 12.5 |

Neither model warned. For the c5.xlarge pool the new model's risk 15 minutes before was at the 0.7th
percentile of that pool's scores (max 0.0073 in the preceding hour, against a 99th percentile of 0.134);
for the m5.large pool the risk was about zero throughout (15.8th to 21.9th percentile) with no price
spike at any step. The old model was no better (percentiles 18.8 to 19.4 for the first pool, 1.9 to 2.5 for the second until
the reclaim step itself, where it read 25.4).

Two events cannot give a rate, and both fall inside the new model's training period, so this is a sanity
check, not an evaluation. But it is the first real ground truth, and it points the wrong way for the
price-spike proxy: AWS reclaims on capacity, and price does not show it. The proxy-based results above
measure how well price behaviour is predicted, not how well reclaims are.

Next step, ready but not run: `scripts/launch_reclaim_sensors.sh` launches 15 tiny Spot instances across 5
instance types and 3 AZs with an 8-hour self-termination cap (about $1 to $2) so reclaims are generated
on purpose and read back with `ml/data/pull_reclaim_labels.py`. The launch was blocked as a billable action
in this session, so it needs your go-ahead. A second real signal worth collecting going forward is the
Spot placement score (`ec2 get-spot-placement-scores`), which returned 1 (low) for 20 units of m5.xlarge in
every eu-north-1 AZ when queried.

## 8. Limits

- The label is still a price-spike proxy. Section 7 is the reason to doubt it as a reclaim predictor.
- One test period (9 days), one season. Series in the same region are correlated, so intervals are
  optimistic.
- Instance types outside the 30 trained on are not served (the predictor returns 404).
- Lift is not precision: at the 10% alarm budget precision is 9.5%.

## 9. Robustness checks (added after the first draft)

**Skill is stable over time (`rolling_origin.py`, `rolling_origin.csv`).** The model was retrained from
scratch at five successive cut-off dates and scored on each following 9-day block, with calibration and
thresholds from the 7 days before each block:

| Test block | Train days | Test positives | Lift | AUROC | In-hour-end AUROC | Clock-only AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 4 to 13 Oct | 73.9 | 2,947 | 98.6x | 0.978 | 0.938 | 0.875 |
| 25 Sep to 4 Oct | 64.9 | 6,883 | 68.1x | 0.982 | 0.948 | 0.876 |
| 16 to 25 Sep | 55.9 | 1,621 | 127.8x | 0.977 | 0.936 | 0.875 |
| 7 to 16 Sep | 46.9 | 2,549 | 104.8x | 0.985 | 0.948 | 0.875 |
| 29 Aug to 7 Sep | 37.9 | 3,025 | 67.6x | 0.974 | 0.925 | 0.875 |

AUROC stays within 0.974 to 0.985 even with only 38 days of training. Lift varies with each block's base
rate, which is why AUROC is the steadier comparison.

**The checkpointing result is not specific to one region or slice (`benchmark/summarize_regions.py`,
`results/regions_summary.csv`).** The same five instance types, fleets of 15, were replayed in all eight
regions with the alarm budget chosen on each region's own calibration slice. At 300 s checkpoints, paired
slowdown difference against Young-Daly (negative favours the model), resampled by whole days:

| Region | Spike events | Model vs Young-Daly | 95% interval | Read |
|---|---:|---:|---|---|
| eu-north-1 | 66 | -0.200 | [-0.222, -0.178] | model better |
| ap-southeast-1 | 22 | -0.066 | [-0.134, -0.015] | model better |
| us-east-2 | 26 | -0.015 | [-0.067, +0.057] | tie |
| us-east-1 | 22 | +0.015 | [-0.051, +0.075] | tie |
| us-west-2 | 7 | -0.009 | [-0.035, +0.014] | too few events |
| eu-central-1 | 3 | 0.000 | [-0.064, +0.061] | too few events |
| eu-west-1 | 1 | -0.005 | [-0.017, +0.006] | too few events |
| ap-south-1 | 1 | -0.137 | [-0.144, -0.130] | too few events (one event decides it) |

Of the four regions with 20 or more events the model wins in two and ties in two, and it never loses
significantly. The other four regions barely had interruptions in the test slice, so no policy can be
separated there. At 10 s checkpoints it ties Young-Daly and reactive wins everywhere, as before.

**Intervals.** `benchmark/recompute_ci.py` replaces the overlapping-window intervals with paired
differences resampled by whole days. For eu-north-1 at 15 nodes the model beats Young-Daly by 0.200
slowdown at 300 s checkpoints [0.178, 0.222] and wins on all 8 test days; at 60 s by 0.068 [0.054, 0.081]
on 98% of days. Eight days is still a short record.

**Checkpoint cost is parameterized, not measured in-region.** An upload test to S3 from this machine
(`checkpoint_write_s3.csv`) ran at about 1.4 MB/s (256 MB in 179 s), which says only that the laptop uplink
is slow. The 10, 60 and 300 s values remain assumptions; the in-region EKS-to-S3 rate needs measuring on the
cluster.
