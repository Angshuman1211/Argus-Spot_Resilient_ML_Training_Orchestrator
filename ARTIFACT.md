# Argus artifact: how to reproduce every result

Everything below runs on a laptop with no AWS access, from files committed in this repository. Only the
data collection steps (marked **AWS**) need an account, and their outputs are committed too.

## Software versions

Analysis environment (`.venv`, used for everything in `benchmark/` and `ml/model/`):

| Package | Version |
|---|---|
| Python | 3.11.9 |
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| torch (CPU) | 2.11.0 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |
| boto3 | 1.42.90 |

Operator and tests (system Python 3.11; the `.venv` lacks `kopf`): kopf 1.37.2, kubernetes 31.0.0, httpx 0.27.0,
boto3 1.34.69, pytest 9.0.3, torch 2.12.0 (CPU). The pinned container dependencies are in
`operator/requirements.txt`, `ml/api/requirements.txt` and `ml/cifar10_job/requirements.txt`.

Cluster: Amazon EKS 1.32 (`terraform/eks.tf`), Terraform AWS provider ~> 5.0, AWS Node Termination Handler in
queue mode, Helm chart in `helm/argus`.

## Tests

```bash
"C:\Program Files\Python311\python.exe" -m pytest -q tests      # 104 pass, 1 skipped (mlflow)
```

## Results, the command that makes each, and where it lands

Seeds are fixed in each script (benchmark seed 7, bootstrap seed 0).

| Result | Command | Output |
|---|---|---|
| Price model, controls, ablations, rolling origin | `python ml/model/wide_experiments.py`, `wide_xgb_ablation.py`, `rolling_origin.py` | `ml/model/wide_results/` |
| Trace benchmark (15/5/1 nodes, all policies) | `cd benchmark && python run_trace_benchmark.py --trace traces/trace_scores_new.csv.gz --out-dir results/trace_oct_new` | `results/trace_oct_new/summary.csv` |
| Same on the calibration slice (picks the alarm budget) | add `--split calib --out-dir results/trace_oct_new_calib` | `results/trace_oct_new_calib/` |
| Periodic interval grid, Young-Daly plus model alarms | `python run_trace_benchmark.py --trace traces/trace_scores_new.csv.gz --out-dir results/trace_oct_new --only-sets N15 --arms-prefix periodic_grid_,hybrid_yd_model_migrate_q0.1,hybrid_yd_model_migrate_q0.2,young_daly --tag extra` | `results/trace_oct_new/summary_extra.csv` |
| Idealized predictor sweep | `python run_trace_benchmark.py --trace traces/trace_scores_new.csv.gz --out-dir results/trace_oct_new --arms-prefix synthetic_yd --tag synth` | `results/trace_oct_new/summary_synth.csv` |
| Paired day intervals | from the repo root: `python benchmark/recompute_ci.py --test results/trace_oct_new --calib results/trace_oct_new_calib` | `results/trace_oct_new/paired_day_ci.csv` |
| Other regions | from the repo root: `python benchmark/summarize_regions.py` | `results/regions_summary.csv` |
| Dollar cost per policy | `python cost_table.py` | `results/trace_oct_new/cost.csv` |
| Analytic model, validation, decision maps | `python analytic_model.py` | `results/analytic/summary.json`, `validation.csv`, `map_*.csv` |
| Multi-node exposure check | `python exposure.py` | `results/analytic/exposure.json` |
| Flush handshake cost, SIGTERM grace period | `python handshake_grace.py` | `results/analytic/handshake_grace.json` |
| Real reclaims: AUROC table | `python ml/model/evaluate_real_reclaims.py` | `ml/model/wide_results/real_reclaim_eval.json` |
| Real reclaims: capacity model, held-out tests | `python ml/model/train_real_label_model.py` | `wide_results/real_label_model.json`, `final/capacity_model.json` |
| Real reclaims: PR-AUC, calibration, lead | `python ml/model/real_label_quality.py` | `wide_results/reclaim_quality.json` |
| In-region checkpoint upload rate (**AWS**) | `python benchmark/measure_checkpoint_write.py` on the EKS nodes | `results/trace/checkpoint_write_eks.csv` |
| Real reclaim labels (**AWS**) | `python scripts/sensor_campaign.py` (see `scripts/start_cloud_runner.py`) | `ml/data/real/` |
| EKS drills (**AWS**) | `scripts/eks_predictive_drill.sh` | `demo/drill/results/` |

`analytic_model.py` reads `runs.csv.gz` from the trace benchmark, which is gitignored for size; run the trace
benchmark first. `real_label_quality.py` must run before `analytic_model.py` for the real-predictor points.

## Paper

The LaTeX source is kept outside this repository. Its build scripts read only the result files listed above:
every number in the text is a macro written by `build_numbers.py`, every table by `build_tables.py`, and every
figure by `make_figures.py`, so a number can change only when its result file changes.

## Data

- `ml/data/real/real_reclaims.csv`: all 160 real reclaims (2 from CloudTrail, 158 from the sensor fleet over
  13, 14 and 25 Oct 2026), with instance type, AZ, launch and reclaim times.
- `ml/data/real/`: per-campaign launch logs, placement scores and rebalance signals.
- `benchmark/traces/`: Spot price traces with the model's causal risk scores, one file per region.
