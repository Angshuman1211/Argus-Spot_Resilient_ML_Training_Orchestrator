# Status of the 19-item fix list (review of 26 Oct 2026)

Each fix from the external review (F1 to F19) with where it is done. Paper text lives in the gitignored
`presentation_files/Paper_v2/`; every result it quotes is a committed file listed in `ARTIFACT.md`.

| ID | Fix | Status | Evidence |
|---|---|---|---|
| F1 | Real predictor output in the harness, threshold sweep, precision and recall | done | `benchmark/trace_sim.py`, `run_trace_benchmark.py`; operating points in `results/trace_oct_new/detection.csv` |
| F2 | Young-Daly, grid of fixed intervals, hybrid arm | done | `results/trace_oct_new/summary_extra.csv` (10 min to 4 h grid, Young-Daly plus model alarms at 10 and 20%) |
| F3 | State how the notice is simulated, explicit C and R | done | `trace_sim.py` docstring, paper Section 3.3 |
| F4 | Realistic checkpoint sizes | done | `results/trace/checkpoint_write_eks.csv` (16 MB to 2 GB in region), 2 GB EKS drill, C swept 10 to 300 s; 5 GB not measured |
| F5 | Realistic rates, trace-driven interruptions | done | nine-day price traces in 8 regions; real reclaims in `ml/data/real/` |
| F6 | Many repetitions, bootstrap intervals, all rates | done | day-resampled intervals in `paired_day_ci.csv`, every checkpoint size in the main table |
| F7 | Analytic model of waste, lead rule, decision map | done | `benchmark/analytic_model.py`, `results/analytic/` (median error 0.3% against the simulator, same winner in 72 of 72 cells), decision map figure |
| F8 | Reframe title, abstract, contributions | done | paper title and abstract state the negative result on real reclaims |
| F9 | Real EKS campaign with natural reclaims, FIS reclaim | partly, blocked | 160 natural reclaims from the sensor fleet; no EKS training campaign and no FIS reclaim, because the old AWS account is suspended and the new one has EC2 blocked |
| F10 | Real labels, temporal split, PR-AUC, calibration, lead time | done | `ml/model/real_label_quality.py`, `wide_results/reclaim_quality.json`; day-ahead test in `real_label_model.json` |
| F11 | Model ablation against simple baselines | done | XGBoost vs Transformer vs clock, volatility, Advisor band; logistic on real labels |
| F12 | Multi-node job, correlated interruptions | done locally, not on GPUs | operator gang mode (`gang: true`) with 8 unit tests and a 3-node minikube check (`demo/gang/`); exposure check `results/analytic/exposure.json`; no multi-node GPU training run (no AWS access) |
| F13 | Related work and positioning table | done | Parcae, SpotServe, SkyPilot, Can't Be Late, Aupy et al., Gainaru et al. added and verified; positioning table in the paper |
| F14 | Dollar cost per arm | done | `benchmark/cost_table.py`, `results/trace_oct_new/cost.csv` |
| F15 | Release code, configs, seeds, logs, versions | done | `ARTIFACT.md` |
| F16 | Abstract, figures, screenshots, author name | done | rewritten abstract; screenshots removed; name renders as MD Rayyan |
| F17 | Move the bug log out of the main text | done | v1 Section 3.4 removed; EKS defects kept as a short lessons paragraph |
| F18 | Flush-marker ablation, grace period check | done (measured, not a second implementation) | `benchmark/handshake_grace.py`, `results/analytic/handshake_grace.json`: the handshake adds 3.5 s on average, under 0.07% of job time; 30 s default grace covers about 0.29B parameters. `train.py` warns when a save would not fit. A direct in-operator checkpoint was not built |
| F19 | $190 example, define the interval behind p, makespan note | done | example removed; p is now per hour; the makespan issue no longer applies (every run completes) |

## Not doable now

- **F9 EKS campaign and FIS reclaim**, and a **multi-node GPU training run** for F12: both need EC2 in an AWS
  account. Account 844641713781 is suspended and 906615256166 has EC2 blocked (25 Oct). Resume once AWS
  restores access (see `docs/pending_work.md`).
