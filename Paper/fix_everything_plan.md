# Argus: Fix Everything Plan (project + paper)

Companion to `v2_revision_plan.md`. That file covers the reviewer comments. This one lists every problem found in a full read of the repo and the submitted paper (`presentation_files/Argus_Paper_NeurIPS/neurips_2026.tex`), with the fix for each.

Owners follow the v2 plan: **A = Angshuman** (systems, infra, harness, real EKS), **B = Rayyan** (model, ML writing). Effort: LOW (under 1 hour), MED (a few hours), HIGH (days).

## Status (branch `week-9-v2-fixes-and-benchmark`)

| Item | State |
|---|---|
| 1.1 operator waits for checkpoint flush | done, 12 tests (`tests/operator`) |
| 1.2 calibrator in API image, refuse uncalibrated | done, tested. Image not rebuilt or pushed |
| 1.3 threshold consistency | done (`demo/spotresilientjob-real.yaml`, README) |
| 1.4 operator: uncordon, retry, AZ from spec, CRD enum, metrics, dead code, mojibake | done. Unused CRD fields (`command`, `checkpointIntervalSteps`) and bare-pod creation still open |
| 1.5 training job: atomic upload, retries, resume same epoch, flag-based SIGTERM | done, 9 tests (`tests/training`) |
| 1.6 pipeline accepts Lambda schema, unknown types dropped, API lifespan | done. Serving still uses a baked-in features.csv |
| 1.7 CI | tests workflow added (installs xgboost and API deps), deploy trigger fixed to `master`. OIDC: `scripts/setup_github_oidc.sh` is written but creating the IAM role was blocked; needs you to run it, then switch `deploy.yml` to the role |
| 2.1 trace-driven benchmark with the real model | done (`benchmark/trace_sim.py`, `docs/trace_benchmark_result.md`) |
| 2.2 non-learned baselines | done (volatility rule, Advisor-tier Young-Daly, random) |
| 2.3 rerun with proper stats | done (CIs by block bootstrap, 3 fleet sizes, 3 checkpoint costs) |
| 2.4 checkpoint size vs write time | measured local disk; S3 upload speed assumed, not measured |
| 2.5 stress test doc | fixed (no speedup claim) |
| 2.6 real-EKS predictive-path run | not done by this session. A cluster `argus-eks` (3 On-Demand nodes) was started at 11:11 IST by another Claude session and is still running; I did not touch it or Terraform |
| 3.1 predictor-as-predictor numbers | done (`predictor_quality.csv`, `detection.csv`) |
| 3.2 multi-seed re-test of rejected features | done on 100x more data, XGBoost and Transformer (`docs/wide_model_result.md`). Both stay rejected; cadence and price-position features help a lot |
| 3.3 real reclaim labels | 2 real reclaims found in CloudTrail and saved (`ml/data/real/real_reclaims.csv`); neither model flagged them. Sensor fleet (`scripts/launch_reclaim_sensors.sh`) written, dry-run validated, launch blocked: needs your approval |
| 3.4 retrain on more data, better model class | done: 8 regions, 768 series, XGBoost, 101x lift on an untouched slice (old model 2.2x on the same windows) |
| 3.5 serve live data, shared feature code | done: `ml/api` (EC2 / S3 / CSV sources), causality tests, image `argus/predict-service:v2-xgb` pushed to ECR. EC2/S3 sources need an IAM policy not applied from here |
| 2.7 benchmark with the retrained model | done on an October slice with a calibration-selected alarm budget: model beats Young-Daly (1.14x vs 1.34x) at 15 nodes / 300 s checkpoints |
| Part 4 paper rewrite | not started, waits for discussion |
| Part 5 hygiene | `terraform.exe`, `mlruns.db`, stray BSL `LICENSE.txt` untracked, `.gitignore` cleaned. History not rewritten (the 90 MB blob remains in old commits) |

Key result that changes the paper: the trace benchmark shows the current model does not beat Young-Daly
periodic checkpointing (details in `docs/trace_benchmark_result.md`), so the framing in Part 4.1 should be the
guideline plus negative result.

---

Order of work: Part 1 (stop the system lying) -> Part 2 (make the benchmark test prediction) -> Part 3 (paper) -> Part 4 (cleanup). Do not rewrite the paper before Parts 1 and 2 are done, or the numbers will change under you.

---

## Part 1. Correctness bugs in the real system

### 1.1 Operator does not wait for the checkpoint before killing the pod  [A, MED] CRITICAL
- **Where:** `operator/controller/handlers.py` lines ~143-177, `operator/controller/scheduler.py::reschedule_pod`.
- **Problem:** `reconcile` writes `_FLUSH_TRIGGER`, then cordons the node and deletes the pod with `grace_period_seconds=0` (SIGKILL). The training pod only polls the trigger every 100 batches, and SIGKILL skips the SIGTERM handler. The checkpoint is a race the pod usually loses.
- **Why it matters:** The paper (section 3.3) says "the operator checkpoints synchronously before migrating". True only in the benchmark harness, not in the real operator. The zero-wasted-compute claim is not demonstrated on the real path.
- **Fix:**
  1. After writing the trigger, poll S3 until the trigger key is deleted by the pod (it deletes it after saving) or `latest_checkpoint.pt` LastModified is newer than the trigger time. Timeout, e.g. 60 s.
  2. Only then cordon and delete the pod, with a normal grace period (not 0) so the SIGTERM handler can also run.
  3. If the wait times out, set phase `Failed` or fall back to deleting with grace, and log it. Increment a new metric `argus_checkpoint_wait_timeouts_total`.
  4. Add a minikube/LocalStack test that proves a checkpoint exists before the pod is deleted.
- **Done when:** a log line shows "checkpoint confirmed" before "Deleting pod", and the test passes.

### 1.2 API image ships without the calibrator  [B, LOW] CRITICAL
- **Where:** `ml/api/Dockerfile`.
- **Problem:** It copies the model, scaler, metadata and features but not `spot_calibrator.joblib`. `app.py` silently falls back to the raw sigmoid (scores around 0.05 to 0.09). The Helm default `riskThreshold: 0.0015` is on the calibrated scale, so on the raw scale the predictive path would fire on almost every poll.
- **Fix:**
  1. Add `COPY ml/model/spot_calibrator.joblib /app/model/spot_calibrator.joblib`.
  2. Make `app.py` fail startup (or `/health` returns 503) if no calibrator is found, instead of silently degrading. Keep an explicit env flag to allow uncalibrated mode for tests.
  3. Add the `calibrated: true/false` field to the `/predict` response.
  4. Rebuild and push the image, then check a live `/predict` returns values near 0.001, not 0.05.

### 1.3 Risk threshold is inconsistent across the repo  [A+B, LOW]
- **Where:** `demo/spotresilientjob.yaml` (0.65), `helm/argus/values.yaml` (0.0015), `handlers.py` default (0.0015), README diagram (0.65), paper text.
- **Problem:** 0.65 never fires against the real model (max calibrated score about 0.0018).
- **Fix:** Pick one real-model threshold and one mock-demo threshold. Put the mock threshold only in `demo/mock-predict.yaml` flows. Update the README diagram, `demo/spotresilientjob.yaml`, and docs so no real-model path uses 0.65. Better long term: express the threshold as a percentile or as a target alarm rate rather than a raw score.

### 1.4 Smaller operator issues  [A, LOW-MED]
- Cordoned nodes are never uncordoned (`scheduler.uncordon_node` exists but is unused). Uncordon after the replacement pod is Running, or use drain semantics instead of manual cordon.
- `Failed` is terminal with no retry or backoff. Add a retry counter and requeue.
- A bare Pod is deleted and nothing recreates it. Either have the operator create the pod from the CRD spec (preferred, makes the CRD real) or document clearly that a controller must exist.
- `controller/checkpoint.py` is dead code and imports from `handlers` (circular risk). Delete it or wire it in.
- `lastCheckpointStep` is just a counter, not a step. Rename it or store the real step from the checkpoint.
- The `spec.command`, `spec.checkpointIntervalSteps`, `spec.image` fields in the CRD are logged but never used. Either use them or remove them from the CRD.
- Mojibake in strings (`â€”` in `handlers.py` line ~99 and `main.py` logging format). Replace with plain `-`.
- Hardcoded `az="eu-north-1a"` in the predict call. Read it from the node label or the CRD.
- `httpx` failure returns silently. Add a metric for predict-service failures so a dead predictor is visible in Grafana.

### 1.5 Training job checkpoint robustness  [B, LOW-MED]
- **Where:** `ml/cifar10_job/train.py`.
- Checkpoints are only saved per epoch or on trigger. The SIGTERM handler saves `current_epoch`, which marks a half-finished epoch as complete on resume (it resumes at epoch+1, skipping the rest of the epoch). Save batch index too, or save to a "partial" key and resume at the same epoch.
- `upload_file` failure is swallowed. If the upload fails the job continues as if it succeeded. Retry and surface failure.
- Write the checkpoint to a temp key and rename, so a kill mid-upload cannot leave a corrupt `latest_checkpoint.pt`.
- `torch.load` without `weights_only=True`.
- The signal handler does an S3 upload inside the handler. Safer to set a flag and let the main loop checkpoint.

### 1.6 Lambda and data pipeline  [B, MED]
- `lambda/price_collector/handler.py` collects `m5.large, m5.xlarge, c5.large, c5.xlarge, t3.medium, t3.large`, but `feature_pipeline.py` `OD_PRICES` covers `m5.xlarge, m5.2xlarge, c5.xlarge, c5.2xlarge, g4dn.xlarge`. Mismatched sets mean unknown types get `od_price=1.0` (silent wrong feature). Align them or fail loudly.
- `feature_pipeline.py` writes a CSV the Lambda outputs do not feed directly (the Lambda writes `az`, the pipeline expects `availability_zone`). Confirm the real path from Lambda output to training data, and write one script that does it end to end.
- The API serves from a static `features.csv` baked into the image. It is not live. Either read recent features from S3 at request time (real system) or state clearly in the paper that serving uses a frozen history.
- `app.py` uses the deprecated `@app.on_event("startup")`. Switch to lifespan.

### 1.7 CI/CD  [A, LOW]
- `.github/workflows/deploy.yml` only triggers on pushes to `main`. Your working branches are `master` and `week-*`. Decide the real default branch and fix the trigger.
- Uses long-lived `AWS_ACCESS_KEY_ID` secrets. Switch to GitHub OIDC role assumption.
- No tests run in CI. Add at least: operator unit tests (mock S3 and k8s), model contract test (metadata matches `feature_config.py`), harness smoke test.

---

## Part 2. Make the benchmark actually test prediction (the core reviewer complaint)

Decision to make first with Angshuman: **Path A (trace-driven, model in the loop)**, per the v2 plan. Items below assume Path A.

### 2.1 Replace the oracle trigger with real model output  [A harness + B model, HIGH] CENTERPIECE
- **Problem:** The "predictive" arm fires at T minus lead, where T is the harness's own future kill time (`harness.py::run_cycle`). That is perfect foresight, zero false positives, zero misses. No learned signal exists in Poisson kills.
- **Fix:**
  1. Build a trace replayer: take real historical Spot price series (`ml/data/raw_spot_prices.csv`, plus more history if obtainable) and derive kill events from them, so the model's price features can correlate with kills.
  2. Because real reclaim labels are not available (AWS IAM restricted, FIS blocked), define the kill rule explicitly and honestly. Options: (a) kills at price-spike events (matches the training label, but circular, say so); (b) kills sampled with probability tied to the Spot Advisor tier of the instance type, with timing from price events. Be explicit in the paper about which one and why.
  3. Step the harness through trace time. At each step, call the real model (`/predict` or in-process) on the features up to that step. Trigger a checkpoint/migration when the score crosses the threshold. No access to future T.
  4. Report per arm: wasted compute, makespan, checkpoint count, completion, plus **detection stats**: recall, false alarms per hour, actual lead time achieved.
  5. Sweep the model threshold (reuse `chaos_threshold_sweep.py` ideas) so the arm moves along its real precision/recall curve.
- **Done when:** predictive wastes some compute (misses happen), has false-alarm overhead, and the numbers differ from the oracle. That is the honest result.

### 2.2 Add a non-learned predictive baseline  [A arm + B tier logic, MED]
- Add an arm that checkpoints/migrates when the instance type's Spot Advisor interruption tier is above a cutoff, or when a simple rule fires (price above rolling mean plus k std). `ml/data/pull_spot_advisor.py` already fetches the data.
- Purpose: show whether the Transformer beats a trivial rule. If it does not, say so.

### 2.3 Re-run the 120 s condition properly  [A, LOW-MED]
- Use a tuned lead (from the sweep) and enough wall-clock budget so predictive completes 5 of 5. Kill the single-run headline in Table 1 (makespan from 1 sample).
- Increase reps (10+) where variance is large. Report mean and a confidence interval, not just std.
- Sweep the lead-time experiment across several interruption rates, not just 20 s. The doc admits the "knee equals the interval" relation was only seen at one rate.

### 2.4 Make the benchmark match the real operator  [A, MED]
- After fixing 1.1, make the harness checkpoint through the same wait-for-confirmation logic as the operator, or run the benchmark against the real operator code in minikube for at least one condition. Otherwise the paper's benchmark and the real system are different systems.
- Add a checkpoint-size knob: measure real write time for larger state (for example a 100 MB to 1 GB model, optimizer state included) and show where the "lead just above write time" guideline moves. This answers reviewer concern 5.
- Add a variable checkpoint-write-time to the sweep so the guideline is not based on a near-instant write.

### 2.5 Stress test doc is self-contradictory  [A, LOW]
- `docs/week8_stress_test.md` claims a ~5x wall-clock speedup "close to the theoretical ceiling" for 3 jobs. The ceiling is 3x. The 5x comes from a sequential control in which one job timed out.
- Fix: rerun with more jobs and reps, compare equal-completion runs, or drop the speedup claim. Do not cite it in the paper as is.

### 2.6 Real-EKS evidence  [A, MED-HIGH]
- Objective 1 is one run, injected warning, reactive path only (NTH plus the SIGTERM handler did the work). Argus's predictor and operator were not involved.
- Fix: capture at least one real-EKS run where the operator's **predictive** path triggers (using a controlled score injection through the real operator) and the pod checkpoints and resumes. Repeat 3+ times and report resume time and lost work each time.
- If FIS gets unblocked, do one AWS-issued reclaim. Otherwise keep the honest "injected" wording.
- Optional, high value: one small multi-node run to answer the "single node, small model" criticism.

---

## Part 3. The model (honest evaluation)

### 3.1 Evaluate it as a predictor  [B, LOW]
- Put AUROC, PR-AUC, precision, recall, F1 at the chosen deployed threshold into the paper, with the PR curve (`docs/figures/pr_curve.png`). Data is already in `docs/week7_model_evaluation.md`.
- State plainly that the old 0.65 default never fires and that the deployed threshold is about 0.0015.
- Report 5-seed mean, not the shipped checkpoint's lucky 0.0480. Make the shipped model a real seed-averaged or median model, or document that it is the best of the seeds and that this is a selection effect.

### 3.2 Re-test the rejected features properly  [B, MED]
- `az_price_divergence` and `instance_interruption_rate` were rejected on single runs. Run `multi_seed_eval.py` with each feature (5+ seeds) and compare with a paired test before ruling them out. The interruption-rate feature is a constant per instance type, so a Transformer over time windows may not use it well. Consider feeding it as a static input.

### 3.3 Better labels  [B, HIGH, optional]
- Use any real interruption signal reachable without admin IAM: the Spot Advisor bucket, public Spot interruption datasets, or the user's own forward-captured NTH/EventBridge events (the paper lists "forward label capture" as future work, so start it now so there is data later).
- If real labels remain out of reach, say so as scope and keep the model labeled advisory.

### 3.4 Model engineering  [B, LOW]
- `FEATURE_COLUMNS` vs `model_metadata.json` is gitignored, so image builds depend on untracked local files. Track a small `model_metadata.json` (it has no secrets) or generate it in CI.
- Decide where model artifacts live (S3 with a version tag) and have the image or pod fetch them, rather than baking local files.

---

## Part 4. The paper (`neurips_2026.tex` and the markdown drafts)

Do this only after Parts 1 to 3 produce final numbers. Every number must be copied from the source CSV, never from an earlier draft.

### 4.1 Reframe the title and contributions  [A+B, MED]
- Current title promises that prediction beats checkpointing. The data says it mostly does not (1 of 5 completion at 120 s, periodic wins makespan). After Part 2, the title should state what the evidence supports. Two honest options:
  - If Part 2 shows the real model helps: "...Trace-Driven Evaluation of Predictive Checkpointing on Spot Instances".
  - If it does not: "When Does Predicting Spot Interruptions Pay? A Negative Result and a Lead-Time Guideline".
- Rewrite the three contributions to match. Contribution 1 (real EKS) should be described as a feasibility check, not a headline result, unless 2.6 strengthens it.

### 4.2 Abstract  [A, LOW]
- Rewrite in plain voice. Remove "Alongside, we further find that...". State the question, method, two or three numbers, and the limit. No fabricated or unverified headline numbers.

### 4.3 Introduction and motivation  [A, LOW-MED]
- Add the missing argument: why predict instead of just making checkpointing fast enough to fit inside the 2-minute notice? (reviewer concern 6). Give a concrete case: checkpoint time greater than 120 s at scale, or multi-node coordination.
- The 16-node A100 cost example uses assumed prices. Mark them as illustrative with a source, and show the arithmetic in a footnote.
- Keep the P(>=1)=1-(1-p)^N point; it was praised.

### 4.4 Section 3.2 (when does prediction pay)  [A+B, MED]
- Replace Table 1 with the new trace-driven results (2.1 to 2.3). Show all four rates in the main text, not only the fastest, with confidence intervals.
- Remove statements that rely on a single completed run. Do not report recovery time for the predictive arm unless n is adequate. The old "5.4 s / 12x faster recovery" is wrong (the table says 2.6 s; 5.4 is the checkpoint count in the lead sweep). Never reintroduce it.

### 4.5 Section 3.3 (lead time)  [A, MED]
- Remove or correct "the operator checkpoints synchronously before migrating" until 1.1 is fixed and true.
- Add the checkpoint-size/write-time experiment (2.4) so the guideline is not based on an instant write.
- Add the multi-rate sweep (2.3).
- Say clearly what the oracle-lead sweep does and does not show, if you keep it as a supplementary result.

### 4.6 Section 3.4 (risk model)  [B, MED]
- Expand from a paragraph to a real section with the PR curve, precision/recall at the deployed threshold, the Spot-Advisor tier baseline (2.2), and the seed variance. Say the label is a proxy and what that implies.
- Rewrite in human voice. No em or en dashes, no colon or semicolon flourishes. Use literal "Section X.Y.Z", not `\S\ref{}` macros (your LaTeX style rules).

### 4.7 Limitations and scope  [A+B, LOW]
- Keep them honest. Add: single node, synthetic job, proxy label, injected reclaim, small sample sizes. State scope, not apology.

### 4.8 Related work  [A+B, MED]
- Currently one dense paragraph. Add the closest work on predictive or proactive checkpointing for preemptible instances, Spot-aware scheduling (for example systems that use price or interruption forecasts), and checkpoint-interval optimization theory (Young/Daly). The periodic baseline should be compared with an optimal-interval baseline, not just 20 steps.

### 4.9 Baselines  [A, LOW-MED]
- Add Young/Daly optimal periodic interval as a stronger periodic baseline. Currently periodic is fixed at 20 steps, which is arbitrary.

### 4.10 Figures and tables  [A, LOW]
- Keep figure text legible at print size (ZNZL's note). Regenerate all figures from the final CSVs. Remove the pooled wasted-compute figure if it hides per-rate differences.
- Table 1 mixes a completion fraction with means taken over fewer runs. Add an `n` column.

### 4.11 Anonymity and format  [A, LOW]
- The tex uses `[preprint]` and has author names. For an anonymous submission, use the review option and remove author info and any identifying URLs. Complete `checklist.tex` honestly.

### 4.12 Human-voice pass  [both, LOW-MED]
- Both authors read each other's sections. Remove phrases flagged as generated: "we consolidate the paper's disclosures here rather than...", "a regime not exotic for...", stacked hedges, and filler transitions. Plain sentences, concrete nouns.

### 4.13 Repo/paper consistency  [both, LOW]
- Fix the README headline table and poster numbers (`README.md` lines ~99-108, `docs/poster_blueprint.md` line ~90). Replace "5.4 s" and "12x faster recovery".
- README says "Status: complete" and "Validated end-to-end on real AWS EKS". Soften to what is true.
- Keep one canonical results file (`benchmark/results/summary_table.csv`) and generate paper tables from it with a script, so the paper cannot drift from the data.

---

## Part 5. Repo hygiene

- `terraform.exe` (90 MB) is committed and bloats the repo (pack is about 189 MB). Remove it from the tree, add `terraform.exe` to `.gitignore`, and if needed purge it from history (this rewrites history, so agree first).
- `LICENSE` and `LICENSE.txt` differ. Keep one.
- `ml/model/mlruns.db` is committed. Ignore it.
- `.terraform.lock.hcl` is both tracked and listed in `.gitignore`. HashiCorp recommends committing the lock file; remove it from `.gitignore`.
- `.gitignore` has garbled bytes (`???`, BOM). Re-save as UTF-8 without BOM.
- Branch confusion: local `master` tracks `origin/week6-real-services`. Rename or re-point branches so `master` means the same thing locally and on GitHub.
- Add a top-level `Makefile` or `scripts/` target for: run tests, run quick benchmark, rebuild all tables and figures from raw data.
- Add unit tests: operator reconcile (mock S3, k8s, predict service), feature pipeline, `temporal_split_indices` no-leak test, `app.py` contract test.
- Terraform: the operator role has a placeholder `ec2.amazonaws.com` trust policy in `iam.tf`. Confirm what is actually applied (IRSA role is separate); remove the unused placeholder to avoid confusion. Document the IAM user `person_b` access scope.
- Cost safety: keep `scripts/verify_teardown.sh` as the last step of every session.

---

## Suggested order and owners

| Step | Items | Owner | Effort |
|---|---|---|---|
| 1 | Decide Path A vs B | both | meeting |
| 2 | 1.2 calibrator in image, 1.3 threshold, 4.13 README numbers | B / A | LOW |
| 3 | 1.1 operator waits for checkpoint, 1.5 training job robustness | A / B | MED |
| 4 | 3.1 model-as-predictor numbers, 2.5 stress test fix | B / A | LOW |
| 5 | 2.1 to 2.4 trace-driven harness, baselines, reruns | A + B | HIGH |
| 6 | 2.6 real-EKS predictive-path run(s) | A | MED-HIGH |
| 7 | 3.2 to 3.4 model follow-ups | B | MED |
| 8 | Part 4 paper rewrite from final numbers | both | MED-HIGH |
| 9 | Part 5 hygiene, tests, CI | A | MED |
| 10 | Human-voice pass, post arXiv v2, pick resubmission venue | both | LOW |

## Definition of done

- Every claim in the paper is backed by a number generated from a file in the repo.
- The real operator and the benchmark do the same thing (checkpoint is confirmed before migration).
- The predictive arm uses the real model, with false alarms and misses visible in the results.
- The README, poster, and paper agree with `summary_table.csv`.
- A reviewer who reads only the abstract and Table 1 would not be able to say "the predictor was never tested".
