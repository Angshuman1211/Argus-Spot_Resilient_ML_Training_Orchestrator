# Week 8 — Stress Test: 3 Concurrent SpotResilientJobs

Reproduce with:
```bash
python benchmark/stress_test.py --rate 300 --step-budget 500 --step-time-sec 0.3
```

## Method

Launches 3 `predictive`-arm jobs (real subprocesses, the same mechanism
`benchmark/harness.py` uses for the Objective 2 sweep) **at once**, via a
`ThreadPoolExecutor`, each into its own run directory so checkpoint/trigger
files never collide. Compared against the same 3 jobs run **sequentially**
(one at a time) at the same rate (300s MTBF — the rate where the Week 7
sweep showed 100% completion across 5 reps) to isolate whether concurrency
itself — CPU contention, checkpoint I/O contention — degrades behavior.

## Result

| | Concurrent (3 at once) | Sequential (one at a time) |
|---|---|---|
| Completed | **3/3** | 2/3 (1 timed out) |
| Wall clock | 278.7s | 1391.0s |
| Mean makespan (completed jobs) | 243.8s | 232.2s |
| Total wasted compute | 0.0s | 0.0s |

**Do not quote a speedup from this table.** Wall clock was 278.7 s concurrent vs 1391.0 s sequential
(about 5x), but 3 parallel jobs cap the speedup at 3x, and the sequential run includes one job that timed out
and sat until its wall-clock deadline. The 5x is that timeout, not parallelism. With n=3 per condition this test
supports only the weaker statements below.

**Per-job makespan increased about 5% under concurrency** (243.8 s vs 232.2 s), a small CPU-contention cost
on 2 completed sequential jobs against 3 concurrent ones. Not a meaningful degradation, and not a tight estimate.

**No cross-job interference of any kind**: every job's checkpoint count,
trigger files, and completion were independent — nothing suggests one job's
migration ever touched another's state. All 3 concurrent jobs finished with
**zero wasted compute each**, matching the single-job predictive behavior
already documented for this rate in `docs/objective2_result.md`.

## The one failure — and why it isn't a concurrency bug

One job **in the sequential control run** (not the concurrent run) timed
out — 1 checkpoint logged, then no further progress within the wall-clock
deadline. This is the predictive arm's own known run-to-run variance
(`docs/objective3_result.md`, `docs/objective2_result.md`'s Week 7 section
document this repeatedly at other rates) showing up in a fresh random
sample, not something concurrency caused — it happened in the run with
*less* resource contention, and the 3 concurrent jobs (more contention, if
anything) all succeeded. With only 3 sequential trials in this sample,
one Poisson-schedule outlier landing badly is expected, not evidence of a
new defect.

## Honest scope

- Synthetic benchmark job (`benchmark/job.py`), not real EKS pods — same
  scope as the rest of the Objective 2 benchmark (mechanics, not a live
  3-pod EKS deployment). A real-cluster version of this test (3 actual
  `SpotResilientJob` CRs on a live or minikube cluster) is a reasonable
  next step but wasn't attempted here — starting Docker Desktop's
  Kubernetes engine and deploying the full operator + predict-service
  stack is a bigger, riskier lift than extending the already-validated
  local harness, so this stayed in the same controlled environment
  Objective 2 used.
- n=3 per condition is small — the ~5% contention overhead and the single
  timeout are both consistent with, not proof against, larger-sample
  behavior. Treat this as a real signal at this sample size, not a tight
  confidence interval.
