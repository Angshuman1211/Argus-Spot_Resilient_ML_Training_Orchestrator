"""
Conservative intervals for the trace benchmark: paired, resampled by whole calendar day.

The first intervals resample blocks of overlapping job start times, which understates uncertainty
because neighbouring windows share the same price history and the same events. The unit of
independent evidence is closer to a day. Here, for each (fleet size, checkpoint cost):

  * the model policy chosen on the CALIBRATION slice is compared with Young-Daly and with the best
    non-learned baseline, run-by-run on the same start times (a paired difference);
  * days (not runs) are resampled with replacement (5,000 draws), pooled over every fleet of that size;
  * the output is the mean slowdown difference, its 95% interval, and the share of days the model won.

    python benchmark/recompute_ci.py --test results/trace_oct_new --calib results/trace_oct_new_calib
Output: <test dir>/paired_day_ci.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 300.0
BASELINES = ["young_daly", "reactive", "periodic_1h", "none"]


def pick_model_policy(calib_summary, N, C):
    c = calib_summary[(calib_summary.N == N) & (calib_summary.C_s == C) & calib_summary.policy.str.startswith("model_")]
    return c.sort_values("slowdown").iloc[0].policy


def day_bootstrap(diff, day, rng, n=5000):
    days = np.unique(day)
    by_day = {d: diff[day == d] for d in days}
    # weight each draw by run count so a day with more starts counts more, as in the pooled mean
    means = []
    for _ in range(n):
        pick = rng.choice(days, len(days), replace=True)
        means.append(np.concatenate([by_day[d] for d in pick]).mean())
    return float(np.mean(diff)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    test_dir = a.test if os.path.isabs(a.test) else os.path.join(HERE, a.test)
    calib_dir = a.calib if os.path.isabs(a.calib) else os.path.join(HERE, a.calib)
    runs = pd.read_csv(os.path.join(test_dir, "runs.csv.gz"))
    calib = pd.read_csv(os.path.join(calib_dir, "summary.csv"))
    runs = runs[runs.completed]
    runs["day"] = (runs["start"] * STEP // 86400).astype(int)
    rng = np.random.default_rng(a.seed)

    rows = []
    for (N, C), g in runs.groupby(["N", "C"]):
        model = pick_model_policy(calib, N, C)
        wide = g.pivot_table(index=["set", "start"], columns="policy", values="slowdown")
        day = (wide.index.get_level_values("start").to_numpy() * STEP // 86400).astype(int)
        for base in BASELINES:
            both = wide[[model, base]].dropna()
            d = day[wide.index.isin(both.index)]
            diff = (both[model] - both[base]).to_numpy()
            mean, lo, hi = day_bootstrap(diff, d, rng)
            win = float(np.mean([np.mean(diff[d == x] < 0) for x in np.unique(d)]))
            rows.append(dict(N=N, C_s=C, model_policy=model, baseline=base, days=len(np.unique(d)), runs=len(diff),
                             model_slowdown=both[model].mean(), baseline_slowdown=both[base].mean(),
                             mean_diff=mean, diff_lo=lo, diff_hi=hi, share_of_days_model_better=win,
                             model_significantly_better=hi < 0, model_significantly_worse=lo > 0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(test_dir, "paired_day_ci.csv"), index=False)
    cols = ["N", "C_s", "model_policy", "baseline", "days", "mean_diff", "diff_lo", "diff_hi", "share_of_days_model_better"]
    print(out[out.baseline == "young_daly"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
