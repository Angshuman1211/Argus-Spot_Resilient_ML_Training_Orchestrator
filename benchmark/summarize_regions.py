"""
Cross-region summary of the trace benchmark (light policy set), same protocol everywhere: the model's
alarm budget is chosen on the region's calibration slice, results are on its test slice, intervals are
paired and resampled by whole days (recompute_ci.py). Regions with very few events cannot separate
policies, so the event count is printed next to every result.

    python benchmark/summarize_regions.py
Output: benchmark/results/regions_summary.csv
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
REGIONS = ["eu-north-1", "eu-west-1", "eu-central-1", "us-east-1", "us-east-2", "us-west-2", "ap-south-1", "ap-southeast-1"]


def dirs(region):
    if region == "eu-north-1":
        return os.path.join(R, "trace_oct_new"), os.path.join(R, "trace_oct_new_calib")
    return os.path.join(R, f"region_{region}", "test"), os.path.join(R, f"region_{region}", "calib")


def main():
    rows = []
    for region in REGIONS:
        test_dir, calib_dir = dirs(region)
        subprocess.run([sys.executable, os.path.join(HERE, "recompute_ci.py"), "--test", test_dir, "--calib", calib_dir],
                       check=True, capture_output=True)
        ci = pd.read_csv(os.path.join(test_dir, "paired_day_ci.csv"))
        summ = pd.read_csv(os.path.join(test_dir, "summary.csv"))
        trace = os.path.join(HERE, "traces", "trace_scores_new.csv.gz" if region == "eu-north-1"
                             else f"trace_scores_{region}.csv.gz")
        t = pd.read_csv(trace, usecols=["split", "spike_now"])
        ev = int(t[t.split == "test"].spike_now.sum())  # spike events in the test slice, whole fleet
        for _, r in ci[ci.baseline == "young_daly"].iterrows():
            g = summ[(summ.N == r.N) & (summ.C_s == r.C_s)].set_index("policy")
            rows.append(dict(region=region, events=ev, N=int(r.N), C_s=r.C_s, model_policy=r.model_policy,
                             model=g.loc[r.model_policy].slowdown, young_daly=g.loc["young_daly"].slowdown,
                             reactive=g.loc["reactive"].slowdown, periodic_1h=g.loc["periodic_1h"].slowdown,
                             none=g.loc["none"].slowdown, diff_vs_young_daly=r.mean_diff, diff_lo=r.diff_lo,
                             diff_hi=r.diff_hi, share_days_model_better=r.share_of_days_model_better))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "regions_summary.csv"), index=False)
    show = out[out.N == 15].copy()
    pd.set_option("display.width", 250)
    print(show[["region", "events", "C_s", "model", "young_daly", "reactive", "none", "diff_vs_young_daly", "diff_lo",
                "diff_hi", "share_days_model_better"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
