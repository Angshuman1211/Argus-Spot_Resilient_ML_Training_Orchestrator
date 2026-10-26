"""
Dollar cost of one 15-node, 12-hour job under each policy (fix F14), from the trace benchmark's runs.

  Spot compute    makespan x the fleet's mean Spot price on the test slice (as run_trace_benchmark.py)
  S3 requests     each checkpoint is a multipart upload of C x rate bytes in 8 MB parts (boto3's default),
                  one PUT per part, plus one GET per part per restore
  S3 storage      the latest checkpoint kept for the job's makespan
  transfer        EC2 to S3 in the same region is free
  On-Demand       the same 12 hours of work at On-Demand prices, with no interruptions

Prices are S3 Standard list prices (USD 0.023 per GB-month, 0.005 per 1,000 PUT, 0.0004 per 1,000 GET);
eu-north-1 differs by cents. The checkpoint size uses the measured in-region rate, so C = 300 s is 34 GB.

    python benchmark/cost_table.py
Output: results/trace_oct_new/cost.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
PUT, GET, GB_MONTH = 0.005 / 1000, 0.0004 / 1000, 0.023
PART_MB = 8.0
POLICIES = ["none", "reactive", "periodic_1h", "periodic_grid_15m", "periodic_30m", "young_daly",
            "model_migrate_q0.1", "hybrid_yd_model_migrate_q0.1", "oracle_migrate"]


def main():
    eks = pd.read_csv(os.path.join(R, "trace", "checkpoint_write_eks.csv"))
    big = eks[eks.size_mb >= 512]
    rate = float((big.size_mb / (big.upload_ms / 1000)).mean())
    runs = pd.concat([pd.read_csv(os.path.join(R, "trace_oct_new", f)) for f in ("runs.csv.gz", "runs_extra.csv.gz")])
    runs = runs[(runs.set == "N15") & runs.policy.isin(POLICIES)].drop_duplicates(["policy", "C", "start"])
    rows = []
    for (pol, C), g in runs.groupby(["policy", "C"]):
        done = g[g.completed]
        size_gb = C * rate / 1000.0
        parts = max(1.0, np.ceil(size_gb * 1000.0 / PART_MB))
        restores = done.kills.mean() + done.n_migr.mean()
        s3_req = done.n_ckpt.mean() * parts * PUT + restores * parts * GET
        s3_store = size_gb * GB_MONTH * (done.makespan.mean() / 3600.0) / 730.0
        spot = done.spot_cost.mean()
        od = g.od_cost.iloc[0]
        rows.append(dict(policy=pol, C_s=C, completion=g.completed.mean(), checkpoint_gb=size_gb,
                         checkpoints=done.n_ckpt.mean(), restores=restores, spot_compute_usd=spot,
                         s3_requests_usd=s3_req, s3_storage_usd=s3_store, total_usd=spot + s3_req + s3_store,
                         on_demand_usd=od, savings_vs_on_demand_pct=100.0 * (1 - (spot + s3_req + s3_store) / od)))
    out = pd.DataFrame(rows).sort_values(["C_s", "total_usd"])
    out.to_csv(os.path.join(R, "trace_oct_new", "cost.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
