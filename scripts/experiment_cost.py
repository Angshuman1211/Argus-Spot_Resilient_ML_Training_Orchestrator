"""
What the real-reclaim sensor fleet cost (a minor point of the 26 Oct review: state the cost of the experiments).

Instance-hours come from the launch logs and reclaim times, exactly as evaluate_real_reclaims.py builds them
(an instance ends at its reclaim or at its 2-hour self-termination cap). Spot prices for these small types are
not stored, so the cost is bounded from above with us-east-1 On-Demand list prices; Spot is cheaper. The EKS
cluster and the cloud runner are not included: their bills were on the suspended account and cannot be read.

    python scripts/experiment_cost.py
Output: ml/data/real/sensor_cost.json
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL = os.path.join(ROOT, "ml", "data", "real")
MAX_LIFE = pd.Timedelta(minutes=120)
# us-east-1 Linux On-Demand list prices, USD per hour
OD = {"c5.large": 0.085, "m5.large": 0.096, "t3.medium": 0.0416, "t3.large": 0.0832, "c6i.large": 0.085,
      "m6i.large": 0.096, "r5.large": 0.126, "r6i.large": 0.126, "c5.xlarge": 0.17, "m5.xlarge": 0.192}


def main():
    rows = []
    for f in glob.glob(os.path.join(REAL, "sensor_campaign_*.csv")) + glob.glob(os.path.join(REAL, "sensor_launches_*.csv")):
        d = pd.read_csv(f)
        d = d[d.status == "launched"]
        col = "event_time_utc" if "event_time_utc" in d else "launched_at_utc"
        rows.append(d[["instance_id", "instance_type", "availability_zone", col]].rename(columns={col: "launched"}))
    inst = pd.concat(rows, ignore_index=True).drop_duplicates("instance_id")
    inst["launched"] = pd.to_datetime(inst["launched"], utc=True, format="ISO8601")
    rec = pd.read_csv(os.path.join(REAL, "real_reclaims.csv"))
    rec["reclaim_time"] = pd.to_datetime(rec["reclaim_time"], utc=True, format="ISO8601")
    inst = inst.merge(rec[["instance_id", "reclaim_time"]], on="instance_id", how="left")
    inst["end"] = inst["reclaim_time"].fillna(inst["launched"] + MAX_LIFE)
    inst["hours"] = (inst["end"] - inst["launched"]).dt.total_seconds() / 3600.0
    missing = sorted(set(inst.instance_type) - set(OD))
    if missing:
        raise SystemExit(f"no list price for {missing}")
    inst["od_usd"] = inst.hours * inst.instance_type.map(OD)
    by_type = inst.groupby("instance_type").agg(instances=("instance_id", "size"), hours=("hours", "sum"),
                                                od_usd=("od_usd", "sum")).round(3)
    out = dict(instances=int(len(inst)), instance_hours=float(inst.hours.sum()),
               upper_bound_usd_on_demand=float(inst.od_usd.sum()),
               by_type=by_type.reset_index().to_dict(orient="records"),
               note="us-east-1 On-Demand list prices as an upper bound; Spot is cheaper. EKS cluster and cloud "
                    "runner not included (billed on the suspended account).")
    json.dump(out, open(os.path.join(REAL, "sensor_cost.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_type"}, indent=1))


if __name__ == "__main__":
    main()
