#!/usr/bin/env python3
"""
Pull REAL Spot interruption-frequency labels from the AWS Spot Instance Advisor.

The price-spike proxy in dataset.py (>1% jump) is a weak label. This is AWS's own
published interruption frequency, per instance-type per region — a real target for
the "predict interruption risk" task (Objective 3).

Granularity note: this is an AGGREGATE frequency bucket per (region, os, type),
NOT a per-timestep event. So it labels *which types/regions are risky*, not
"will THIS instance be interrupted in the next 15 min." Use it either to reframe
the task as risk-tier classification, or as a strong prior/feature alongside the
price time-series. Coordinate with Person B on which.

Buckets (r): 0=<5%  1=5-10%  2=10-15%  3=15-20%  4=>20% interruption rate.

Usage:
  python pull_spot_advisor.py                      # -> spot_interruption_labels.csv
  python pull_spot_advisor.py --region eu-north-1  # filter one region
  python pull_spot_advisor.py --upload-s3          # also push raw JSON + CSV to the feature store
"""
import argparse
import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone

DATA_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"
FEATURE_BUCKET = __import__("os").environ.get("FEATURE_STORE_BUCKET", "")  # argus-feature-store-<account id>


def fetch():
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        return json.load(r)


def to_rows(data, region_filter=None, os_filter="Linux"):
    ranges = {b["index"]: b for b in data["ranges"]}          # bucket meta
    meta = data.get("instance_types", {})                      # cores / ram_gb
    rows = []
    for region, os_map in data["spot_advisor"].items():
        if region_filter and region != region_filter:
            continue
        for os_name, types in os_map.items():
            if os_filter and os_name != os_filter:
                continue
            for itype, v in types.items():
                b = ranges[v["r"]]
                m = meta.get(itype, {})
                rows.append({
                    "region": region,
                    "os": os_name,
                    "instance_type": itype,
                    "interruption_bucket": v["r"],           # 0..4 ordinal target
                    "interruption_label": b["label"],        # "<5%" .. ">20%"
                    "interruption_rate_max_pct": b["max"],   # upper bound of the bucket
                    "savings_pct": v.get("s"),               # % cheaper than on-demand
                    "vcpus": m.get("cores"),
                    "ram_gb": m.get("ram_gb"),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="filter to one region (default: all)")
    ap.add_argument("--os", default="Linux", help="Linux (default) or Windows")
    ap.add_argument("--out", default="spot_interruption_labels.csv")
    ap.add_argument("--upload-s3", action="store_true")
    args = ap.parse_args()

    data = fetch()
    rows = to_rows(data, args.region, args.os)
    if not rows:
        sys.exit(f"No rows (region={args.region}, os={args.os}).")

    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    dist = Counter(r["interruption_label"] for r in rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"Regions: {len({r['region'] for r in rows})}  OS: {args.os}  as-of: {datetime.now(timezone.utc).date()}")
    print("Interruption-bucket distribution:")
    for lbl in ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]:
        print(f"  {lbl:>7}: {dist.get(lbl, 0)}")

    if args.upload_s3:
        import boto3
        if not FEATURE_BUCKET:
            sys.exit("set FEATURE_STORE_BUCKET (argus-feature-store-<account id>) to upload")
        s3 = boto3.client("s3")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s3.upload_file(args.out, FEATURE_BUCKET, f"labels/spot_interruption_labels_{stamp}.csv")
        with open("/tmp/spot_advisor_raw.json", "w") as f:
            json.dump(data, f)
        s3.upload_file("/tmp/spot_advisor_raw.json", FEATURE_BUCKET, f"labels/spot_advisor_raw_{stamp}.json")
        print(f"Uploaded to s3://{FEATURE_BUCKET}/labels/ (dated {stamp})")


if __name__ == "__main__":
    main()
