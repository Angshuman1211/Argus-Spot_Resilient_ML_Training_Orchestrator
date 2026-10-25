"""
Wide Spot price pull: many regions and instance types, plus On-Demand prices.

The single-region, 5-type dataset (about 300 positive windows) is the main limit on the
risk model. Spot price history is public and free through describe_spot_price_history, so
more regions and families mean more series and more spike events to learn from.

On-Demand prices come from the AWS Pricing API so normalized_ratio stays correct for the
new types (the old pipeline silently used a made-up default for unknown types).

    AWS_PROFILE=argus-demo python ml/data/fetch_spot_prices_wide.py
Output (gitignored, ml/data/*.csv):
    ml/data/raw_spot_prices_wide.csv   region, timestamp, instance_type, availability_zone, spot_price
    ml/data/od_prices_wide.csv         region, instance_type, od_price
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))

REGIONS = {
    "eu-north-1": "EU (Stockholm)",
    "eu-west-1": "EU (Ireland)",
    "eu-central-1": "EU (Frankfurt)",
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-2": "US West (Oregon)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
}

INSTANCE_TYPES = [
    "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge",
    "m6i.large", "m6i.xlarge", "m6i.2xlarge", "m7i.xlarge",
    "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge",
    "c6i.large", "c6i.xlarge", "c6i.2xlarge", "c7i.xlarge",
    "r5.large", "r5.xlarge", "r5.2xlarge", "r6i.large", "r6i.xlarge",
    "t3.medium", "t3.large", "t3.xlarge",
    "g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g5.2xlarge", "p3.2xlarge", "inf1.xlarge",
]

CFG = Config(retries={"max_attempts": 8, "mode": "adaptive"})


def fetch_prices(region, days=90):
    ec2 = boto3.client("ec2", region_name=region, config=CFG)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows = []
    pages = ec2.get_paginator("describe_spot_price_history").paginate(
        InstanceTypes=INSTANCE_TYPES, ProductDescriptions=["Linux/UNIX"], StartTime=start, EndTime=end)
    for page in pages:
        for r in page["SpotPriceHistory"]:
            rows.append((region, r["Timestamp"], r["InstanceType"], r["AvailabilityZone"], float(r["SpotPrice"])))
    return rows


def fetch_od_prices(region, location):
    pricing = boto3.client("pricing", region_name="us-east-1", config=CFG)
    out = {}
    for itype in INSTANCE_TYPES:
        resp = pricing.get_products(ServiceCode="AmazonEC2", MaxResults=10, Filters=[
            {"Type": "TERM_MATCH", "Field": f, "Value": v} for f, v in (
                ("location", location), ("instanceType", itype), ("operatingSystem", "Linux"),
                ("tenancy", "Shared"), ("preInstalledSw", "NA"), ("capacitystatus", "Used"))])
        for item in resp["PriceList"]:
            terms = json.loads(item)["terms"].get("OnDemand", {})
            for term in terms.values():
                for dim in term["priceDimensions"].values():
                    usd = float(dim["pricePerUnit"].get("USD", 0))
                    if usd > 0:
                        out[itype] = usd
    return out


def main():
    prices, od_rows = [], []
    for region, location in REGIONS.items():
        t0 = time.time()
        try:
            rows = fetch_prices(region)
            od = fetch_od_prices(region, location)
        except Exception as e:  # keep going: one denied or throttled region should not lose the rest
            print(f"{region}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            continue
        prices.extend(rows)
        od_rows.extend((region, k, v) for k, v in od.items())
        print(f"{region}: {len(rows):>7} price records, {len(od)} On-Demand prices, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(prices, columns=["region", "timestamp", "instance_type", "availability_zone", "spot_price"])
    df = df.sort_values(["region", "instance_type", "availability_zone", "timestamp"]).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, "raw_spot_prices_wide.csv"), index=False)
    pd.DataFrame(od_rows, columns=["region", "instance_type", "od_price"]).to_csv(
        os.path.join(HERE, "od_prices_wide.csv"), index=False)
    n_series = df.groupby(["region", "instance_type", "availability_zone"]).ngroups
    print(f"\n{len(df)} records, {n_series} series, {df.timestamp.min()} -> {df.timestamp.max()}")


if __name__ == "__main__":
    main()
