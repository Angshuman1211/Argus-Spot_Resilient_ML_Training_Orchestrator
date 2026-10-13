"""
Keeps a CSV of recent Spot price updates fresh in S3 for the prediction service's s3object source.

Run it where EC2 credentials exist (a laptop, a cron host). The service itself then needs only
s3:GetObject on one object, which the operator's existing IRSA role already allows for the checkpoints bucket.

    AWS_PROFILE=argus-demo python scripts/price_feeder.py --once
    AWS_PROFILE=argus-demo python scripts/price_feeder.py --every 300 --hours 4
"""
import argparse
import io
import os
import sys
import time

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "api"))
import price_source  # noqa: E402

TYPES = ["m5.xlarge", "m5.2xlarge", "c5.xlarge", "c5.2xlarge", "g4dn.xlarge"]


def publish(region, bucket, key):
    df = price_source.Ec2PriceSource(region).fetch(TYPES)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    boto3.client("s3", region_name=region).put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode(),
                                                      ContentType="text/csv")
    return len(df), df["timestamp"].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="eu-north-1")
    ap.add_argument("--bucket", default="argus-checkpoints-844641713781")
    ap.add_argument("--key", default="prices/latest.csv")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--hours", type=float, default=4.0)
    a = ap.parse_args()
    end = time.time() + a.hours * 3600
    while True:
        n, newest = publish(a.region, a.bucket, a.key)
        print(f"{time.strftime('%H:%M:%S')} published {n} records (newest update {newest}) to s3://{a.bucket}/{a.key}", flush=True)
        if a.once or time.time() + a.every > end:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
