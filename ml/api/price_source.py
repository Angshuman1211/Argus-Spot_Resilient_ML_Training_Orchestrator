"""
Where the prediction service gets recent Spot price updates from.

The old service answered from a features.csv baked into the image (Jan-Apr data), so its
"live" risk score was frozen in time. Every source here returns the same thing: a DataFrame of
raw price-update records [timestamp, instance_type, availability_zone, spot_price], enough
history for the model's longest feature (72 hours) plus its 2-hour window.

  ec2  describe_spot_price_history. One paginated call, exact update records.
       Needs ec2:DescribeSpotPriceHistory on the pod's role.
  s3   the Lambda price collector's snapshots (raw/YYYY/MM/DD/HH/*.csv). Each snapshot lists the
       latest update per series, so deduplicating (timestamp, type, AZ) across snapshots
       reconstructs every update. Needs s3:GetObject and s3:ListBucket on the feature store.
  s3object  one CSV object kept fresh by scripts/price_feeder.py (needs only s3:GetObject on it).
  csv  a static file, for local runs and tests.
"""
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

COLUMNS = ["timestamp", "instance_type", "availability_zone", "spot_price"]
HISTORY_DAYS = 5  # model needs 72 h of lookback + a 2 h window; keep a safety margin


def _normalize(df):
    df = df.rename(columns={"az": "availability_zone"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["spot_price"] = df["spot_price"].astype(float)
    return (df[COLUMNS].drop_duplicates(["timestamp", "instance_type", "availability_zone"])
            .sort_values("timestamp").reset_index(drop=True))


class Ec2PriceSource:
    name = "ec2"

    def __init__(self, region, client=None):
        import boto3
        self.region = region
        self.ec2 = client or boto3.client("ec2", region_name=region)

    def fetch(self, instance_types):
        end = datetime.now(timezone.utc)
        pages = self.ec2.get_paginator("describe_spot_price_history").paginate(
            InstanceTypes=list(instance_types), ProductDescriptions=["Linux/UNIX"],
            StartTime=end - timedelta(days=HISTORY_DAYS), EndTime=end)
        rows = [(r["Timestamp"], r["InstanceType"], r["AvailabilityZone"], r["SpotPrice"])
                for p in pages for r in p["SpotPriceHistory"]]
        return _normalize(pd.DataFrame(rows, columns=COLUMNS))


class S3PriceSource:
    name = "s3"

    def __init__(self, bucket, client=None, prefix="raw/", workers=16):
        import boto3
        self.bucket, self.prefix, self.workers = bucket, prefix, workers
        self.s3 = client or boto3.client("s3")
        self._cache = {}  # key -> DataFrame, snapshots never change once written

    def _keys(self, now):
        keys = []
        for d in range(HISTORY_DAYS + 1):
            day = now - timedelta(days=d)
            pre = f"{self.prefix}{day:%Y/%m/%d}/"
            for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=pre):
                keys += [o["Key"] for o in page.get("Contents", [])]
        return keys

    def _read(self, key):
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(body))

    def fetch(self, instance_types):
        now = datetime.now(timezone.utc)
        keys = [k for k in self._keys(now) if k not in self._cache]
        with ThreadPoolExecutor(self.workers) as pool:
            for k, df in zip(keys, pool.map(self._read, keys)):
                self._cache[k] = df
        horizon = now - timedelta(days=HISTORY_DAYS + 1)
        self._cache = {k: v for k, v in self._cache.items() if k >= f"{self.prefix}{horizon:%Y/%m/%d}"}
        df = _normalize(pd.concat(self._cache.values(), ignore_index=True))
        return df[df["instance_type"].isin(set(instance_types))]


class S3ObjectPriceSource:
    """A single CSV object that something else keeps fresh (see scripts/price_feeder.py).

    For clusters whose pod role can read one bucket but not call the EC2 API or read the Lambda
    collector's bucket, and where adding IAM permissions is not an option."""
    name = "s3object"

    def __init__(self, uri, client=None):
        import boto3
        assert uri.startswith("s3://"), uri
        self.bucket, self.key = uri[5:].split("/", 1)
        self.s3 = client or boto3.client("s3")

    def fetch(self, instance_types):
        body = self.s3.get_object(Bucket=self.bucket, Key=self.key)["Body"].read()
        df = _normalize(pd.read_csv(io.BytesIO(body)))
        return df[df["instance_type"].isin(set(instance_types))]


class CsvPriceSource:
    name = "csv"

    def __init__(self, path):
        self.path = path

    def fetch(self, instance_types):
        df = pd.read_csv(self.path)
        if "availability_zone" not in df.columns and "az" not in df.columns:
            raise ValueError(f"{self.path} has no availability zone column")
        return _normalize(df[df["instance_type"].isin(set(instance_types))].copy())


class CachedSource:
    """Wraps a source with a time-to-live so request bursts do not hammer the upstream."""

    def __init__(self, source, ttl_s=300):
        self.source, self.ttl_s = source, ttl_s
        self.name = source.name
        self._df, self._at, self._lock = None, 0.0, threading.Lock()

    def get(self, instance_types):
        with self._lock:
            if self._df is None or time.time() - self._at > self.ttl_s:
                self._df = self.source.fetch(instance_types)
                self._at = time.time()
            return self._df, self._at


def from_env(region):
    kind = os.getenv("PRICE_SOURCE", "ec2")
    if kind == "ec2":
        return Ec2PriceSource(region)
    if kind == "s3":
        return S3PriceSource(os.environ["FEATURE_STORE_BUCKET"])
    if kind == "s3object":
        return S3ObjectPriceSource(os.environ["PRICE_S3_URI"])
    if kind == "csv":
        return CsvPriceSource(os.environ["PRICE_CSV"])
    raise ValueError(f"unknown PRICE_SOURCE {kind!r}")
