"""
Measures real torch.save time against checkpoint size on this machine's disk, and maps
checkpoint size to the checkpoint cost C used by trace_sim under stated upload speeds.

The benchmark's C values (10, 60, 300 s) are not arbitrary: with the training state of
a model (weights plus optimizer state) and a sustained upload rate, C is size / rate.
Local disk time is measured here. S3 upload speed is NOT measured (no AWS access from
this environment), so the mapping uses stated rates and says so.

    python benchmark/measure_checkpoint_write.py
Output: benchmark/results/trace/checkpoint_write.csv
"""
import os
import tempfile
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "trace", "checkpoint_write.csv")

SIZES_MB = [4, 64, 256, 1024]
UPLOAD_MB_S = [100, 500, 1000]  # stated sustained upload rates, not measured
# training state bytes per parameter: fp32 weights (4) + Adam moments (8) = 12
BYTES_PER_PARAM = 12
MODELS = [("CIFAR CNN (this repo)", 0.3e6), ("ResNet-50", 25.6e6), ("BERT-large", 340e6),
          ("1.3B LLM", 1.3e9), ("7B LLM", 7e9)]


def measure(size_mb):
    n = int(size_mb * 1024 * 1024 / 4)
    state = {"w": torch.randn(n)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        t0 = time.perf_counter()
        torch.save(state, path)
        with open(path, "rb+") as f:
            os.fsync(f.fileno())
        dt = time.perf_counter() - t0
        return dt, os.path.getsize(path) / 1e6 / dt


def measure_s3(size_mb, bucket, region):
    """Upload `size_mb` of random bytes with boto3's multipart transfer, then delete the object.
    Network-bound: from a laptop this measures the laptop's uplink, not in-region EKS-to-S3 speed."""
    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3", region_name=region)
    key = f"tmp-bench/ckpt_{size_mb}mb_{int(time.time())}.bin"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blob.bin")
        with open(path, "wb") as f:
            remaining = size_mb * 1024 * 1024
            while remaining > 0:
                n = min(remaining, 64 * 1024 * 1024)
                f.write(os.urandom(n))
                remaining -= n
        cfg = TransferConfig(multipart_threshold=8 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024,
                             max_concurrency=16)
        try:
            t0 = time.perf_counter()
            s3.upload_file(path, bucket, key, Config=cfg)
            dt = time.perf_counter() - t0
            t1 = time.perf_counter()
            s3.download_file(bucket, key, os.path.join(d, "back.bin"), Config=cfg)
            dt_dl = time.perf_counter() - t1
        finally:
            s3.delete_object(Bucket=bucket, Key=key)
            # a failed multipart upload leaves billable parts behind: always abort them
            for u in s3.list_multipart_uploads(Bucket=bucket, Prefix=key).get("Uploads", []):
                s3.abort_multipart_upload(Bucket=bucket, Key=u["Key"], UploadId=u["UploadId"])
    return dt, size_mb / dt, dt_dl, size_mb / dt_dl


def main():
    rows = []
    if "--s3" in __import__("sys").argv:
        bucket = (os.environ.get("CHECKPOINT_BUCKET")
                  or "argus-checkpoints-" + __import__("boto3").client("sts").get_caller_identity()["Account"])
        region = os.environ.get("AWS_REGION", "eu-north-1")
        for mb in (4, 64, 256):
            up, up_rate, dn, dn_rate = measure_s3(mb, bucket, region)
            rows.append(dict(kind="measured_s3_from_this_machine", size_mb=mb, seconds=round(up, 2),
                             mb_per_s=round(up_rate, 1), download_seconds=round(dn, 2),
                             download_mb_per_s=round(dn_rate, 1)))
            print(f"S3 {mb:>5} MB  upload {up:6.2f} s ({up_rate:6.1f} MB/s)  download {dn:6.2f} s ({dn_rate:6.1f} MB/s)")
        s3_out = os.path.join(os.path.dirname(OUT), "checkpoint_write_s3.csv")
        pd.DataFrame(rows).to_csv(s3_out, index=False)
        print(f"wrote {s3_out}")
        return
    for mb in SIZES_MB:
        dt, mbps = measure(mb)
        rows.append(dict(kind="measured_local_disk", size_mb=mb, seconds=round(dt, 3), mb_per_s=round(mbps, 1)))
        print(f"{mb:>5} MB  {dt:6.2f} s  {mbps:7.1f} MB/s")
    for name, params in MODELS:
        size_mb = params * BYTES_PER_PARAM / 1e6
        for rate in UPLOAD_MB_S:
            rows.append(dict(kind="estimated_upload", model=name, size_mb=round(size_mb, 1),
                             upload_mb_s=rate, seconds=round(size_mb / rate, 1)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
