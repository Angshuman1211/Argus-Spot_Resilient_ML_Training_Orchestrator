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


def main():
    rows = []
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
