"""
Feature builder for the wide Spot dataset (many regions, many instance types).

Same 13 base features as feature_pipeline.py, plus candidate extras that are tested by
multi-seed ablation in ml/model/wide_experiments.py (never assumed to help):

  az_price_divergence          price vs the mean of sibling AZs, same region and type
  instance_interruption_rate   AWS Spot Advisor max monthly interruption fraction (static)
  n_updates_1h                 how many price updates AWS published in the last hour
  mins_since_change            minutes since the last price update (capped at 24 h)
  price_vs_6h_max / _24h_min   price relative to its recent high and low

Output is a compact npz instead of a CSV (about 20M rows): one float32 matrix, a per
series offset table, the proxy spike label, and the absolute 5-minute grid index.

    python ml/data/feature_pipeline_wide.py
Output: ml/data/features_wide2.npz, ml/data/features_wide_meta.csv (both gitignored)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from spot_features import (  # noqa: E402
    ALL_FEATURES, BASE_FEATURES, EXTRA_FEATURES, MEMORY_FEATURES, STEP_S, series_features, to_epoch_s,
)


def advisor_rates(labels_dir):
    files = sorted(glob.glob(os.path.join(labels_dir, "spot_interruption_labels_*.csv")))
    lab = pd.read_csv(files[-1])
    lab = lab[lab["os"] == "Linux"].drop_duplicates(["region", "instance_type"])
    return {(r.region, r.instance_type): r.interruption_rate_max_pct / 100.0 for r in lab.itertuples()}


def main():
    raw = pd.read_csv(os.path.join(HERE, "raw_spot_prices_wide.csv"), parse_dates=["timestamp"])
    od = pd.read_csv(os.path.join(HERE, "od_prices_wide.csv"))
    od_map = {(r.region, r.instance_type): r.od_price for r in od.itertuples()}
    adv = advisor_rates(os.path.join(HERE, "labels"))
    raw["ts_s"] = to_epoch_s(raw["timestamp"])
    k_end = int(raw["ts_s"].max() // STEP_S)

    metas, mats, spikes, ks = [], [], [], []
    offset, skipped = 0, 0
    for (region, itype, az), g in raw.sort_values("timestamp").groupby(["region", "instance_type", "availability_zone"]):
        if (region, itype) not in od_map:
            skipped += 1
            continue
        k0, mat, spike = series_features(g["ts_s"].to_numpy(), g["spot_price"].to_numpy(), k_end,
                                         od_map[(region, itype)], adv.get((region, itype), 0.0))
        metas.append(dict(region=region, instance_type=itype, availability_zone=az, offset=offset,
                          length=len(mat), k0=k0, od_price=od_map[(region, itype)],
                          advisor_rate=adv.get((region, itype), np.nan)))
        mats.append(mat); spikes.append(spike)
        ks.append(np.arange(k0, k0 + len(mat), dtype=np.int32))
        offset += len(mat)
    meta = pd.DataFrame(metas)

    # cross-AZ divergence: compare each AZ's price with the mean over sibling AZs at the same step
    ai = ALL_FEATURES.index("az_price_divergence")
    pi = ALL_FEATURES.index("spot_price")
    for (region, itype), grp in meta.groupby(["region", "instance_type"]):
        if len(grp) < 2:
            continue
        k_lo = int(grp.k0.min())
        K = k_end - k_lo + 1
        stack = np.full((len(grp), K), np.nan, dtype=np.float64)
        for j, r in enumerate(grp.itertuples()):
            stack[j, r.k0 - k_lo:r.k0 - k_lo + r.length] = mats[meta.index.get_loc(r.Index)][:, pi]
        mean = np.nanmean(stack, axis=0)
        for j, r in enumerate(grp.itertuples()):
            m = mats[meta.index.get_loc(r.Index)]
            ref = mean[r.k0 - k_lo:r.k0 - k_lo + r.length]
            m[:, ai] = np.nan_to_num((m[:, pi] - ref) / np.where(ref == 0, np.nan, ref))

    X = np.concatenate(mats)
    np.savez(os.path.join(HERE, "features_wide2.npz"), X=X, spike=np.concatenate(spikes),
             k=np.concatenate(ks), features=np.array(ALL_FEATURES))
    meta.to_csv(os.path.join(HERE, "features_wide_meta.csv"), index=False)
    print(f"{len(meta)} series ({skipped} skipped, no On-Demand price), {len(X):,} rows, "
          f"{int(np.concatenate(spikes).sum()):,} spike steps, {X.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
