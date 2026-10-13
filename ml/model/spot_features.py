"""
Spot price features, shared by training and serving.

Single source of truth for how a (instance type, AZ) price history becomes model input, so the
two can never drift apart (train/serve skew was this project's most expensive recurring bug).
Everything here is causal: the features at 5-minute step t use only price updates up to and
including step t. tests/ml/test_spot_features.py enforces that by recomputing on truncated
histories.

Price updates arrive as (epoch seconds, price) records. AWS now publishes Spot prices about once
an hour, so most steps carry no update and the series is forward-filled.
"""
import numpy as np
import pandas as pd

STEP_S = 300
SEQ_LENGTH = 24
HORIZON = 3

def to_epoch_s(ts):
    """Datetimes -> integer epoch seconds, independent of the datetime64 unit (ns/us/s), which
    differs between pandas versions. `astype("int64") // 10**9` silently breaks on pandas 3."""
    ts = pd.to_datetime(ts, utc=True)
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")


BASE_FEATURES = [
    "spot_price", "price_delta",
    "rolling_mean_15m", "rolling_std_15m",
    "rolling_mean_1h", "rolling_std_1h",
    "rolling_mean_6h", "rolling_std_6h",
    "normalized_ratio",
    "sin_time_day", "cos_time_day",
    "sin_day_week", "cos_day_week",
]
EXTRA_FEATURES = [
    "az_price_divergence", "instance_interruption_rate", "n_updates_1h",
    "mins_since_change", "price_vs_6h_max", "price_vs_24h_min",
]
# Longer-memory features added after the first ablation showed that price position relative to
# its recent range and the update cadence carry most of the signal (docs/wide_model_result.md).
MEMORY_FEATURES = [
    "price_vs_24h_max", "price_vs_72h_min", "price_vs_72h_max", "range_pos_24h",
    "n_updates_24h", "n_up_moves_24h", "last_change_pct", "mins_since_up", "mins_since_spike",
]
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES + MEMORY_FEATURES


def series_features(ts_s, price, k_end, od_price, adv_rate):
    """ts_s: epoch seconds of raw price updates (sorted). Returns (k0, matrix, spike)."""
    k_raw = (ts_s // STEP_S).astype(np.int64)
    k0, n = int(k_raw[0]), int(k_end - k_raw[0] + 1)
    idx = k_raw - k0
    counts = np.bincount(idx, minlength=n).astype(np.float32)

    # price at each grid step = the last update at or before it (same as resample().ffill())
    last_upd = np.maximum.accumulate(np.where(counts > 0, np.arange(n), -1))
    # last update's price within each bin that has updates
    bin_price = np.full(n, np.nan)
    bin_price[idx] = price  # later updates overwrite earlier ones within a bin
    p = pd.Series(bin_price).ffill().to_numpy()

    s = pd.Series(p)
    f = {}
    f["spot_price"] = p
    f["price_delta"] = s.diff().fillna(0).to_numpy()
    for name, w in (("15m", 3), ("1h", 12), ("6h", 72)):
        r = s.rolling(w, min_periods=1)
        f[f"rolling_mean_{name}"] = r.mean().to_numpy()
        f[f"rolling_std_{name}"] = r.std().fillna(0).to_numpy()
    f["normalized_ratio"] = p / od_price

    abs_k = k0 + np.arange(n)
    sec_of_day = (abs_k * STEP_S) % 86400
    day_of_week = ((abs_k * STEP_S) // 86400 + 3) % 7  # 1970-01-01 was a Thursday (3, Monday = 0)
    f["sin_time_day"] = np.sin(2 * np.pi * sec_of_day / 86400)
    f["cos_time_day"] = np.cos(2 * np.pi * sec_of_day / 86400)
    f["sin_day_week"] = np.sin(2 * np.pi * day_of_week / 7)
    f["cos_day_week"] = np.cos(2 * np.pi * day_of_week / 7)

    f["instance_interruption_rate"] = np.full(n, adv_rate)
    f["n_updates_1h"] = pd.Series(counts).rolling(12, min_periods=1).sum().to_numpy()
    mins = (np.arange(n) - last_upd) * 5.0
    f["mins_since_change"] = np.minimum(mins, 1440.0)
    f["price_vs_6h_max"] = p / s.rolling(72, min_periods=1).max().to_numpy()
    f["price_vs_24h_min"] = p / s.rolling(288, min_periods=1).min().to_numpy()
    f["az_price_divergence"] = np.zeros(n)  # filled once sibling AZs are aligned

    # longer memory (all causal: each uses only rows up to and including this step)
    max24 = s.rolling(288, min_periods=1).max().to_numpy()
    min24 = s.rolling(288, min_periods=1).min().to_numpy()
    f["price_vs_24h_max"] = p / max24
    f["price_vs_72h_min"] = p / s.rolling(864, min_periods=1).min().to_numpy()
    f["price_vs_72h_max"] = p / s.rolling(864, min_periods=1).max().to_numpy()
    f["range_pos_24h"] = (p - min24) / np.where(max24 - min24 > 0, max24 - min24, 1.0)
    f["n_updates_24h"] = pd.Series(counts).rolling(288, min_periods=1).sum().to_numpy()
    prev_p = np.concatenate([[p[0]], p[:-1]])
    up = (p > prev_p).astype(np.float32)
    f["n_up_moves_24h"] = pd.Series(up).rolling(288, min_periods=1).sum().to_numpy()
    chg = np.where(p != prev_p, p / prev_p - 1.0, np.nan)
    f["last_change_pct"] = pd.Series(chg).ffill().fillna(0).to_numpy()

    def mins_since(mask):
        last = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
        return np.where(last >= 0, np.minimum((np.arange(n) - last) * 5.0, 4320.0), 4320.0)

    f["mins_since_up"] = mins_since(up > 0)
    f["mins_since_spike"] = mins_since(p > prev_p * 1.01)

    prev = np.concatenate([[np.nan], p[:-1]])
    spike = (p > prev * 1.01).astype(np.uint8)
    spike[0] = 0
    mat = np.stack([f[c] for c in ALL_FEATURES], axis=1).astype(np.float32)
    return k0, mat, spike


# --------------------------------------------------------------------------- window transform

LAGS = (1, 3, 12)
DIFF_COLS = ["spot_price", "normalized_ratio", "rolling_std_1h", "n_updates_1h", "mins_since_change"]

CLOCK_FEATURES = ["sin_time_day", "cos_time_day", "sin_day_week", "cos_day_week"]
# Lean set: what the multi-seed ablation (docs/wide_model_result.md) showed is needed. No clock
# features (cadence features make them redundant), no cross-AZ divergence and no Advisor rate
# (no measurable effect), so serving needs only one series' own price history.
LEAN_FEATURES = [c for c in BASE_FEATURES if c not in CLOCK_FEATURES] +     ["n_updates_1h", "mins_since_change", "price_vs_6h_max", "price_vs_24h_min"] + MEMORY_FEATURES


def window_features(W, names):
    """
    W: [B, SEQ_LENGTH, F] windows of the columns `names` (rows are consecutive 5-minute steps,
    the last row is "now"). Returns ([B, D] float32 matrix, list of D column names): last-step
    values, changes over 1/3/12 steps for the cadence columns, and price-delta extremes.
    """
    W = np.asarray(W, dtype=np.float32)
    idx = {n: i for i, n in enumerate(names)}
    parts, out = [], []
    for n in names:
        parts.append(W[:, -1, idx[n]]); out.append(n)
    for n in DIFF_COLS:
        if n in idx:
            for lag in LAGS:
                parts.append(W[:, -1, idx[n]] - W[:, -1 - lag, idx[n]]); out.append(f"{n}_d{lag}")
    delta = W[:, :, idx["price_delta"]]
    parts += [delta.max(axis=1), delta.min(axis=1), (delta != 0).sum(axis=1).astype(np.float32)]
    out += ["delta_max", "delta_min", "n_nonzero_delta"]
    return np.stack(parts, axis=1).astype(np.float32), out
