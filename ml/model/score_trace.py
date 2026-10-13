"""
Scores the calibration + held-out test windows with the SHIPPED model and writes a
causal replay trace for benchmark/trace_sim.py.

Each row is one decision instant for one (instance_type, az) series: the 5-minute
step at which a window of SEQ_LENGTH steps ends. Everything in a row is knowable at
that instant, except the two label columns (future_spike, spike_now), which the
simulator uses only to place interruption events, never to decide anything.

Splits come from dataset.val_calibration_test_split, the exact calibration/test split
calibrate_and_finalize.py uses, so thresholds can be chosen on `calib` rows and
evaluated on `test` rows the model, scaler and calibrator never saw.

Usage:
    python ml/model/score_trace.py
Output:
    benchmark/traces/trace_scores.csv.gz
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from dataset import SpotPriceDataset, val_calibration_test_split
from feature_config import SEQ_LENGTH
from transformer import SpotInterruptionPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FEATURES_CSV = os.path.join(HERE, "..", "data", "features.csv")
OUT_PATH = os.path.join(REPO, "benchmark", "traces", "trace_scores.csv.gz")


def load_model():
    with open(os.path.join(HERE, "model_metadata.json")) as f:
        meta = json.load(f)
    model = SpotInterruptionPredictor(
        num_features=meta["num_features"], d_model=meta["d_model"],
        nhead=meta["nhead"], num_layers=meta["num_layers"],
    )
    model.load_state_dict(torch.load(os.path.join(HERE, "spot_transformer.pt"), map_location="cpu"))
    model.eval()
    scaler = joblib.load(os.path.join(HERE, "spot_scaler.joblib"))
    calibrator = joblib.load(os.path.join(HERE, "spot_calibrator.joblib"))
    return model, scaler, calibrator, meta


def calibrate(calibrator, raw_prob):
    eps = 1e-7
    clipped = np.clip(raw_prob, eps, 1 - eps)
    logit = np.log(clipped / (1 - clipped))
    return calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]


def main():
    model, scaler, calibrator, meta = load_model()
    horizon = meta.get("prediction_horizon", 3)
    ds = SpotPriceDataset(FEATURES_CSV, scaler=scaler, prediction_horizon=horizon,
                          spike_threshold=meta.get("spike_threshold", 1.01))

    # Re-derive the same per-group row order the dataset used, to recover timestamps.
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    df = df.sort_values(["instance_type", "availability_zone", "timestamp"])
    df["prev_price"] = df.groupby(["instance_type", "availability_zone"])["spot_price"].shift(1)
    df["spike_now"] = (df["spot_price"] > df["prev_price"] * meta.get("spike_threshold", 1.01)).astype(int)
    df["vol_cv"] = (df["rolling_std_1h"] / df["rolling_mean_1h"].replace(0, np.nan)).fillna(0)

    groups = [(key, g.reset_index(drop=True)) for key, g in df.groupby(["instance_type", "availability_zone"])]
    assert len(groups) == len(ds.group_ranges)

    calib_idx, test_idx = val_calibration_test_split(ds.group_ranges)
    wanted = {i: "calib" for i in calib_idx}
    wanted.update({i: "test" for i in test_idx})
    order = sorted(wanted)

    probs = np.zeros(len(order), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(order), 2048):
            batch = order[s:s + 2048]
            x = torch.tensor(ds.sequences[batch], dtype=torch.float32)
            probs[s:s + len(batch)] = torch.sigmoid(model(x)).numpy().flatten()
    risk = calibrate(calibrator, probs)

    # Map each flat window index back to (group, row) so timestamps line up.
    starts = np.array([r[0] for r in ds.group_ranges])
    rows = []
    for k, flat in enumerate(order):
        gi = int(np.searchsorted(starts, flat, side="right") - 1)
        (itype, az), g = groups[gi]
        j = flat - ds.group_ranges[gi][0]
        decision_row = j + SEQ_LENGTH - 1
        r = g.iloc[decision_row]
        rows.append((itype, az, r["timestamp"], wanted[flat], float(risk[k]), float(probs[k]),
                     int(ds.labels[flat]), int(r["spike_now"]), float(r["vol_cv"]),
                     float(r["spot_price"]), float(r.get("instance_interruption_rate", 0.0))))

    out = pd.DataFrame(rows, columns=[
        "instance_type", "availability_zone", "timestamp", "split", "risk_score", "raw_prob",
        "future_spike", "spike_now", "vol_cv", "spot_price", "advisor_rate",
    ])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"wrote {len(out)} rows -> {OUT_PATH}")
    print(out.groupby("split").agg(rows=("risk_score", "size"), positives=("future_spike", "sum"),
                                   spikes=("spike_now", "sum"), t0=("timestamp", "min"), t1=("timestamp", "max")))


if __name__ == "__main__":
    main()
