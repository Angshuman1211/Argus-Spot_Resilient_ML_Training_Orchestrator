"""
Replay traces for benchmark/run_trace_benchmark.py built from the wide dataset (Jul-Oct 2026).

One fleet of 15 series (the five instance types the original study used, three AZs, eu-north-1),
scored twice on the SAME windows: by the retrained gradient-boosting model and by the old shipped
Transformer. Same file layout as score_trace.py, so the simulator runs unchanged on either.

    python ml/model/score_trace_wide.py
Output: benchmark/traces/trace_scores_new.csv.gz, benchmark/traces/trace_scores_old.csv.gz
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import spot_features as sf
import wide_experiments as we
import wide_xgboost as wx
from transformer import SpotInterruptionPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "benchmark", "traces")


def new_model_scores(data, starts, meta, booster):
    X, names = wx.tabular(data, sf.LEAN_FEATURES, starts)
    raw = booster.predict(xgb.DMatrix(X, feature_names=names))
    p = np.clip(raw, 1e-7, 1 - 1e-7)
    z = meta["platt"]["coef"] * np.log(p / (1 - p)) + meta["platt"]["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def old_model_scores(data, starts):
    scaler = joblib.load(os.path.join(HERE, "spot_scaler.joblib"))
    calibrator = joblib.load(os.path.join(HERE, "spot_calibrator.joblib"))
    m = json.load(open(os.path.join(HERE, "model_metadata.json")))
    model = SpotInterruptionPredictor(m["num_features"], m["d_model"], m["nhead"], m["num_layers"])
    model.load_state_dict(torch.load(os.path.join(HERE, "spot_transformer.pt"), map_location="cpu"))
    model.to(data.device)
    Xg = data.to_device(we.BASE, scaler.mean_, scaler.scale_)
    raw = 1 / (1 + np.exp(-we.predict_logits(model, data, Xg, starts)))
    p = np.clip(raw, 1e-7, 1 - 1e-7)
    return calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def main(region="eu-north-1", types=None, tag=None, with_old=True):
    """Build replay traces for one region's fleet. Default reproduces the original eu-north-1 pair."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = we.WideData(dev)
    types = set(types) if types else set(we.OLD_TYPES)
    meta = json.load(open(os.path.join(HERE, "final", "meta.json")))
    booster = xgb.Booster()
    booster.load_model(os.path.join(HERE, "final", "model.json"))

    sel = data.meta[(data.meta.region == region) & (data.meta.instance_type.isin(types))]
    # at most three AZs per type so every region's fleet is the same size (15 for five types)
    sel = sel.sort_values(["instance_type", "availability_zone"]).groupby("instance_type").head(3)
    ids = set(sel.index)
    assert len(ids) >= 5, f"{region}: only {len(ids)} series for {sorted(types)}"
    print(f"{region}: {len(ids)} series", flush=True)
    rows = {"new": [], "old": []} if with_old else {"new": []}
    names = we.BASE
    ci_price = data.names.index("spot_price")
    ci_std, ci_mean = data.names.index("rolling_std_1h"), data.names.index("rolling_mean_1h")
    for split in ("calib", "test"):
        keep = np.isin(data.series[split], list(ids))
        starts = data.starts[split][keep]
        sid = data.series[split][keep]
        last = starts + we.SEQ - 1
        base = pd.DataFrame({
            "instance_type": data.meta.instance_type.to_numpy()[sid],
            "availability_zone": data.meta.availability_zone.to_numpy()[sid],
            "timestamp": pd.to_datetime(data.k[last].astype("int64") * 300, unit="s", utc=True),
            "split": split,
            "future_spike": data.labels[split][keep].astype(int),
            "spike_now": data.spike[last].astype(int),
            "vol_cv": np.nan_to_num(data.X[last, ci_std] / np.where(data.X[last, ci_mean] == 0, np.nan,
                                                                    data.X[last, ci_mean])),
            "spot_price": data.X[last, ci_price],
            "advisor_rate": data.meta.advisor_rate.to_numpy()[sid],
        })
        scored = [("new", new_model_scores(data, starts, meta, booster))]
        if with_old:
            scored.append(("old", old_model_scores(data, starts)))
        for which, scores in scored:
            d = base.copy()
            d["risk_score"] = scores
            d["raw_prob"] = scores
            rows[which].append(d)
    os.makedirs(OUT, exist_ok=True)
    for which in rows:
        df = pd.concat(rows[which], ignore_index=True)
        name = f"trace_scores_{which}.csv.gz" if tag is None else f"trace_scores_{tag}.csv.gz"
        path = os.path.join(OUT, name)
        df.to_csv(path, index=False, compression="gzip")
        t = df[df.split == "test"]
        print(f"{which}: {len(df)} rows -> {path}; test spikes {int(t.spike_now.sum())}, "
              f"span {t.timestamp.min()} .. {t.timestamp.max()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # python score_trace_wide.py <region> [tag]: new model only, the 5 original types
        main(region=sys.argv[1], tag=sys.argv[2] if len(sys.argv) > 2 else sys.argv[1], with_old=False)
    else:
        main()
