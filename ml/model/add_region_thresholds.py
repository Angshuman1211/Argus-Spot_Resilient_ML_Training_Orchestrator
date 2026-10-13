"""
Adds per-region alarm-budget thresholds to final/meta.json.

An alarm budget is the share of (series, 5-minute step) pairs that should alarm. The thresholds are
quantiles of the CALIBRATED score on the calibration slice only, per region, so a deployment can
say "alarm on about 10% of steps in eu-north-1" without touching test data. The trace benchmark
(docs/trace_benchmark_result.md) selects the budget; this turns that choice into a score threshold.

    python ml/model/add_region_thresholds.py
"""
import json
import os

import numpy as np
import xgboost as xgb

import spot_features as sf
import wide_experiments as we
import wide_xgboost as wx

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [0.01, 0.02, 0.05, 0.1, 0.2]


def main():
    meta_path = os.path.join(HERE, "final", "meta.json")
    meta = json.load(open(meta_path))
    booster = xgb.Booster()
    booster.load_model(os.path.join(HERE, "final", "model.json"))
    data = we.WideData(device="cpu")
    X, names = wx.tabular(data, sf.LEAN_FEATURES, data.starts["calib"])
    raw = booster.predict(xgb.DMatrix(X, feature_names=names))
    p = np.clip(raw, 1e-7, 1 - 1e-7)
    z = meta["platt"]["coef"] * np.log(p / (1 - p)) + meta["platt"]["intercept"]
    risk = 1.0 / (1.0 + np.exp(-z))

    region = data.meta.region.to_numpy()[data.series["calib"]]
    itype = data.meta.instance_type.to_numpy()[data.series["calib"]]
    out = {}
    for r in sorted(set(region)):
        m = region == r
        out[r] = {f"alarm_budget_{b}": float(np.quantile(risk[m], 1 - b)) for b in BUDGETS}
    m5 = (region == "eu-north-1") & np.isin(itype, sorted(we.OLD_TYPES))
    out["eu-north-1_old5_types"] = {f"alarm_budget_{b}": float(np.quantile(risk[m5], 1 - b)) for b in BUDGETS}
    meta["thresholds_by_region"] = out
    json.dump(meta, open(meta_path, "w"), indent=1)
    for r in ("eu-north-1", "eu-north-1_old5_types"):
        print(r, {k: round(v, 4) for k, v in out[r].items()})


if __name__ == "__main__":
    main()
