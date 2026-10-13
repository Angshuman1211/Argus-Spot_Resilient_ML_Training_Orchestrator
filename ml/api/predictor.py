"""
Turns recent Spot price updates into a calibrated interruption-risk score.

Pure logic with an injectable price source, so it is unit-testable without AWS or a web server.
Features come from ml/model/spot_features.py, the same code training uses, and the model bundle
(model.json + meta.json) is written by ml/model/train_final.py.
"""
import json
import math
import os
import sys
import time

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
import spot_features as sf  # noqa: E402


class NoHistory(Exception):
    """No (or too little) price history for the requested instance type and AZ."""


class Predictor:
    def __init__(self, model_dir, source, region):
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(model_dir, "model.json"))
        self.source = source              # price_source.CachedSource
        self.region = region
        self.features = self.meta["feature_names"]
        self.col_idx = [sf.ALL_FEATURES.index(c) for c in self.features]
        self.od = self.meta["od_prices"].get(region, {})
        self.platt = self.meta["platt"]
        if sf.SEQ_LENGTH != self.meta["seq_length"]:
            raise ValueError("serving seq_length does not match the trained model")

    @property
    def instance_types(self):
        return sorted(self.od)

    def calibrate(self, p_raw):
        p = min(max(float(p_raw), 1e-7), 1 - 1e-7)
        z = self.platt["coef"] * math.log(p / (1 - p)) + self.platt["intercept"]
        return 1.0 / (1.0 + math.exp(-z))

    def score_series(self, ts_s, price, od_price, now_s):
        """Raw model probability for one series' update history at time now_s."""
        k_end = int(now_s // sf.STEP_S)
        _, mat, _ = sf.series_features(ts_s, price, k_end, od_price, 0.0)
        if len(mat) < sf.SEQ_LENGTH:
            raise NoHistory(f"only {len(mat)} steps of history, need {sf.SEQ_LENGTH}")
        window = mat[-sf.SEQ_LENGTH:, self.col_idx][None]
        x, names = sf.window_features(window, self.features)
        if names != self.meta["window_feature_names"]:
            raise ValueError("window feature layout differs from the trained model")
        return float(self.booster.predict(xgb.DMatrix(x, feature_names=names))[0])

    def predict(self, instance_type, az, now_s=None):
        if instance_type not in self.od:
            raise NoHistory(f"model has no On-Demand price for {instance_type} in {self.region}")
        now_s = now_s if now_s is not None else time.time()
        df, fetched_at = self.source.get(self.instance_types)
        sub = df[(df["instance_type"] == instance_type) & (df["availability_zone"] == az)]
        if sub.empty:
            raise NoHistory(f"no price history for {instance_type} in {az}")
        ts_s = sf.to_epoch_s(sub["timestamp"]).to_numpy()
        keep = ts_s <= now_s  # never let a record dated after "now" influence the score
        sub, ts_s = sub[keep], ts_s[keep]
        if sub.empty:
            raise NoHistory(f"no price updates for {instance_type} in {az} at or before now")
        raw = self.score_series(ts_s, sub["spot_price"].to_numpy(), self.od[instance_type], now_s)
        return dict(
            instance_type=instance_type, az=az, risk_score=round(self.calibrate(raw), 6),
            raw_score=round(raw, 6), calibrated=True, source=self.source.name,
            last_price_update_age_s=int(now_s - ts_s.max()), source_age_s=int(now_s - fetched_at),
        )
