"""Predictor and API behaviour with a tiny synthetic model and a fake price source (no AWS)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("fastapi")

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "ml", "model"))
sys.path.insert(0, os.path.join(ROOT, "ml", "api"))

import spot_features as sf  # noqa: E402
from predictor import NoHistory, Predictor  # noqa: E402

NOW = 1_760_000_000.0
REGION = "eu-north-1"


class FakeSource:
    name = "fake"

    def __init__(self, df):
        self.df = df
        self.calls = 0

    def get(self, instance_types):
        self.calls += 1
        return self.df, NOW - 30


def make_prices(hours=130, itype="m5.xlarge", az="eu-north-1a", seed=0):
    rng = np.random.default_rng(seed)
    ts = (NOW - (hours - np.arange(hours)) * 3600).astype(np.int64)
    price = 0.05 * np.exp(np.cumsum(rng.normal(0, 0.01, hours)))
    return pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="s", utc=True), "instance_type": itype,
                         "availability_zone": az, "spot_price": price})


@pytest.fixture
def bundle(tmp_path):
    names = sf.LEAN_FEATURES
    W = np.random.default_rng(0).normal(size=(400, sf.SEQ_LENGTH, len(names))).astype(np.float32)
    X, wnames = sf.window_features(W, names)
    y = (X[:, 0] > 0.3).astype(int)
    booster = xgb.train(dict(objective="binary:logistic", max_depth=3, tree_method="hist"),
                        xgb.DMatrix(X, label=y, feature_names=wnames), 20)
    booster.save_model(str(tmp_path / "model.json"))
    meta = dict(seq_length=sf.SEQ_LENGTH, feature_names=names, window_feature_names=wnames,
                platt=dict(coef=1.0, intercept=-2.0), od_prices={REGION: {"m5.xlarge": 0.2}},
                thresholds={"best_f1_on_calibration": 0.05}, metrics={"x": 1})
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return str(tmp_path)


def test_predict_returns_a_calibrated_probability(bundle):
    p = Predictor(bundle, FakeSource(make_prices()), REGION)
    out = p.predict("m5.xlarge", "eu-north-1a", now_s=NOW)
    assert 0.0 < out["risk_score"] < 1.0 and out["calibrated"] is True
    assert out["source"] == "fake" and out["source_age_s"] == 30


def test_platt_calibration_is_monotone_and_bounded(bundle):
    p = Predictor(bundle, FakeSource(make_prices()), REGION)
    vals = [p.calibrate(x) for x in (0.0, 1e-4, 0.01, 0.3, 0.9, 1.0)]
    assert vals == sorted(vals) and 0 < vals[0] and vals[-1] < 1


def test_prediction_ignores_updates_published_after_now(bundle):
    prices = make_prices()
    future = prices.tail(1).copy()
    future["timestamp"] = future["timestamp"] + pd.Timedelta(hours=5)
    future["spot_price"] = 99.0
    a = Predictor(bundle, FakeSource(prices), REGION).predict("m5.xlarge", "eu-north-1a", now_s=NOW)
    b = Predictor(bundle, FakeSource(pd.concat([prices, future])), REGION).predict(
        "m5.xlarge", "eu-north-1a", now_s=NOW)
    assert a["risk_score"] == pytest.approx(b["risk_score"], abs=1e-9)


def test_unknown_instance_type_and_missing_az_raise_nohistory(bundle):
    p = Predictor(bundle, FakeSource(make_prices()), REGION)
    with pytest.raises(NoHistory):
        p.predict("p5.48xlarge", "eu-north-1a", now_s=NOW)
    with pytest.raises(NoHistory):
        p.predict("m5.xlarge", "eu-north-1z", now_s=NOW)


def test_feature_layout_mismatch_is_refused(bundle):
    meta_path = os.path.join(bundle, "meta.json")
    meta = json.load(open(meta_path))
    meta["window_feature_names"] = list(reversed(meta["window_feature_names"]))
    json.dump(meta, open(meta_path, "w"))
    p = Predictor(bundle, FakeSource(make_prices()), REGION)
    with pytest.raises(ValueError):
        p.predict("m5.xlarge", "eu-north-1a", now_s=NOW)


def test_api_endpoints(bundle, monkeypatch):
    from fastapi.testclient import TestClient
    import app as api

    monkeypatch.setattr(api, "_load", lambda: None)
    api.predictor = Predictor(bundle, FakeSource(make_prices()), REGION)
    api.load_error = None
    monkeypatch.setattr(api.predictor, "predict",
                        lambda t, a, now_s=None, _orig=api.predictor.predict: _orig(t, a, now_s=NOW))
    with TestClient(api.app) as c:
        api.predictor = Predictor(bundle, FakeSource(make_prices()), REGION)
        orig = api.predictor.predict
        api.predictor.predict = lambda t, a, now_s=None: orig(t, a, now_s=NOW)
        assert c.get("/health").json()["status"] == "ok"
        r = c.get("/predict", params={"instance_type": "m5.xlarge", "az": "eu-north-1a"})
        assert r.status_code == 200 and 0 < r.json()["risk_score"] < 1
        assert c.post("/predict", json={"instance_type": "m5.xlarge", "az": "eu-north-1a"}).status_code == 200
        assert c.get("/predict", params={"instance_type": "x1.32xlarge", "az": "a"}).status_code == 404
        assert c.get("/model").json()["region"] == REGION


def test_api_reports_why_it_is_not_ready(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as api

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # empty dir: no model bundle
    monkeypatch.setattr(api, "MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PRICE_SOURCE", "csv")
    monkeypatch.setenv("PRICE_CSV", str(tmp_path / "none.csv"))
    with TestClient(api.app) as c:
        r = c.get("/health")
        assert r.status_code == 503 and "meta.json" in r.json()["detail"]
