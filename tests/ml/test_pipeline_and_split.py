"""Data pipeline, leakage-free splits, loss weighting and the serving contract."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "ml", "model"))
sys.path.insert(0, os.path.join(ROOT, "ml", "data"))

from dataset import temporal_split_indices, val_calibration_test_split  # noqa: E402
from feature_config import FEATURE_COLUMNS, PREDICTION_HORIZON, SEQ_LENGTH  # noqa: E402
import feature_pipeline as fp  # noqa: E402


# ------------------------------------------------------------------ raw price schema

def _raw(**over):
    base = {
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"], utc=True),
        "instance_type": ["m5.xlarge", "m5.xlarge"],
        "availability_zone": ["eu-north-1a", "eu-north-1a"],
        "spot_price": [0.05, 0.06],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_lambda_schema_is_renamed_to_training_schema():
    df = _raw().rename(columns={"availability_zone": "az"})
    df["product"] = "Linux/UNIX"
    out = fp.normalize_raw_prices(df)
    assert "availability_zone" in out.columns and "product_desc" in out.columns
    assert "az" not in out.columns


def test_unknown_instance_types_are_dropped_not_given_a_default_price():
    df = pd.concat([_raw(), _raw(instance_type=["t3.medium", "t3.medium"])])
    out = fp.normalize_raw_prices(df)
    assert set(out.instance_type) == {"m5.xlarge"}


def test_duplicate_rows_from_overlapping_lambda_windows_are_removed():
    df = pd.concat([_raw(), _raw()])
    assert len(fp.normalize_raw_prices(df)) == 2


def test_missing_required_column_fails_loudly():
    with pytest.raises(ValueError):
        fp.normalize_raw_prices(_raw().drop(columns=["spot_price"]))


def test_process_ts_group_resamples_to_five_minutes_and_scales_by_od_price():
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:17"], utc=True)
    g = pd.DataFrame({"timestamp": ts, "instance_type": "m5.xlarge",
                      "availability_zone": "eu-north-1a", "spot_price": [0.05, 0.06]})
    out = fp.process_ts_group(g)
    assert out["timestamp"].diff().dropna().eq(pd.Timedelta("5min")).all()
    assert out["normalized_ratio"].iloc[0] == pytest.approx(0.05 / fp.OD_PRICES["m5.xlarge"])


# ------------------------------------------------------------------------ the splits

GROUPS = [(0, 1000), (1000, 1800), (1800, 3000)]


def test_train_and_val_windows_never_share_a_timestep():
    train, val = temporal_split_indices(GROUPS)
    for start, end in GROUPS:
        tr = [i for i in train if start <= i < end]
        va = [i for i in val if start <= i < end]
        # a window i covers rows i .. i+SEQ_LENGTH+PREDICTION_HORIZON-1 of its group
        last_train_row = max(tr) + SEQ_LENGTH + PREDICTION_HORIZON - 1
        first_val_row = min(va)
        assert last_train_row < first_val_row


def test_validation_is_strictly_later_than_training_within_each_group():
    train, val = temporal_split_indices(GROUPS)
    for start, end in GROUPS:
        assert max(i for i in train if start <= i < end) < min(i for i in val if start <= i < end)


def test_calibration_and_test_are_disjoint_and_ordered():
    calib, test = val_calibration_test_split(GROUPS)
    assert not set(calib) & set(test)
    for start, end in GROUPS:
        c = [i for i in calib if start <= i < end]
        t = [i for i in test if start <= i < end]
        assert max(c) + SEQ_LENGTH + PREDICTION_HORIZON - 1 < min(t)


def test_calibration_and_test_never_touch_training_windows():
    train, _ = temporal_split_indices(GROUPS)
    calib, test = val_calibration_test_split(GROUPS)
    assert not set(train) & (set(calib) | set(test))


# ----------------------------------------------------------------------- focal loss

def test_focal_loss_alpha_weights_positives_more_than_negatives():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from train import FocalLoss

    loss = FocalLoss(alpha=0.75, gamma=0.0)  # gamma 0 -> weighted BCE
    logit = torch.tensor([[0.0]])
    pos = loss(logit, torch.tensor([[1.0]])).item()
    neg = loss(logit, torch.tensor([[0.0]])).item()
    assert pos / neg == pytest.approx(0.75 / 0.25)


# ---------------------------------------------------------------- serving contract

def test_shipped_metadata_matches_the_feature_contract():
    path = os.path.join(ROOT, "ml", "model", "model_metadata.json")
    if not os.path.exists(path):
        pytest.skip("model_metadata.json is gitignored and not present in this checkout")
    meta = json.load(open(path))
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert meta["seq_length"] == SEQ_LENGTH
    assert meta["num_features"] == len(FEATURE_COLUMNS)


def test_api_refuses_to_serve_without_a_calibrator(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(ROOT, "ml", "api"))
    needed = [os.path.join(ROOT, "ml", "model", f) for f in
              ("spot_transformer.pt", "spot_scaler.joblib", "model_metadata.json")]
    if not all(os.path.exists(p) for p in needed):
        pytest.skip("model artifacts are gitignored and not present in this checkout")
    import app as api

    real_exists = os.path.exists
    monkeypatch.setattr(api.os.path, "exists",
                        lambda p: False if str(p).endswith("spot_calibrator.joblib") else real_exists(p))
    monkeypatch.delenv("ALLOW_UNCALIBRATED", raising=False)
    api._load_model()
    assert api.is_ready is False

    monkeypatch.setenv("ALLOW_UNCALIBRATED", "1")
    api._load_model()
    assert api.is_ready is True and api.calibrator is None
