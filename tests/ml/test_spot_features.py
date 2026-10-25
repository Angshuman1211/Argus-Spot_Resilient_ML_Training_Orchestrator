"""Causality and shape tests for ml/model/spot_features.py (shared by training and serving)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml", "model"))

import spot_features as sf  # noqa: E402

BASE_TS = 1_760_000_000  # arbitrary epoch seconds


def synthetic_updates(n_hours=400, seed=0):
    """Hourly price updates with a random walk, like current AWS Spot cadence."""
    rng = np.random.default_rng(seed)
    ts = BASE_TS + np.arange(n_hours) * 3600 + rng.integers(0, 40, n_hours)
    price = 0.05 * np.exp(np.cumsum(rng.normal(0, 0.01, n_hours)))
    return ts.astype(np.int64), price


def full_features(ts, price, k_end=None):
    k_end = k_end if k_end is not None else int(ts[-1] // sf.STEP_S)
    return sf.series_features(ts, price, k_end, od_price=0.2, adv_rate=0.05)


def test_features_do_not_use_future_updates():
    """Recomputing on a history cut at step c must reproduce row c of the full computation."""
    ts, price = synthetic_updates()
    k0, full, _ = full_features(ts, price)
    rng = np.random.default_rng(1)
    for cut_rec in rng.integers(1100 // 12 + 80, len(ts) - 2, 6):
        cut_k = int(ts[cut_rec] // sf.STEP_S)  # the step in which record `cut_rec` lands
        keep = ts // sf.STEP_S <= cut_k
        k0c, part, _ = full_features(ts[keep], price[keep], k_end=cut_k)
        assert k0c == k0
        row = cut_k - k0
        assert part.shape[0] == row + 1
        diff = np.abs(part[row] - full[row])
        bad = [sf.ALL_FEATURES[i] for i in np.flatnonzero(diff > 1e-4)
               if sf.ALL_FEATURES[i] != "az_price_divergence"]
        assert not bad, f"features at step {row} changed when future records were removed: {bad}"


def test_spike_label_is_a_price_jump_at_that_step_only():
    ts = np.array([BASE_TS, BASE_TS + 3600, BASE_TS + 7200]).astype(np.int64)
    price = np.array([0.100, 0.103, 0.103])
    k0, _, spike = full_features(ts, price)
    k_jump = int(ts[1] // sf.STEP_S) - k0
    assert spike[k_jump] == 1 and spike.sum() == 1


def test_updates_are_counted_in_their_own_step():
    ts, price = synthetic_updates(50)
    k0, mat, _ = full_features(ts, price)
    i = sf.ALL_FEATURES.index("mins_since_change")
    j = sf.ALL_FEATURES.index("n_updates_1h")
    first_rec_row = int(ts[10] // sf.STEP_S) - k0
    assert mat[first_rec_row, i] == 0.0           # an update just landed
    assert mat[first_rec_row, j] >= 1.0
    assert mat[first_rec_row + 5, i] == 25.0      # five steps later: 25 minutes since


def test_window_features_shapes_and_names():
    names = sf.LEAN_FEATURES
    W = np.random.default_rng(0).normal(size=(7, sf.SEQ_LENGTH, len(names))).astype(np.float32)
    X, out = sf.window_features(W, names)
    assert X.shape == (7, len(out)) and len(set(out)) == len(out)
    assert out[:len(names)] == names
    assert "n_nonzero_delta" in out and "mins_since_change_d3" in out


def test_window_features_last_step_and_lag_values():
    names = ["price_delta", "spot_price", "mins_since_change"]
    W = np.zeros((1, sf.SEQ_LENGTH, 3), dtype=np.float32)
    W[0, :, 1] = np.arange(sf.SEQ_LENGTH)           # spot_price rises by 1 per step
    W[0, 5, 0] = 2.0
    X, out = sf.window_features(W, names)
    row = dict(zip(out, X[0]))
    assert row["spot_price"] == sf.SEQ_LENGTH - 1
    assert row["spot_price_d3"] == 3.0
    assert row["delta_max"] == 2.0 and row["n_nonzero_delta"] == 1.0


def test_lean_feature_set_is_a_subset_of_all_features():
    assert set(sf.LEAN_FEATURES) <= set(sf.ALL_FEATURES)
    assert not set(sf.CLOCK_FEATURES) & set(sf.LEAN_FEATURES)
