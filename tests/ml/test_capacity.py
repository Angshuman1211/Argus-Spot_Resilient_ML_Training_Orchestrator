"""Capacity (real-label) reclaim model and the /reclaim_risk endpoint. No AWS needed."""
import json
import math
import os
import sys
from unittest import mock

import pytest

pytest.importorskip("fastapi")

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "ml", "model"))
sys.path.insert(0, os.path.join(ROOT, "ml", "api"))

import capacity  # noqa: E402

SHIPPED = os.path.join(ROOT, "ml", "model", "final")


@pytest.fixture
def model():
    return capacity.CapacityModel.load(SHIPPED)


def test_shipped_bundle_has_a_capacity_model_with_ranges(model):
    assert model is not None
    assert set(model.ranges) == {"placement", "age_min"}
    assert model.info()["operating_points"]


def test_score_matches_the_logistic_formula(model):
    m = model.m
    p, age = 2.0, 30.0
    z = m["intercept"] + sum(w * (v - mu) / s for w, v, mu, s in zip(m["coef"], (p, age), m["mean"], m["scale"]))
    out = model.score(p, age)
    assert out["reclaim_risk"] == pytest.approx(1 / (1 + math.exp(-z)), abs=1e-6)
    assert out["inputs_clamped"] is False


def test_risk_falls_with_better_placement_and_older_instances(model):
    # what the real reclaims showed: low placement score and young instances are the risky ones
    assert model.score(1, 10)["reclaim_risk"] > model.score(3, 10)["reclaim_risk"]
    assert model.score(2, 5)["reclaim_risk"] > model.score(2, 100)["reclaim_risk"]


def test_inputs_outside_the_training_range_are_clamped_not_extrapolated(model):
    hi_place = model.ranges["placement"][1]
    out = model.score(9, 30)
    assert out["inputs_clamped"] is True
    assert out["reclaim_risk"] == model.score(hi_place, 30)["reclaim_risk"]
    old = model.score(2, 10_000)
    assert old["inputs_clamped"] is True
    assert old["reclaim_risk"] == model.score(2, model.ranges["age_min"][1])["reclaim_risk"]


def test_missing_bundle_returns_none(tmp_path):
    assert capacity.CapacityModel.load(str(tmp_path)) is None


def test_ec2_placement_source_maps_zone_ids_and_caches():
    ec2 = mock.Mock()
    ec2.describe_availability_zones.return_value = {"AvailabilityZones": [
        {"ZoneId": "eun1-az1", "ZoneName": "eu-north-1a"}, {"ZoneId": "eun1-az2", "ZoneName": "eu-north-1b"}]}
    ec2.get_spot_placement_scores.return_value = {"SpotPlacementScores": [
        {"AvailabilityZoneId": "eun1-az1", "Score": 3}, {"AvailabilityZoneId": "eun1-az2", "Score": 1}]}
    src = capacity.EC2PlacementSource("eu-north-1", client=ec2)
    assert src.get("m5.large", "eu-north-1b") == 1
    assert src.get("m5.large", "eu-north-1a") == 3
    assert src.get("m5.large", "eu-north-1c") is None
    assert ec2.get_spot_placement_scores.call_count == 1          # second and third lookups hit the cache
    kw = ec2.get_spot_placement_scores.call_args.kwargs
    assert kw["SingleAvailabilityZone"] is True and kw["TargetCapacity"] == 1


def test_bad_placement_source_name_is_rejected(monkeypatch):
    monkeypatch.setenv("PLACEMENT_SOURCE", "bogus")
    with pytest.raises(ValueError):
        capacity.placement_source_from_env("eu-north-1")


def test_reclaim_risk_endpoint(model, monkeypatch):
    from fastapi.testclient import TestClient
    import app as api

    monkeypatch.setattr(api, "_load", lambda: None)
    with TestClient(api.app) as c:
        monkeypatch.setattr(api, "capacity_model", model)
        monkeypatch.setattr(api, "placement", capacity.NoPlacementSource())
        q = {"instance_type": "m5.large", "az": "eu-north-1a", "instance_age_min": 12}
        r = c.get("/reclaim_risk", params={**q, "placement_score": 1})
        assert r.status_code == 200
        body = r.json()
        assert 0 < body["reclaim_risk"] < 1 and body["placement_source"] == "caller"
        # no placement score from the caller and no source configured: say so instead of guessing
        r = c.get("/reclaim_risk", params=q)
        assert r.status_code == 404 and "placement_score" in r.json()["detail"]
        assert c.get("/reclaim_risk", params={**q, "placement_score": 2, "instance_age_min": -1}).status_code == 422

        src = mock.Mock(name="src")
        src.name, src.get.return_value = "ec2", 3
        monkeypatch.setattr(api, "placement", src)
        body = c.get("/reclaim_risk", params=q).json()
        assert body["placement_score"] == 3 and body["placement_source"] == "ec2"

        monkeypatch.setattr(api, "capacity_model", None)
        monkeypatch.setattr(api, "capacity_error", "no capacity_model.json in the model bundle")
        r = c.get("/reclaim_risk", params=q)
        assert r.status_code == 503 and "capacity_model.json" in r.json()["detail"]


def test_shipped_operating_points_are_ordered(model):
    ops = model.info()["operating_points"]
    thr = [o["threshold"] for o in ops]
    rec = [o["recall"] for o in ops]
    assert thr == sorted(thr, reverse=True) and rec == sorted(rec)
    json.dumps(ops)
