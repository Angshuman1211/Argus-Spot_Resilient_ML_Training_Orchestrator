"""
Reclaim risk from capacity signals: AWS's Spot placement score and the instance's age.

This is the model trained on REAL reclaim labels from the sensor campaign (ml/model/train_real_label_model.py,
bundle ml/model/final/capacity_model.json). On those reclaims it ranked far better than the price model
(AUROC about 0.74 to 0.77 against 0.53, docs/wide_model_result.md Section 7b), but it was fitted on two days
of small instances in four regions, so treat it as a first real-label model, not a finished one.

Inputs are clamped to the range seen in training (placement scores there only took values 1 to 3, ages
up to the sensors' 2 hour cap) so the model never extrapolates to a lower risk than it has evidence for.
"""
import json
import math
import os
import time


class CapacityModel:
    def __init__(self, path):
        with open(path) as f:
            self.m = json.load(f)
        self.features = self.m["features"]
        if self.features != ["placement", "age_min"]:
            raise ValueError(f"unexpected capacity model features {self.features}")
        self.ranges = self.m.get("ranges", {})

    @classmethod
    def load(cls, model_dir):
        """Returns the model, or None if the bundle has no capacity model."""
        path = os.path.join(model_dir, "capacity_model.json")
        return cls(path) if os.path.exists(path) else None

    def score(self, placement, age_min):
        x, clamped = [], False
        for name, v in zip(self.features, (float(placement), float(age_min))):
            lo, hi = self.ranges.get(name, (-math.inf, math.inf))
            c = min(max(v, lo), hi)
            clamped |= c != v
            x.append(c)
        z = self.m["intercept"] + sum(
            w * (v - mu) / s for w, v, mu, s in zip(self.m["coef"], x, self.m["mean"], self.m["scale"]))
        return dict(reclaim_risk=round(1.0 / (1.0 + math.exp(-z)), 6), inputs_clamped=clamped)

    def info(self):
        keys = ("features", "ranges", "operating_points", "trained_on", "note")
        return {k: self.m.get(k) for k in keys}


class NoPlacementSource:
    name = "none"

    def get(self, instance_type, az):
        return None


class EC2PlacementSource:
    """
    Single-AZ Spot placement score (1 low .. 10 high) for one unit of an instance type, the same query the
    sensor campaign recorded. Needs ec2:GetSpotPlacementScores and ec2:DescribeAvailabilityZones.
    """
    name = "ec2"

    def __init__(self, region, ttl_s=300, client=None):
        if client is None:
            import boto3
            client = boto3.client("ec2", region_name=region)
        self.ec2, self.region, self.ttl_s = client, region, ttl_s
        self._zone_name = None
        self._cache = {}  # instance_type -> (fetched_at, {az_name: score})

    def _zones(self):
        if self._zone_name is None:
            zones = self.ec2.describe_availability_zones()["AvailabilityZones"]
            self._zone_name = {z["ZoneId"]: z["ZoneName"] for z in zones}
        return self._zone_name

    def get(self, instance_type, az):
        hit = self._cache.get(instance_type)
        if hit is None or time.time() - hit[0] > self.ttl_s:
            resp = self.ec2.get_spot_placement_scores(
                InstanceTypes=[instance_type], TargetCapacity=1, TargetCapacityUnitType="units",
                SingleAvailabilityZone=True, RegionNames=[self.region])
            names = self._zones()
            scores = {names.get(r.get("AvailabilityZoneId"), r.get("AvailabilityZoneId")): r["Score"]
                      for r in resp["SpotPlacementScores"]}
            hit = (time.time(), scores)
            self._cache[instance_type] = hit
        return hit[1].get(az)


def placement_source_from_env(region):
    kind = os.getenv("PLACEMENT_SOURCE", "none")
    if kind == "ec2":
        return EC2PlacementSource(region, ttl_s=int(os.getenv("PLACEMENT_TTL_SECONDS", "300")))
    if kind == "none":
        return NoPlacementSource()
    raise ValueError(f"PLACEMENT_SOURCE must be ec2 or none, not {kind!r}")
