"""
Real Spot reclaim labels from CloudTrail.

AWS records a BidEvictedEvent management event whenever it reclaims one of YOUR Spot instances,
at no cost and with no setup. Joining it to the instance's RunInstances record gives
(instance type, AZ, launch time, reclaim time): a genuine label, unlike the price-spike proxy the
risk model was trained on. CloudTrail keeps 90 days of events, so run this regularly.

Only instances launched in this account are visible, so volume depends on how much Spot you run
(see scripts/launch_reclaim_sensors.sh for a small, strictly time-capped sensor fleet).

    AWS_PROFILE=argus-demo python ml/data/pull_reclaim_labels.py
Output: ml/data/real/real_reclaims.csv (merged with any earlier pull, never overwritten blindly)
"""
import json
import os

import boto3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real", "real_reclaims.csv")  # tracked: tiny and irreplaceable (CloudTrail keeps 90 days)
REGION = os.environ.get("AWS_REGION", "eu-north-1")


def events(ct, name, **kw):
    token = None
    while True:
        args = dict(LookupAttributes=[{"AttributeKey": "EventName", "AttributeValue": name}], MaxResults=50, **kw)
        if token:
            args["NextToken"] = token
        r = ct.lookup_events(**args)
        for e in r["Events"]:
            yield json.loads(e["CloudTrailEvent"])
        token = r.get("NextToken")
        if not token:
            return


def launch_info(ct, instance_id):
    r = ct.lookup_events(LookupAttributes=[{"AttributeKey": "ResourceName", "AttributeValue": instance_id}],
                         MaxResults=50)
    for e in r["Events"]:
        if e["EventName"] != "RunInstances":
            continue
        ev = json.loads(e["CloudTrailEvent"])
        for item in ev["responseElements"]["instancesSet"]["items"]:
            if item["instanceId"] == instance_id:
                return dict(instance_type=item["instanceType"], availability_zone=item["placement"]["availabilityZone"],
                            lifecycle=item.get("instanceLifecycle"), launch_time=ev["eventTime"])
    return {}


def main():
    ct = boto3.client("cloudtrail", region_name=REGION)
    rows = []
    for ev in events(ct, "BidEvictedEvent"):
        ids = []
        for key in ("requestParameters", "serviceEventDetails", "responseElements"):
            block = ev.get(key) or {}
            ids = block.get("instanceIdSet") or block.get("instanceIds") or []
            if ids:
                break
        for iid in ids:
            info = launch_info(ct, iid)
            rows.append(dict(instance_id=iid, region=ev["awsRegion"], reclaim_time=ev["eventTime"], **info))
    new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
    if new.empty:
        print("no BidEvictedEvent records in the last 90 days")
        return
    new = new.drop_duplicates("instance_id").sort_values("reclaim_time").reset_index(drop=True)
    new["launch_time"] = pd.to_datetime(new["launch_time"], utc=True, format="ISO8601")
    new["reclaim_time"] = pd.to_datetime(new["reclaim_time"], utc=True, format="ISO8601")
    new["minutes_to_reclaim"] = (new["reclaim_time"] - new["launch_time"]).dt.total_seconds() / 60
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new.to_csv(OUT, index=False)
    print(new.to_string(index=False))


if __name__ == "__main__":
    main()
