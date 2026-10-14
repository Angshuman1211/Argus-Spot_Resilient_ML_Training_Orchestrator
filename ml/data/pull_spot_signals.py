"""
Pull AWS's own Spot signals recorded by scripts/setup_spot_signals.sh (EventBridge -> CloudWatch Logs):
  rebalance     EC2 Instance Rebalance Recommendation (AWS's free early warning)
  interruption  EC2 Spot Instance Interruption Warning (the 2-minute notice)
for every campaign region, merged with any earlier pull (the log group keeps 30 days).

    AWS_PROFILE=argus-sensor python ml/data/pull_spot_signals.py
Output: ml/data/real/spot_signals.csv  [instance_id, region, signal, time], spot_signals_coverage.json
"""
import json
import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real", "spot_signals.csv")
COVER = os.path.join(HERE, "real", "spot_signals_coverage.json")
GROUP = "/aws/events/argus-spot-signals"
REGIONS = os.environ.get("AWS_REGIONS", "eu-north-1,us-east-1,us-east-2,ap-southeast-1").split(",")
KIND = {"EC2 Instance Rebalance Recommendation": "rebalance", "EC2 Spot Instance Interruption Warning": "interruption"}


def main():
    rows, coverage = [], {}
    for region in REGIONS:
        logs = boto3.client("logs", region_name=region)
        n = 0
        try:
            for page in logs.get_paginator("filter_log_events").paginate(logGroupName=GROUP):
                for e in page["events"]:
                    ev = json.loads(e["message"])
                    kind = KIND.get(ev.get("detail-type"))
                    iid = (ev.get("detail") or {}).get("instance-id")
                    if kind and iid:
                        rows.append(dict(instance_id=iid, region=ev.get("region", region), signal=kind, time=ev["time"]))
                        n += 1
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            print(f"{region}: no signal log group (setup_spot_signals.sh not run there)")
        print(f"{region}: {n} signal events")
        # when recording started: only sensors launched after this could have received a signal we saw
        g = [x for x in logs.describe_log_groups(logGroupNamePrefix=GROUP)["logGroups"] if x["logGroupName"] == GROUP]
        if g:
            coverage[region] = pd.Timestamp(g[0]["creationTime"], unit="ms", tz="UTC").isoformat()
    new = pd.DataFrame(rows, columns=["instance_id", "region", "signal", "time"])
    if os.path.exists(OUT):
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
    new = new.drop_duplicates(["instance_id", "signal", "time"]).sort_values("time")
    new.to_csv(OUT, index=False)
    old = json.load(open(COVER)) if os.path.exists(COVER) else {}
    json.dump({**coverage, **old}, open(COVER, "w"), indent=1)  # keep the earliest start per region
    print(f"{len(new)} signals in {OUT}: " + str(new.signal.value_counts().to_dict()))


if __name__ == "__main__":
    main()
