"""
Bounded reclaim-sensor campaign: keeps one tiny Spot instance alive in each (type, AZ) pool for a fixed
window, replacing any that AWS reclaims, so real reclaim events (CloudTrail BidEvictedEvent) accumulate
quickly. Strictly capped; every limit is enforced here and by the instances themselves:

  * campaign length: --hours (default 6); at the end every sensor is terminated
  * total launches: --max-launches (default 150)
  * every instance also self-terminates (shutdown timer, shutdown behaviour = terminate), at most 2 hours
    after launch and never later than the campaign end, even if this process dies
  * no public IP, no inbound rules, IMDSv2, tagged argus-sensor=true

Capacity failures (InsufficientInstanceCapacity) are logged as labels too: that pool had no Spot capacity.

    AWS_PROFILE=argus-demo python scripts/sensor_campaign.py --hours 6
Output: ml/data/real/sensor_campaign_<utc>.csv (every launch and failure)
"""
import argparse
import csv
import datetime as dt
import os
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "eu-north-1")
TYPES = ["c5.large", "m5.large", "t3.medium", "c6i.large", "m6i.large"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "data", "real")
MAX_LIFE_MIN = 120
BACKOFF_AFTER_FAILURE_S = 600  # do not hammer a pool that just reported no capacity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--max-launches", type=int, default=150)
    ap.add_argument("--poll-seconds", type=int, default=240)
    ap.add_argument("--end-utc", help="absolute end, e.g. 2026-10-13T22:41 (overrides --hours; used when resuming)")
    ap.add_argument("--resume-tag", help="timestamp of an earlier run's CSVs to append to, e.g. 20261013T144104Z")
    a = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=REGION)
    azs = [z["ZoneName"] for z in ec2.describe_availability_zones()["AvailabilityZones"]]
    ami = sorted(ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]}, {"Name": "state", "Values": ["available"]}])["Images"],
        key=lambda i: i["CreationDate"])[-1]["ImageId"]
    sg = [g for g in ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["default"]}])["SecurityGroups"]
          if g.get("VpcId")][0]["GroupId"]
    subnets = {az: ec2.describe_subnets(Filters=[{"Name": "availability-zone", "Values": [az]},
                                                 {"Name": "default-for-az", "Values": ["true"]}])["Subnets"][0]["SubnetId"]
               for az in azs}

    if a.end_utc:
        t_end = dt.datetime.fromisoformat(a.end_utc).replace(tzinfo=dt.timezone.utc).timestamp()
    else:
        t_end = time.time() + a.hours * 3600
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = a.resume_tag or f"{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}"
    log_path = os.path.join(OUT_DIR, f"sensor_campaign_{tag}.csv")
    zone_id = {z["ZoneId"]: z["ZoneName"] for z in ec2.describe_availability_zones()["AvailabilityZones"]}
    score_path = log_path.replace("sensor_campaign_", "placement_scores_")
    # resuming after a restart: append to the same files and keep counting launches toward the cap
    launches = 0
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            launches = sum(1 for row in csv.reader(f) if row and row[-1] == "launched")
    score_log = open(score_path, "a", newline="")
    score_w = csv.writer(score_log)
    if score_log.tell() == 0:
        score_w.writerow(["time_utc", "instance_type", "availability_zone", "placement_score"])
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["instance_id", "instance_type", "availability_zone", "event_time_utc", "status"])
    last_fail = {}
    print(f"campaign until {dt.datetime.utcfromtimestamp(t_end):%H:%M}Z, log {log_path}", flush=True)

    def active():
        r = ec2.describe_instances(Filters=[{"Name": "tag:argus-sensor", "Values": ["true"]},
                                            {"Name": "instance-state-name", "Values": ["pending", "running"]}])
        return {(i["InstanceType"], i["Placement"]["AvailabilityZone"]) for res in r["Reservations"] for i in res["Instances"]}

    while time.time() < t_end:
        have = active()
        for az in azs:
            for t in TYPES:
                pool = (t, az)
                if pool in have or launches >= a.max_launches:
                    continue
                if time.time() - last_fail.get(pool, 0) < BACKOFF_AFTER_FAILURE_S:
                    continue
                life = int(min(MAX_LIFE_MIN, (t_end - time.time()) / 60))
                if life < 10:
                    continue
                now = dt.datetime.utcnow().isoformat() + "Z"
                try:
                    r = ec2.run_instances(
                        ImageId=ami, InstanceType=t, MinCount=1, MaxCount=1,
                        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnets[az], "AssociatePublicIpAddress": False,
                                            "Groups": [sg]}],
                        InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                            "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}},
                        InstanceInitiatedShutdownBehavior="terminate",
                        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                        UserData=f"#!/bin/bash\nshutdown -h +{life}\n",
                        TagSpecifications=[{"ResourceType": "instance", "Tags": [
                            {"Key": "argus-sensor", "Value": "true"}, {"Key": "Name", "Value": "argus-sensor"}]}])
                    iid = r["Instances"][0]["InstanceId"]
                    launches += 1
                    w.writerow([iid, t, az, now, "launched"])
                except ClientError as e:
                    code = e.response["Error"]["Code"]
                    last_fail[pool] = time.time()
                    w.writerow(["", t, az, now, f"failed:{code}"])
                log.flush()
        # AWS's own capacity estimate (1 low .. 10 high) for one unit of each type, per AZ: a real signal that
        # price history does not contain, recorded alongside the reclaims it may predict
        now = dt.datetime.utcnow().isoformat() + "Z"
        for t in TYPES:
            try:
                for r in ec2.get_spot_placement_scores(InstanceTypes=[t], TargetCapacity=1, TargetCapacityUnitType="units",
                                                       SingleAvailabilityZone=True, RegionNames=[REGION])["SpotPlacementScores"]:
                    score_w.writerow([now, t, zone_id.get(r.get("AvailabilityZoneId"), r.get("AvailabilityZoneId")), r["Score"]])
            except ClientError as e:
                score_w.writerow([now, t, "", f"error:{e.response['Error']['Code']}"])
        score_log.flush()
        print(f"{dt.datetime.utcnow():%H:%M}Z active {len(active())} launches {launches}", flush=True)
        time.sleep(a.poll_seconds)

    ids = [i["InstanceId"] for res in ec2.describe_instances(Filters=[
        {"Name": "tag:argus-sensor", "Values": ["true"]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]}])["Reservations"] for i in res["Instances"]]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    print(f"campaign over: {launches} launches, terminated {len(ids)} remaining", flush=True)


if __name__ == "__main__":
    main()
