"""
Bounded reclaim-sensor campaign: keeps tiny Spot instances alive across (type, AZ) pools of one region for a fixed
window, replacing any that AWS reclaims, so real reclaim events (CloudTrail BidEvictedEvent) accumulate
quickly. Strictly capped; every limit is enforced here and by the instances themselves:

  * campaign length: --hours (default 6) or --end-utc; at the end every sensor in the region is terminated
  * total launches: --max-launches (default 150)
  * every instance also self-terminates (shutdown timer, shutdown behaviour = terminate), at most 2 hours
    after launch and never later than the campaign end, even if this process dies
  * no public IP, no inbound rules, IMDSv2, tagged argus-sensor=true

Capacity failures (InsufficientInstanceCapacity) are logged as labels too: that pool had no Spot capacity.
Quota failures (MaxSpotInstanceCountExceeded, VcpuLimitExceeded) are logged but are not labels; launching pauses
briefly instead. Run one process per region; they are independent.

    AWS_PROFILE=argus-sensor python scripts/sensor_campaign.py --region us-east-1 --hours 10
    AWS_PROFILE=argus-sensor python scripts/sensor_campaign.py --region us-east-1 --stop-only
Output: ml/data/real/sensor_campaign_<tag>.csv (every launch and failure), placement_scores_<tag>.csv
"""
import argparse
import csv
import datetime as dt
import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# small 2-vCPU types the risk model has prices for in every campaign region; yesterday's five plus three more
TYPES = ["c5.large", "m5.large", "t3.medium", "c6i.large", "m6i.large", "r5.large", "r6i.large", "t3.large"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "data", "real")
MAX_LIFE_MIN = 120
BACKOFF_AFTER_FAILURE_S = 600  # do not hammer a pool that just reported no capacity
QUOTA_PAUSE_S = 300  # the account hit its Spot quota: wait for sensors to end before launching more
QUOTA_CODES = {"MaxSpotInstanceCountExceeded", "VcpuLimitExceeded", "InstanceLimitExceeded"}
SENSOR_FILTER = [{"Name": "tag:argus-sensor", "Values": ["true"]},
                 {"Name": "instance-state-name", "Values": ["pending", "running"]}]


def sensor_ids(ec2):
    return [i["InstanceId"] for res in ec2.describe_instances(Filters=SENSOR_FILTER)["Reservations"] for i in res["Instances"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-north-1"))
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--max-launches", type=int, default=150)
    ap.add_argument("--per-pool", type=int, default=1, help="sensors kept alive per (type, AZ) pool")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--poll-seconds", type=int, default=240)
    ap.add_argument("--end-utc", help="absolute end, e.g. 2026-10-13T22:41 (overrides --hours; used when resuming)")
    ap.add_argument("--resume-tag", help="tag of an earlier run's CSVs to append to, e.g. us-east-1_20261014T101500Z")
    ap.add_argument("--stop-only", action="store_true", help="terminate every sensor in the region and exit")
    a = ap.parse_args()
    region, types = a.region, a.types.split(",")

    ec2 = boto3.client("ec2", region_name=region)
    if a.stop_only:
        ids = sensor_ids(ec2)
        if ids:
            ec2.terminate_instances(InstanceIds=ids)
        print(f"{region}: terminated {len(ids)} sensors", flush=True)
        return

    # started while offline (a reboot before Wi-Fi is back)? wait for the network instead of crashing
    while True:
        try:
            ec2.describe_availability_zones()
            break
        except (BotoCoreError, ClientError) as e:
            print(f"{dt.datetime.utcnow():%H:%M}Z {region} waiting for network: {e}", flush=True)
            time.sleep(60)
    zones = ec2.describe_availability_zones(Filters=[{"Name": "zone-type", "Values": ["availability-zone"]}])["AvailabilityZones"]
    azs = [z["ZoneName"] for z in zones]
    zone_id = {z["ZoneId"]: z["ZoneName"] for z in zones}
    ami = sorted(ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]}, {"Name": "state", "Values": ["available"]}])["Images"],
        key=lambda i: i["CreationDate"])[-1]["ImageId"]
    sg = [g for g in ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["default"]}])["SecurityGroups"]
          if g.get("VpcId")][0]["GroupId"]
    subnets = {}
    for az in azs:  # an AZ without a default subnet is skipped, not fatal
        s = ec2.describe_subnets(Filters=[{"Name": "availability-zone", "Values": [az]},
                                          {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
        if s:
            subnets[az] = s[0]["SubnetId"]
    # only pools where the type is actually offered
    offered = {(o["InstanceType"], o["Location"]) for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": types}])["InstanceTypeOfferings"]}
    pools = [(t, az) for az in subnets for t in types if (t, az) in offered]

    if a.end_utc:
        t_end = dt.datetime.fromisoformat(a.end_utc).replace(tzinfo=dt.timezone.utc).timestamp()
    else:
        t_end = time.time() + a.hours * 3600
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = a.resume_tag or f"{region}_{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}"
    log_path = os.path.join(OUT_DIR, f"sensor_campaign_{tag}.csv")
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
    last_fail, quota_until = {}, 0.0
    print(f"{region}: {len(pools)} pools, campaign until {dt.datetime.utcfromtimestamp(t_end):%H:%M}Z, log {log_path}", flush=True)

    def active():
        r = ec2.describe_instances(Filters=SENSOR_FILTER)
        have = {}
        for res in r["Reservations"]:
            for i in res["Instances"]:
                k = (i["InstanceType"], i["Placement"]["AvailabilityZone"])
                have[k] = have.get(k, 0) + 1
        return have

    def launch(t, az, life):
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
        return r["Instances"][0]["InstanceId"]

    while time.time() < t_end:
        try:
            have = active()
            # fill the emptiest pools first, so a quota limit spreads sensors evenly
            for pool in sorted(pools, key=lambda p: have.get(p, 0)):
                t, az = pool
                if launches >= a.max_launches or time.time() < quota_until:
                    break
                if have.get(pool, 0) >= a.per_pool or time.time() - last_fail.get(pool, 0) < BACKOFF_AFTER_FAILURE_S:
                    continue
                life = int(min(MAX_LIFE_MIN, (t_end - time.time()) / 60))
                if life < 10:
                    continue
                now = dt.datetime.utcnow().isoformat() + "Z"
                try:
                    iid = launch(t, az, life)
                    launches += 1
                    have[pool] = have.get(pool, 0) + 1
                    w.writerow([iid, t, az, now, "launched"])
                except ClientError as e:
                    code = e.response["Error"]["Code"]
                    w.writerow(["", t, az, now, f"failed:{code}"])
                    if code in QUOTA_CODES:
                        quota_until = time.time() + QUOTA_PAUSE_S
                    else:
                        last_fail[pool] = time.time()
                log.flush()
            # AWS's own capacity estimate (1 low .. 10 high) for one unit of each type, per AZ: a real signal that
            # price history does not contain, recorded alongside the reclaims it may predict
            now = dt.datetime.utcnow().isoformat() + "Z"
            for t in types:
                try:
                    for r in ec2.get_spot_placement_scores(InstanceTypes=[t], TargetCapacity=1, TargetCapacityUnitType="units",
                                                           SingleAvailabilityZone=True, RegionNames=[region])["SpotPlacementScores"]:
                        score_w.writerow([now, t, zone_id.get(r.get("AvailabilityZoneId"), r.get("AvailabilityZoneId")), r["Score"]])
                except ClientError as e:
                    score_w.writerow([now, t, "", f"error:{e.response['Error']['Code']}"])
            score_log.flush()
            print(f"{dt.datetime.utcnow():%H:%M}Z {region} active {sum(active().values())} launches {launches}", flush=True)
        except (BotoCoreError, ClientError) as e:  # network drop or throttling: keep going, the next poll retries
            print(f"{dt.datetime.utcnow():%H:%M}Z {region} poll error: {e}", flush=True)
        time.sleep(a.poll_seconds)

    ids = sensor_ids(ec2)
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    print(f"campaign over: {launches} launches, terminated {len(ids)} remaining", flush=True)


if __name__ == "__main__":
    main()
