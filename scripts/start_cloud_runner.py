"""
Moves the running sensor campaign from this laptop to a small on-demand EC2 machine, so it keeps going with the
laptop off. Stop the local campaign processes first (the watchdog will not restart them once the state says
mode=cloud), then:

    AWS_PROFILE=argus-sensor python scripts/start_cloud_runner.py

It uploads the campaign code, ml/data/real/campaign_state.json and the campaign's CSVs and logs to
s3://<bucket>/campaign/, and starts one t3.small (tagged argus-runner=true, never argus-sensor) with role
argus-campaign-runner. The machine resumes every region's campaign from those CSVs, syncs results back every
2 minutes, and terminates itself 10 minutes after the campaign ends (plus a hard backstop).
Needs scripts/setup_cloud_runner.sh to have been run once.
"""
import glob
import json
import os
import sys

import boto3

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REAL = os.path.join(ROOT, "ml", "data", "real")
REGION = "eu-north-1"
ROLE = "argus-campaign-runner"


def main():
    sts = boto3.client("sts")
    bucket = f"argus-checkpoints-{sts.get_caller_identity()['Account']}"
    ec2, s3 = boto3.client("ec2", region_name=REGION), boto3.client("s3", region_name=REGION)

    running = ec2.describe_instances(Filters=[{"Name": "tag:argus-runner", "Values": ["true"]},
                                              {"Name": "instance-state-name", "Values": ["pending", "running"]}])["Reservations"]
    if running:
        sys.exit(f"a runner is already up: {running[0]['Instances'][0]['InstanceId']}")

    state_path = os.path.join(REAL, "campaign_state.json")
    state = json.load(open(state_path, encoding="utf-8-sig"))
    state["mode"] = "cloud"
    state["bucket"] = bucket
    json.dump(state, open(state_path, "w", encoding="utf-8"), indent=1)

    s3.upload_file(os.path.join(ROOT, "scripts", "sensor_campaign.py"), bucket, "campaign/code/sensor_campaign.py")
    s3.upload_file(os.path.join(ROOT, "scripts", "cloud_runner_boot.sh"), bucket, "campaign/code/cloud_runner_boot.sh")
    s3.upload_file(state_path, bucket, "campaign/state.json")
    n = 0
    for tag in state["regions"].values():
        for f in glob.glob(os.path.join(REAL, f"*{tag}*")):
            s3.upload_file(f, bucket, f"campaign/data/{os.path.basename(f)}")
            n += 1
    print(f"uploaded code, state and {n} data files to s3://{bucket}/campaign/")

    ami = sorted(ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]}, {"Name": "state", "Values": ["available"]}])["Images"],
        key=lambda i: i["CreationDate"])[-1]["ImageId"]
    subnet = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"][0]["SubnetId"]
    hours_left = 30  # hard backstop: the machine powers off after this many hours whatever happens
    user_data = (f"#!/bin/bash\nshutdown -h +{hours_left * 60}\nexport AWS_DEFAULT_REGION={REGION}\n"
                 f"aws s3 cp s3://{bucket}/campaign/code/cloud_runner_boot.sh /opt/boot.sh\nbash /opt/boot.sh {bucket}\n")
    r = ec2.run_instances(
        ImageId=ami, InstanceType="t3.small", MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": ROLE},
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "AssociatePublicIpAddress": True}],  # outbound only
        InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
        UserData=user_data,
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "argus-runner", "Value": "true"},
                                                                 {"Key": "Name", "Value": "argus-campaign-runner"}]}])
    print(f"runner {r['Instances'][0]['InstanceId']} starting in {REGION}; results appear in s3://{bucket}/campaign/data/")


if __name__ == "__main__":
    main()
