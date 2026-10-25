"""
Checks the Spot vCPU quota requests and says so once when AWS answers. Run by scripts/resume_campaign.ps1 on every
watchdog pass (every 10 minutes and at logon), so it survives reboots and Wi-Fi drops: the last seen status per region
is kept in ml/data/real/quota_watch.json, and only a change is logged and announced.

    AWS_PROFILE=argus-sensor python scripts/quota_watch.py
"""
import datetime as dt
import json
import os
import subprocess

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
STATE = os.path.join(ROOT, "ml", "data", "real", "quota_watch.json")
LOG = os.path.join(ROOT, "ml", "data", "real", "watchdog.log")
REGIONS = os.environ.get("AWS_REGIONS", "eu-north-1,us-east-1,us-east-2,ap-southeast-1").split(",")
QUOTA = "L-34B43A08"  # All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests, in vCPUs
ANSWERED = {"APPROVED", "DENIED", "CASE_CLOSED", "NOT_APPROVED", "INVALID_REQUEST"}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z quota: {msg}\n")


def check(region):
    sq = boto3.client("service-quotas", region_name=region)
    value = sq.get_service_quota(ServiceCode="ec2", QuotaCode=QUOTA)["Quota"]["Value"]
    reqs = [r for r in sq.list_requested_service_quota_change_history(ServiceCode="ec2")["RequestedQuotas"]
            if r["QuotaCode"] == QUOTA]
    last = max(reqs, key=lambda r: r["Created"]) if reqs else {}
    return {"quota": value, "status": last.get("Status"), "desired": last.get("DesiredValue"), "case": last.get("CaseId")}


def main():
    try:
        with open(STATE, encoding="utf-8") as f:
            seen = json.load(f)
    except (OSError, ValueError):
        seen = {}
    news = []
    for region in REGIONS:
        try:
            now = check(region)
        except (BotoCoreError, ClientError) as e:  # offline or throttled: try again on the next pass
            log(f"{region} not checked ({type(e).__name__})")
            continue
        before = seen.get(region, {})
        if now != before:
            log(f"{region} quota {now['quota']:.0f} vCPU, request {now['status']} (wanted {now['desired']}, case {now['case']})")
            raised = now["quota"] > before.get("quota", now["quota"])
            if raised or (now["status"] in ANSWERED and before.get("status") not in ANSWERED):
                news.append(f"{region}: quota now {now['quota']:.0f} vCPU, request {now['status']}")
        seen[region] = now
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=1, default=str)
    os.replace(tmp, STATE)
    if news and os.name == "nt":
        msg = "AWS answered the Spot quota request:\n" + "\n".join(news) + \
              "\nA running campaign uses the new quota by itself. Tell Claude so it can check."
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                          os.path.join(ROOT, "scripts", "alarm.ps1"), "-Message", msg, "-Seconds", "10"])


if __name__ == "__main__":
    main()
