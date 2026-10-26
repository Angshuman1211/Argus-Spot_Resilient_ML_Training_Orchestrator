"""
Unattended finish for a sensor campaign: wait for its end time, stop any remaining sensors, wait for CloudTrail to
catch up (events appear up to about 15 minutes late), then pull labels, re-run the real-reclaim evaluation, train the
real-label model, rebuild the paper's numbers, tables and figures (the paper lives in the gitignored
presentation_files/Paper_v2), compile it, and commit and push the result files.

Every step is logged; a failing step is recorded and the next ones still run when they can. Finished steps are
checkpointed in ml/data/real/post_campaign_progress.json, so after a reboot, a crash or a Wi-Fi drop the watchdog
(scripts/resume_campaign.ps1) restarts this script and it carries on after the last finished step.

    python scripts/post_campaign.py --end-utc 2026-10-15T00:00 --regions eu-north-1,us-east-1 --python <python> --tectonic <tectonic.exe>
"""
import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOG = os.path.join(ROOT, "ml", "data", "real", "post_campaign.log")
PROGRESS = os.path.join(ROOT, "ml", "data", "real", "post_campaign_progress.json")


class Progress:
    """Steps already finished for one campaign (keyed by its end time), written atomically after each step."""

    def __init__(self, end_utc):
        self.key = end_utc
        try:
            with open(PROGRESS, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        self.done = list(d.get("done", [])) if d.get("end_utc") == end_utc else []

    def step(self, name, fn):
        """Runs fn() unless this step already finished before a restart; records it only if it succeeded."""
        if name in self.done:
            log(f"SKIP {name} (finished before a restart)")
            return True
        ok = fn()
        if ok:
            self.done.append(name)
            tmp = PROGRESS + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"end_utc": self.key, "done": self.done}, f, indent=1)
            os.replace(tmp, PROGRESS)
        return ok


def log(msg):
    line = f"{dt.datetime.utcnow():%H:%M:%S}Z {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, cmd, cwd=ROOT, env=None):
    log(f"START {name}: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
    log(f"{'OK' if r.returncode == 0 else 'FAILED (' + str(r.returncode) + ')'} {name}\n{tail}")
    return r.returncode == 0


def wait_online(max_hours=12):
    """Block until AWS and GitHub are reachable (Wi-Fi back after a drop or a reboot)."""
    t0, said = time.time(), False
    while time.time() - t0 < max_hours * 3600:
        try:
            for host in ("sts.amazonaws.com", "github.com"):
                socket.create_connection((host, 443), timeout=10).close()
            if said:
                log("network is back")
            return True
        except OSError:
            if not said:
                log("no network; waiting for it")
                said = True
            time.sleep(60)
    return False


def run_net(name, cmd, cwd=ROOT, env=None, tries=6, wait_s=600):
    """A step that needs the network: wait for it, and retry a failure (a drop mid-step) a few times."""
    for i in range(tries):
        wait_online()
        if run(name if i == 0 else f"{name} (retry {i})", cmd, cwd=cwd, env=env):
            return True
        time.sleep(wait_s)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-utc", required=True, help="campaign end, e.g. 2026-10-13T22:41")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tectonic", default=None)
    ap.add_argument("--cloudtrail-wait-min", type=int, default=20)
    ap.add_argument("--regions", default="eu-north-1", help="comma-separated campaign regions")
    ap.add_argument("--s3-data", default=None, help="campaign ran on the cloud runner: sync its results from this S3 prefix")
    a = ap.parse_args()
    end = dt.datetime.fromisoformat(a.end_utc)
    regions = a.regions.split(",")
    env = dict(os.environ, AWS_PROFILE=os.environ.get("AWS_PROFILE", "argus-sensor"), PYTHONIOENCODING="utf-8",
               AWS_REGIONS=a.regions)
    prog = Progress(a.end_utc)
    log(f"waiting until {end:%Y-%m-%d %H:%M}Z" + (f" (resuming; done: {', '.join(prog.done)})" if prog.done else ""))
    first_stop, checked = None, 0.0
    while dt.datetime.utcnow() < end:
        # the spending stop can end the campaign early: every region then leaves a stopped_<tag>.txt in S3
        if a.s3_data and time.time() - checked > 600:
            checked = time.time()
            r = subprocess.run(["aws", "s3", "ls", a.s3_data], capture_output=True, text=True, env=env)
            stopped = {line.split("stopped_", 1)[1].split("_")[0] for line in r.stdout.splitlines() if "stopped_" in line}
            if stopped and first_stop is None:
                first_stop = time.time()
                log(f"spending stop reached in {', '.join(sorted(stopped))}")
            if stopped >= set(regions) or (first_stop and time.time() - first_stop > 5400):
                log("campaign ended early (spending stop); finishing now")
                if os.name == "nt":  # keep the laptop awake while the finish runs
                    subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-File",
                                      os.path.join(ROOT, "scripts", "keep_awake.ps1"), "-Hours", "2"])
                break
        time.sleep(30)

    py = a.python
    for r in regions:  # the campaigns terminate their own sensors; this is the backstop
        prog.step(f"stop remaining sensors in {r}", lambda r=r: run_net(
            f"stop remaining sensors in {r}", [py, "scripts/sensor_campaign.py", "--region", r, "--stop-only"], env=env))
    # wait for CloudTrail until end + N minutes: after a restart only the part of the wait that is left
    ready = min(end, dt.datetime.utcnow()) + dt.timedelta(minutes=a.cloudtrail_wait_min)  # early stop: from now
    left = (ready - dt.datetime.utcnow()).total_seconds()
    if left > 0:
        log(f"waiting {left / 60:.0f} minutes for CloudTrail")
        time.sleep(left)

    ok = []
    if a.s3_data:  # the runner syncs every 2 minutes and makes a final sync 10 minutes after the end
        ok.append(prog.step("sync campaign data from S3", lambda: run_net(
            "sync campaign data from S3", ["aws", "s3", "sync", a.s3_data, "ml/data/real/"], env=env)))
    for name, cmd, net in [
        ("pull reclaim labels", [py, "ml/data/pull_reclaim_labels.py"], True),
        ("pull spot signals (rebalance recommendations)", [py, "ml/data/pull_spot_signals.py"], True),
        ("evaluate against real reclaims", [py, "ml/model/evaluate_real_reclaims.py"], True),
        ("train real-label model", [py, "ml/model/train_real_label_model.py"], False),
        ("paper numbers", [py, "presentation_files/Paper_v2/build_numbers.py"], False),
        ("paper tables", [py, "presentation_files/Paper_v2/build_tables.py"], False),
        ("paper figures", [py, "presentation_files/Paper_v2/make_figures.py"], False),
    ]:
        ok.append(prog.step(name, lambda name=name, cmd=cmd, net=net: (run_net if net else run)(name, cmd, env=env)))
    if a.tectonic and os.path.exists(a.tectonic):
        ok.append(prog.step("compile paper", lambda: run(
            "compile paper", [a.tectonic, "-X", "compile", "argus_v2.tex"], cwd=os.path.join(ROOT, "presentation_files", "Paper_v2"), env=env)))
    run("stage results", ["git", "add", "-f", "ml/data/real", "ml/model/wide_results", "ml/model/final"], env=env)
    # after a restart this may run a second time with nothing new to commit: that is fine, but still push
    nothing_new = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, env=env).returncode == 0
    committed = nothing_new or run("commit", ["git", "commit", "-q", "-m", f"real reclaim results, {len(regions)} regions, real-label model"], env=env)
    ok.append(committed and run_net("push", ["git", "push", "-q"], env=env))
    log("ALL STEPS OK" if all(ok) else "SOME STEPS FAILED: " + str(ok))
    problems = final_check(regions, env)  # is everything really finished and saved?
    for p in problems:
        log(f"CHECK FAILED: {p}")
    log("DONE")
    if os.name == "nt":  # wake whoever is at the laptop, only now that everything has been checked
        if all(ok) and not problems:
            msg = ("All done and saved: sensors and cloud runner stopped, results pulled, model retrained, "
                   "paper rebuilt, committed and pushed to GitHub.")
        else:
            msg = "Finished, but NOT everything is done:\n" + "\n".join(
                [f"- step failed ({sum(not x for x in ok)} of {len(ok)})"] * (not all(ok)) + [f"- {p}" for p in problems]
            ) + "\nSee ml/data/real/post_campaign.log"
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                          os.path.join(ROOT, "scripts", "alarm.ps1"), "-Message", msg])


def final_check(regions, env):
    """Everything that must be true before the alarm says 'done'. Returns what is not."""
    problems = []
    try:
        import boto3
        sess = boto3.Session(profile_name=env.get("AWS_PROFILE"))
        for r in regions:
            ec2 = sess.client("ec2", region_name=r)
            n = sum(len(x["Instances"]) for x in ec2.describe_instances(Filters=[
                {"Name": "tag:argus-sensor", "Values": ["true"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]}])["Reservations"])
            if n:
                problems.append(f"{n} sensors still running in {r}")
        ec2 = sess.client("ec2", region_name="eu-north-1")
        if ec2.describe_instances(Filters=[{"Name": "tag:argus-runner", "Values": ["true"]},
                                           {"Name": "instance-state-name", "Values": ["pending", "running"]}])["Reservations"]:
            problems.append("cloud runner machine still running")
    except Exception as e:  # noqa: BLE001 - report, never crash before the alarm
        problems.append(f"could not check AWS: {e}")
    # results saved: nothing left unstaged in the result folders, and the branch is pushed
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "ml/data/real", "ml/model/wide_results",
                            "ml/model/final"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # log files keep growing after the commit (this very check writes to one), so they do not count
    if any(not line.startswith("??") and not line.rstrip().endswith((".log", ".out", ".err")) for line in dirty.splitlines()):
        problems.append("some result files are not committed")
    subprocess.run(["git", "fetch", "-q"], cwd=ROOT, env=env)
    ahead = subprocess.run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if ahead != "0":
        problems.append(f"{ahead or 'some'} commits not pushed to GitHub")
    return problems


if __name__ == "__main__":
    main()
