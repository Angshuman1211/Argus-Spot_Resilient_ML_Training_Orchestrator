"""
Unattended finish for a sensor campaign: wait for its end time, stop any remaining sensors, wait for CloudTrail to
catch up (events appear up to about 15 minutes late), then pull labels, re-run the real-reclaim evaluation, train the
real-label model, rebuild the paper's numbers, tables and figures, compile it, and commit and push the results.

Every step is logged; a failing step is recorded and the next ones still run when they can.

    python scripts/post_campaign.py --end-utc 2026-10-15T00:00 --regions eu-north-1,us-east-1 --python <python> --tectonic <tectonic.exe>
"""
import argparse
import datetime as dt
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOG = os.path.join(ROOT, "ml", "data", "real", "post_campaign.log")


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
    log(f"waiting until {end:%Y-%m-%d %H:%M}Z")
    while dt.datetime.utcnow() < end:
        time.sleep(30)

    py = a.python
    for r in regions:  # the campaigns terminate their own sensors; this is the backstop
        run_net(f"stop remaining sensors in {r}", [py, "scripts/sensor_campaign.py", "--region", r, "--stop-only"], env=env)
    log(f"waiting {a.cloudtrail_wait_min} minutes for CloudTrail")
    time.sleep(a.cloudtrail_wait_min * 60)

    ok = []
    if a.s3_data:  # the runner syncs every 2 minutes and makes a final sync 10 minutes after the end
        ok.append(run_net("sync campaign data from S3", ["aws", "s3", "sync", a.s3_data, "ml/data/real/"], env=env))
    ok.append(run_net("pull reclaim labels", [py, "ml/data/pull_reclaim_labels.py"], env=env))
    ok.append(run_net("evaluate against real reclaims", [py, "ml/model/evaluate_real_reclaims.py"], env=env))
    ok.append(run("train real-label model", [py, "ml/model/train_real_label_model.py"], env=env))
    ok.append(run("paper numbers", [py, "Paper/v2/build_numbers.py"], env=env))
    ok.append(run("paper tables", [py, "Paper/v2/build_tables.py"], env=env))
    ok.append(run("paper figures", [py, "Paper/v2/make_figures.py"], env=env))
    if a.tectonic:
        ok.append(run("compile paper", [a.tectonic, "-X", "compile", "argus_v2.tex"], cwd=os.path.join(ROOT, "Paper", "v2"), env=env))
    run("stage results", ["git", "add", "-f", "Paper/v2", "ml/data/real", "ml/model/wide_results", "ml/model/final"], env=env)
    # after a restart this may run a second time with nothing new to commit: that is fine, but still push
    nothing_new = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, env=env).returncode == 0
    committed = nothing_new or run("commit", ["git", "commit", "-q", "-m", f"real reclaim results, {len(regions)} regions, real-label model"], env=env)
    ok.append(committed and run_net("push", ["git", "push", "-q"], env=env))
    log("ALL STEPS OK" if all(ok) else "SOME STEPS FAILED: " + str(ok))
    log("DONE")


if __name__ == "__main__":
    main()
