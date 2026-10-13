"""
Unattended finish for a sensor campaign: wait for its end time, stop any remaining sensors, wait for CloudTrail to
catch up (events appear up to about 15 minutes late), then pull labels, re-run the real-reclaim evaluation, train the
real-label model, rebuild the paper's numbers, tables and figures, compile it, and commit and push the results.

Every step is logged; a failing step is recorded and the next ones still run when they can.

    python scripts/post_campaign.py --end-utc 2026-10-13T22:41 --python <python with deps> --tectonic <tectonic.exe>
"""
import argparse
import datetime as dt
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-utc", required=True, help="campaign end, e.g. 2026-10-13T22:41")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tectonic", default=None)
    ap.add_argument("--cloudtrail-wait-min", type=int, default=20)
    a = ap.parse_args()
    end = dt.datetime.fromisoformat(a.end_utc)
    env = dict(os.environ, AWS_PROFILE=os.environ.get("AWS_PROFILE", "argus-demo"), PYTHONIOENCODING="utf-8")
    log(f"waiting until {end:%Y-%m-%d %H:%M}Z")
    while dt.datetime.utcnow() < end:
        time.sleep(30)

    run("stop remaining sensors", ["bash", os.path.join("scripts", "stop_reclaim_sensors.sh")], env=env)
    log(f"waiting {a.cloudtrail_wait_min} minutes for CloudTrail")
    time.sleep(a.cloudtrail_wait_min * 60)

    py = a.python
    run("pull reclaim labels", [py, "ml/data/pull_reclaim_labels.py"], env=env)
    run("evaluate against real reclaims", [py, "ml/model/evaluate_real_reclaims.py"], env=env)
    run("train real-label model", [py, "ml/model/train_real_label_model.py"], env=env)
    run("paper numbers", [py, "Paper/v2/build_numbers.py"], env=env)
    run("paper tables", [py, "Paper/v2/build_tables.py"], env=env)
    run("paper figures", [py, "Paper/v2/make_figures.py"], env=env)
    if a.tectonic:
        run("compile paper", [a.tectonic, "-X", "compile", "argus_v2.tex"], cwd=os.path.join(ROOT, "Paper", "v2"), env=env)
    run("stage results", ["git", "add", "-f", "Paper/v2", "ml/data/real", "ml/model/wide_results", "ml/model/final"], env=env)
    if run("commit", ["git", "commit", "-q", "-m", "real reclaim campaign results and real-label model"], env=env):
        run("push", ["git", "push", "-q"], env=env)
    log("DONE")


if __name__ == "__main__":
    main()
