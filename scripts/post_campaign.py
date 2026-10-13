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
    ap.add_argument("--delete-root-key", action="store_true",
                    help="after every step succeeded, delete the root access key used by this AWS profile (irreversible)")
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
    ok = []
    ok.append(run("pull reclaim labels", [py, "ml/data/pull_reclaim_labels.py"], env=env))
    ok.append(run("evaluate against real reclaims", [py, "ml/model/evaluate_real_reclaims.py"], env=env))
    ok.append(run("train real-label model", [py, "ml/model/train_real_label_model.py"], env=env))
    ok.append(run("paper numbers", [py, "Paper/v2/build_numbers.py"], env=env))
    ok.append(run("paper tables", [py, "Paper/v2/build_tables.py"], env=env))
    ok.append(run("paper figures", [py, "Paper/v2/make_figures.py"], env=env))
    if a.tectonic:
        ok.append(run("compile paper", [a.tectonic, "-X", "compile", "argus_v2.tex"], cwd=os.path.join(ROOT, "Paper", "v2"), env=env))
    run("stage results", ["git", "add", "-f", "Paper/v2", "ml/data/real", "ml/model/wide_results", "ml/model/final"], env=env)
    # after a restart this may run a second time with nothing new to commit: that is fine, but still push
    nothing_new = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, env=env).returncode == 0
    committed = nothing_new or run("commit", ["git", "commit", "-q", "-m", "real reclaim campaign results and real-label model"], env=env)
    ok.append(committed and run("push", ["git", "push", "-q"], env=env))
    log("ALL STEPS OK" if all(ok) else "SOME STEPS FAILED: " + str(ok))

    if a.delete_root_key:
        delete_root_key(env, all(ok))
    log("DONE")


def delete_root_key(env, all_ok):
    """Irreversible, so guarded: only if every step above worked, only for the ROOT user, only the key this profile uses."""
    prof = env.get("AWS_PROFILE", "argus-demo")
    if not all_ok:
        log("root key NOT deleted: a step failed, so the key is kept for fixing it")
        return
    who = subprocess.run(["aws", "sts", "get-caller-identity", "--profile", prof, "--query", "Arn", "--output", "text"],
                         capture_output=True, text=True, env=env)
    key = subprocess.run(["aws", "configure", "get", "aws_access_key_id", "--profile", prof],
                         capture_output=True, text=True, env=env).stdout.strip()
    if not who.stdout.strip().endswith(":root") or not key:
        log(f"root key NOT deleted: caller is {who.stdout.strip() or 'unknown'} (expected the root user) or no key id found")
        return
    r = subprocess.run(["aws", "iam", "delete-access-key", "--access-key-id", key, "--profile", prof],
                       capture_output=True, text=True, env=env)
    log(f"root access key ending {key[-4:]} " + ("DELETED" if r.returncode == 0 else f"delete failed: {r.stderr.strip()[:200]}"))


if __name__ == "__main__":
    main()
