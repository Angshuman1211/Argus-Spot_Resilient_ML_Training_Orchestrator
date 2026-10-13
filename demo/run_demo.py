"""
Local live demo of Argus: starts the real prediction API, scores every instance/AZ pair, then
prints the headline results of the trace-driven benchmark.

    python demo/run_demo.py             # full walkthrough, pauses between steps
    python demo/run_demo.py --no-pause  # for recording or CI
    python demo/run_demo.py --live      # fetch fresh prices from EC2 (needs AWS credentials)

Without --live it scores a small committed price snapshot (demo/price_snapshot.csv), so no AWS
access is needed. The model is local (ml/model/final). Scores computed on an old snapshot are
for show only: the features forward-fill the last known price up to "now".
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(ROOT, "ml", "api")
SNAPSHOT = os.path.join(ROOT, "demo", "price_snapshot.csv")
RESULTS = os.path.join(ROOT, "benchmark", "results")
PORT = int(os.environ.get("ARGUS_DEMO_PORT", "8765"))
REGION = "eu-north-1"
TYPES = ["m5.xlarge", "m5.2xlarge", "c5.xlarge", "c5.2xlarge", "g4dn.xlarge"]
AZS = ["eu-north-1a", "eu-north-1b", "eu-north-1c"]


def step(title, pause):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if pause:
        input("(enter to continue) ")


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30) as r:
        return json.load(r)


def start_api(live):
    env = dict(os.environ, AWS_REGION=REGION, PRICE_SOURCE="ec2" if live else "csv", PRICE_CSV=SNAPSHOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=API_DIR, env=env,
    )
    for _ in range(90):
        try:
            get("/health")
            return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            if proc.poll() is not None:
                raise SystemExit("API exited during startup; check the output above.")
            time.sleep(1)
    proc.terminate()
    raise SystemExit("API did not become ready in 90 s.")


def show_predictions():
    info = get("/model")
    thr = info["thresholds_by_region"]["eu-north-1_old5_types"]["alarm_budget_0.1"]
    rows = []
    for it in TYPES:
        for az in AZS:
            r = get(f"/predict?instance_type={it}&az={az}")
            rows.append((it, az, r["risk_score"], r["last_price_update_age_s"] // 60))
    rows.sort(key=lambda x: -x[2])
    print(f"{'instance':<14}{'az':<14}{'risk':>8}{'last update':>14}   action (alarm threshold {thr:.3f})")
    for it, az, s, age in rows:
        action = "ALARM -> checkpoint + migrate" if s >= thr else "ok"
        print(f"{it:<14}{az:<14}{s:>8.4f}{age:>10} min   {action}")


def show_benchmark():
    new = pd.read_csv(os.path.join(RESULTS, "trace_oct_new", "summary.csv"))
    old = pd.read_csv(os.path.join(RESULTS, "trace_oct_old", "summary.csv"))
    det = pd.read_csv(os.path.join(RESULTS, "trace_oct_new", "detection.csv"))
    d = det[(det.signal == "model") & (det.budget == 0.1)].iloc[0]
    print("Replay of the last 9 days of eu-north-1 Spot prices: 15 nodes, 66 price-spike events (proxy for reclaims)")
    print(f"New model at the 10% alarm budget catches {d.recall:.0%} of events with {d.precision:.0%} precision.\n")
    rows = [("no protection", "none", new), ("reactive on 2-min notice", "reactive", new),
            ("periodic, Young-Daly", "young_daly", new),
            ("OLD model alarms", "model_migrate_q0.1", old), ("NEW model alarms", "model_migrate_q0.1", new),
            ("oracle (perfect warning)", "oracle_migrate", new)]
    print(f"{'15-node job, slowdown vs ideal':<32}{'ckpt 10 s':>11}{'ckpt 60 s':>11}{'ckpt 300 s':>12}")
    for label, pol, src in rows:
        vals = [src[(src.N == 15) & (src.C_s == c) & (src.policy == pol)].slowdown.iloc[0] for c in (10, 60, 300)]
        print(f"{label:<32}{vals[0]:>11.3f}{vals[1]:>11.3f}{vals[2]:>12.3f}")
    print(
        "\nHonest reading: when a checkpoint fits the 2-minute notice, reactive is best. When it does not\n"
        "(300 s, large models), reactive collapses and the retrained model beats periodic checkpointing.\n"
        "The old model's skill did not survive six months out of time. Kills here are price spikes, not\n"
        "confirmed reclaims, and the two real reclaims on record showed no price signal\n"
        "(docs/wide_model_result.md)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--live", action="store_true", help="fetch fresh prices from EC2 (needs AWS credentials)")
    args = ap.parse_args()
    pause = not args.no_pause

    step("1/3  Start the prediction service (gradient boosting + Platt calibration)", pause)
    proc = start_api(args.live)
    try:
        step("2/3  Score every instance type / AZ and decide what the operator would do", pause)
        show_predictions()
        step("3/3  Does prediction pay? Trace-driven benchmark with the real model in the loop", pause)
        show_benchmark()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
