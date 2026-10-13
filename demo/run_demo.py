"""
Local live demo of Argus: starts the real prediction API, scores every instance/AZ pair,
then prints the headline trace-benchmark results (read from benchmark/results/trace/summary.csv).

    python demo/run_demo.py            # full walkthrough, pauses between steps
    python demo/run_demo.py --no-pause # for recording or CI

No AWS access needed: the model, scaler, calibrator and feature history are all local.
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
SUMMARY = os.path.join(ROOT, "benchmark", "results", "trace", "summary.csv")
QUALITY = os.path.join(ROOT, "benchmark", "results", "trace", "predictor_quality.csv")
FEATURES = os.path.join(ROOT, "ml", "data", "features.csv")
PORT = 8000
THRESHOLD = 0.0015  # operator alarm threshold on the calibrated scale (README)


def step(title, pause):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if pause:
        input("(enter to continue) ")


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.load(r)


def start_api():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=API_DIR,
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
    pairs = pd.read_csv(FEATURES, usecols=["instance_type", "availability_zone"]).drop_duplicates()
    rows = []
    for it, az in pairs.itertuples(index=False):
        r = get(f"/predict?instance_type={it}&az={az}")
        rows.append((it, az, r["risk_score"]))
    rows.sort(key=lambda x: -x[2])
    print(f"{'instance':<14}{'az':<14}{'risk':>8}   action (threshold {THRESHOLD})")
    for it, az, s in rows:
        action = "ALARM -> checkpoint + migrate" if s >= THRESHOLD else "ok"
        print(f"{it:<14}{az:<14}{s:>8.4f}   {action}")


def show_benchmark():
    q = pd.read_csv(QUALITY)
    t = q[q.split == "test"].set_index("signal")
    print("Predictor quality on the held-out test slice (proxy labels: price-spike events)")
    print(t[["positives", "auroc", "lift"]].round(3).to_string())

    s = pd.read_csv(SUMMARY)
    s = s[(s.N == 1) & (s.C_s == 60.0)]
    keep = ["none", "reactive", "young_daly", "model_migrate_q0.002", "random_migrate_q0.002", "oracle_migrate"]
    s = s[s.policy.isin(keep)].set_index("policy").loc[keep]
    print("\nTrace benchmark, 1 job, 60 s checkpoint write (slowdown 1.0 = never interrupted)")
    print(s[["slowdown", "wasted_frac", "cost_savings_pct"]].round(3).to_string())
    print(
        "\nHonest reading: the model has real signal (AUROC ~0.75 vs 0.44 for a volatility rule) but at\n"
        "this recall/precision it does not beat reactive (checkpoint fits the 2-min notice) or\n"
        "Young-Daly periodic checkpointing. The oracle row shows the headroom a better predictor could capture."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    pause = not ap.parse_args().no_pause

    step("1/3  Start the prediction service (Transformer + Platt calibration)", pause)
    proc = start_api()
    try:
        step("2/3  Score every instance type / AZ and decide what the operator would do", pause)
        show_predictions()
        step("3/3  Does prediction pay? Trace-driven benchmark on real model scores", pause)
        show_benchmark()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
