"""
Two design checks on the checkpoint path (fix F18).

1. What the decoupled flush handshake costs. The operator writes a request id to S3, the pod looks for it
   every FLUSH_POLL_SECONDS, saves, and writes the id back; the operator looks for the reply every
   CHECKPOINT_WAIT_POLL_SECONDS. A direct design (the operator execs into the pod and saves itself) would skip
   both polls but couples the operator to the training code and needs pods/exec rights. The handshake's extra
   latency is the two poll waits; it is simulated here (uniform poll phase), compared with the two real-cluster
   drills (acknowledgement time minus the measured save time), and pushed through the analytic model to see
   what it does to a job's slowdown.
2. Whether a checkpoint fits the SIGTERM path. On a real reclaim the Node Termination Handler drains the node
   when the 2-minute notice arrives, the pod gets SIGTERM, and SIGKILL follows after the pod's grace period
   (90 s in our manifests, 30 s by Kubernetes default) or when the node goes, whichever is first. At the
   measured in-region S3 rate, which model sizes can still flush?

    python benchmark/handshake_grace.py
Output: results/analytic/handshake_grace.json
"""
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd

import analytic_model as am

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "results", "analytic")


def _default(path, name):
    m = re.search(rf'{name}\s*=\s*\w+\(os\.environ\.get\("{name}",\s*"([0-9.]+)"\)\)', open(path, encoding="utf-8").read())
    return float(m.group(1))


def _drill_ack(tag):
    """Seconds from the operator's last flush request to its confirmation, from the drill's operator log."""
    p = os.path.join(ROOT, "demo", "drill", "results", f"{tag}.operator.log")
    lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    ts = lambda l: dt.datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S,%f")
    trig = [i for i, l in enumerate(lines) if "[CHECKPOINT] Trigger" in l][-1]
    ok = next(l for l in lines[trig:] if "checkpoint confirmed on S3" in l)
    return (ts(ok) - ts(lines[trig])).total_seconds()


def main():
    os.makedirs(OUT, exist_ok=True)
    pod_poll = _default(os.path.join(ROOT, "ml", "cifar10_job", "train.py"), "FLUSH_POLL_SECONDS")
    op_poll = _default(os.path.join(ROOT, "operator", "controller", "handlers.py"), "CHECKPOINT_WAIT_POLL_SECONDS")
    grace_ours = 90.0
    rng = np.random.default_rng(0)
    extra = rng.uniform(0, pod_poll, 100000) + rng.uniform(0, op_poll, 100000)

    eks = pd.read_csv(os.path.join(HERE, "results", "trace", "checkpoint_write_eks.csv"))
    big = eks[eks.size_mb >= 512]
    rate = float((big.size_mb / (big.upload_ms / 1000)).mean())  # MB/s per node, measured on the EKS nodes

    drills = {}
    for tag, mb in (("drill-small", 4.0), ("drill-big", 2048.0)):
        ack = _drill_ack(tag)
        drills[tag] = dict(ack_s=ack, checkpoint_mb=mb, upload_s_at_measured_rate=mb / rate,
                           ack_minus_upload_s=ack - mb / rate)

    # what the extra latency does to a 15-node job, at the reference reclaim rate of the trace benchmark
    s = json.load(open(os.path.join(OUT, "summary.json")))["reference"]
    lam = 1.0 / (s["mtbf_min"] * 60.0)
    R = 120.0
    impact = []
    for C in (10.0, 60.0, 300.0):
        base = am.s_predictor(lam, C, R, s["recall"], s["precision"])
        slow = am.s_predictor(lam, C + float(extra.mean()), R, s["recall"], s["precision"])
        impact.append(dict(C=C, slowdown_direct=base, slowdown_handshake=slow, added_pct=100 * (slow / base - 1)))

    # SIGTERM path: largest checkpoint that can be written before SIGKILL
    budgets = {"kubernetes default grace (30 s)": 30.0, "our manifests (90 s)": grace_ours,
               "the whole 2-minute notice (120 s)": 120.0}
    fits = {k: dict(seconds=v, max_checkpoint_gb=v * rate / 1000.0, max_params_billion=v * rate / 1000.0 / 12.0)
            for k, v in budgets.items()}
    models = [("ResNet-50", 25.6e6), ("BERT-large", 340e6), ("1.3B LLM", 1.3e9), ("7B LLM", 7e9)]
    table = []
    for name, p in models:
        sec = p * 12 / 1e6 / rate
        table.append(dict(model=name, params=p, flush_s=sec, **{k: bool(sec <= v) for k, v in budgets.items()}))

    out = dict(pod_poll_s=pod_poll, operator_poll_s=op_poll,
               handshake_extra_s=dict(mean=float(extra.mean()), p95=float(np.percentile(extra, 95)), max=pod_poll + op_poll),
               measured_rate_mb_s=rate, drills=drills, slowdown_impact=impact, sigterm_budget=fits, models=table)
    json.dump(out, open(os.path.join(OUT, "handshake_grace.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
