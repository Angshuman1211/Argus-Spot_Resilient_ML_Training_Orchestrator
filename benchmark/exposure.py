"""
Does a synchronous N-node job really see 1 - (1 - p)^N of the interruptions? That formula assumes nodes
are reclaimed independently. Two checks:

  trace      on the eu-north-1 price-spike trace the simulator replays, the measured rate of steps with
             at least one event on any node of an N-node job, against the independence prediction from
             each node's own rate (N = 1, 5, 15, the same node sets as the benchmark)
  real       on the sensor campaign's real reclaims: how many reclaims land in the same 15-minute window
             in one region, against a Poisson count with the same mean (index of dispersion; 1 = no clustering),
             and the share of reclaims that share their window with a reclaim in another pool

    python benchmark/exposure.py
Output: results/analytic/exposure.json
"""
import json
import os

import numpy as np
import pandas as pd

import trace_sim as ts
from run_trace_benchmark import node_sets
from trace_sim import STEP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "results", "analytic")


def trace_check(trace_path, seed=7, n5=8):
    trace = ts.load_trace(trace_path)
    nodes = sorted(set(zip(trace.instance_type, trace.availability_zone)))
    rows = []
    for name, ns in node_sets(nodes, np.random.default_rng(seed), n5):
        f = ts.build_fleet(trace, ns, "test")
        p_node = f.spike.mean(axis=1)                        # per-node probability of an event in a step
        indep = 1.0 - np.prod(1.0 - p_node)                  # independence prediction for the job
        observed = f.spike.any(axis=0).mean()
        rows.append(dict(set=name, N=len(ns), p_node_mean=float(p_node.mean()), independent=float(indep),
                         observed=float(observed), ratio=float(observed / indep) if indep else np.nan,
                         events_per_kill_step=float(f.spike.sum() / max(f.spike.any(axis=0).sum(), 1))))
    d = pd.DataFrame(rows)
    by_n = d.groupby("N").agg(independent=("independent", "mean"), observed=("observed", "mean"),
                              ratio=("ratio", "mean"), events_per_kill_step=("events_per_kill_step", "mean"))
    return {int(n): r.to_dict() for n, r in by_n.iterrows()}


def real_check(window_min=15):
    r = pd.read_csv(os.path.join(ROOT, "ml", "data", "real", "real_reclaims.csv"), parse_dates=["reclaim_time"])
    r["pool"] = r.instance_type + "|" + r.availability_zone
    r["win"] = r.reclaim_time.dt.floor(f"{window_min}min")
    # only windows in which sensors were running in that region (the CloudTrail reclaims fall outside them)
    st = pd.read_csv(os.path.join(ROOT, "ml", "model", "wide_results", "real_reclaim_steps.csv.gz"), parse_dates=["t"])
    st["win"] = st.t.dt.floor(f"{window_min}min")
    active = st.groupby("region").win.unique()
    out = {}
    shared, total = 0, 0
    for region, g in r.groupby("region"):
        if region not in active or len(g) < 5:
            continue
        span = pd.DatetimeIndex(sorted(active[region]))
        g = g[g.win.isin(span)]
        counts = g.groupby("win").size().reindex(span, fill_value=0)
        m, v = counts.mean(), counts.var()
        per_win_pools = g.groupby("win").pool.nunique()
        sh = int(sum(n for w, n in g.groupby("win").size().items() if per_win_pools[w] >= 2))
        out[region] = dict(reclaims=int(len(g)), windows=int(len(span)), mean=float(m), var=float(v),
                           dispersion=float(v / m) if m else np.nan, reclaims_sharing_window_with_other_pool=sh)
        shared += sh
        total += len(g)
    out["all"] = dict(reclaims=total, share_with_other_pool=shared / max(total, 1))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(trace=trace_check(os.path.join(HERE, "traces", "trace_scores_new.csv.gz")), real=real_check())
    json.dump(res, open(os.path.join(OUT, "exposure.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
