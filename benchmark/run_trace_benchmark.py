"""
Trace-driven benchmark: does prediction pay when it must actually predict?

Replays the held-out test slice of real eu-north-1 Spot price series. Compares:
  none | periodic | Young-Daly | Advisor-tier Young-Daly | reactive-on-notice |
  model (checkpoint+migrate, checkpoint-only) | volatility rule | random | oracle.

Thresholds are chosen on the CALIBRATION split, as alarm budgets (fraction of node-steps
that alarm), then applied unchanged to the TEST split. Every signal arm gets the same
budgets, so the comparison is "who spends an alarm better", not "who alarms more".

    python benchmark/run_trace_benchmark.py            # full sweep
    python benchmark/run_trace_benchmark.py --quick    # small, for a smoke test

Outputs (benchmark/results/trace/): summary.csv, detection.csv, predictor_quality.csv,
runs.csv.gz, and figures.
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import trace_sim as ts
from trace_sim import Policy, STEP

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "traces", "trace_scores.csv.gz")
OUT = os.path.join(HERE, "results", "trace")

BUDGETS = [0.0005, 0.001, 0.002, 0.005, 0.01]
CKPT_ONLY_BUDGETS = [0.002, 0.01]
HYBRID_BUDGETS = [0.002, 0.01]
SYN_RECALL = [0.2, 0.5, 0.8]
SYN_PRECISION = [0.1, 0.3, 0.6]
OD_PRICE = {"m5.xlarge": 0.204, "m5.2xlarge": 0.408, "c5.xlarge": 0.188,
            "c5.2xlarge": 0.376, "g4dn.xlarge": 0.605}


def node_sets(all_nodes, rng, n5_subsets):
    sets = [("N15", list(all_nodes))]
    for s in range(n5_subsets):
        idx = sorted(rng.choice(len(all_nodes), 5, replace=False))
        sets.append((f"N5_{s}", [all_nodes[i] for i in idx]))
    for i, node in enumerate(all_nodes):
        sets.append((f"N1_{i}", [node]))
    return sets


def policies_for(calib, C, W):
    """Policy list for one fleet and checkpoint cost, thresholds fitted on `calib` only."""
    calib_span = calib.L * STEP
    mtbf_calib = ts.fleet_mtbf_from_events(calib, calib_span)
    pols = [
        Policy("none", "none"),
        Policy("periodic_30m", "periodic", tau=1800.0),
        Policy("periodic_1h", "periodic", tau=3600.0),
        Policy("young_daly", "periodic", tau=ts.young_daly_tau(C, mtbf_calib)),
        Policy("advisor_young_daly", "periodic", tau=ts.young_daly_tau(C, ts.advisor_mtbf(calib))),
        Policy("reactive", "reactive"),
        Policy("oracle_migrate", "oracle", action="migrate"),
    ]
    for q in BUDGETS:
        pols.append(Policy(f"model_migrate_q{q}", "signal", signal="risk",
                           thr=ts.threshold_for_budget(calib.risk, q), action="migrate"))
        pols.append(Policy(f"vol_migrate_q{q}", "signal", signal="vol",
                           thr=ts.threshold_for_budget(calib.vol, q), action="migrate"))
        pols.append(Policy(f"random_migrate_q{q}", "random", thr=q, action="migrate"))
    tau_yd = ts.young_daly_tau(C, mtbf_calib)
    for q in HYBRID_BUDGETS:
        pols.append(Policy(f"hybrid_yd_model_migrate_q{q}", "hybrid", tau=tau_yd, signal="risk",
                           thr=ts.threshold_for_budget(calib.risk, q), action="migrate"))
        pols.append(Policy(f"hybrid_yd_model_ckpt_q{q}", "hybrid", tau=tau_yd, signal="risk",
                           thr=ts.threshold_for_budget(calib.risk, q), action="ckpt"))
    kappa = max(int(calib.spike.sum()), 1) / (calib.n * calib.L)  # kills per node-step
    for rec, prec in itertools.product(SYN_RECALL, SYN_PRECISION):
        fa = kappa * rec * (1.0 / prec - 1.0)
        pols.append(Policy(f"synthetic_r{rec}_p{prec}", "synthetic", thr=rec, fa=fa, action="migrate"))
    for q in CKPT_ONLY_BUDGETS:
        pols.append(Policy(f"model_ckpt_q{q}", "signal", signal="risk",
                           thr=ts.threshold_for_budget(calib.risk, q), action="ckpt"))
    return pols


def bootstrap_ci(df, col, rng, n_boot=400):
    blocks = [g[col].to_numpy() for _, g in df.groupby("block")]
    if len(blocks) < 2:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        means.append(np.concatenate([blocks[i] for i in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(args):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    trace = ts.load_trace(TRACE)
    all_nodes = sorted(set(zip(trace.instance_type, trace.availability_zone)))
    sets = node_sets(all_nodes, rng, args.n5_subsets)
    if args.quick:
        sets = [s for s in sets if s[0] in ("N15", "N5_0", "N1_0")]
    W = args.work_hours * 3600.0
    R = args.restart_s
    max_len = int(args.max_makespan_factor * W / STEP)
    block_len = max(1, int(round(args.work_hours / args.start_every_h)))  # one block ~ one job length

    records, detection = [], []
    t0 = time.time()
    for set_name, nodes in sets:
        n_nodes = len(nodes)
        calib = ts.build_fleet(trace, nodes, "calib")
        test = ts.build_fleet(trace, nodes, "test")
        last_start = test.L - 1 - max_len
        if last_start < 1:
            print(f"skip {set_name}: test window too short")
            continue
        starts = list(range(0, last_start, int(args.start_every_h * 3600 / STEP)))
        spot = float(test.price.sum())
        od = float(sum(OD_PRICE[n[0]] for n in nodes))
        for C in args.ckpt_s:
            for pol in policies_for(calib, C, W):
                for si, s in enumerate(starts):
                    o = ts.simulate(test, pol, s, W, C, R, seed=args.seed + si)
                    o.update(set=set_name, N=n_nodes, C=C, policy=pol.name, start=s, block=f"{set_name}-{si // block_len}",
                             spot_cost=(o["makespan"] / 3600.0) * spot if o["completed"] else float("nan"),
                             od_cost=(W / 3600.0) * od)
                    records.append(o)
        if set_name == "N15":
            for q in BUDGETS:
                for sig, fl in (("risk", "model"), ("vol", "volatility")):
                    thr = ts.threshold_for_budget(getattr(calib, "risk" if sig == "risk" else "vol"), q)
                    d = ts.detection_stats(test, sig, thr)
                    d.update(signal=fl, budget=q)
                    detection.append(d)
        print(f"{set_name}: {len(starts)} starts, {time.time() - t0:.0f}s elapsed", flush=True)

    runs = pd.DataFrame(records)
    runs["slowdown"] = runs["makespan"] / W
    runs["wasted_frac"] = runs["wasted"] / W
    runs["cost_savings_pct"] = 100.0 * (1.0 - runs["spot_cost"] / runs["od_cost"])
    runs.to_csv(os.path.join(OUT, "runs.csv.gz"), index=False, compression="gzip")

    rows = []
    for (N, C, pol), g in runs.groupby(["N", "C", "policy"]):
        done = g[g.completed]
        lo, hi = bootstrap_ci(done, "slowdown", rng) if len(done) else (np.nan, np.nan)
        rows.append(dict(
            N=N, C_s=C, policy=pol, runs=len(g), completion=g.completed.mean(),
            slowdown=done.slowdown.mean(), slowdown_lo=lo, slowdown_hi=hi,
            wasted_frac=g.wasted_frac.mean(), kills=g.kills.mean(), avoided=g.avoided.mean(),
            n_ckpt=g.n_ckpt.mean(), n_migr=g.n_migr.mean(), alarms=g.alarms.mean(),
            cost_savings_pct=done.cost_savings_pct.mean(),
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(detection).to_csv(os.path.join(OUT, "detection.csv"), index=False)
    predictor_quality(trace)
    print(f"done in {time.time() - t0:.0f}s; wrote {len(runs)} runs")


def predictor_quality(trace):
    """Window-level AUROC / AUPRC on the test split for model vs non-learned signals."""
    rows = []
    for split in ("calib", "test"):
        x = trace[trace.split == split]
        y = x.future_spike.to_numpy()
        for name, col in (("model", "risk_score"), ("volatility_rule", "vol_cv"), ("advisor_prior", "advisor_rate")):
            rows.append(dict(split=split, signal=name, windows=len(x), positives=int(y.sum()),
                             base_rate=float(y.mean()), auprc=average_precision_score(y, x[col]),
                             lift=average_precision_score(y, x[col]) / float(y.mean()),
                             auroc=roc_auc_score(y, x[col])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "predictor_quality.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work-hours", type=float, default=12.0)
    ap.add_argument("--restart-s", type=float, default=120.0)
    ap.add_argument("--ckpt-s", type=float, nargs="+", default=[10.0, 60.0, 300.0])
    ap.add_argument("--start-every-h", type=float, default=3.0)
    ap.add_argument("--n5-subsets", type=int, default=8)
    ap.add_argument("--max-makespan-factor", type=float, default=3.0)
    run(ap.parse_args())
