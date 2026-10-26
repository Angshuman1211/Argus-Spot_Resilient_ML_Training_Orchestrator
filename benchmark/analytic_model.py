"""
Analytic model of expected slowdown for each checkpoint policy, checked against the trace simulator,
and the decision map it implies (which policy is best for a given checkpoint time, MTBF and predictor).

Quantities (all in seconds of wall time):
  lam    job reclaim rate (a synchronous job dies when any node is reclaimed), 1 / MTBF
  C      checkpoint time, R restart time, n = 120 s notice
  r, p   predictor recall and precision against reclaims; alarms arrive at rate a = r * lam / p
  lead   how far ahead the predictor warns; an alarm saves the job only if the checkpoint
         finishes first, so useful recall is r_eff = r * P(lead >= C)   (the lead-time rule)

First-order slowdown (makespan / work), each the inverse of the useful fraction of wall time:
  none       (1/lam + R) (exp(lam W) - 1) / W                       restart from scratch
  periodic   (1 + C/tau) / (1 - lam ((tau + C)/2 + R))               Young-Daly optimum near sqrt(2 C / lam)
  reactive   1 / (1 - lam (n + R))          if C <= n, else none    flush on the notice, idle, restart
  predictor  1 / (1 - a (C + R) - lam (1 - r_eff) (1/b + R))         every alarm migrates; a miss loses
             with b = a + lam (1 - r_eff)                             the work since the last alarm or restart
  hybrid     periodic tau underneath the predictor: a miss loses min(periodic age, alarm age),
             E = (1 - exp(-b tau)) / b, plus C / tau of checkpoint overhead

The slowdown is convex in lam, and real reclaims cluster in time, so the check against the simulator
evaluates each formula at the reclaim rate of every job's own window and averages over jobs, as the
simulator does. Evaluated at the long-run mean rate it understates loss when rates swing day to day.
  oracle     1 / (1 - lam (C + R))

    python benchmark/analytic_model.py            # validation + decision map, results/analytic/
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import trace_sim as ts
from run_trace_benchmark import node_sets
from trace_sim import STEP, HORIZON_STEPS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "analytic")
NOTICE = 120.0
BIG = float("inf")


# ------------------------------------------------------------------------------------------ the model

def s_none(lam, W, R):
    if lam <= 0:
        return 1.0
    x = lam * W
    return BIG if x > 50 else (1.0 / lam + R) * math.expm1(x) / W


def s_periodic(lam, C, R, tau):
    lost = lam * ((tau + C) / 2.0 + R)
    return BIG if lost >= 1 else (1.0 + C / tau) / (1.0 - lost)


def best_tau(lam, C, R):
    """Interval that minimises s_periodic (a 1-D search; Young-Daly sqrt(2C/lam) is its first-order value)."""
    if lam <= 0:
        return BIG
    yd = math.sqrt(2.0 * C / lam)
    grid = yd * np.exp(np.linspace(-2.5, 2.5, 201))
    vals = [s_periodic(lam, C, R, t) for t in grid]
    return float(grid[int(np.argmin(vals))])


def s_reactive(lam, W, C, R, notice=NOTICE):
    if C > notice:
        return s_none(lam, W, R)
    lost = lam * (notice + R)
    return BIG if lost >= 1 else 1.0 / (1.0 - lost)


def s_predictor(lam, C, R, r, p, p_lead=1.0):
    if r <= 0 or p <= 0:
        return BIG
    return s_predictor_rate(lam, C, R, r * lam / p, r * p_lead)


def s_predictor_rate(lam, C, R, a, r_eff):
    """Same, from the alarm rate itself. The work lost on a miss is its age since the last durable state,
    which an alarm or a restart after an earlier miss resets: mean 1 / (a + lam (1 - r_eff))."""
    b = a + lam * (1.0 - r_eff)
    if b <= 0:
        return 1.0
    lost = a * (C + R) + lam * (1.0 - r_eff) * (1.0 / b + R)
    return BIG if lost >= 1 else 1.0 / (1.0 - lost)


def s_hybrid(lam, C, R, r, p, tau, p_lead=1.0):
    if r <= 0 or p <= 0:
        return s_periodic(lam, C, R, tau)
    a = r * lam / p
    r_eff = r * p_lead
    b = a + lam * (1.0 - r_eff)
    age = (1.0 - math.exp(-b * tau)) / b
    lost = a * (C + R) + lam * (1.0 - r_eff) * (age + R)
    return BIG if lost >= 1 else (1.0 + C / tau) / (1.0 - lost)


def s_hybrid_best(lam, C, R, r, p, p_lead=1.0):
    yd = math.sqrt(2.0 * C / lam)
    return min(s_hybrid(lam, C, R, r, p, t, p_lead) for t in yd * np.exp(np.linspace(-2.5, 3.5, 121)))


def s_oracle(lam, C, R):
    lost = lam * (C + R)
    return BIG if lost >= 1 else 1.0 / (1.0 - lost)


# ------------------------------------------------------------------- measuring lam, r, p, lead on a fleet

def job_alarm_stats(fleet, thr, C, cooldown_steps=HORIZON_STEPS):
    """Job-level view of a thresholded risk signal, as the simulator applies it: per-node alarms with a
    15-minute cooldown, a reclaim step is caught if every reclaimed node alarmed 1-3 steps earlier
    early enough for a C-second checkpoint to finish. Returns alarm rate, recall, precision and leads."""
    n, L = fleet.risk.shape
    alarm = np.zeros((n, L), bool)
    for j in range(n):
        last = -10 ** 9
        for i in np.flatnonzero(fleet.risk[j] > thr):
            if i - last >= cooldown_steps:
                alarm[j, i] = True
                last = i
    job_alarm_steps = alarm.any(axis=0)
    caught, leads = 0, []
    for i in fleet.kill_steps:
        nodes = np.flatnonzero(fleet.spike[:, i])
        best = []
        for j in nodes:
            hits = [k for k in range(max(0, i - HORIZON_STEPS), i) if alarm[j, k]]
            best.append((i - min(hits)) * STEP if hits else 0.0)
        lead = min(best) if best else 0.0
        leads.append(lead)
        if lead > 0 and lead >= C:
            caught += 1
    useful = 0
    for k in np.flatnonzero(job_alarm_steps):
        if any(fleet.spike[:, k + 1:k + 1 + HORIZON_STEPS].any(axis=0)):
            useful += 1
    span = L * STEP
    n_alarm = int(job_alarm_steps.sum())
    n_kill = max(len(fleet.kill_steps), 1)
    return dict(alarm_rate=n_alarm / span, recall=caught / n_kill, precision=useful / max(n_alarm, 1), leads=leads,
                alarm_steps=np.flatnonzero(job_alarm_steps))


def fleet_rate(fleet):
    return len(fleet.kill_steps) / (fleet.L * STEP)


# ------------------------------------------------------------------------------------------ validation

def _count(steps, lo, hi):
    return int(np.searchsorted(steps, hi) - np.searchsorted(steps, lo))


def validate(trace_path, runs_path, W, R, seed=7, n5=8, window=1.5):
    """Analytic slowdown per job window (reclaim and alarm rates measured over `window` x W after the
    job's start, no simulator output used), averaged over the same starts the simulator ran."""
    trace = ts.load_trace(trace_path)
    all_nodes = sorted(set(zip(trace.instance_type, trace.availability_zone)))
    sets = node_sets(all_nodes, np.random.default_rng(seed), n5)
    runs = pd.read_csv(runs_path)
    sim = (runs.groupby(["set", "C", "policy"])
               .agg(sim=("slowdown", "mean"), completion=("completed", "mean"), n=("slowdown", "size")).reset_index())
    starts_of = runs.groupby("set").start.unique()
    win_steps = int(window * W / STEP)
    rows = []
    for name, nodes in sets:
        calib = ts.build_fleet(trace, nodes, "calib")
        test = ts.build_fleet(trace, nodes, "test")
        kills = np.asarray(test.kill_steps)
        starts = np.sort(starts_of[name])
        lams = np.array([_count(kills, s, s + win_steps) / (win_steps * STEP) for s in starts])
        mtbf_cal = ts.fleet_mtbf_from_events(calib, calib.L * STEP)  # the simulator's Young-Daly uses node events
        for C in (10.0, 60.0, 300.0):
            f = {
                "none": lambda l: s_none(l, W, R),
                "reactive": lambda l: s_reactive(l, W, C, R),
                "periodic_30m": lambda l: s_periodic(l, C, R, 1800.0),
                "periodic_1h": lambda l: s_periodic(l, C, R, 3600.0),
                "young_daly": lambda l: s_periodic(l, C, R, ts.young_daly_tau(C, mtbf_cal)),
                "oracle_migrate": lambda l: s_oracle(l, C, R),
            }
            pred = {k: [g(l) for l in lams] for k, g in f.items()}
            for q in (0.01, 0.05, 0.1):
                st = job_alarm_stats(test, ts.threshold_for_budget(calib.risk, q), C)
                al = st["alarm_steps"]
                pred[f"model_migrate_q{q}"] = [
                    s_predictor_rate(l, C, R, _count(al, s, s + win_steps) / (win_steps * STEP), st["recall"])
                    for l, s in zip(lams, starts)]
            for pol, vals in pred.items():
                m = sim[(sim.set == name) & (sim.C == C) & (sim.policy == pol)]
                if m.empty:
                    continue
                m = m.iloc[0]
                vals = np.asarray(vals, float)
                lam = fleet_rate(test)
                rows.append(dict(set=name, N=len(nodes), C=C, policy=pol, mtbf_h=1 / lam / 3600 if lam else np.nan,
                                 analytic=float(vals.mean()) if np.isfinite(vals).all() else BIG,
                                 analytic_mean_rate=f[pol](lam) if pol in f else np.nan,
                                 sim=m.sim, completion=m.completion))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- decision map

POLICIES = ["reactive", "periodic", "predictor", "hybrid"]


def best_policy(lam, W, C, R, r, p, p_lead):
    v = {"reactive": s_reactive(lam, W, C, R),
         "periodic": s_periodic(lam, C, R, best_tau(lam, C, R)),
         "predictor": s_predictor(lam, C, R, r, p, p_lead),
         "hybrid": s_hybrid_best(lam, C, R, r, p, p_lead)}
    k = min(v, key=v.get)
    return k, v


def lead_cdf(leads):
    leads = np.asarray([x for x in leads if x > 0])
    return (lambda C: float((leads >= C).mean()) if len(leads) else 0.0)


MAP_RECALL = 0.65  # the trace model (0.69) and the real-reclaim capacity model (0.64) both sit here


def decision_maps(lam_ref, W, R, model_rp, p_lead_fn, real_lead_fn=None):
    """Best policy over checkpoint time x precision (recall fixed at MAP_RECALL), once with the trace model's
    lead and once with a measured real-reclaim lead distribution, and over checkpoint time x MTBF."""
    Cs = np.exp(np.linspace(math.log(1), math.log(3600), 70))
    ps = np.linspace(0.02, 1.0, 50)
    mt = np.exp(np.linspace(math.log(600), math.log(30 * 86400), 60))
    a_rows, b_rows = [], []
    for C in Cs:
        pl = p_lead_fn(C)
        for p in ps:
            k, v = best_policy(lam_ref, W, C, R, MAP_RECALL, p, pl)
            a_rows.append(dict(C=C, precision=p, lead="trace", best=k, **{f"s_{x}": v[x] for x in v}))
            if real_lead_fn is not None:
                k, v = best_policy(lam_ref, W, C, R, MAP_RECALL, p, real_lead_fn(C))
                a_rows.append(dict(C=C, precision=p, lead="real", best=k, **{f"s_{x}": v[x] for x in v}))
        for m in mt:
            k, v = best_policy(1 / m, W, C, R, model_rp[0], model_rp[1], pl)
            b_rows.append(dict(C=C, mtbf_s=m, best=k, **{f"s_{x}": v[x] for x in v}))
    return pd.DataFrame(a_rows), pd.DataFrame(b_rows)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    W, R = args.work_hours * 3600, args.restart_s
    val = validate(args.trace, args.runs, W, R)
    val.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    # No protection is left out: the simulator averages only the jobs that finish before the trace ends,
    # a survivor bias the closed form does not have. Cells where a policy cannot finish are left out too.
    ok = val[(val.policy != "none") & (val.completion >= 0.9) & np.isfinite(val.analytic) & (val.analytic < 5)]
    rel = (ok.analytic - ok.sim).abs() / ok.sim
    summ = {"cells": int(len(ok)), "cells_total": int((val.policy != "none").sum()),
            "median_rel_err": float(rel.median()), "p90_rel_err": float(rel.quantile(0.9)),
            "spearman": float(ok[["analytic", "sim"]].corr(method="spearman").iloc[0, 1]),
            "by_policy": {p: float(((g.analytic - g.sim).abs() / g.sim).median()) for p, g in ok.groupby("policy")},
            "by_nodes": {int(n): float(((g.analytic - g.sim).abs() / g.sim).median()) for n, g in ok.groupby("N")},
            "n15_signed": {f"{p}@{int(c)}": float((g.analytic.iloc[0] - g.sim.iloc[0]) / g.sim.iloc[0])
                           for (p, c), g in ok[ok.set == "N15"].groupby(["policy", "C"])}}
    # does the model pick the same winner as the simulator in each (set, C) cell?
    agree, total = 0, 0
    fam = lambda p: "model" if p.startswith("model_") else p
    for (_, _), g in ok.groupby(["set", "C"]):
        g = g.assign(f=g.policy.map(fam)).groupby("f").agg(analytic=("analytic", "min"), sim=("sim", "min"))
        if len(g) >= 3:
            total += 1
            agree += int(g.analytic.idxmin() == g.sim.idxmin())
    summ["winner_agreement"] = agree / max(total, 1)
    summ["winner_cells"] = total

    # reference point: the 15-node fleet on the test slice and the model at the 10% alarm budget
    trace = ts.load_trace(args.trace)
    nodes = sorted(set(zip(trace.instance_type, trace.availability_zone)))
    calib, test = ts.build_fleet(trace, nodes, "calib"), ts.build_fleet(trace, nodes, "test")
    lam = fleet_rate(test)
    st = job_alarm_stats(test, ts.threshold_for_budget(calib.risk, 0.1), 0.0)
    p_model = st["precision"]
    r_model = st["recall"]
    pl = lead_cdf(st["leads"])
    summ["reference"] = dict(N=len(nodes), mtbf_min=1 / lam / 60, recall=r_model, precision=p_model,
                             lead_median_s=float(np.median([x for x in st["leads"] if x > 0])) if any(st["leads"]) else None,
                             p_lead_ge_300=pl(300.0), p_lead_ge_600=pl(600.0))
    # crossover: smallest C at which the analytic predictor beats the analytic best periodic at the reference point
    cross = None
    for C in np.exp(np.linspace(0, math.log(3600), 400)):
        sp = s_predictor(lam, C, R, r_model, p_model, pl(C))
        sper = s_periodic(lam, C, R, best_tau(lam, C, R))
        sre = s_reactive(lam, W, C, R)
        if sp < min(sper, sre):
            cross = float(C)
            break
    summ["predictor_beats_both_from_C_s"] = cross
    # quality needed: at each C, the smallest r = p at which a predictor (lead always enough) beats periodic and reactive
    need = []
    for C in (30.0, 60.0, 120.0, 300.0, 600.0, 1800.0):
        q_need = None
        for q in np.linspace(0.01, 1.0, 100):
            if s_predictor(lam, C, R, q, q) < min(s_periodic(lam, C, R, best_tau(lam, C, R)), s_reactive(lam, W, C, R)):
                q_need = float(q)
                break
        need.append(dict(C=C, quality_needed=q_need))
    summ["quality_needed"] = need
    # the predictors measured on REAL reclaims (ml/model/real_label_quality.py): recall, precision and lead
    # distribution at a 10% alarm budget, put through the same formulas at the reference reclaim rate
    rq = os.path.join(HERE, "..", "ml", "model", "wide_results", "reclaim_quality.json")
    if os.path.exists(rq):
        real = {}
        for name, v in json.load(open(rq))["predictors"].items():
            op = v.get("operating")
            if not op:
                continue
            leads = np.asarray(op["leads_min"]) * 60.0
            pts = []
            for C in (60.0, 120.0, 300.0, 600.0, 1800.0):
                pl_ = float((leads >= C).mean()) if len(leads) else 0.0
                v_ = {"reactive": s_reactive(lam, W, C, R), "periodic": s_periodic(lam, C, R, best_tau(lam, C, R)),
                      "predictor": s_predictor(lam, C, R, op["recall"], op["precision"], pl_),
                      "hybrid": s_hybrid_best(lam, C, R, op["recall"], op["precision"], pl_)}
                pts.append(dict(C=C, p_lead=pl_, best=min(v_, key=v_.get), **v_))
            real[name] = dict(recall=op["recall"], precision=op["precision"], points=pts)
        summ["real_reclaim_predictors"] = real
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    real_pl = None
    cap = summ.get("real_reclaim_predictors", {}).get("capacity model, held-out pool")
    if cap:
        leads = np.asarray(json.load(open(rq))["predictors"]["capacity model, held-out pool"]["operating"]["leads_min"]) * 60
        real_pl = lambda C: float((leads >= C).mean())
    a, b = decision_maps(lam, W, R, (r_model, p_model), pl, real_pl)
    a.to_csv(os.path.join(OUT, "map_quality.csv"), index=False)
    b.to_csv(os.path.join(OUT, "map_mtbf.csv"), index=False)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.path.join(HERE, "traces", "trace_scores_new.csv.gz"))
    ap.add_argument("--runs", default=os.path.join(HERE, "results", "trace_oct_new", "runs.csv.gz"))
    ap.add_argument("--work-hours", type=float, default=12.0)
    ap.add_argument("--restart-s", type=float, default=120.0)
    main(ap.parse_args())
