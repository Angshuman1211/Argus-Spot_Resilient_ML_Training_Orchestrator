"""
Trace-driven checkpoint-policy simulator.

Replaces the Poisson-kill harness for the question "does PREDICTION pay?". Poisson
kills have no signal to learn, and the old predictive arm fired at (kill time - lead),
i.e. it was an oracle. Here:

  * interruptions come from a real trace: price-spike events (>1% jump), the same
    proxy the model was trained on, on real eu-north-1 Spot price series;
  * a policy only sees what it could see at decision time: the shipped model's
    calibrated risk score for each node (ml/model/score_trace.py), or a non-learned
    signal computed from the same history;
  * the job is a synchronous N-node job: a reclaim of ANY node kills it.

Real reclaim labels are not available, so "kill" means "proxy spike event". That is
a stated limitation of the whole study, not hidden here.

Time model (seconds, trace time, 5-minute decision grid):
  a decision at grid step i uses features up to step i, so a trigger there can protect
  against kills at steps i+1.. i+3 (the 5-15 minute horizon the label was built on).
  Kills at step i are processed before decisions at step i.

Job model: needs W seconds of compute. While running, work accrues 1:1. A checkpoint
pauses the job for C seconds (synchronous snapshot, the same behaviour the operator now
enforces by waiting for the flush) and is durable only if it completes. A kill loses all
work since the last durable checkpoint and costs R seconds to restart. A migration is a
checkpoint followed by an R-second move to fresh capacity; the moved nodes cannot be
reclaimed for the prediction horizon plus R (assumption, stated in the write-up). After a
reactive (notice-driven) checkpoint the node is drained, so the job idles until the kill.
"""
import bisect
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STEP = 300.0  # seconds per trace step
HORIZON_STEPS = 3  # label horizon: spikes within the next 3 steps (15 min)
INF = float("inf")


# ----------------------------------------------------------------------------- data

@dataclass
class Fleet:
    """Dense per-node arrays on a shared 5-minute grid, for one split."""
    nodes: list                 # [(instance_type, az), ...]
    k0: int                     # absolute grid index of column 0
    risk: np.ndarray            # [n, L] calibrated model risk score
    vol: np.ndarray             # [n, L] non-learned volatility signal
    spike: np.ndarray           # [n, L] bool, kill event at this step
    future: np.ndarray          # [n, L] bool, label: a spike in the next 3 steps
    advisor_rate: np.ndarray    # [n] AWS Spot Advisor max interruption fraction per month
    price: np.ndarray           # [n] mean spot price, $/hr
    kill_steps: list = field(default_factory=list)  # sorted steps with >=1 kill

    @property
    def L(self):
        return self.risk.shape[1]

    @property
    def n(self):
        return len(self.nodes)


def load_trace(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["k"] = (df["timestamp"].astype("int64") // 10**9 // int(STEP)).astype(int)
    return df


def build_fleet(trace, nodes, split):
    """Intersect the nodes' spans for `split` and return dense arrays over it."""
    sub = {}
    for key in nodes:
        g = trace[(trace.instance_type == key[0]) & (trace.availability_zone == key[1])
                  & (trace.split == split)].sort_values("k")
        if g.empty:
            raise ValueError(f"no {split} rows for {key}")
        sub[key] = g
    k0 = max(g.k.min() for g in sub.values())
    k1 = min(g.k.max() for g in sub.values())
    if k1 <= k0:
        raise ValueError("nodes have no common window")
    idx = np.arange(k0, k1 + 1)
    cols = {c: [] for c in ("risk_score", "vol_cv", "spike_now", "future_spike")}
    adv, price = [], []
    for key in nodes:
        g = sub[key].drop_duplicates("k").set_index("k").reindex(idx)
        if g["risk_score"].isna().any():
            raise ValueError(f"gap in trace for {key}")
        for c in cols:
            cols[c].append(g[c].to_numpy())
        adv.append(float(g["advisor_rate"].iloc[0]))
        price.append(float(g["spot_price"].mean()))
    spike = np.array(cols["spike_now"]).astype(bool)
    fleet = Fleet(
        nodes=list(nodes), k0=int(k0),
        risk=np.array(cols["risk_score"]), vol=np.array(cols["vol_cv"]),
        spike=spike, future=np.array(cols["future_spike"]).astype(bool),
        advisor_rate=np.array(adv), price=np.array(price),
    )
    fleet.kill_steps = [int(i) for i in np.flatnonzero(spike.any(axis=0))]
    return fleet


# --------------------------------------------------------------------------- policy

@dataclass
class Policy:
    name: str
    kind: str                    # none | periodic | reactive | signal | hybrid | oracle | random | synthetic
    tau: float = 0.0             # periodic interval (s of compute); also active for kind == hybrid
    signal: str = "risk"         # risk | vol   (kind == signal | hybrid)
    thr: float = INF             # alarm threshold; per-node-step probability (random); recall (synthetic)
    action: str = "migrate"      # ckpt | migrate
    fa: float = 0.0              # synthetic: false-alarm probability per node-step


def young_daly_tau(C, mtbf):
    """Optimal periodic checkpoint interval, sqrt(2 * C * MTBF)."""
    return math.sqrt(2.0 * C * mtbf)


def fleet_mtbf_from_events(fleet, span_seconds):
    n_events = int(fleet.spike.sum())
    return span_seconds / max(n_events, 1)


def advisor_mtbf(fleet):
    """Fleet MTBF implied by the Spot Advisor's published monthly interruption bands."""
    month = 30 * 86400.0
    lam = 0.0
    for p in fleet.advisor_rate:
        p = min(max(float(p), 1e-4), 0.99)
        lam += -math.log(1.0 - p) / month
    return 1.0 / lam


def threshold_for_budget(scores, budget):
    """Threshold that alarms on `budget` of calibration node-steps."""
    return float(np.quantile(np.asarray(scores).ravel(), 1.0 - budget))


# ------------------------------------------------------------------------ simulator

def simulate(fleet, pol, start_i, W, C, R, notice=120.0, cooldown=HORIZON_STEPS * STEP,
             protect=HORIZON_STEPS * STEP, seed=0):
    """Run one job from grid column `start_i`. Returns a dict of outcomes."""
    rng = np.random.default_rng(seed)
    n = fleet.n
    T_end = (fleet.L - 1) * STEP
    t = start_i * STEP
    t_start = t

    work = saved = snapshot = 0.0
    mode, mode_end, migrating, drain = "run", INF, None, False
    last_alarm = np.full(n, -INF)
    prot = np.full(n, -1.0)
    last_ckpt_start = t
    next_dec = math.ceil(t / STEP) * STEP

    kill_steps = fleet.kill_steps
    notice_times = [i * STEP - notice for i in kill_steps] if pol.kind == "reactive" else []
    ni = bisect.bisect_left(notice_times, t - 1e-9)

    out = dict(wasted=0.0, n_ckpt=0, n_migr=0, kills=0, avoided=0, alarms=0, dropped=0,
               ckpt_time=0.0, restart_time=0.0, completed=False, makespan=float("nan"))

    def start_ckpt(nodes=None, is_drain=False):
        nonlocal mode, mode_end, snapshot, last_ckpt_start, migrating, drain
        mode, mode_end, snapshot, last_ckpt_start = "ckpt", t + C, work, t
        migrating, drain = nodes, is_drain

    while True:
        cands = [next_dec]
        cands.append(t + (W - work) if mode == "run" else mode_end)
        if pol.tau > 0 and mode == "run":
            cands.append(max(t, last_ckpt_start + pol.tau))
        if ni < len(notice_times) and mode == "run":
            cands.append(max(t, notice_times[ni]))
        t_next = min(cands)
        if t_next > T_end:
            out["work_done"] = work
            return out
        if mode == "run":
            work += t_next - t
        t = t_next
        if mode == "run" and work >= W - 1e-6:
            out.update(completed=True, makespan=t - t_start, work_done=W)
            return out

        # 1. a checkpoint or restart finishes
        if mode != "run" and t >= mode_end - 1e-9:
            if mode == "ckpt":
                saved = snapshot
                out["n_ckpt"] += 1
                out["ckpt_time"] += C
                if migrating:
                    out["n_migr"] += 1
                    out["restart_time"] += R
                    for j in migrating:
                        prot[j] = t + R + protect
                    mode, mode_end, migrating = "restart", t + R, None
                elif drain:
                    # reactive path: the node is being drained after the flush, so the
                    # job stops computing until the reclaim lands (SIGTERM -> flush -> exit)
                    mode, mode_end, drain = "idle", INF, False
                else:
                    mode, mode_end = "run", INF
            else:
                mode, mode_end = "run", INF

        on_grid = abs(t - next_dec) < 1e-9
        i = int(round(t / STEP))

        # 2. kills at this step
        if on_grid:
            killed = False
            for j in np.flatnonzero(fleet.spike[:, i]):
                if prot[j] >= t:
                    out["avoided"] += 1
                else:
                    killed = True
            if killed:
                out["kills"] += 1
                if mode != "restart":
                    out["wasted"] += work - saved
                    work = saved
                    out["restart_time"] += R
                    mode, mode_end, migrating = "restart", t + R, None

        # 3. reactive notice fires
        if ni < len(notice_times) and abs(t - notice_times[ni]) < 1e-9:
            ni += 1
            if mode == "run":
                start_ckpt(is_drain=True)
        elif ni < len(notice_times) and notice_times[ni] < t - 1e-9:
            ni = bisect.bisect_left(notice_times, t - 1e-9)

        # 4. decision step
        if on_grid:
            next_dec += STEP
            alarmed = _alarmed(fleet, pol, i, t, last_alarm, cooldown, rng)
            if len(alarmed):
                if mode == "run":
                    out["alarms"] += 1
                    last_alarm[alarmed] = t
                    start_ckpt(list(alarmed) if pol.action == "migrate" else None)
                else:
                    out["dropped"] += 1

        # 5. periodic timer
        if pol.tau > 0 and mode == "run" and t >= last_ckpt_start + pol.tau - 1e-9:
            start_ckpt()


def _alarmed(fleet, pol, i, t, last_alarm, cooldown, rng):
    if pol.kind in ("signal", "hybrid"):
        sig = fleet.risk if pol.signal == "risk" else fleet.vol
        hit = sig[:, i] > pol.thr
    elif pol.kind == "synthetic":
        # a predictor with a chosen recall and false-alarm rate, 10 minutes of lead:
        # shows how good a predictor must be before it pays, independent of this model
        hit = rng.random(fleet.n) < pol.fa
        if i + 2 < fleet.L:
            hit = hit | (fleet.spike[:, i + 2] & (rng.random(fleet.n) < pol.thr))
    elif pol.kind == "random":
        hit = rng.random(fleet.n) < pol.thr
    elif pol.kind == "oracle":
        # perfect foresight with the minimum 5-minute lead: the old benchmark's assumption
        hit = fleet.spike[:, i + 1] if i + 1 < fleet.L else np.zeros(fleet.n, bool)
    else:
        return np.array([], dtype=int)
    ok = hit & (t - last_alarm >= cooldown)
    return np.flatnonzero(ok)


# ---------------------------------------------------------------------- signal quality

def detection_stats(fleet, signal, thr, horizon=HORIZON_STEPS):
    """How well a thresholded signal anticipates kills, with no cooldown or action."""
    sig = fleet.risk if signal == "risk" else fleet.vol
    alarm = sig > thr
    n, L = alarm.shape
    kills = np.argwhere(fleet.spike)
    caught, leads = 0, []
    for node, i in kills:
        lo = max(0, i - horizon)
        win = alarm[node, lo:i]
        if win.any():
            caught += 1
            first = lo + int(np.argmax(win))
            leads.append((i - first) * STEP)
    n_alarm = int(alarm.sum())
    # an alarm is useful if a kill follows within the horizon
    useful = 0
    for node in range(n):
        s = fleet.spike[node]
        csum = np.concatenate([[0], np.cumsum(s)])
        for i in np.flatnonzero(alarm[node]):
            hi = min(L, i + 1 + horizon)
            if csum[hi] - csum[i + 1] > 0:
                useful += 1
    days = n * L * STEP / 86400.0
    return dict(
        kills=len(kills), caught=caught, recall=caught / max(len(kills), 1),
        alarms=n_alarm, alarms_per_node_day=n_alarm / days,
        precision=useful / max(n_alarm, 1), mean_lead_s=float(np.mean(leads)) if leads else float("nan"),
    )
