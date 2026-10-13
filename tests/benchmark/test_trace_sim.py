"""Known-answer tests for benchmark/trace_sim.py on tiny synthetic fleets."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmark"))

import trace_sim as ts  # noqa: E402
from trace_sim import Fleet, Policy, STEP, simulate  # noqa: E402


def make_fleet(L, spikes=(), risk=None, n=1):
    """spikes: list of (node, step). risk: dict {(node, step): score}."""
    spike = np.zeros((n, L), bool)
    for node, i in spikes:
        spike[node, i] = True
    r = np.full((n, L), 0.0001)
    for (node, i), v in (risk or {}).items():
        r[node, i] = v
    f = Fleet(
        nodes=[("x", f"az{j}") for j in range(n)], k0=0, risk=r, vol=np.zeros((n, L)),
        spike=spike, future=np.zeros((n, L), bool), advisor_rate=np.full(n, 0.05),
        price=np.full(n, 0.1),
    )
    f.kill_steps = [int(i) for i in np.flatnonzero(spike.any(axis=0))]
    return f


W = 3600.0


def run(fleet, pol, C=10.0, R=60.0, start=0, **kw):
    return simulate(fleet, pol, start, W, C, R, **kw)


def test_no_interruptions_means_makespan_equals_work():
    f = make_fleet(100)
    out = run(f, Policy("none", "none"))
    assert out["completed"] and out["makespan"] == pytest.approx(W)
    assert out["wasted"] == 0 and out["kills"] == 0


def test_no_protection_loses_everything_since_start():
    f = make_fleet(100, spikes=[(0, 6)])  # kill at t=1800 s
    out = run(f, Policy("none", "none"))
    assert out["kills"] == 1
    assert out["wasted"] == pytest.approx(1800.0)
    assert out["makespan"] == pytest.approx(W + 1800.0 + 60.0)  # redo + restart


def test_periodic_bounds_loss_by_interval():
    f = make_fleet(100, spikes=[(0, 6)])
    out = run(f, Policy("p", "periodic", tau=600.0))
    assert out["kills"] == 1
    assert out["wasted"] <= 600.0 + 1e-6
    assert out["n_ckpt"] >= 2


def test_reactive_saves_everything_when_checkpoint_fits_in_notice():
    f = make_fleet(100, spikes=[(0, 6)])
    out = run(f, Policy("r", "reactive"), C=60.0, notice=120.0)
    assert out["kills"] == 1 and out["wasted"] == pytest.approx(0.0)
    assert out["n_ckpt"] == 1


def test_reactive_fails_when_checkpoint_longer_than_notice():
    f = make_fleet(100, spikes=[(0, 6)])
    out = run(f, Policy("r", "reactive"), C=300.0, notice=120.0)
    # the job pauses at the notice (t=1680) and the kill aborts the checkpoint: the 1680 s
    # of work done by then is lost, and nothing was saved
    assert out["wasted"] == pytest.approx(1680.0)
    assert out["n_ckpt"] == 0


def test_signal_policy_migrates_and_avoids_a_predicted_kill():
    # high score at step 4, kill at step 6 (10 min later)
    f = make_fleet(100, spikes=[(0, 6)], risk={(0, 4): 0.01})
    pol = Policy("m", "signal", thr=0.005, action="migrate")
    out = run(f, pol, C=10.0, R=60.0)
    assert out["kills"] == 0 and out["avoided"] == 1
    assert out["wasted"] == 0 and out["n_migr"] == 1
    assert out["makespan"] == pytest.approx(W + 10.0 + 60.0)


def test_signal_policy_cannot_see_the_future():
    # score only rises AFTER the kill: policy must lose the work
    f = make_fleet(100, spikes=[(0, 6)], risk={(0, 7): 0.01})
    pol = Policy("m", "signal", thr=0.005, action="migrate")
    out = run(f, pol)
    assert out["kills"] == 1 and out["wasted"] == pytest.approx(1800.0)


def test_false_alarm_costs_overhead_and_nothing_else():
    f = make_fleet(100, risk={(0, 3): 0.01})
    pol = Policy("m", "signal", thr=0.005, action="migrate")
    out = run(f, pol, C=10.0, R=60.0)
    assert out["kills"] == 0 and out["n_migr"] == 1
    assert out["makespan"] == pytest.approx(W + 70.0)


def test_cooldown_suppresses_repeat_alarms():
    risk = {(0, i): 0.01 for i in range(2, 8)}
    f = make_fleet(100, risk=risk)
    pol = Policy("m", "signal", thr=0.005, action="ckpt")
    out = run(f, pol, cooldown=3 * STEP)
    assert out["n_ckpt"] == 2  # steps 2 and 5, not six


def test_any_node_kills_a_synchronous_job():
    f = make_fleet(100, spikes=[(2, 6)], n=3)
    out = run(f, Policy("none", "none"))
    assert out["kills"] == 1 and out["wasted"] == pytest.approx(1800.0)


def test_alarm_on_a_different_node_does_not_protect_the_killed_node():
    f = make_fleet(100, spikes=[(0, 6)], risk={(1, 4): 0.01}, n=2)
    pol = Policy("m", "signal", thr=0.005, action="migrate")
    out = run(f, pol)
    assert out["kills"] == 1  # node 1 was moved, node 0 was not


def test_ckpt_only_action_keeps_the_kill_but_bounds_the_loss():
    f = make_fleet(100, spikes=[(0, 6)], risk={(0, 4): 0.01})
    pol = Policy("c", "signal", thr=0.005, action="ckpt")
    out = run(f, pol, C=10.0)
    # checkpoint at t=1200 (10 s pause), kill at 1800: 600 s elapsed minus the 10 s pause
    assert out["kills"] == 1 and out["wasted"] == pytest.approx(590.0)


def test_oracle_beats_everything_with_zero_waste():
    f = make_fleet(100, spikes=[(0, 6), (0, 9)])
    out = run(f, Policy("o", "oracle", action="migrate"))
    assert out["kills"] == 0 and out["wasted"] == 0 and out["avoided"] == 2


def test_incomplete_when_trace_ends_first():
    f = make_fleet(5)
    out = run(f, Policy("none", "none"))
    assert out["completed"] is False


def test_young_daly_formula():
    assert ts.young_daly_tau(60.0, 86400.0) == pytest.approx(np.sqrt(2 * 60 * 86400))


def test_threshold_for_budget_hits_the_alarm_fraction():
    scores = np.linspace(0, 1, 1000)
    thr = ts.threshold_for_budget(scores, 0.05)
    assert (scores > thr).mean() == pytest.approx(0.05, abs=0.002)


def test_detection_stats_counts_lead_and_recall():
    f = make_fleet(50, spikes=[(0, 20)], risk={(0, 18): 0.01})
    d = ts.detection_stats(f, "risk", 0.005)
    assert d["kills"] == 1 and d["caught"] == 1
    assert d["mean_lead_s"] == pytest.approx(2 * STEP)
    assert d["alarms"] == 1 and d["precision"] == 1.0
