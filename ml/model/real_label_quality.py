"""
Precision-recall, calibration and lead time of each reclaim predictor on the REAL reclaim labels
(the sensor campaign; steps built by evaluate_real_reclaims.py). AUROC alone hides all three.

Unit: an (instance, 5 minute step), label = AWS reclaims that instance within the next 15 minutes.
  PR-AUC        average precision, with its base rate and a 95% interval resampled by whole hours
  calibration   for the scores that claim to be probabilities: expected calibration error over 10
                equal-count bins, Brier score against always predicting the base rate, reliability bins
  operating     at a 10% alarm budget (threshold = the 90th percentile of the score over all steps):
                recall per reclaim (an alarm in the reclaim's last 15 minutes), precision per alarm
                (a reclaim follows within 15 minutes), and lead = reclaim time minus the first alarm
                of the run of alarms that ends at the reclaim. A reclaim lands within 5 minutes after the
                instance's last step, so leads are rounded to that grid (taken as the last step + 2.5 min).
The capacity model (logistic on placement + age) is scored out of sample, by held-out pool and by day.

    python ml/model/real_label_quality.py
Output: wide_results/reclaim_quality.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from train_real_label_model import FEATURE_SETS, cross_val, forward

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "wide_results")
BUDGET = 0.10
STEP_MIN = 5.0


def ap_ci(d, score, rng, n=500):
    hours = d.hour.unique()
    by = {h: d.index[d.hour == h] for h in hours}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[h] for h in rng.choice(hours, len(hours))])
        y = d.label.loc[idx]
        if 0 < y.sum() < len(y):
            vals.append(average_precision_score(y, score.loc[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def calibration(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    rows, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        mp, my = float(p[chunk].mean()), float(y[chunk].mean())
        rows.append(dict(mean_predicted=mp, observed=my, steps=int(len(chunk))))
        ece += len(chunk) / len(y) * abs(mp - my)
    base = y.mean()
    return dict(ece=float(ece), brier=float(np.mean((p - y) ** 2)), brier_base_rate=float(np.mean((base - y) ** 2)),
                mean_predicted=float(p.mean()), base_rate=float(base), bins=rows)


def operating_point(d, score, budget=BUDGET):
    s = score.loc[d.index]
    thr = float(np.quantile(s, 1 - budget))
    alarm = s > thr
    if alarm.mean() < budget / 3:  # heavy ties at the top (placement takes few values): include the tied level
        alarm = s >= thr
    d = d.assign(alarm=alarm.values)
    leads, caught, n_rec = [], 0, 0
    for _, g in d.groupby("instance"):
        g = g.sort_values("t")
        if not g.label.any():
            continue
        n_rec += 1
        reclaim = g.t.iloc[-1] + pd.Timedelta(minutes=STEP_MIN / 2)
        if not g.alarm[g.label.astype(bool)].any():
            continue
        caught += 1
        # walk back from the last step through the unbroken run of alarms
        a = g.alarm.to_numpy()
        k = len(a) - 1
        while k >= 0 and not a[k]:
            k -= 1
        while k > 0 and a[k - 1]:
            k -= 1
        leads.append((reclaim - g.t.iloc[k]).total_seconds() / 60.0)
    leads = np.asarray(leads)
    q = lambda x: float(np.percentile(leads, x)) if len(leads) else None
    return dict(threshold=thr, alarm_rate=float(alarm.mean()), recall=caught / max(n_rec, 1),
                precision=float(d.label[d.alarm.astype(bool)].mean()) if d.alarm.any() else 0.0, reclaims=n_rec, caught=caught,
                lead_min=dict(p10=q(10), median=q(50), p90=q(90),
                              share_ge={str(m): float((leads >= m).mean()) if len(leads) else 0.0 for m in (2, 5, 10, 15, 30)}),
                leads_min=[float(x) for x in leads])


def main():
    d = pd.read_csv(os.path.join(W, "real_reclaim_steps.csv.gz"), parse_dates=["t", "hour"])
    d["log_risk"] = np.log(d["risk"] + 1e-9)
    d = d.dropna(subset=["placement", "age_min", "log_risk"]).reset_index(drop=True)
    d["label"] = d["label"].astype(int)
    rng = np.random.default_rng(0)
    feats = FEATURE_SETS["placement+age"]
    cap_pool = cross_val(d, feats, "logistic", "pool")
    first = sorted(d.day.unique())[0]
    cap_fwd = forward(d, feats, "logistic", first)

    scorers = {
        "price model": (d["risk"], True),
        "placement score": (-d["placement"], False),
        "instance age (control)": (-d["age_min"], False),
        "pool prior": (d["pool_prior"], True),
        "capacity model, held-out pool": (cap_pool, True),
        "capacity model, train day 1, test later days": (cap_fwd, True),
    }
    out = dict(steps=len(d), positives=int(d.label.sum()), base_rate=float(d.label.mean()),
               reclaims=int(d[d.label == 1].instance.nunique()), alarm_budget=BUDGET, predictors={})
    for name, (score, is_prob) in scorers.items():
        ok = score.notna()
        sub, s = d[ok], score[ok]
        ap = float(average_precision_score(sub.label, s))
        r = dict(steps=int(len(sub)), positives=int(sub.label.sum()), base_rate=float(sub.label.mean()),
                 auprc=ap, auprc_lift=ap / float(sub.label.mean()), auprc_ci=ap_ci(sub, s, rng),
                 operating=operating_point(sub, s))
        if is_prob:
            r["calibration"] = calibration(sub.label, s)
        out["predictors"][name] = r
    if "rebalance" in d and d.rebalance.notna().any():
        sub = d[d.rebalance.notna()]
        s = sub.rebalance.astype(float)
        out["predictors"]["rebalance recommendation"] = dict(
            steps=int(len(sub)), positives=int(sub.label.sum()), base_rate=float(sub.label.mean()),
            auprc=float(average_precision_score(sub.label, s)),
            auprc_lift=float(average_precision_score(sub.label, s)) / float(sub.label.mean()),
            precision_when_received=float(sub.label[s > 0].mean()) if (s > 0).any() else None)
    json.dump(out, open(os.path.join(W, "reclaim_quality.json"), "w"), indent=1)
    print(f"{'predictor':<46}{'AP':>6}{'lift':>6}{'recall':>8}{'prec':>7}{'lead med':>9}{'ECE':>8}")
    for k, v in out["predictors"].items():
        op = v.get("operating", {})
        print(f"{k:<46}{v['auprc']:>6.3f}{v['auprc_lift']:>6.1f}{op.get('recall', np.nan):>8.2f}{op.get('precision', np.nan):>7.3f}"
              f"{(op.get('lead_min') or {}).get('median') or np.nan:>9.1f}{v.get('calibration', {}).get('ece', np.nan):>8.4f}")


if __name__ == "__main__":
    main()
