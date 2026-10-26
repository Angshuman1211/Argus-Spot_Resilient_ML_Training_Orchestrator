"""
A risk model trained on REAL reclaim labels (the sensor campaign), evaluated two honest ways.

Unit: an (instance, 5 minute step); label: AWS reclaims that instance within the next 15 minutes
(built by evaluate_real_reclaims.py into wide_results/real_reclaim_steps.csv.gz).

Features available at the step: AWS's Spot placement score, instance age, the price model's risk, and the
pool's type and AZ. Two cross-validations, because they answer different questions:
  by hour        hold out whole hours: does it work on new time, in pools it has seen?
  by pool        hold out a whole (type, AZ) pool: does it work on a pool it has never seen?
                 (pool identity cannot help here, so this is the stricter and more useful test)
Models: logistic regression, and a small gradient-boosted model. Baselines: placement score alone, price risk alone,
the pool's leave-one-out reclaim rate (by hour only), random.

    python ml/model/train_real_label_model.py
Output: wide_results/real_label_model.json, final/capacity_model.json (logistic model fitted on all data)
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "wide_results")
FEATURE_SETS = {
    "placement": ["placement"],
    "placement+age": ["placement", "age_min"],
    "placement+age+price": ["placement", "age_min", "log_risk"],
    "price only": ["log_risk"],
}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 and not np.isnan(s).any() else np.nan


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "logistic":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    m = xgb.XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.1, subsample=0.8, min_child_weight=2,
                          tree_method="hist", verbosity=0).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def cross_val(d, feats, kind, group_col, n_folds=5, seed=0):
    groups = d[group_col].unique()
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(groups), min(n_folds, len(groups)))
    pred = pd.Series(np.nan, index=d.index)
    for f in folds:
        te = d[group_col].isin(f)
        if d.loc[~te, "label"].nunique() < 2 or te.sum() == 0:
            continue
        pred[te] = fit_predict(kind, d.loc[~te, feats].to_numpy(), d.loc[~te, "label"].to_numpy(), d.loc[te, feats].to_numpy())
    return pred


def forward(d, feats, kind, first_day):
    """Fit on the first campaign day only, score every later day: the out-of-time test. (With a short last day,
    testing on the last day alone left a handful of reclaims and an uninformative interval.)"""
    pred = pd.Series(np.nan, index=d.index)
    tr, te = d.day == first_day, d.day > first_day
    if d.loc[tr, "label"].nunique() == 2 and te.any():
        pred[te] = fit_predict(kind, d.loc[tr, feats].to_numpy(), d.loc[tr, "label"].to_numpy(), d.loc[te, feats].to_numpy())
    return pred


def boot_ci(d, score, rng, n=500):
    hours = d.hour.unique()
    by = {h: d[d.hour == h] for h in hours}
    vals = []
    for _ in range(n):
        s = pd.concat([by[h] for h in rng.choice(hours, len(hours))])
        a = auc(s.label, score.loc[s.index])
        if not np.isnan(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    d = pd.read_csv(os.path.join(W, "real_reclaim_steps.csv.gz"), parse_dates=["t", "hour"])
    d["log_risk"] = np.log(d["risk"] + 1e-9)
    d = d.dropna(subset=["placement", "age_min", "log_risk"]).reset_index(drop=True)
    d["label"] = d["label"].astype(int)
    print(f"{len(d)} steps, {int(d.label.sum())} positive, {d.pool.nunique()} pools, {d.hour.nunique()} distinct hours")
    rng = np.random.default_rng(0)
    schemes = [("by_hour", "hour"), ("by_pool", "pool")]
    if "day" in d and d.day.nunique() >= 2:
        schemes.append(("forward_day", "day"))  # train on the first day, test on every later day
    if "region" in d and d.region.nunique() >= 2:
        schemes.append(("by_region", "region"))  # hold out a whole region: does it transfer?
    out = {"steps": len(d), "positives": int(d.label.sum()), "pools": int(d.pool.nunique()), "hours": int(d.hour.nunique()),
           **{s: {} for s, _ in schemes}}
    rows = []
    for scheme, col in schemes:
        if scheme == "forward_day":
            last = sorted(d.day.unique())[0]  # the first day: training data for the out-of-time test
            ev = d[d.day > last]
            out["forward_day_test"] = dict(train_day=str(last), test_days=[str(x) for x in sorted(ev.day.unique())],
                                           steps=len(ev), positives=int(ev.label.sum()))
        else:
            ev = d
        for name, feats in FEATURE_SETS.items():
            for kind in (("logistic", "boosted") if len(feats) > 1 else ("logistic",)):
                p = forward(d, feats, kind, last) if scheme == "forward_day" else cross_val(d, feats, kind, col)
                a = auc(d.label[p.notna()], p[p.notna()])
                ap = average_precision_score(d.label[p.notna()], p[p.notna()]) if p.notna().any() else np.nan
                lo, hi = boot_ci(d[p.notna()], p, rng)
                key = f"{name} ({kind})"
                out[scheme][key] = dict(auroc=a, auprc=float(ap), ci=[lo, hi])
                rows.append((scheme, key, a, lo, hi))
        # baselines that need no fitting
        for label, score in (("placement score alone", -d["placement"]), ("price model alone", d["risk"]),
                             ("instance age alone", -d["age_min"])):
            a = auc(ev.label, score[ev.index])  # on the same steps the models were scored on
            lo, hi = boot_ci(ev, score, rng)
            out[scheme][label] = dict(auroc=a, ci=[lo, hi])
            rows.append((scheme, label, a, lo, hi))
    out["by_hour"]["pool reclaim rate (leave one hour out)"] = dict(auroc=auc(d.label, d["pool_prior"]))
    print(f"{'scheme':<12}{'model':<36}{'AUROC':>7}   95% CI")
    for scheme, key, a, lo, hi in rows:
        print(f"{scheme:<12}{key:<36}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]")

    # final logistic model on all data, saved for use as a capacity-risk signal
    feats = FEATURE_SETS["placement+age"]
    sc = StandardScaler().fit(d[feats])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(d[feats]), d.label)
    # the serving side clamps inputs to the range seen here rather than extrapolate (placement only took 1 to 3)
    ranges = {f: [float(d[f].min()), float(d[f].max())] for f in feats}
    # thresholds for a few alarm budgets (share of steps alarmed), in-sample: two features, little room to overfit
    p_all = lr.predict_proba(sc.transform(d[feats]))[:, 1]
    ops = []
    for budget in (0.05, 0.10, 0.20):
        thr = float(np.quantile(p_all, 1 - budget))
        hit = p_all > thr
        ops.append(dict(alarm_budget=budget, threshold=round(thr, 4), alarm_rate=round(float(hit.mean()), 4),
                        recall=round(float(d.label[hit].sum() / d.label.sum()), 3),
                        precision=round(float(d.label[hit].mean()), 3)))
    model = dict(features=feats, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                 intercept=float(lr.intercept_[0]), ranges=ranges, operating_points=ops,
                 trained_on=dict(steps=len(d), positives=int(d.label.sum()), regions=sorted(d.region.unique()),
                                 days=sorted(str(x) for x in d.day.unique())),
                 note="P(reclaim within 15 min | AWS placement score, instance age in minutes), sensor campaign "
                      f"on {', '.join(sorted(str(x) for x in d.day.unique()))}; small instances only, see "
                      "docs/wide_model_result.md Section 7b and ml/model/wide_results/reclaim_quality.json")
    os.makedirs(os.path.join(HERE, "final"), exist_ok=True)
    json.dump(model, open(os.path.join(HERE, "final", "capacity_model.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(W, "real_label_model.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
