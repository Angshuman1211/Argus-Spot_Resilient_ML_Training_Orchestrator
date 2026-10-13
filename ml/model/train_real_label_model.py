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
    out = {"steps": len(d), "positives": int(d.label.sum()), "pools": int(d.pool.nunique()), "hours": int(d.hour.nunique()),
           "by_hour": {}, "by_pool": {}}
    rows = []
    for scheme, col in (("by_hour", "hour"), ("by_pool", "pool")):
        for name, feats in FEATURE_SETS.items():
            for kind in (("logistic", "boosted") if len(feats) > 1 else ("logistic",)):
                p = cross_val(d, feats, kind, col)
                a = auc(d.label[p.notna()], p[p.notna()])
                ap = average_precision_score(d.label[p.notna()], p[p.notna()]) if p.notna().any() else np.nan
                lo, hi = boot_ci(d[p.notna()], p, rng)
                key = f"{name} ({kind})"
                out[scheme][key] = dict(auroc=a, auprc=float(ap), ci=[lo, hi])
                rows.append((scheme, key, a, lo, hi))
        # baselines that need no fitting
        for label, score in (("placement score alone", -d["placement"]), ("price model alone", d["risk"]),
                             ("instance age alone", -d["age_min"])):
            a = auc(d.label, score)
            lo, hi = boot_ci(d, score, rng)
            out[scheme][label] = dict(auroc=a, ci=[lo, hi])
            rows.append((scheme, label, a, lo, hi))
    out["by_hour"]["pool reclaim rate (leave one hour out)"] = dict(auroc=auc(d.label, d["pool_prior"]))
    print(f"{'scheme':<9}{'model':<36}{'AUROC':>7}   95% CI")
    for scheme, key, a, lo, hi in rows:
        print(f"{scheme:<9}{key:<36}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]")

    # final logistic model on all data, saved for use as a capacity-risk signal
    feats = FEATURE_SETS["placement+age"]
    sc = StandardScaler().fit(d[feats])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(d[feats]), d.label)
    model = dict(features=feats, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                 intercept=float(lr.intercept_[0]), trained_on=dict(steps=len(d), positives=int(d.label.sum())),
                 note="P(reclaim within 15 min | AWS placement score, instance age in minutes), eu-north-1 sensor data")
    os.makedirs(os.path.join(HERE, "final"), exist_ok=True)
    json.dump(model, open(os.path.join(HERE, "final", "capacity_model.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(W, "real_label_model.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
