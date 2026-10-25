"""
Trains and exports the production risk model (gradient boosting on engineered window features).

  * hyperparameters: best calibration-AUPRC row of wide_results/xgb_tune.csv (chosen without test)
  * the SHIPPED model is always seed 0, declared before looking at test results; seeds 1-4 are
    trained only to report variability, never to pick a winner
  * Platt calibration and every threshold are fitted on the calibration slice (natural class balance)
  * the test slice is evaluated once, at the end, together with controls: a clock-only rule and the
    old shipped Transformer on the very same windows

    python ml/model/train_final.py
Output: ml/model/final/model.json, ml/model/final/meta.json, ml/model/wide_results/final_eval.json
"""
import datetime
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import spot_features as sf
import wide_experiments as we
import wide_xgboost as wx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "final")
BUDGETS = [0.0005, 0.001, 0.002, 0.005, 0.01]


def best_params():
    t = pd.read_csv(os.path.join(we.RESULTS, "xgb_tune.csv")).sort_values("calib_auprc", ascending=False)
    r = t.iloc[0]
    return dict(max_depth=int(r.max_depth), learning_rate=float(r.learning_rate),
                colsample_bytree=float(r.colsample_bytree), min_child_weight=float(r.min_child_weight),
                neg_per_pos=int(r.neg_per_pos), rounds=int(r.rounds))


def fit(data, params, seed, names_out):
    rng = np.random.RandomState(seed)
    tr_s, tr_y = data.starts["train"], data.labels["train"]
    pos, neg = tr_s[tr_y], tr_s[~tr_y]
    sel = np.concatenate([pos, rng.choice(neg, params["neg_per_pos"] * len(pos), replace=False)])
    Xtr, names = wx.tabular(data, sf.LEAN_FEATURES, sel)
    ytr = np.isin(sel, pos)
    ca = np.arange(0, len(data.starts["calib"]), 4)
    Xca, _ = wx.tabular(data, sf.LEAN_FEATURES, data.starts["calib"][ca])
    yca = data.labels["calib"][ca]
    names_out[:] = names
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=names)
    dca = xgb.DMatrix(Xca, label=yca, feature_names=names)
    p = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", device="cuda",
             max_depth=params["max_depth"], eta=params["learning_rate"], subsample=0.8,
             colsample_bytree=params["colsample_bytree"], min_child_weight=params["min_child_weight"], seed=seed)
    return xgb.train(p, dtr, 2000, evals=[(dca, "calib")], early_stopping_rounds=40, verbose_eval=False)


def predict(booster, data, split, names):
    X, _ = wx.tabular(data, sf.LEAN_FEATURES, data.starts[split])
    return booster.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, booster.best_iteration + 1))


def metrics(y, s, mask=None):
    if mask is not None:
        y, s = y[mask], s[mask]
    base = float(y.mean())
    ap = average_precision_score(y, s)
    return dict(positives=int(y.sum()), windows=len(y), base_rate=base, auprc=ap, lift=ap / base,
                auroc=roc_auc_score(y, s))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = we.WideData(device="cpu")
    params = best_params()
    print("params (chosen on calibration only):", params, flush=True)
    names = []
    seeds = {}
    for seed in range(5):
        seeds[seed] = fit(data, params, seed, names)
        print(f"seed {seed}: best iteration {seeds[seed].best_iteration}", flush=True)
    booster = seeds[0]  # declared in advance, not picked on test

    yc, yt = data.labels["calib"], data.labels["test"]
    pc = predict(booster, data, "calib", names)
    pt_raw = predict(booster, data, "test", names)

    def logit(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc), yc)
    cal = lambda p: platt.predict_proba(logit(p))[:, 1]  # noqa: E731
    pc_cal, pt_cal = cal(pc), cal(pt_raw)

    prec, rec, thr = precision_recall_curve(yc, pc_cal)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    t_f1 = float(thr[int(np.argmax(f1[:-1]))])
    thresholds = {"best_f1_on_calibration": t_f1}
    for b in BUDGETS:
        thresholds[f"alarm_budget_{b}"] = float(np.quantile(pc_cal, 1.0 - b))

    minute = ((data.k[data.starts["test"] + we.SEQ - 1] * 300) % 3600) // 60
    phase = minute >= 45
    hit = pt_cal >= t_f1
    test = {
        "all": metrics(yt, pt_cal),
        "in_hour_end_windows": metrics(yt, pt_cal, phase),
        "eu-north-1": metrics(yt, pt_cal, data.region_mask("test", "eu-north-1")),
        "eu-north-1_old5_types": metrics(yt, pt_cal, data.region_mask("test", "eu-north-1", we.OLD_TYPES)),
        "brier": float(np.mean((pt_cal - yt) ** 2)),
        "brier_always_predict_base_rate": float(np.mean((yc.mean() - yt) ** 2)),
        "at_best_f1_threshold": dict(threshold=t_f1, precision=float(yt[hit].mean()) if hit.any() else None,
                                     recall=float(hit[yt].mean()), alarm_rate=float(hit.mean())),
    }
    # reliability: predicted vs observed by quantile bin of the calibrated score
    bins = np.quantile(pt_cal, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(pt_cal, bins[1:-1]), 0, 9)
    test["reliability_deciles"] = [dict(bin=i, mean_predicted=float(pt_cal[idx == i].mean()),
                                        observed=float(yt[idx == i].mean())) for i in range(10)]
    controls = {"clock_only": metrics(yt, phase.astype(float))}
    old_path = os.path.join(we.RESULTS, "old_model_on_wide.json")
    if os.path.exists(old_path):
        controls["old_shipped_transformer"] = json.load(open(old_path))
    variability = []
    for seed, b in seeds.items():
        p = predict(b, data, "test", names)
        variability.append(dict(seed=seed, **{k: v for k, v in metrics(yt, p).items() if k in ("lift", "auroc", "auprc")}))

    od = pd.read_csv(os.path.join(sf_data_dir(), "od_prices_wide.csv"))
    od_prices = {r: dict(zip(g.instance_type, g.od_price.astype(float))) for r, g in od.groupby("region")}
    meta_df = data.meta
    meta = dict(
        model="xgboost on engineered 2-hour window features",
        label="price rises more than 1% within the next 3 five-minute steps (proxy for a capacity event, not a reclaim)",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seq_length=sf.SEQ_LENGTH, horizon=sf.HORIZON,
        feature_names=sf.LEAN_FEATURES, window_feature_names=names,
        platt=dict(coef=float(platt.coef_[0, 0]), intercept=float(platt.intercept_[0])),
        thresholds=thresholds, od_prices=od_prices, xgb_params=params, best_iteration=int(booster.best_iteration),
        training_data=dict(series=len(meta_df), regions=sorted(meta_df.region.unique()),
                           train_windows=int(len(data.starts["train"])),
                           train_positives=int(data.labels["train"].sum()),
                           calib_positives=int(yc.sum()), test_positives=int(yt.sum())),
        metrics=dict(test=test, controls=controls, seed_variability=variability),
    )
    booster.save_model(os.path.join(OUT, "model.json"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(we.RESULTS, "final_eval.json"), "w") as f:
        json.dump(dict(test=test, controls=controls, seed_variability=variability, thresholds=thresholds,
                       params=params), f, indent=1)
    t = test
    print(f"TEST all: lift {t['all']['lift']:.1f}x AUROC {t['all']['auroc']:.3f} | in-hour-end AUROC "
          f"{t['in_hour_end_windows']['auroc']:.3f} lift {t['in_hour_end_windows']['lift']:.1f}x | "
          f"eu-north-1 {t['eu-north-1']['lift']:.1f}x | brier {t['brier']:.5f} vs {t['brier_always_predict_base_rate']:.5f}")
    print("at best-F1:", t["at_best_f1_threshold"])
    print("seed variability:", [(v["seed"], round(v["lift"], 1)) for v in variability])


def sf_data_dir():
    return os.path.join(HERE, "..", "data")


if __name__ == "__main__":
    main()
