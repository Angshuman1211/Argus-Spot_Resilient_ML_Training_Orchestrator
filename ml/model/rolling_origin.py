"""
Rolling-origin evaluation: does the model's skill hold up across time, not just on one test slice?

Five folds step back from the end of the data in 9-day blocks. For each fold the model is retrained
from scratch on everything before the test block (minus a 7-day calibration period and a purge gap),
Platt calibration and the 10% alarm threshold come from the calibration period only, and the test block
is scored once. Same features, hyperparameters and metrics as the shipped model.

    python ml/model/rolling_origin.py
Output: ml/model/wide_results/rolling_origin.csv
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import spot_features as sf
import wide_experiments as we
import wide_xgboost as wx

DAY = 288  # five-minute steps per day
PURGE = we.SEQ + we.HORIZON
CALIB_DAYS, TEST_DAYS, FOLDS = 7, 9, 5
BUDGET = 0.1


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    data = we.WideData(device="cpu")
    # every valid window across all series, with its decision step and label
    starts, series = [], []
    for sid, m in enumerate(data.meta.itertuples()):
        n = m.length - we.SEQ - we.HORIZON + 1
        if n > 0:
            starts.append(m.offset + np.arange(n))
            series.append(np.full(n, sid))
    starts, series = np.concatenate(starts), np.concatenate(series)
    cs = np.concatenate([[0], np.cumsum(data.spike.astype(np.int64))])
    label = (cs[starts + we.SEQ + we.HORIZON] - cs[starts + we.SEQ]) > 0
    k_dec = data.k[starts + we.SEQ - 1].astype(np.int64)
    minute = ((k_dec * 300) % 3600) // 60
    k_end = int(k_dec.max())
    params = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", device="cuda",
                  max_depth=3, eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=0)

    rows = []
    rng = np.random.RandomState(0)
    for f in range(FOLDS):
        test_hi = k_end - f * TEST_DAYS * DAY
        test_lo = test_hi - TEST_DAYS * DAY
        calib_lo = test_lo - CALIB_DAYS * DAY
        tr = k_dec < calib_lo - PURGE
        ca = (k_dec >= calib_lo) & (k_dec < test_lo - PURGE)
        te = (k_dec >= test_lo) & (k_dec <= test_hi)
        if te.sum() == 0 or label[te].sum() < 10 or label[tr].sum() < 100:
            print(f"fold {f}: skipped (not enough data)")
            continue
        pos, neg = starts[tr & label], starts[tr & ~label]
        sel = np.concatenate([pos, rng.choice(neg, 10 * len(pos), replace=False)])
        Xtr, names = wx.tabular(data, sf.LEAN_FEATURES, sel)
        ytr = np.isin(sel, pos)
        ca_idx = np.flatnonzero(ca)
        ca_sub = ca_idx[::4]
        Xcs, _ = wx.tabular(data, sf.LEAN_FEATURES, starts[ca_sub])
        booster = xgb.train(params, xgb.DMatrix(Xtr, label=ytr, feature_names=names), 2000,
                            evals=[(xgb.DMatrix(Xcs, label=label[ca_sub], feature_names=names), "calib")],
                            early_stopping_rounds=40, verbose_eval=False)
        it = (0, booster.best_iteration + 1)

        def score(mask):
            X, _ = wx.tabular(data, sf.LEAN_FEATURES, starts[mask])
            return booster.predict(xgb.DMatrix(X, feature_names=names), iteration_range=it)

        pc, pt = score(ca), score(te)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc), label[ca])
        cc, tt = platt.predict_proba(logit(pc))[:, 1], platt.predict_proba(logit(pt))[:, 1]
        thr = np.quantile(cc, 1 - BUDGET)
        yt = label[te]
        hit = tt >= thr
        phase = minute[te] >= 45
        base = float(yt.mean())
        ap = average_precision_score(yt, tt)
        d0 = pd.to_datetime(test_lo * 300, unit="s", utc=True)
        d1 = pd.to_datetime(test_hi * 300, unit="s", utc=True)
        rows.append(dict(
            fold=f, test_start=d0.date().isoformat(), test_end=d1.date().isoformat(),
            train_days=round((calib_lo - int(k_dec.min())) / DAY, 1), train_positives=int((tr & label).sum()),
            test_positives=int(yt.sum()), rounds=booster.best_iteration + 1, base_rate=base, auprc=ap,
            lift=ap / base, auroc=roc_auc_score(yt, tt), in_hour_end_auroc=roc_auc_score(yt[phase], tt[phase]),
            clock_only_auroc=roc_auc_score(yt, phase.astype(float)),
            recall_at_10pct_budget=float(hit[yt].mean()), precision_at_10pct_budget=float(yt[hit].mean()),
            realized_alarm_rate=float(hit.mean()),
        ))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(we.RESULTS, "rolling_origin.csv"), index=False)


if __name__ == "__main__":
    main()
