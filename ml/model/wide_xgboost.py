"""
Gradient-boosting baseline on the wide dataset, same windows, labels, splits and metrics as
wide_experiments.py. The earlier XGBoost baseline lost badly, but it had about 300 positives and
only last-step features. This one gets 20k positives and window summary features.

    python ml/model/wide_xgboost.py
Output: ml/model/wide_results/xgboost.csv
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import wide_experiments as we

import spot_features as sf


def tabular(data, cols, starts, chunk=200_000):
    """Window -> flat feature vector, via the same function the serving path uses."""
    ci = [data.names.index(c) for c in cols]
    outs, names = [], None
    for i in range(0, len(starts), chunk):
        st = starts[i:i + chunk]
        W = data.X[st[:, None] + np.arange(we.SEQ)][:, :, ci]
        X, names = sf.window_features(W, cols)
        outs.append(X)
    return np.concatenate(outs), names


def main():
    data = we.WideData(device="cpu")
    cols = we.ABLATIONS["all19"]
    rng = np.random.RandomState(0)
    tr_s, tr_y = data.starts["train"], data.labels["train"]
    pos, neg = tr_s[tr_y], tr_s[~tr_y]
    sel = np.concatenate([pos, rng.choice(neg, 30 * len(pos), replace=False)])
    Xtr, names = tabular(data, cols, sel)
    ytr = np.isin(sel, pos)
    ca = np.arange(0, len(data.starts["calib"]), 4)
    Xca, _ = tabular(data, cols, data.starts["calib"][ca])
    yca = data.labels["calib"][ca]
    Xte, _ = tabular(data, cols, data.starts["test"])
    yte = data.labels["test"]

    rows = []
    for depth in (4, 6, 8):
        t0 = time.time()
        clf = xgb.XGBClassifier(n_estimators=600, max_depth=depth, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", device="cuda",
                                eval_metric="aucpr", early_stopping_rounds=40, n_jobs=8)
        clf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
        p = clf.predict_proba(Xte)[:, 1]
        row = dict(model=f"xgboost_depth{depth}", rounds=clf.best_iteration + 1, seconds=round(time.time() - t0))
        row.update({f"all_{k}": v for k, v in we.metrics(yte, p).items()})
        m = data.region_mask("test", "eu-north-1")
        row.update({f"eu-north-1_{k}": v for k, v in we.metrics(yte[m], p[m]).items()})
        rows.append(row)
        print(f"depth {depth}: test lift {row['all_lift']:.2f}x AUROC {row['all_auroc']:.3f} "
              f"(eu-north-1 {row['eu-north-1_lift']:.2f}x)  {row['seconds']}s", flush=True)
    os.makedirs(we.RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(we.RESULTS, "xgboost.csv"), index=False)


if __name__ == "__main__":
    main()
