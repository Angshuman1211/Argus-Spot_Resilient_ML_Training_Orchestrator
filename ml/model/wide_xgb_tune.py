"""
Hyperparameter search for the final gradient-boosting model. Selection uses the CALIBRATION slice
only; the test slice is touched once, afterwards, for the chosen configuration.

The lean feature set drops everything the ablation showed to be unnecessary: the four clock
features (once update cadence is known the clock adds nothing), cross-AZ divergence and the
Advisor rate (no measurable effect), and so needs no cross-series data and no Advisor file at
serving time.

    python ml/model/wide_xgb_tune.py
Output: ml/model/wide_results/xgb_tune.csv
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import wide_experiments as we
import wide_xgb_ablation as ab
import spot_features as sf
import wide_xgboost as wx

LEAN = sf.LEAN_FEATURES

GRID = dict(max_depth=[3, 4, 6], learning_rate=[0.03, 0.06], colsample_bytree=[0.5, 0.8],
            neg_per_pos=[10, 30, 100], min_child_weight=[1, 5])


def main():
    data = we.WideData(device="cpu")
    ca_idx = np.arange(0, len(data.starts["calib"]), 4)
    Xca, _ = wx.tabular(data, LEAN, data.starts["calib"][ca_idx])
    yca = data.labels["calib"][ca_idx]
    tr_s, tr_y = data.starts["train"], data.labels["train"]
    pos, neg = tr_s[tr_y], tr_s[~tr_y]

    cache = {}
    rows = []
    keys = list(GRID)
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        npp = cfg["neg_per_pos"]
        if npp not in cache:
            rng = np.random.RandomState(0)
            sel = np.concatenate([pos, rng.choice(neg, npp * len(pos), replace=False)])
            Xtr, _ = wx.tabular(data, LEAN, sel)
            cache[npp] = (Xtr, np.isin(sel, pos))
        Xtr, ytr = cache[npp]
        t0 = time.time()
        clf = xgb.XGBClassifier(n_estimators=800, subsample=0.8, tree_method="hist", device="cuda",
                                eval_metric="aucpr", early_stopping_rounds=40, random_state=0,
                                max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                                colsample_bytree=cfg["colsample_bytree"], min_child_weight=cfg["min_child_weight"])
        clf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
        s = ab.score(yca, clf.predict_proba(Xca)[:, 1])
        rows.append(dict(**cfg, rounds=clf.best_iteration + 1, calib_auprc=s["auprc"], calib_lift=s["lift"],
                         calib_auroc=s["auroc"], seconds=round(time.time() - t0)))
        pd.DataFrame(rows).sort_values("calib_auprc", ascending=False).to_csv(
            os.path.join(we.RESULTS, "xgb_tune.csv"), index=False)
    best = pd.DataFrame(rows).sort_values("calib_auprc", ascending=False).head(5)
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
