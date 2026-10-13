"""
Multi-seed feature ablation for the gradient-boosting model, with the controls that matter.

Why the controls: AWS now publishes Spot prices once an hour, so every price spike lands on the
:00 step and every positive window ends at :45, :50 or :55. A rule that only knows "the top of the
hour is near" already reaches AUROC 0.876 (lift 4x) on this data. A model that beats a baseline
must beat THAT, and skill should also be reported among the windows where a spike is possible at
all (positives only exist there). Reported per run, test slice, natural class balance:

  all_*    every test window (the number comparable to the old 21.8x lift)
  phase_*  only windows ending in the last 15 minutes of the hour (the real discrimination task)

    python ml/model/wide_xgb_ablation.py
Output: ml/model/wide_results/xgb_ablation.csv, xgb_controls.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

import wide_experiments as we
import wide_xgboost as wx

SETS = {
    "base13": we.BASE,
    "base13+az_divergence": we.BASE + ["az_price_divergence"],
    "base13+advisor_rate": we.BASE + ["instance_interruption_rate"],
    "base13+update_density": we.BASE + ["n_updates_1h", "mins_since_change"],
    "base13+price_extremes": we.BASE + ["price_vs_6h_max", "price_vs_24h_min"],
    "all19": we.ABLATIONS["all19"],
    "all19_no_clock": [c for c in we.ABLATIONS["all19"]
                       if c not in ("sin_time_day", "cos_time_day", "sin_day_week", "cos_day_week")],
    "all28": we.ABLATIONS["all28"],
    "all28_no_clock": [c for c in we.ABLATIONS["all28"]
                       if c not in ("sin_time_day", "cos_time_day", "sin_day_week", "cos_day_week")],
    "extremes_only": we.BASE + ["price_vs_6h_max", "price_vs_24h_min", "price_vs_24h_max",
                                "price_vs_72h_min", "price_vs_72h_max", "range_pos_24h"],
    "cadence_only": we.BASE + ["n_updates_1h", "mins_since_change", "n_updates_24h", "n_up_moves_24h"],
    "all28_minus_advisor_az": [c for c in we.ABLATIONS["all28"]
                               if c not in ("az_price_divergence", "instance_interruption_rate")],
}


def score(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    base = float(y.mean())
    ap = average_precision_score(y, p)
    return dict(positives=int(y.sum()), base_rate=base, auprc=ap, lift=ap / base, auroc=roc_auc_score(y, p))


def main(seeds=(0, 1, 2), only=None):
    data = we.WideData(device="cpu")
    te = data.starts["test"]
    yte = data.labels["test"]
    minute = ((data.k[te + we.SEQ - 1] * 300) % 3600) // 60
    phase = minute >= 45

    controls = {"clock_only_all": score(yte, phase.astype(float)),
                "phase_windows": int(phase.sum()), "phase_positives": int(yte[phase].sum()),
                "windows_outside_phase_positives": int(yte[~phase].sum())}
    os.makedirs(we.RESULTS, exist_ok=True)
    with open(os.path.join(we.RESULTS, "xgb_controls.json"), "w") as f:
        json.dump(controls, f, indent=2)
    print("controls:", controls, flush=True)

    out_csv = os.path.join(we.RESULTS, "xgb_ablation.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["features"], r["seed"]) for r in rows}
    for name, cols in SETS.items():
        if only and name not in only:
            continue
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            rng = np.random.RandomState(seed)
            tr_s, tr_y = data.starts["train"], data.labels["train"]
            pos, neg = tr_s[tr_y], tr_s[~tr_y]
            sel = np.concatenate([pos, rng.choice(neg, 30 * len(pos), replace=False)])
            Xtr, _ = wx.tabular(data, cols, sel)
            ytr = np.isin(sel, pos)
            ca = np.arange(0, len(data.starts["calib"]), 4)
            Xca, _ = wx.tabular(data, cols, data.starts["calib"][ca])
            yca = data.labels["calib"][ca]
            Xte, _ = wx.tabular(data, cols, te)
            clf = xgb.XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, tree_method="hist", device="cuda",
                                    eval_metric="aucpr", early_stopping_rounds=40, random_state=seed)
            clf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
            p = clf.predict_proba(Xte)[:, 1]
            pca = clf.predict_proba(Xca)[:, 1]
            row = dict(features=name, n_features=len(cols), seed=seed, rounds=clf.best_iteration + 1,
                       seconds=round(time.time() - t0))
            for k, v in score(yte, p).items():
                row[f"all_{k}"] = v
            for k, v in score(yca, pca).items():
                row[f"calib_{k}"] = v
            for k, v in score(yte, p, phase).items():
                row[f"phase_{k}"] = v
            m = data.region_mask("test", "eu-north-1")
            for k, v in score(yte, p, m).items():
                row[f"eu_north_1_{k}"] = v
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{name:24s} seed {seed}: all lift {row['all_lift']:.1f}x AUROC {row['all_auroc']:.3f} | "
                  f"in-phase AUROC {row['phase_auroc']:.3f} lift {row['phase_lift']:.1f}x  ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    import sys
    main(only=sys.argv[1:] or None)
