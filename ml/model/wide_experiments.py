"""
Experiments on the wide Spot dataset (ml/data/feature_pipeline_wide.py).

Why this exists: the shipped model was trained on about 300 positive windows from one region.
Its measured quality swings with the random seed (9.5x to 22.6x lift) and with the evaluation
slice (21.8x on test, 2.1x on calibration). With roughly 100x more spike events, the feature
questions that were left unresolved ("needs a multi-seed comparison") can finally be answered.

Method (same as the original, with two fixes):
  * windows are generated lazily on the GPU from one big array, never materialized;
  * every series is split by time into train / calibration / test (80 / 10 / 10) with a purge
    gap of SEQ_LENGTH + HORIZON so no window spans a boundary;
  * the scaler is fitted on TRAINING rows only (the old pipeline fitted it on all data);
  * early stopping uses calibration AUPRC; Platt calibration is fitted on the calibration
    slice; every reported number is on the test slice, which nothing was tuned on;
  * training uses all positive windows plus a fixed ratio of random negatives per epoch, while
    calibration and test keep the natural class balance.

    python ml/model/wide_experiments.py --ablation --seeds 0 1 2
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from losses import FocalLoss
from transformer import SpotInterruptionPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "wide_results")
SEQ, HORIZON = 24, 3
PURGE = SEQ + HORIZON

BASE = ["spot_price", "price_delta", "rolling_mean_15m", "rolling_std_15m", "rolling_mean_1h",
        "rolling_std_1h", "rolling_mean_6h", "rolling_std_6h", "normalized_ratio",
        "sin_time_day", "cos_time_day", "sin_day_week", "cos_day_week"]
ABLATIONS = {
    "base13": BASE,
    "+az_divergence": BASE + ["az_price_divergence"],
    "+advisor_rate": BASE + ["instance_interruption_rate"],
    "+update_density": BASE + ["n_updates_1h", "mins_since_change"],
    "+price_extremes": BASE + ["price_vs_6h_max", "price_vs_24h_min"],
    "all19": BASE + ["az_price_divergence", "instance_interruption_rate", "n_updates_1h",
                     "mins_since_change", "price_vs_6h_max", "price_vs_24h_min"],
}
MEMORY = ["price_vs_24h_max", "price_vs_72h_min", "price_vs_72h_max", "range_pos_24h",
          "n_updates_24h", "n_up_moves_24h", "last_change_pct", "mins_since_up", "mins_since_spike"]
ABLATIONS["all28"] = ABLATIONS["all19"] + MEMORY
OLD_TYPES = {"m5.xlarge", "m5.2xlarge", "c5.xlarge", "c5.2xlarge", "g4dn.xlarge"}


class WideData:
    def __init__(self, device):
        self.device = device
        z = np.load(os.path.join(DATA, os.environ.get("ARGUS_WIDE_NPZ", "features_wide2.npz")))
        self.names = list(z["features"])
        self.X, self.spike, self.k = z["X"], z["spike"], z["k"]
        self.meta = pd.read_csv(os.path.join(DATA, "features_wide_meta.csv"))
        cs = np.concatenate([[0], np.cumsum(self.spike.astype(np.int64))])
        parts = {"train": [], "calib": [], "test": []}
        series = {"train": [], "calib": [], "test": []}
        for sid, m in enumerate(self.meta.itertuples()):
            n = m.length - SEQ - HORIZON + 1
            if n < 10 * PURGE:
                continue
            b1, b2 = int(0.8 * n), int(0.9 * n)
            for name, lo, hi in (("train", 0, b1), ("calib", b1 + PURGE, b2), ("test", b2 + PURGE, n)):
                parts[name].append(m.offset + np.arange(lo, hi))
                series[name].append(np.full(max(hi - lo, 0), sid))
        self.starts = {k: np.concatenate(v) for k, v in parts.items()}
        self.series = {k: np.concatenate(v) for k, v in series.items()}
        # label of a window starting at s: any spike in rows s+SEQ .. s+SEQ+HORIZON-1
        self.labels = {k: ((cs[v + SEQ + HORIZON] - cs[v + SEQ]) > 0) for k, v in self.starts.items()}
        self.train_rows = np.concatenate([np.arange(s, s + SEQ) for s in self._train_ranges()])
        self._ar = torch.arange(SEQ, device=device)

    def _train_ranges(self):
        # a thin sample of train starts is enough to estimate feature mean/std
        return self.starts["train"][::50]

    def fit_scaler(self, cols):
        ci = [self.names.index(c) for c in cols]
        rows = self.X[self.train_rows][:, ci].astype(np.float64)
        return rows.mean(axis=0), rows.std(axis=0) + 1e-8

    def to_device(self, cols, mean, std):
        ci = [self.names.index(c) for c in cols]
        x = (self.X[:, ci] - mean.astype(np.float32)) / std.astype(np.float32)
        return torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=self.device)

    def batch(self, Xg, starts):
        idx = torch.as_tensor(starts, device=self.device)[:, None] + self._ar
        return Xg[idx]

    def region_mask(self, split, region=None, types=None):
        m = self.meta.iloc[self.series[split]]
        ok = np.ones(len(m), bool)
        if region:
            ok &= (m.region == region).to_numpy()
        if types:
            ok &= m.instance_type.isin(types).to_numpy()
        return ok


@torch.no_grad()
def predict_logits(model, data, Xg, starts, bs=16384):
    model.eval()
    out = []
    for i in range(0, len(starts), bs):
        out.append(model(data.batch(Xg, starts[i:i + bs])).squeeze(-1).float().cpu())
    return torch.cat(out).numpy()


def metrics(y, score):
    base = float(y.mean())
    ap = average_precision_score(y, score)
    return dict(positives=int(y.sum()), windows=len(y), base_rate=base, auprc=ap,
                lift=ap / base if base > 0 else float("nan"), auroc=roc_auc_score(y, score))


def platt(logit_calib, y_calib):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit_calib.reshape(-1, 1), y_calib)
    return lr


def train_one(data, cols, seed, max_epochs=12, patience=3, neg_per_pos=30, batch=1024, lr=5e-4,
              d_model=128, nhead=2, layers=2, alpha=0.75, gamma=1.0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mean, std = data.fit_scaler(cols)
    Xg = data.to_device(cols, mean, std)
    model = SpotInterruptionPredictor(len(cols), d_model, nhead, layers).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = FocalLoss(alpha=alpha, gamma=gamma)

    tr_s, tr_y = data.starts["train"], data.labels["train"]
    pos, neg = tr_s[tr_y], tr_s[~tr_y]
    ca_s, ca_y = data.starts["calib"], data.labels["calib"]
    ca_sub = np.arange(0, len(ca_s), 2)  # stride 2 keeps early stopping cheap; calibration uses all

    best, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        t0 = time.time()
        model.train()
        s = np.concatenate([pos, rng.choice(neg, min(len(neg), neg_per_pos * len(pos)), replace=False)])
        rng.shuffle(s)
        y_all = np.isin(s, pos)
        run = 0.0
        for i in range(0, len(s) - batch + 1, batch):
            xb = data.batch(Xg, s[i:i + batch])
            yb = torch.as_tensor(y_all[i:i + batch], dtype=torch.float32, device=data.device).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item()
        lg = predict_logits(model, data, Xg, ca_s[ca_sub])
        ap = average_precision_score(ca_y[ca_sub], lg)
        if verbose:
            print(f"  epoch {ep + 1:2d}  loss {run / (len(s) // batch):.4f}  calib AUPRC {ap:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, Xg, (mean, std), best


def evaluate(model, data, Xg):
    """Full natural-distribution calibration and test metrics, overall and per subset."""
    lc = predict_logits(model, data, Xg, data.starts["calib"])
    lt = predict_logits(model, data, Xg, data.starts["test"])
    yc, yt = data.labels["calib"], data.labels["test"]
    cal = platt(np.clip(lc, -30, 30), yc)
    pt = cal.predict_proba(np.clip(lt, -30, 30).reshape(-1, 1))[:, 1]
    pc = cal.predict_proba(np.clip(lc, -30, 30).reshape(-1, 1))[:, 1]

    prec, rec, thr = precision_recall_curve(yc, pc)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    t_star = thr[int(np.argmax(f1[:-1]))]
    hit = pt >= t_star
    out = {"calib": metrics(yc, lc), "all": metrics(yt, lt),
           "brier_test": float(np.mean((pt - yt) ** 2)),
           "f1_threshold": float(t_star), "test_precision": float(yt[hit].mean()) if hit.any() else float("nan"),
           "test_recall": float(hit[yt].mean()) if yt.any() else float("nan"),
           "test_alarm_rate": float(hit.mean())}
    subsets = {"eu-north-1": data.region_mask("test", "eu-north-1"),
               "eu-north-1_old5": data.region_mask("test", "eu-north-1", OLD_TYPES)}
    for name, m in subsets.items():
        if yt[m].sum() > 0:
            out[name] = metrics(yt[m], lt[m])
    return out, (lc, lt, pt)


def flat(prefix, d):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def run_ablation(seeds, names, max_epochs, out_csv, neg_per_pos):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = WideData(dev)
    print(f"device {dev}; train windows {len(data.starts['train']):,} "
          f"({int(data.labels['train'].sum()):,} positive); calib {len(data.starts['calib']):,} "
          f"({int(data.labels['calib'].sum()):,}); test {len(data.starts['test']):,} "
          f"({int(data.labels['test'].sum()):,})", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict("records")
    done = {(r["experiment"], r["seed"]) for r in rows}
    for name in names:
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            print(f"== {name} seed {seed}", flush=True)
            model, Xg, _, best = train_one(data, ABLATIONS[name], seed, max_epochs, neg_per_pos=neg_per_pos)
            res, _ = evaluate(model, data, Xg)
            row = dict(experiment=name, seed=seed, features=len(ABLATIONS[name]), best_calib_auprc=best,
                       seconds=round(time.time() - t0), brier_test=res["brier_test"],
                       test_precision=res["test_precision"], test_recall=res["test_recall"],
                       test_alarm_rate=res["test_alarm_rate"])
            for sub in ("calib", "all", "eu-north-1", "eu-north-1_old5"):
                if sub in res:
                    row.update(flat(sub, res[sub]))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"   test lift {row['all_lift']:.1f}x  AUROC {row['all_auroc']:.3f}  "
                  f"(eu-north-1 lift {row.get('eu-north-1_lift', float('nan')):.1f}x)  {row['seconds']}s", flush=True)
            del model, Xg
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--names", nargs="+", default=list(ABLATIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(RESULTS, "ablation.csv"))
    args = ap.parse_args()
    run_ablation(args.seeds, args.names, args.epochs, args.out, args.neg_per_pos)
