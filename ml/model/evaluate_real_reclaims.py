"""
Do any of our predictors anticipate REAL Spot reclaims?

Data: the sensor campaign (scripts/sensor_campaign.py) keeps one tiny Spot instance alive in each
(instance type, AZ) pool and CloudTrail records every reclaim (ml/data/pull_reclaim_labels.py). That gives
labelled pool-time steps:

    step (pool, t, 5 min)  is "at risk" while a sensor is alive in the pool
    label                  1 if AWS reclaimed a sensor in that pool within the next 15 minutes

Predictors scored on the same steps (all causal, computed from information available at t):
    price model      the retrained model's calibrated risk from the pool's price history (ml/api/predictor.py)
    placement score  AWS's Spot placement score for the pool (1 = low capacity); lower means riskier
    proxy spike      did the price jump more than 1% in the next 15 minutes (NOT causal: it asks how well the
                     price-spike proxy the model was trained on coincides with real reclaims)
    pool prior       reclaim rate of the pool over the whole campaign, leave-one-hour-out
Intervals resample whole hours. Run again as the campaign collects more events.

    AWS_PROFILE=argus-demo python ml/model/evaluate_real_reclaims.py
Output: ml/model/wide_results/real_reclaim_eval.json and printed tables.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL = os.path.join(ROOT, "ml", "data", "real")
sys.path.insert(0, os.path.join(ROOT, "ml", "api"))
sys.path.insert(0, HERE)

import price_source  # noqa: E402
import spot_features as sf  # noqa: E402
from predictor import Predictor  # noqa: E402

STEP = 300
HORIZON_S = 900
MAX_LIFE_S = 120 * 60


def load_instances():
    rows = []
    for f in glob.glob(os.path.join(REAL, "sensor_campaign_*.csv")) + glob.glob(os.path.join(REAL, "sensor_launches_*.csv")):
        d = pd.read_csv(f)
        d = d[d.status == "launched"]
        rows.append(d[["instance_id", "instance_type", "availability_zone", "event_time_utc" if "event_time_utc" in d
                       else "launched_at_utc"]].rename(columns=lambda c: "launched" if c.endswith("_utc") else c))
    inst = pd.concat(rows, ignore_index=True).drop_duplicates("instance_id")
    inst["launched"] = pd.to_datetime(inst["launched"], utc=True, format="ISO8601")
    rec = pd.read_csv(os.path.join(REAL, "real_reclaims.csv"))
    rec["reclaim_time"] = pd.to_datetime(rec["reclaim_time"], utc=True, format="ISO8601")
    inst = inst.merge(rec[["instance_id", "reclaim_time"]], on="instance_id", how="left")
    now = pd.Timestamp.now(tz="UTC")
    inst["end"] = inst["reclaim_time"].fillna(
        (inst["launched"] + pd.Timedelta(seconds=MAX_LIFE_S)).where(lambda s: s < now, now))
    return inst


def main():
    inst = load_instances()
    reclaimed = inst.reclaim_time.notna()
    print(f"{len(inst)} sensor instances, {int(reclaimed.sum())} reclaimed "
          f"({reclaimed.mean():.0%}); median minutes alive when reclaimed: "
          f"{((inst.reclaim_time - inst.launched).dt.total_seconds() / 60)[reclaimed].median():.1f}")

    types = sorted(inst.instance_type.unique())
    prices = price_source.Ec2PriceSource("eu-north-1").fetch(types)
    plc = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(REAL, "placement_scores_*.csv"))], ignore_index=True)
    plc = plc[pd.to_numeric(plc.placement_score, errors="coerce").notna()]
    plc["time_utc"] = pd.to_datetime(plc.time_utc, utc=True, format="ISO8601")
    plc["placement_score"] = plc.placement_score.astype(float)

    pred = Predictor(os.path.join(HERE, "final"), None, "eu-north-1")
    t0 = inst.launched.min().floor("5min")
    t1 = min(inst.end.max(), pd.Timestamp.now(tz="UTC")).floor("5min")
    steps = pd.date_range(t0, t1, freq="5min")

    series = {k: g.sort_values("timestamp") for k, g in prices.groupby(["instance_type", "availability_zone"])}
    rows = []
    for (itype, az), g in inst.groupby(["instance_type", "availability_zone"]):
        ps = series.get((itype, az))
        if ps is None or itype not in pred.od:
            continue
        ts_s = sf.to_epoch_s(ps["timestamp"]).to_numpy()
        price = ps["spot_price"].to_numpy()
        pl = plc[(plc.instance_type == itype) & (plc.availability_zone == az)].sort_values("time_utc")
        rec_times = g.reclaim_time.dropna().dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        for t in steps:
            alive = ((g.launched <= t) & (g.end > t)).any()
            if not alive:
                continue
            ts = t.timestamp()
            keep = ts_s <= ts
            if keep.sum() == 0:
                continue
            raw = pred.score_series(ts_s[keep], price[keep], pred.od[itype], ts)
            risk = pred.calibrate(raw)
            future = (ts_s > ts) & (ts_s <= ts + HORIZON_S)
            p_now = price[keep][-1]
            spike = bool((price[future] > p_now * 1.01).any()) if future.any() else False
            plc_now = pl[pl.time_utc <= t]
            label = bool(((rec_times > np.datetime64(t.tz_convert(None))) &
                          (rec_times <= np.datetime64((t + pd.Timedelta(seconds=HORIZON_S)).tz_convert(None)))).any())
            rows.append(dict(pool=f"{itype}|{az}", t=t, hour=t.floor("h"), label=label, risk=risk,
                             placement=float(plc_now.placement_score.iloc[-1]) if len(plc_now) else np.nan,
                             proxy_spike=spike))
    d = pd.DataFrame(rows)
    d["neg_placement"] = -d["placement"]
    # leave-one-hour-out pool prior
    prior = []
    for _, r in d.iterrows():
        o = d[(d.pool == r.pool) & (d.hour != r.hour)]
        prior.append(o.label.mean() if len(o) else d.label.mean())
    d["pool_prior"] = prior
    print(f"{len(d)} at-risk pool steps, {int(d.label.sum())} followed by a real reclaim within 15 min "
          f"({d.label.mean():.1%})")

    rng = np.random.default_rng(0)
    hours = d.hour.unique()
    by_hour = {h: d[d.hour == h] for h in hours}

    def score(col, sub):
        sub = sub.dropna(subset=[col])
        if sub.label.nunique() < 2:
            return np.nan, np.nan
        return roc_auc_score(sub.label, sub[col].astype(float)), average_precision_score(sub.label, sub[col].astype(float))

    out = {"instances": len(inst), "reclaimed": int(reclaimed.sum()), "steps": len(d), "positive_steps": int(d.label.sum()),
           "base_rate": float(d.label.mean()), "predictors": {}}
    print(f"\n{'predictor':<22}{'AUROC':>8}{'95% CI':>18}{'AUPRC':>8}   (base rate {d.label.mean():.3f})")
    for name, col in (("price model", "risk"), ("placement score", "neg_placement"), ("proxy spike (future)", "proxy_spike"),
                      ("pool prior", "pool_prior")):
        auc, ap = score(col, d)
        boots = []
        for _ in range(500):
            pick = rng.choice(hours, len(hours))
            s = score(col, pd.concat([by_hour[h] for h in pick]))[0]
            if not np.isnan(s):
                boots.append(s)
        lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
        out["predictors"][name] = dict(auroc=auc, ci=[lo, hi], auprc=ap)
        print(f"{name:<22}{auc:>8.3f}   [{lo:.3f}, {hi:.3f}]{ap:>8.3f}")
    with open(os.path.join(HERE, "wide_results", "real_reclaim_eval.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    d.to_csv(os.path.join(HERE, "wide_results", "real_reclaim_steps.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    main()
