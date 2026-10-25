"""
Do any of our predictors anticipate REAL Spot reclaims?

Data: the sensor campaign (scripts/sensor_campaign.py) keeps tiny Spot instances alive across (instance type,
AZ) pools and CloudTrail records every reclaim (ml/data/pull_reclaim_labels.py). The unit is an (instance, 5 minute
step) while the instance is alive; the label is 1 if AWS reclaims THAT instance within the next 15 minutes.

Predictors, all computed from information available at the step:
    price model      the retrained model's calibrated risk from the pool's price history (ml/api/predictor.py)
    placement score  AWS's Spot placement score for the pool at that time (lower is riskier)
    instance age     minutes since launch (lower is riskier). A CONTROL: every real reclaim in our data hit a young
                     instance, so a predictor must beat age, or work inside an age band, to be credited
    pool prior       reclaim rate of the pool over the campaign, leave-one-hour-out
    proxy spike      did the price jump more than 1% in the next 15 minutes (NOT causal; it asks how well the
                     price-spike proxy coincides with real reclaims)
    learned          logistic regression on [placement, age, price risk], cross-validated by whole hours
Intervals resample whole hours. Re-run as the campaign collects more events.

    AWS_PROFILE=argus-demo python ml/model/evaluate_real_reclaims.py
Output: ml/model/wide_results/real_reclaim_eval.json, real_reclaim_steps.csv.gz
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL = os.path.join(ROOT, "ml", "data", "real")
sys.path.insert(0, os.path.join(ROOT, "ml", "api"))
sys.path.insert(0, HERE)

import price_source  # noqa: E402
import spot_features as sf  # noqa: E402
from predictor import Predictor  # noqa: E402

HORIZON = pd.Timedelta(minutes=15)
MAX_LIFE = pd.Timedelta(minutes=120)


def load_instances():
    rows = []
    for f in glob.glob(os.path.join(REAL, "sensor_campaign_*.csv")) + glob.glob(os.path.join(REAL, "sensor_launches_*.csv")):
        d = pd.read_csv(f)
        d = d[d.status == "launched"]
        col = "event_time_utc" if "event_time_utc" in d else "launched_at_utc"
        rows.append(d[["instance_id", "instance_type", "availability_zone", col]].rename(columns={col: "launched"}))
    inst = pd.concat(rows, ignore_index=True).drop_duplicates("instance_id")
    inst["launched"] = pd.to_datetime(inst["launched"], utc=True, format="ISO8601")
    rec = pd.read_csv(os.path.join(REAL, "real_reclaims.csv"))
    rec["reclaim_time"] = pd.to_datetime(rec["reclaim_time"], utc=True, format="ISO8601")
    inst = inst.merge(rec[["instance_id", "reclaim_time"]], on="instance_id", how="left")
    now = pd.Timestamp.now(tz="UTC")
    cap = (inst["launched"] + MAX_LIFE).where(lambda s: s < now, now)
    inst["end"] = inst["reclaim_time"].fillna(cap)
    inst["region"] = inst["availability_zone"].str[:-1]  # "us-east-1a" -> "us-east-1"
    return inst


def fetch_prices(region, types, start):
    """Spot price updates from `start` (AWS keeps 90 days), so a re-run days later still covers early sensors."""
    import boto3
    ec2 = boto3.client("ec2", region_name=region)
    pages = ec2.get_paginator("describe_spot_price_history").paginate(
        InstanceTypes=list(types), ProductDescriptions=["Linux/UNIX"], StartTime=start.to_pydatetime(),
        EndTime=pd.Timestamp.now(tz="UTC").to_pydatetime())
    rows = [(r["Timestamp"], r["InstanceType"], r["AvailabilityZone"], r["SpotPrice"]) for p in pages for r in p["SpotPriceHistory"]]
    return price_source._normalize(pd.DataFrame(rows, columns=price_source.COLUMNS))


def add_rebalance(d, inst):
    """AWS's Rebalance Recommendation as a predictor: 1 once the instance has received one (at or before the step),
    else 0. NaN for instances launched before signal recording started in their region (we cannot know)."""
    d["rebalance"] = np.nan
    sig_path, cov_path = os.path.join(REAL, "spot_signals.csv"), os.path.join(REAL, "spot_signals_coverage.json")
    if not (os.path.exists(sig_path) and os.path.exists(cov_path)):
        return {}
    sig = pd.read_csv(sig_path)
    sig["time"] = pd.to_datetime(sig["time"], utc=True, format="ISO8601")
    cover = {r: pd.Timestamp(t) for r, t in json.load(open(cov_path)).items()}
    first_reb = sig[sig.signal == "rebalance"].groupby("instance_id")["time"].min()
    covered = inst[[pd.notna(cover.get(r)) and l >= cover[r] for r, l in zip(inst.region, inst.launched)]]
    m = d.instance.isin(covered.instance_id)
    reb_t = d.loc[m, "instance"].map(first_reb)
    d.loc[m, "rebalance"] = (reb_t.notna() & (reb_t <= d.loc[m, "t"])).astype(float)
    # how often and how early it warned before a real reclaim
    rec = covered[covered.reclaim_time.notna()].set_index("instance_id")
    lead = (rec.reclaim_time - rec.index.map(first_reb).to_series(index=rec.index)).dt.total_seconds() / 60
    warned = lead[lead > 0]
    notreb = covered[covered.reclaim_time.isna()]
    info = dict(covered_instances=len(covered), covered_reclaims=len(rec), reclaims_warned=int(len(warned)),
                median_lead_min=float(warned.median()) if len(warned) else None,
                false_warnings=int(notreb.instance_id.isin(first_reb.index).sum()), unreclaimed_covered=len(notreb))
    print(f"rebalance recommendation: warned before {info['reclaims_warned']} of {len(rec)} covered reclaims "
          f"(median lead {info['median_lead_min']} min); sent to {info['false_warnings']} of {len(notreb)} never-reclaimed")
    return info


def auc(y, s):
    ok = ~pd.isna(s)
    y, s = np.asarray(y)[ok], np.asarray(s, dtype=float)[ok]
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def main():
    inst = load_instances()
    got = inst.reclaim_time.notna()
    mins = ((inst.reclaim_time - inst.launched).dt.total_seconds() / 60)[got]
    print(f"{len(inst)} sensor instances, {int(got.sum())} reclaimed; minutes alive when reclaimed: "
          f"median {mins.median():.1f}, max {mins.max():.1f}")
    surv = inst[~got]
    print(f"{len(surv)} not reclaimed; of those, {(((surv.end - surv.launched).dt.total_seconds() / 60) > 30).sum()} "
          f"lived more than 30 minutes")

    plc = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(REAL, "placement_scores_*.csv"))], ignore_index=True)
    plc = plc[pd.to_numeric(plc.placement_score, errors="coerce").notna()]
    plc["time_utc"] = pd.to_datetime(plc.time_utc, utc=True, format="ISO8601")
    plc["placement_score"] = plc.placement_score.astype(float)
    series, preds = {}, {}
    for region, g in inst.groupby("region"):
        prices = fetch_prices(region, sorted(g.instance_type.unique()), g.launched.min() - pd.Timedelta(days=4))
        series.update({k: v.sort_values("timestamp") for k, v in prices.groupby(["instance_type", "availability_zone"])})
        preds[region] = Predictor(os.path.join(HERE, "final"), None, region)

    rows = []
    for r in inst.itertuples():
        ps = series.get((r.instance_type, r.availability_zone))
        pred = preds[r.region]
        if ps is None or r.instance_type not in pred.od:
            continue
        ts_s = sf.to_epoch_s(ps["timestamp"]).to_numpy()
        price = ps["spot_price"].to_numpy()
        pl = plc[(plc.instance_type == r.instance_type) & (plc.availability_zone == r.availability_zone)].sort_values("time_utc")
        for t in pd.date_range(r.launched.ceil("5min"), r.end.floor("5min"), freq="5min", inclusive="left"):
            keep = ts_s <= t.timestamp()
            if keep.sum() == 0:
                continue
            raw = pred.score_series(ts_s[keep], price[keep], pred.od[r.instance_type], t.timestamp())
            fut = (ts_s > t.timestamp()) & (ts_s <= (t + HORIZON).timestamp())
            now_p = price[keep][-1]
            pn = pl[pl.time_utc <= t]
            rows.append(dict(instance=r.instance_id, pool=f"{r.instance_type}|{r.availability_zone}", region=r.region,
                             t=t, hour=t.floor("h"), day=str(t.date()),
                             label=bool(pd.notna(r.reclaim_time) and t < r.reclaim_time <= t + HORIZON),
                             risk=pred.calibrate(raw), age_min=(t - r.launched).total_seconds() / 60,
                             placement=float(pn.placement_score.iloc[-1]) if len(pn) else np.nan,
                             proxy_spike=bool((price[fut] > now_p * 1.01).any()) if fut.any() else False))
    d = pd.DataFrame(rows)
    sig_info = add_rebalance(d, inst)
    d["neg_placement"] = -d["placement"]
    d["neg_age"] = -d["age_min"]
    # the pool's reclaim rate in every OTHER hour (leave one hour out), from pool and pool-hour sums
    lab = d.label.astype(float)
    pool_s, pool_n = lab.groupby(d.pool).transform("sum"), lab.groupby(d.pool).transform("count")
    ph_s, ph_n = lab.groupby([d.pool, d.hour]).transform("sum"), lab.groupby([d.pool, d.hour]).transform("count")
    rest = pool_n - ph_n
    d["pool_prior"] = ((pool_s - ph_s) / rest.where(rest > 0)).fillna(lab.mean())
    # learned model, cross-validated by whole hours so nothing from the test hour is in training
    feats = ["placement", "age_min", "risk"]
    dd = d.dropna(subset=feats).copy()
    dd["learned"] = np.nan
    hours = dd.hour.unique()
    folds = np.array_split(np.random.default_rng(0).permutation(hours), 5)
    for f in folds:
        te = dd.hour.isin(f)
        if dd[~te].label.nunique() < 2 or te.sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(dd.loc[~te, feats].assign(risk=lambda x: np.log(x.risk + 1e-9)),
                                                        dd.loc[~te, "label"])
        dd.loc[te, "learned"] = lr.predict_proba(dd.loc[te, feats].assign(risk=lambda x: np.log(x.risk + 1e-9)))[:, 1]
    d = d.merge(dd[["instance", "t", "learned"]], on=["instance", "t"], how="left")

    print(f"{len(d)} instance-steps, {int(d.label.sum())} followed by a real reclaim within 15 min ({d.label.mean():.1%})")
    rng = np.random.default_rng(0)
    hrs = d.hour.unique()
    by_hour = {h: d[d.hour == h] for h in hrs}

    def boot(col, sub_fn=lambda x: x):
        vals = []
        for _ in range(500):
            s = sub_fn(pd.concat([by_hour[h] for h in rng.choice(hrs, len(hrs))]))
            a = auc(s.label, s[col])
            if not np.isnan(a):
                vals.append(a)
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)

    out = {"instances": len(inst), "reclaimed": int(got.sum()), "steps": len(d), "positive_steps": int(d.label.sum()),
           "base_rate": float(d.label.mean()), "median_minutes_alive_when_reclaimed": float(mins.median()), "predictors": {}}
    print(f"\n{'predictor':<24}{'AUROC':>8}{'95% CI':>18}")
    names = [("price model", "risk"), ("placement score", "neg_placement"), ("instance age (control)", "neg_age"),
             ("pool prior", "pool_prior"), ("proxy spike (future)", "proxy_spike"), ("learned (cv by hour)", "learned")]
    if d["rebalance"].notna().any():
        names.append(("rebalance recommendation", "rebalance"))
        out["rebalance"] = sig_info
    for name, col in names:
        a = auc(d.label, d[col])
        lo, hi = boot(col)
        out["predictors"][name] = dict(auroc=a, ci=[lo, hi], auprc=float(average_precision_score(d.label[d[col].notna()], d[col].dropna())))
        print(f"{name:<24}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]")

    young = d[d.age_min <= 30]
    out["within_first_30_minutes"] = {"steps": len(young), "positives": int(young.label.sum()), "auroc": {}}
    print(f"\nwithin the first 30 minutes of an instance's life ({len(young)} steps, {int(young.label.sum())} positive):")
    for name, col in names:
        a = auc(young.label, young[col])
        lo, hi = boot(col, lambda x: x[x.age_min <= 30])
        out["within_first_30_minutes"]["auroc"][name] = dict(auroc=a, ci=[lo, hi])
        print(f"  {name:<24}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]")
    # each region on its own (point estimates; regions with few reclaims are noisy)
    out["by_region"] = {}
    print(f"\n{'region':<16}{'inst':>6}{'recl':>6}{'steps':>7}{'pos':>5}  " + "  ".join(n.split(" ")[0][:8] for n, _ in names))
    for region, g in d.groupby("region"):
        gi = inst[inst.region == region]
        out["by_region"][region] = dict(instances=len(gi), reclaimed=int(gi.reclaim_time.notna().sum()), steps=len(g),
                                        positive_steps=int(g.label.sum()), auroc={n: auc(g.label, g[c]) for n, c in names})
        print(f"{region:<16}{len(gi):>6}{int(gi.reclaim_time.notna().sum()):>6}{len(g):>7}{int(g.label.sum()):>5}  "
              + "  ".join(f"{out['by_region'][region]['auroc'][n]:>8.3f}" for n, _ in names))
    out["regions"] = sorted(out["by_region"])
    out["days"] = sorted(d.day.unique())
    with open(os.path.join(HERE, "wide_results", "real_reclaim_eval.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    d.to_csv(os.path.join(HERE, "wide_results", "real_reclaim_steps.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    main()
