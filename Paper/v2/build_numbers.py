"""
Writes Paper/v2/numbers.tex: every number quoted in the paper as a LaTeX macro, read from the result files
(never typed). The paper text uses \\liftFinal, \\auroc... so a number can only change by changing its data.

    python Paper/v2/build_numbers.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
W = os.path.join(ROOT, "ml", "model", "wide_results")
R = os.path.join(ROOT, "benchmark", "results")
macros = {}


def put(name, value, fmt="{:.1f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


# ----------------------------------------------------------------------------------------------- data
meta = pd.read_csv(os.path.join(ROOT, "ml", "data", "features_wide_meta.csv"))
fin = json.load(open(os.path.join(W, "final_eval.json")))
fm = json.load(open(os.path.join(ROOT, "ml", "model", "final", "meta.json")))
td = fm["training_data"]
put("nSeries", fmt_int(len(meta)), "{}")
put("nRegions", len(td["regions"]), "{:d}")
put("nTypes", meta.instance_type.nunique(), "{:d}")
put("nTrainPos", fmt_int(td["train_positives"]), "{}")
put("nCalibPos", fmt_int(td["calib_positives"]), "{}")
put("nTestPos", fmt_int(td["test_positives"]), "{}")
put("nTrainWindows", f"{td['train_windows'] / 1e6:.1f}\\,M", "{}")

# --------------------------------------------------------------------------------------- final model
t = fin["test"]
put("liftFinal", t["all"]["lift"])
put("aurocFinal", t["all"]["auroc"], "{:.3f}")
put("liftPhase", t["in_hour_end_windows"]["lift"])
put("aurocPhase", t["in_hour_end_windows"]["auroc"], "{:.3f}")
put("liftEuNorth", t["eu-north-1"]["lift"])
put("brierFinal", t["brier"], "{:.5f}")
put("brierBase", t["brier_always_predict_base_rate"], "{:.5f}")
lifts = [v["lift"] for v in fin["seed_variability"]]
put("liftSeedMin", min(lifts))
put("liftSeedMax", max(lifts))
c = fin["controls"]
put("liftClock", c["clock_only"]["lift"])
put("aurocClock", c["clock_only"]["auroc"], "{:.3f}")
o = c["old_shipped_transformer"]
put("liftOldAll", o["all"]["lift"])
put("aurocOldAll", o["all"]["auroc"], "{:.3f}")
put("liftOldSame", o["eu-north-1_old5"]["lift"], "{:.2f}")
put("aurocOldSame", o["eu-north-1_old5"]["auroc"], "{:.3f}")

# ---------------------------------------------------------------------------------- feature ablation
x = pd.read_csv(os.path.join(W, "xgb_ablation.csv")).groupby("features").mean(numeric_only=True)
key = {"Base": "base13", "Az": "base13+az_divergence", "Adv": "base13+advisor_rate", "Cad": "base13+update_density",
       "Ext": "base13+price_extremes", "AllNineteen": "all19", "AllTwentyEight": "all28"}
for k, v in key.items():
    put(f"xgbLift{k}", x.loc[v, "all_lift"])
    put(f"xgbAuroc{k}", x.loc[v, "all_auroc"], "{:.3f}")
    put(f"xgbPhase{k}", x.loc[v, "phase_auroc"], "{:.3f}")
a = pd.read_csv(os.path.join(W, "ablation.csv")).groupby("experiment").agg(
    lift=("all_lift", "mean"), mn=("all_lift", "min"), mx=("all_lift", "max"), auroc=("all_auroc", "mean"))
tkey = {"Base": "base13", "Az": "+az_divergence", "Adv": "+advisor_rate", "Cad": "+update_density", "Ext": "+price_extremes",
        "AllNineteen": "all19"}
for k, v in tkey.items():
    put(f"trfLift{k}", a.loc[v, "lift"])
    put(f"trfMin{k}", a.loc[v, "mn"])
    put(f"trfMax{k}", a.loc[v, "mx"])
    put(f"trfAuroc{k}", a.loc[v, "auroc"], "{:.3f}")

# ---------------------------------------------------------------------------------------- rolling origin
ro = pd.read_csv(os.path.join(W, "rolling_origin.csv"))
put("rollAurocMin", ro.auroc.min(), "{:.3f}")
put("rollAurocMax", ro.auroc.max(), "{:.3f}")
put("rollTrainDaysMin", ro.train_days.min())
put("rollFolds", len(ro), "{:d}")

# ------------------------------------------------------------------------- benchmark (eu-north-1, Oct)
te = pd.read_csv(os.path.join(R, "trace_oct_new", "summary.csv"))
ca = pd.read_csv(os.path.join(R, "trace_oct_new_calib", "summary.csv"))
old = pd.read_csv(os.path.join(R, "trace_oct_old", "summary.csv"))
det = pd.read_csv(os.path.join(R, "trace_oct_new", "detection.csv"))
deto = pd.read_csv(os.path.join(R, "trace_oct_old", "detection.csv"))
model_arms = [p for p in te.policy.unique() if p.startswith("model_")]
cname = {10: "Ten", 60: "Sixty", 300: "ThreeHundred"}
for C, nm in cname.items():
    best = ca[(ca.N == 15) & (ca.C_s == C) & ca.policy.isin(model_arms)].sort_values("slowdown").iloc[0].policy
    g = te[(te.N == 15) & (te.C_s == C)].set_index("policy")
    put(f"slowModel{nm}", g.loc[best, "slowdown"], "{:.2f}")
    put(f"slowYD{nm}", g.loc["young_daly", "slowdown"], "{:.2f}")
    put(f"slowReactive{nm}", g.loc["reactive", "slowdown"], "{:.2f}")
    put(f"slowPeriodic{nm}", g.loc["periodic_1h", "slowdown"], "{:.2f}")
    put(f"slowNone{nm}", g.loc["none", "slowdown"], "{:.2f}")
    put(f"slowOracle{nm}", g.loc["oracle_migrate", "slowdown"], "{:.2f}")
    go = old[(old.N == 15) & (old.C_s == C)].set_index("policy")
    put(f"slowOldModel{nm}", go.loc[best, "slowdown"], "{:.2f}")
    for tag, val in (("Model", g.loc[best, "slowdown"]), ("YD", g.loc["young_daly", "slowdown"]),
                     ("Reactive", g.loc["reactive", "slowdown"]), ("Periodic", g.loc["periodic_1h", "slowdown"])):
        put(f"ov{tag}{nm}", (val - 1) * 100, "{:.0f}")
pq = pd.read_csv(os.path.join(R, "trace_oct_new", "predictor_quality.csv"))
pqt = pq[pq.split == "test"].set_index("signal")
put("aurocVol", pqt.loc["volatility_rule", "auroc"], "{:.2f}")
put("aurocAdvisorRule", pqt.loc["advisor_prior", "auroc"], "{:.2f}")
put("slowAdvisorYDThreeHundred", te[(te.N == 15) & (te.C_s == 300) & (te.policy == "advisor_young_daly")].iloc[0].slowdown, "{:.2f}")
put("slowVolThreeHundred", te[(te.N == 15) & (te.C_s == 300) & (te.policy == "vol_migrate_q0.1")].iloc[0].slowdown, "{:.2f}")
ci = pd.read_csv(os.path.join(R, "trace_oct_new", "paired_day_ci.csv"))
cr = ci[(ci.N == 15) & (ci.baseline == "young_daly")].set_index("C_s")
for C, nm in cname.items():
    put(f"diffYD{nm}", -cr.loc[float(C), "mean_diff"], "{:.3f}")
    put(f"diffYDLo{nm}", -cr.loc[float(C), "diff_hi"], "{:.3f}")
    put(f"diffYDHi{nm}", -cr.loc[float(C), "diff_lo"], "{:.3f}")
    put(f"daysBetter{nm}", cr.loc[float(C), "share_of_days_model_better"] * 100, "{:.0f}")
put("benchDays", cr.days.iloc[0], "{:d}")
d10 = det[(det.signal == "model") & (det.budget == 0.1)].iloc[0]
o10 = deto[(deto.signal == "model") & (deto.budget == 0.1)].iloc[0]
put("events", d10.kills, "{:d}")
put("recallNew", d10.recall * 100, "{:.0f}")
put("precNew", d10.precision * 100, "{:.1f}")
put("recallOld", o10.recall * 100, "{:.0f}")
put("precOld", o10.precision * 100, "{:.1f}")

# -------------------------------------------------------------------------------- regions
rg = pd.read_csv(os.path.join(R, "regions_summary.csv"))
r300 = rg[(rg.N == 15) & (rg.C_s == 300)].set_index("region")
for reg, nm in (("eu-north-1", "EuNorth"), ("ap-southeast-1", "ApSoutheast"), ("us-east-1", "UsEastOne"), ("us-east-2", "UsEastTwo")):
    put(f"regDiff{nm}", -r300.loc[reg, "diff_vs_young_daly"], "{:.3f}")
    put(f"regLo{nm}", -r300.loc[reg, "diff_hi"], "{:.3f}")
    put(f"regHi{nm}", -r300.loc[reg, "diff_lo"], "{:.3f}")
    put(f"regEvents{nm}", r300.loc[reg, "events"], "{:d}")

# -------------------------------------------------------------------------- in-region checkpoint speed
eks = pd.read_csv(os.path.join(R, "trace", "checkpoint_write_eks.csv"))
big = eks[eks.size_mb >= 512]
put("awsRateMean", (big.size_mb / (big.upload_ms / 1000)).mean(), "{:.0f}")
put("awsRateMax", (big.size_mb / (big.upload_ms / 1000)).max(), "{:.0f}")
big2 = eks[(eks.size_mb == 2048) & (eks.concurrency == 10)].iloc[0]
put("awsTwoGbSeconds", big2.upload_ms / 1000, "{:.0f}")

# ------------------------------------------- idealized predictor on top of Young-Daly (15 nodes, C = 300 s)
sy = pd.read_csv(os.path.join(R, "trace_oct_new", "summary_synth.csv"))
sy = sy[(sy.N == 15) & (sy.C_s == 300)].set_index("policy").slowdown
put("synthBest", sy["synthetic_yd_r0.95_p0.9"], "{:.2f}")      # 95% recall, 90% precision
put("synthSixty", sy["synthetic_yd_r0.95_p0.6"], "{:.2f}")     # 95% recall, 60% precision
put("synthLowP", sy["synthetic_yd_r0.95_p0.05"], "{:.2f}")     # 95% recall, 5% precision
put("synthLowPLowR", sy["synthetic_yd_r0.2_p0.05"], "{:.2f}")  # 20% recall, 5% precision
_det = det[det.signal == "model"].set_index("budget")
put("opRecallTen", _det.loc[0.1, "recall"] * 100, "{:.0f}")
put("opPrecTen", _det.loc[0.1, "precision"] * 100, "{:.0f}")
put("opRecallTwenty", _det.loc[0.2, "recall"] * 100, "{:.0f}")
put("opPrecTwenty", _det.loc[0.2, "precision"] * 100, "{:.0f}")
_t = te[(te.N == 15) & (te.C_s == 300)].set_index("policy").slowdown
put("opSlowTen", _t["model_migrate_q0.1"], "{:.2f}")
put("opSlowTwenty", _t["model_migrate_q0.2"], "{:.2f}")
put("opSlowTiny", _t["model_migrate_q0.002"], "{:.2f}")
put("ydSlowThreeHundred", _t["young_daly"], "{:.2f}")

# ---------------------------------------------------------------------- real reclaims (if present)
rr = os.path.join(W, "real_reclaim_eval.json")
if os.path.exists(rr):
    e = json.load(open(rr))
    put("realInstances", e["instances"], "{:d}")
    put("realReclaimed", e["reclaimed"], "{:d}")
    put("realSteps", fmt_int(e["steps"]), "{}")
    put("realPosSteps", e["positive_steps"], "{:d}")
    put("realMedianMin", e["median_minutes_alive_when_reclaimed"], "{:.0f}")
    put("realYoungSteps", e["within_first_30_minutes"]["steps"], "{:d}")
    put("realYoungPos", e["within_first_30_minutes"]["positives"], "{:d}")
    for nm, k in (("Price", "price model"), ("Place", "placement score"), ("Age", "instance age (control)"),
                  ("Prior", "pool prior"), ("Proxy", "proxy spike (future)"), ("Learned", "learned (cv by hour)")):
        p_ = e["predictors"][k]
        put(f"realAuroc{nm}", p_["auroc"], "{:.2f}")
        put(f"realLo{nm}", p_["ci"][0], "{:.2f}")
        put(f"realHi{nm}", p_["ci"][1], "{:.2f}")
        y = e["within_first_30_minutes"]["auroc"][k]
        put(f"realYoung{nm}", y["auroc"], "{:.2f}")
    if "rebalance recommendation" in e["predictors"]:  # AWS's free early warning, scored on covered sensors
        p_, rb = e["predictors"]["rebalance recommendation"], e["rebalance"]
        put("realAurocReb", p_["auroc"], "{:.2f}")
        put("realLoReb", p_["ci"][0], "{:.2f}")
        put("realHiReb", p_["ci"][1], "{:.2f}")
        put("realRebCovered", rb["covered_reclaims"], "{:d}")
        put("realRebWarned", rb["reclaims_warned"], "{:d}")
        put("realRebLead", rb["median_lead_min"] or 0, "{:.0f}")
        put("realRebFalse", rb["false_warnings"], "{:d}")
        put("realRebUnrec", rb["unreclaimed_covered"], "{:d}")
    # launch attempts that failed for lack of capacity, over every sensor session (the first session wrote a different file name)
    real_dir = os.path.join(ROOT, "ml", "data", "real")
    failed = 0
    for f in glob.glob(os.path.join(real_dir, "sensor_campaign_*.csv")) + glob.glob(os.path.join(real_dir, "sensor_launches_*.csv")):
        failed += int((pd.read_csv(f)["status"].astype(str) == "failed:InsufficientInstanceCapacity").sum())
    put("realCapFail", failed, "{:d}")
    sensor_types = {"c5.large", "m5.large", "t3.medium", "c6i.large", "m6i.large", "r5.large", "r6i.large", "t3.large"}
    rec = pd.read_csv(os.path.join(real_dir, "real_reclaims.csv"))
    put("realTypesReclaimed", rec[rec.instance_type.isin(sensor_types)].instance_type.nunique(), "{:d}")
lm = os.path.join(W, "real_label_model.json")
if os.path.exists(lm):
    m_ = json.load(open(lm))
    put("realByPoolPlaceAge", m_["by_pool"]["placement+age (logistic)"]["auroc"], "{:.2f}")
    put("realByPoolPlaceAgeLo", m_["by_pool"]["placement+age (logistic)"]["ci"][0], "{:.2f}")
    put("realByPoolPlaceAgeHi", m_["by_pool"]["placement+age (logistic)"]["ci"][1], "{:.2f}")
    for scheme, nm in (("forward_day", "Fwd"), ("by_region", "Region")):
        if scheme in m_:
            for key, short in (("placement+age (logistic)", "PlaceAge"), ("placement score alone", "Place")):
                put(f"real{nm}{short}", m_[scheme][key]["auroc"], "{:.2f}")
                put(f"real{nm}{short}Lo", m_[scheme][key]["ci"][0], "{:.2f}")
                put(f"real{nm}{short}Hi", m_[scheme][key]["ci"][1], "{:.2f}")
if os.path.exists(rr) and "by_region" in e:
    put("realNRegions", len(e["by_region"]), "{:d}")
    put("realNDays", len(e["days"]), "{:d}")
    for reg, nm in (("eu-north-1", "EuNorth"), ("us-east-1", "UsEastOne"), ("us-east-2", "UsEastTwo"),
                    ("ap-southeast-1", "ApSoutheast")):
        if reg in e["by_region"]:
            put(f"realPlace{nm}", e["by_region"][reg]["auroc"]["placement score"], "{:.2f}")
            put(f"realRecl{nm}", e["by_region"][reg]["reclaimed"], "{:d}")

# ------------------------------------------------------------------ real-cluster drills (from the logs)
import datetime as _dt
import re as _re

DR = os.path.join(ROOT, "demo", "drill", "results")


def _ts(line):
    m = _re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)", line)
    return _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + _dt.timedelta(milliseconds=int(m.group(2))) if m else None


for tag, nm in (("drill-small", "Small"), ("drill-big", "Big")):
    ack, resume = "TBD", "TBD"
    op = os.path.join(DR, f"{tag}.operator.log")
    con = os.path.join(DR, f"{tag}.console.log")
    if os.path.exists(op):
        lines = open(op, encoding="utf-8", errors="ignore").read().splitlines()
        # the log may hold earlier attempts: use the LAST trigger and the first events after it
        last = max(i for i, l in enumerate(lines) if "[CHECKPOINT] Trigger" in l) if any("[CHECKPOINT] Trigger" in l for l in lines) else 0
        lines = lines[last:]
        t_trig = next((_ts(l) for l in lines if "[CHECKPOINT] Trigger" in l), None)
        t_ok = next((_ts(l) for l in lines if "checkpoint confirmed on S3" in l), None)
        t_del = next((_ts(l) for l in lines if "[SCHEDULER] Pod" in l and "deleted" in l and "Deleting" not in l), None)
        if t_trig and t_ok:
            ack = f"{(t_ok - t_trig).total_seconds():.1f}"
        if t_del and os.path.exists(con):
            m = _re.search(r"replacement pod ready (\d\d:\d\d:\d\d)", open(con, encoding="utf-8", errors="ignore").read())
            if m:
                ready = _dt.datetime.combine(t_del.date(), _dt.datetime.strptime(m.group(1), "%H:%M:%S").time())
                resume = f"{max((ready - t_del).total_seconds(), 0):.0f}"
    put(f"drillAck{nm}", ack, "{}")
    put(f"drillResume{nm}", resume, "{}")

with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    f.write("% generated by build_numbers.py from the result files; do not edit by hand\n")
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(f"wrote {len(macros)} macros")
