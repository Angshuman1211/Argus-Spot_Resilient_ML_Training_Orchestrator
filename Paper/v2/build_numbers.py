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
put("s3RateMean", (big.size_mb / (big.upload_ms / 1000)).mean(), "{:.0f}")
put("s3RateMax", (big.size_mb / (big.upload_ms / 1000)).max(), "{:.0f}")
big2 = eks[(eks.size_mb == 2048) & (eks.concurrency == 10)].iloc[0]
put("s3TwoGbSeconds", big2.upload_ms / 1000, "{:.0f}")

# ---------------------------------------------------------------------- real reclaims (if present)
rr = os.path.join(W, "real_reclaim_eval.json")
if os.path.exists(rr):
    e = json.load(open(rr))
    put("realInstances", e["instances"], "{:d}")
    put("realReclaimed", e["reclaimed"], "{:d}")
    put("realSteps", fmt_int(e["steps"]), "{}")
    put("realPosSteps", e["positive_steps"], "{:d}")
    for nm, k in (("Price", "price model"), ("Place", "placement score"), ("Proxy", "proxy spike (future)"), ("Prior", "pool prior")):
        p = e["predictors"][k]
        put(f"realAuroc{nm}", p["auroc"], "{:.2f}")
        put(f"realLo{nm}", p["ci"][0], "{:.2f}")
        put(f"realHi{nm}", p["ci"][1], "{:.2f}")

with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    f.write("% generated by build_numbers.py from the result files; do not edit by hand\n")
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(f"wrote {len(macros)} macros")
