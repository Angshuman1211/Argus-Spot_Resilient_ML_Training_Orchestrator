"""
Writes the LaTeX tables of the paper to Paper/v2/tables/*.tex from the result files, so no cell is typed by hand.

    python Paper/v2/build_tables.py
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
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# ----------------------------------------------------- Table 1: checkpoint size and time at the measured rate
eks = pd.read_csv(os.path.join(R, "trace", "checkpoint_write_eks.csv"))
big = eks[eks.size_mb >= 512]
rate = float((big.size_mb / (big.upload_ms / 1000)).mean())  # MB/s per node, measured on the EKS nodes
models = [("CIFAR-10 CNN (this study)", 0.3e6), ("ResNet-50", 25.6e6), ("BERT-large", 340e6),
          ("1.3B-parameter LLM", 1.3e9), ("7B-parameter LLM", 7e9)]
rows = ["\\begin{tabular}{lrrrr}", "\\toprule",
        "Model & Parameters & Training state & Time to S3 & Fits in 2 min? \\\\", "\\midrule"]
for name, p in models:
    gb = p * 12 / 1e9  # fp32 weights (4 B) + Adam moments (8 B)
    sec = gb * 1000 / rate
    nice_p = f"{p / 1e9:.1f}B" if p >= 1e9 else f"{p / 1e6:.1f}M"
    state = f"{gb * 1000:.0f} MB" if gb < 1 else f"{gb:.1f} GB"
    t = f"{sec:.0f} s" if sec >= 1 else "under 1 s"
    rows.append(f"{name} & {nice_p} & {state} & {t} & {'yes' if sec <= 120 else 'no'} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
write("table_ckpt.tex", rows)

# ------------------------------------------------------------------- Table 2: 15-node benchmark
te = pd.read_csv(os.path.join(R, "trace_oct_new", "summary.csv"))
ca = pd.read_csv(os.path.join(R, "trace_oct_new_calib", "summary.csv"))
old = pd.read_csv(os.path.join(R, "trace_oct_old", "summary.csv"))
arms = [p for p in te.policy.unique() if p.startswith("model_")]
Cs = [10, 60, 300]
chosen = {C: ca[(ca.N == 15) & (ca.C_s == C) & ca.policy.isin(arms)].sort_values("slowdown").iloc[0].policy for C in Cs}


def cell(df, pol, C, ci=True):
    r = df[(df.N == 15) & (df.C_s == C) & (df.policy == pol)].iloc[0]
    return f"{r.slowdown:.2f}"


rows = ["\\begin{tabular}{lrrr}", "\\toprule", "Policy (15 nodes) & 10 s & 60 s & 300 s \\\\", "\\midrule"]
lines = [("No protection", "none", te), ("Reactive on the 2-minute notice", "reactive", te),
         ("Periodic, every hour", "periodic_1h", te), ("Periodic, Young-Daly interval", "young_daly", te)]
for label, pol, df in lines:
    rows.append(f"{label} & " + " & ".join(cell(df, pol, C) for C in Cs) + " \\\\")
rows.append("Alarms from the old model, same budget & " + " & ".join(cell(old, chosen[C], C) for C in Cs) + " \\\\")
best = {C: min(float(cell(te, p, C)) for _, p, _ in lines) for C in Cs}
vals = [float(cell(te, chosen[C], C)) for C in Cs]
rows.append("\\textbf{Alarms from the retrained model} & " +
            " & ".join((f"\\textbf{{{v:.2f}}}" if v < best[C] else f"{v:.2f}") for v, C in zip(vals, Cs)) + " \\\\")
rows.append("Oracle, 5 minutes of perfect warning & " + " & ".join(cell(te, "oracle_migrate", C) for C in Cs) + " \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
write("table_bench.tex", rows)

# --------------------------------------------------------------- Table 3: feature ablation (appendix)
x = pd.read_csv(os.path.join(W, "xgb_ablation.csv")).groupby("features").agg(
    lift=("all_lift", "mean"), sd=("all_lift", "std"), auroc=("all_auroc", "mean"), phase=("phase_auroc", "mean"))
tr = pd.read_csv(os.path.join(W, "ablation.csv")).groupby("experiment").agg(lift=("all_lift", "mean"), auroc=("all_auroc", "mean"))
names = [("13 base features", "base13", "base13"), ("+ cross-AZ price divergence", "base13+az_divergence", "+az_divergence"),
         ("+ Spot Advisor interruption rate", "base13+advisor_rate", "+advisor_rate"),
         ("+ update cadence", "base13+update_density", "+update_density"),
         ("+ price extremes", "base13+price_extremes", "+price_extremes"), ("all 19 features", "all19", "all19"),
         ("all 28 features", "all28", None)]
rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
        "& \\multicolumn{3}{c}{Boosted trees} & \\multicolumn{2}{c}{Transformer} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
        "Features & Lift & AUROC & Hour-end AUROC & Lift & AUROC \\\\", "\\midrule"]
for label, xk, tk in names:
    t = f"{tr.loc[tk, 'lift']:.1f} & {tr.loc[tk, 'auroc']:.3f}" if tk and tk in tr.index else "& "
    rows.append(f"{label} & {x.loc[xk, 'lift']:.1f} & {x.loc[xk, 'auroc']:.3f} & {x.loc[xk, 'phase']:.3f} & {t} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
write("table_ablation.tex", rows)

# ----------------------------------------------------------------------- Table 4: rolling origin
ro = pd.read_csv(os.path.join(W, "rolling_origin.csv"))
rows = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Test block & Train days & Positives & Lift & AUROC & Hour-end AUROC \\\\", "\\midrule"]
for _, r in ro.iterrows():
    rows.append(f"{r.test_start[5:]} to {r.test_end[5:]} & {r.train_days:.0f} & {int(r.test_positives):,} & {r.lift:.1f} & {r.auroc:.3f} & {r.in_hour_end_auroc:.3f} \\\\".replace(",", "{,}"))
rows += ["\\bottomrule", "\\end{tabular}"]
write("table_rolling.tex", rows)

# ----------------------------------------------------------------------- Table 5: regions
rg = pd.read_csv(os.path.join(R, "regions_summary.csv"))
d = rg[(rg.N == 15) & (rg.C_s == 300)].sort_values("events", ascending=False)
rows = ["\\begin{tabular}{lrrrrl}", "\\toprule", "Region & Events & Model & Young-Daly & Difference [95\\% interval] & Read \\\\", "\\midrule"]
for _, r in d.iterrows():
    diff, lo, hi = -r.diff_vs_young_daly, -r.diff_hi, -r.diff_lo
    if r.events < 20:
        read = "too few events"
    else:
        read = "model better" if lo > 0 else ("model worse" if hi < 0 else "tie")
    rows.append(f"{r.region} & {int(r.events)} & {r.model:.2f} & {r.young_daly:.2f} & {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] & {read} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
write("table_regions.tex", rows)

# ------------------------------------------------------------- Table 6: real reclaims (if evaluated)
rr = os.path.join(W, "real_reclaim_eval.json")
if os.path.exists(rr):
    e = json.load(open(rr))
    rows = [r"\begin{tabular}{lrrr}", r"\toprule", r"Predictor & AUROC & 95\% interval & First 30 min \\", r"\midrule"]
    order = [("Price-history model (retrained)", "price model"), ("AWS Spot placement score", "placement score"),
             ("Reclaim rate of the pool (leave one hour out)", "pool prior"),
             ("Instance age, younger is riskier (control)", "instance age (control)"),
             ("Logistic model on placement, age, price risk", "learned (cv by hour)"),
             ("Price spike in the next 15 min (not causal)", "proxy spike (future)")]
    for label, k in order:
        p = e["predictors"][k]
        y = e["within_first_30_minutes"]["auroc"][k]["auroc"]
        rows.append(f"{label} & {p['auroc']:.2f} & [{p['ci'][0]:.2f}, {p['ci'][1]:.2f}] & {y:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_real.tex", rows)
print("tables written to", OUT, "| measured S3 rate used:", round(rate), "MB/s")
