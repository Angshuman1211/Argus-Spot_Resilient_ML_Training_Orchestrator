"""
Figures for the v2 paper. Vector PDFs sized for the NeurIPS text width (5.5 in) with type that stays legible
at that size (8 to 9 pt), single y-axis everywhere, and every plotted number read from a CSV in the repo.

    python Paper/v2/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures")
R = os.path.join(ROOT, "benchmark", "results")
W = os.path.join(ROOT, "ml", "model", "wide_results")

INK, INK2, GRID = "#111111", "#4a4a47", "#e2e1dc"
BLUE, ORANGE, GREEN, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#e34948"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------- Figure 1: architecture

def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def arrow(ax, p, q, color=INK2, style="-|>", ls="-", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bands = [(0.70, 0.30, "#e8f1fb", "Prediction"), (0.36, 0.30, "#fdf0e7", "Orchestration"),
             (0.00, 0.32, "#e8f5ee", "Training and storage")]
    for y, h, c, name in bands:
        ax.add_patch(FancyBboxPatch((0.0, y), 1.0, h - 0.02, boxstyle="round,pad=0,rounding_size=0.02", fc=c, ec="none"))
        ax.text(0.012, y + h - 0.045, name, fontsize=9, fontweight="bold", va="top", color=INK2)
    # prediction band
    bw, bh, y1 = 0.215, 0.13, 0.755
    xs = [0.02, 0.265, 0.51, 0.755]
    for x, t in zip(xs, ["EC2 Spot price\nupdates", "Shared feature\ncode", "Risk model\n(boosted trees)", "/predict API\nrisk 0 to 1"]):
        box(ax, x, y1, bw, bh, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + bw, y1 + bh / 2), (b, y1 + bh / 2))
    # orchestration band
    y2 = 0.43
    box(ax, 0.02, y2, 0.20, 0.13, "Operator\npolls each minute")
    box(ax, 0.285, y2, 0.18, 0.13, "Risk above\nthreshold?", fc="#fff8e6")
    box(ax, 0.53, y2, 0.25, 0.13, "Write flush trigger,\nwait for confirmation", fs=8.2)
    box(ax, 0.845, y2, 0.135, 0.13, "Cordon,\ndelete pod", fs=8.2)
    arrow(ax, (0.22, y2 + 0.065), (0.285, y2 + 0.065))
    arrow(ax, (0.465, y2 + 0.065), (0.53, y2 + 0.065))
    ax.text(0.4975, y2 + 0.085, "yes", fontsize=8, ha="center", va="bottom", color=INK2)
    arrow(ax, (0.78, y2 + 0.065), (0.845, y2 + 0.065))
    # the operator reads the score: elbow route from the API down and across to the operator
    xa = xs[3] + bw / 2
    ax.plot([xa, xa], [y1, 0.675], color=INK2, lw=1.2)
    ax.plot([xa, 0.12], [0.675, 0.675], color=INK2, lw=1.2)
    arrow(ax, (0.12, 0.675), (0.12, y2 + 0.13))
    ax.text(0.50, 0.683, "operator reads the score", fontsize=8, color=INK2, ha="center", va="bottom")
    # training band
    y3 = 0.07
    box(ax, 0.02, y3, 0.215, 0.13, "Training pod\n(polls trigger)")
    box(ax, 0.32, y3, 0.215, 0.13, "S3 checkpoints", fc="#ffffff")
    box(ax, 0.62, y3, 0.215, 0.13, "Replacement pod\nresumes")
    arrow(ax, (0.235, y3 + 0.085), (0.32, y3 + 0.085))
    arrow(ax, (0.32, y3 + 0.045), (0.235, y3 + 0.045))
    arrow(ax, (0.535, y3 + 0.065), (0.62, y3 + 0.065))
    ax.text(0.5, y3 - 0.045, "reactive path: AWS 2-minute notice, drain, SIGTERM, flush", fontsize=8, ha="center", color=RED)
    arrow(ax, (0.62, y2), (0.45, y3 + 0.13), color=BLUE, rad=0.0, ls="--")      # trigger -> S3
    ax.text(0.585, 0.32, "flush request", fontsize=8, color=BLUE, ha="center")
    arrow(ax, (0.91, y2), (0.74, y3 + 0.13), color=INK2, ls="--")                 # delete -> replacement
    save(fig, "fig1_architecture")


# ------------------------------------------------------------------------ Figure 2: overhead vs checkpoint cost

def day_ci(runs, policy, C, rng, n=2000):
    g = runs[(runs.policy == policy) & (runs.C == C) & runs.completed]
    day = (g.start * 300 // 86400).astype(int).to_numpy()
    v = g.slowdown.to_numpy()
    days = np.unique(day)
    means = [np.concatenate([v[day == d] for d in rng.choice(days, len(days))]).mean() for _ in range(n)]
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fig_main():
    test = pd.read_csv(os.path.join(R, "csweep", "test", "summary.csv"))
    calib = pd.read_csv(os.path.join(R, "csweep", "calib", "summary.csv"))
    runs = pd.read_csv(os.path.join(R, "csweep", "test", "runs.csv.gz"))
    rng = np.random.default_rng(0)
    model_arms = [p for p in test.policy.unique() if p.startswith("model_")]
    Cs = sorted(test.C_s.unique())
    chosen = {C: calib[(calib.C_s == C) & calib.policy.isin(model_arms)].sort_values("slowdown").iloc[0].policy for C in Cs}
    series = {"reactive on notice": ("reactive", GRAY, "o"), "periodic, 1 hour": ("periodic_1h", "#b9b8b2", "s"),
              "periodic, Young-Daly": ("young_daly", ORANGE, "s"), "model alarms": (None, BLUE, "o")}
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for label, (pol, color, mk) in series.items():
        pts = []
        for C in Cs:
            p = pol or chosen[C]
            m, lo, hi = day_ci(runs, p, C, rng)
            pts.append((C, (m - 1) * 100, (lo - 1) * 100, (hi - 1) * 100))
        x, y, lo, hi = map(np.array, zip(*pts))
        ax.plot(x, y, color=color, marker=mk, ms=4.5, lw=1.8, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
    ax.axvline(120, color=RED, lw=1.0, ls="--")
    ax.text(125, 440, "2-minute\nnotice", color=RED, fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.5, 700)
    ax.set_ylim(1.5, 600)
    ax.set_xticks([5, 10, 30, 60, 120, 300, 600])
    ax.set_xticklabels(["5", "10", "30", "60", "120", "300", "600"])
    ax.set_yticks([2, 5, 10, 30, 100, 300])
    ax.set_yticklabels(["2%", "5%", "10%", "30%", "100%", "300%"])
    ax.set_xlabel("checkpoint time (seconds)")
    ax.set_ylabel("extra time to finish the job")
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig2_overhead_vs_checkpoint")


# ------------------------------------------------------------------------ Figure 3: predictor vs controls

def fig_predictor():
    x = pd.read_csv(os.path.join(W, "xgb_ablation.csv")).groupby("features").mean(numeric_only=True)
    tr = pd.read_csv(os.path.join(W, "ablation.csv")).groupby("experiment").mean(numeric_only=True)
    fin = pd.read_csv(os.path.join(W, "final_eval_flat.csv")) if os.path.exists(os.path.join(W, "final_eval_flat.csv")) else None
    import json
    f = json.load(open(os.path.join(W, "final_eval.json")))
    rows = [
        ("clock only (no model)", f["controls"]["clock_only"]["auroc"], 0.5, GRAY),
        ("old model, tested 6 months later", f["controls"]["old_shipped_transformer"]["all"]["auroc"], np.nan, GRAY),
        ("Transformer, 13 base features", tr.loc["base13", "all_auroc"], np.nan, "#b9b8b2"),
        ("boosted trees, 13 base features", x.loc["base13", "all_auroc"], x.loc["base13", "phase_auroc"], BLUE),
        ("+ update cadence", x.loc["base13+update_density", "all_auroc"], x.loc["base13+update_density", "phase_auroc"], BLUE),
        ("+ price extremes", x.loc["base13+price_extremes", "all_auroc"], x.loc["base13+price_extremes", "phase_auroc"], BLUE),
        ("shipped model (22 features)", f["test"]["all"]["auroc"], f["test"]["in_hour_end_windows"]["auroc"], ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharey=True)
    y = np.arange(len(rows))
    for ax, idx, title in ((axes[0], 1, "all test windows"), (axes[1], 2, "hour-end windows only")):
        vals = [r[idx] for r in rows]
        ax.barh(y, [v if not np.isnan(v) else 0 for v in vals], color=[r[3] for r in rows], height=0.62)
        for yi, v in zip(y, vals):
            if not np.isnan(v):
                ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_xlim(0.4, 1.12)
        ax.axvline(0.5, color=INK2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUROC")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].invert_yaxis()
    save(fig, "fig3_predictor")


# --------------------------------------------------------------------------------- Figure 4: regions

def fig_regions():
    rg = pd.read_csv(os.path.join(R, "regions_summary.csv"))
    d = rg[(rg.N == 15) & (rg.C_s == 300)].sort_values("events", ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        informative = r.events >= 20
        color = BLUE if informative else "#b9b8b2"
        lo, hi = -r.diff_hi * 100, -r.diff_lo * 100
        mid = -r.diff_vs_young_daly * 100
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, solid_capstyle="round")
        ax.plot([mid], [yi], "o", color=color, ms=6)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.region}  ({int(r.events)} event{'s' if r.events != 1 else ''})" for _, r in d.iterrows()])
    ax.invert_yaxis()
    ax.set_xlabel("overhead removed vs Young-Daly (points of job time)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig4_regions")


if __name__ == "__main__":
    fig_architecture()
    fig_main()
    fig_predictor()
    fig_regions()
    print("wrote figures to", OUT)
