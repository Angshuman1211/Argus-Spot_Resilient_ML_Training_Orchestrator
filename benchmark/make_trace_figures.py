"""
Figures and headline tables for the trace-driven benchmark. Reads benchmark/results/trace/
summary.csv (written by run_trace_benchmark.py); every number in the plots comes from it.

    python benchmark/make_trace_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "trace")
FIG = os.path.join(HERE, "results", "figures")

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT = "#2a78d6"      # categorical slot 1: the learned model
NEUTRAL = "#8a8985"     # everything that is not the model
HIGHLIGHT = "#eb6834"   # categorical slot 2: the best non-learned baseline

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

LABELS = {
    "none": "no protection",
    "reactive": "reactive on 2-min notice",
    "periodic_1h": "periodic, every 1 h",
    "young_daly": "periodic, Young-Daly interval",
    "advisor_young_daly": "periodic, Advisor-tier interval",
    "model_migrate_q0.002": "model alarms, checkpoint + migrate",
    "model_ckpt_q0.002": "model alarms, checkpoint only",
    "hybrid_yd_model_migrate_q0.002": "Young-Daly + model alarms",
    "random_migrate_q0.002": "random alarms (same budget)",
    "vol_migrate_q0.002": "volatility rule (same budget)",
    "oracle_migrate": "oracle (5 min of perfect warning)",
}
MODEL_ARMS = {"model_migrate_q0.002", "model_ckpt_q0.002", "hybrid_yd_model_migrate_q0.002"}


ORDER = ["none", "reactive", "periodic_1h", "young_daly", "advisor_young_daly", "random_migrate_q0.002",
         "vol_migrate_q0.002", "model_ckpt_q0.002", "model_migrate_q0.002",
         "hybrid_yd_model_migrate_q0.002", "oracle_migrate"]


def fig_policies(s):
    # One fixed policy order for every panel. With sharey=True a per-panel sort would
    # let the last panel's tick labels overwrite the others and mislabel the bars.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    y = np.arange(len(ORDER))
    for ax, C in zip(axes, (10, 60, 300)):
        d = s[(s.N == 15) & (s.C_s == C)].set_index("policy").loc[ORDER]
        colors = [ACCENT if p in MODEL_ARMS else (HIGHLIGHT if p == "young_daly" else NEUTRAL) for p in ORDER]
        ax.barh(y, d.slowdown - 1.0, left=1.0, color=colors, height=0.62)
        lo, hi = d.slowdown_lo.to_numpy(), d.slowdown_hi.to_numpy()
        ax.errorbar(d.slowdown, y, xerr=[np.nan_to_num(d.slowdown - lo), np.nan_to_num(hi - d.slowdown)],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for yi, v, h in zip(y, d.slowdown, d.slowdown_hi):
            ax.text(max(v, h if np.isfinite(h) else v) + 0.07, yi, f"{v:.2f}x", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[p] for p in ORDER])
        ax.set_title(f"checkpoint takes {C} s", loc="left", fontsize=11, color=INK)
        ax.axvline(1.0, color=INK2, lw=0.8)
        ax.set_xlim(0.95, 4.9)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("makespan / ideal work time (1.0 = no loss)")
    axes[0].invert_yaxis()
    fig.suptitle("15-node synchronous job on the held-out price trace: slowdown by policy "
                 "(mean, 95% block-bootstrap CI). Blue = model-driven, orange = Young-Daly baseline",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "trace_policy_slowdown.png"), dpi=170)
    plt.close(fig)


def fig_frontier(s, synth):
    syn = synth[synth.policy.str.startswith("synthetic_yd_") & (synth.N == 15)].copy()
    syn[["recall", "precision"]] = syn.policy.str.extract(r"synthetic_yd_r([\d.]+)_p([\d.]+)").astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, C in zip(axes, (10, 60, 300)):
        d = syn[syn.C_s == C]
        piv = d.pivot(index="precision", columns="recall", values="slowdown").sort_index(ascending=False)
        yd = float(s[(s.N == 15) & (s.C_s == C) & (s.policy == "young_daly")].slowdown.iloc[0])
        react = float(s[(s.N == 15) & (s.C_s == C) & (s.policy == "reactive")].slowdown.iloc[0])
        base = min(yd, react)
        im = ax.imshow(piv.values, cmap="Blues_r", vmin=1.0, vmax=max(2.0, float(piv.values.max())),
                       aspect="auto")
        for (i, j), v in np.ndenumerate(piv.values):
            wins = v < base
            ax.text(j, i, f"{v:.2f}" + ("*" if wins else ""), ha="center", va="center", fontsize=10,
                    color=INK if v > 1.45 else "white", fontweight="bold" if wins else "normal")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{r:.0%}" for r in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{p:.0%}" for p in piv.index])
        ax.set_xlabel("recall (kills warned 10 min ahead)")
        ax.set_ylabel("precision (alarms that were real)")
        ax.set_title(f"checkpoint {C} s. Best baseline {base:.2f}x", loc="left", fontsize=11, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("How good must a predictor be? Young-Daly checkpointing plus a synthetic predictor that "
                 "migrates on alarm. * beats the best non-predictive policy", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, 0.01, "The shipped model measures about 19% recall at 8% precision on this trace "
             "(detection.csv), near the bottom-left cells.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(os.path.join(FIG, "trace_predictor_frontier.png"), dpi=170)
    plt.close(fig)


def fig_fleet(s):
    pols = [("none", "no protection", NEUTRAL), ("young_daly", "Young-Daly periodic", HIGHLIGHT),
            ("reactive", "reactive on notice", INK2), ("model_migrate_q0.002", "model alarms", ACCENT)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, C in zip(axes, (60, 300)):
        for p, label, col in pols:
            d = s[(s.policy == p) & (s.C_s == C)].sort_values("N")
            ax.plot(d.N, d.slowdown, marker="o", ms=6, lw=2, color=col)
            if C == 300 and p in ("none", "reactive"):
                if p == "none":
                    ax.text(d.N.iloc[-1] + 0.3, d.slowdown.iloc[-1], "no protection = reactive here\n(checkpoint exceeds the notice)",
                            va="center", fontsize=9, color=INK)
                continue
            ax.text(d.N.iloc[-1] + 0.3, d.slowdown.iloc[-1] + {"reactive": -0.06, "young_daly": 0.05}.get(p, 0), label,
                    va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 15])
        ax.set_xticklabels(["1", "5", "15"])
        ax.set_xlim(0.8, 60)
        ax.set_xlabel("nodes in the synchronous job")
        ax.set_title(f"checkpoint takes {C} s", loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("makespan / ideal work time")
    fig.suptitle("Interruption exposure grows with job size; so does the cost of each policy",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "trace_fleet_size.png"), dpi=170)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = pd.read_csv(os.path.join(OUT, "summary.csv"))
    fig_policies(s)
    fig_frontier(s, pd.read_csv(os.path.join(OUT, "summary_synth.csv")))
    fig_fleet(s)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
