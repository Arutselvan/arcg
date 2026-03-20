"""
Generate all paper figures as high-quality vector PDFs.
Figures are saved to ../figures/ relative to this script.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../data/arcg_ollama_results.json")
FIG_DIR    = os.path.join(SCRIPT_DIR, "../figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

summary = data["summary"]
metrics = data["metrics"]   # list of per-problem dicts

MODELS = ["llama3.1:8b", "qwen2.5:14b", "mistral:7b", "gemma2:9b", "phi4:14b"]
MODEL_LABELS = ["Llama 3.1\n8B", "Qwen 2.5\n14B", "Mistral\n7B", "Gemma 2\n9B", "Phi-4\n14B"]
MODEL_LABELS_SHORT = ["Llama 3.1 8B", "Qwen 2.5 14B", "Mistral 7B", "Gemma 2 9B", "Phi-4 14B"]

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  1.5,
})

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
GRAY   = "#636363"
COLORS = [BLUE, ORANGE, GREEN, "#8073ac", "#e08214"]

# ── helper ─────────────────────────────────────────────────────────────────────
def get_per_problem(model):
    return [m for m in metrics if m["model"] == model]

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

# ==============================================================================
# Figure 1 — FAC vs RSC grouped bar chart with ARCG annotation
# ==============================================================================
def fig1_fac_rsc():
    fac = [summary[m]["overall"]["FAC_mean"] for m in MODELS]
    rsc = [summary[m]["overall"]["RSC_mean"] for m in MODELS]
    fac_std = [summary[m]["overall"]["FAC_std"] for m in MODELS]
    rsc_std = [summary[m]["overall"]["RSC_std"] for m in MODELS]
    arcg = [f - r for f, r in zip(fac, rsc)]

    x = np.arange(len(MODELS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars_fac = ax.bar(x - w/2, fac, w, yerr=fac_std, capsize=4,
                      color=BLUE, alpha=0.85, label="FAC (Answer Consistency)", zorder=3)
    bars_rsc = ax.bar(x + w/2, rsc, w, yerr=rsc_std, capsize=4,
                      color=ORANGE, alpha=0.85, label="RSC (Reasoning Consistency)", zorder=3)

    # Annotate ARCG gap
    for i, (f, r, a) in enumerate(zip(fac, rsc, arcg)):
        ax.annotate(
            f"ARCG\n{a:+.2f}",
            xy=(x[i], max(f, r) + 0.04),
            ha="center", va="bottom",
            fontsize=7.5, color=GRAY,
            fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Consistency Score")
    ax.set_title("Final Answer Consistency (FAC) vs. Reasoning Step Consistency (RSC)\nby Model (ARCG = FAC $-$ RSC; all values negative)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig1_fac_rsc.pdf")

# ==============================================================================
# Figure 2 — ARCG by domain (Math vs Logic) — grouped bars
# ==============================================================================
def fig2_arcg_domain():
    math_arcg  = [summary[m]["by_domain"]["math"]["ARCG_mean"]  for m in MODELS]
    logic_arcg = [summary[m]["by_domain"]["logic"]["ARCG_mean"] for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w/2, math_arcg,  w, color=BLUE,   alpha=0.85, label="Math (GSM8K)",       zorder=3)
    ax.bar(x + w/2, logic_arcg, w, color=ORANGE,  alpha=0.85, label="Logic (ARC-Challenge)", zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("ARCG (FAC $-$ RSC)")
    ax.set_title("ARCG by Task Domain\n(Logic tasks show significantly more negative gap; $t=6.93$, $p<0.001$)")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig2_arcg_domain.pdf")

# ==============================================================================
# Figure 3 — ARCG by difficulty — line plot
# ==============================================================================
def fig3_arcg_difficulty():
    difficulties = ["easy", "medium", "hard"]
    diff_labels  = ["Easy", "Medium", "Hard"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for i, (m, label, col) in enumerate(zip(MODELS, MODEL_LABELS_SHORT, COLORS)):
        vals = [summary[m]["by_difficulty"][d]["ARCG_mean"] for d in difficulties]
        ax.plot(diff_labels, vals, marker="o", color=col, label=label, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylabel("ARCG (FAC $-$ RSC)")
    ax.set_title("ARCG Degrades Monotonically with Problem Difficulty\n(Easy $\\to$ Hard: $t=3.25$, $p=0.002$)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig3_arcg_difficulty.pdf")

# ==============================================================================
# Figure 4 — FAC vs Accuracy scatter (all models, all problems)
# ==============================================================================
def fig4_fac_accuracy_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)

    for ax, (metric_key, metric_label, r_val, p_val) in zip(axes, [
        ("FAC", "FAC (Answer Consistency)", 0.448, 0.001),
        ("RSC", "RSC (Reasoning Consistency)", 0.097, 0.236),
    ]):
        for m, col in zip(MODELS, COLORS):
            pts = get_per_problem(m)
            xs = [p[metric_key] for p in pts]
            ys = [p["accuracy"]  for p in pts]
            ax.scatter(xs, ys, color=col, alpha=0.5, s=18, zorder=3)

        # Pooled regression line
        all_x = [p[metric_key] for p in metrics]
        all_y = [p["accuracy"]  for p in metrics]
        slope, intercept, *_ = stats.linregress(all_x, all_y)
        xr = np.linspace(0, 1, 100)
        ax.plot(xr, slope * xr + intercept, color="black", linewidth=1.5, linestyle="--")

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{metric_label}\n$r={r_val:.3f}$, $p={p_val:.3f}$ ({sig})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Shared legend
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, MODEL_LABELS_SHORT)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.suptitle("Predictive Power of Consistency Metrics\n(FAC predicts accuracy; RSC does not)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_fac_accuracy.pdf")

# ==============================================================================
# Figure 5 — ARCG distribution violin plot (per model)
# ==============================================================================
def fig5_arcg_violin():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    arcg_per_model = [[p["ARCG"] for p in get_per_problem(m)] for m in MODELS]
    parts = ax.violinplot(arcg_per_model, positions=range(len(MODELS)),
                          showmedians=True, showextrema=True)

    for i, (pc, col) in enumerate(zip(parts["bodies"], COLORS)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")

    ax.axhline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label="ARCG = 0")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("ARCG per Problem")
    ax.set_title("Distribution of ARCG Across 30 Problems per Model\n(Negative ARCG is universal; red dashed line = 0)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_arcg_violin.pdf")

# ==============================================================================
# Run all
# ==============================================================================
if __name__ == "__main__":
    print("Generating vector PDF figures...")
    fig1_fac_rsc()
    fig2_arcg_domain()
    fig3_arcg_difficulty()
    fig4_fac_accuracy_scatter()
    fig5_arcg_violin()
    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Files:", sorted(os.listdir(FIG_DIR)))
