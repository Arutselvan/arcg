"""
ARCG Experiment  --  Step 6: Analyze Results and Generate Figures
==================================================================
Reads data/experiment_results.json and computes all ARCG metrics:

  FAC  (Final Answer Consistency)
       Fraction of (problem, model) pairs where all paraphrases
       produce the same extracted answer.

  RSC  (Reasoning Step Consistency)
       Mean pairwise cosine similarity of CoT reasoning chains
       across paraphrases, computed via sentence-transformers.

  ARCG (Answer-Reasoning Consistency Gap)
       ARCG = FAC - RSC
       Positive: answers more consistent than reasoning (post-hoc rationalization).
       Negative: reasoning more consistent than answers (execution fragility).

  ARC  (Answer-Reasoning Coherence)
       Fraction of responses where the extracted answer is logically
       supported by the reasoning chain (judged by keyword/pattern match).

Statistical tests:
  - Paired t-test: FAC vs RSC per model (ARCG significance)
  - Kruskal-Wallis: ARCG across models
  - Pearson correlation: FAC vs accuracy, RSC vs accuracy
  - Mann-Whitney U: ARCG by domain (math vs logic)
  - One-way ANOVA: ARCG by difficulty (easy / medium / hard)

Figures (all vector PDF, publication quality):
  fig1_fac_rsc_comparison.pdf    Grouped bar: FAC and RSC per model with ARCG annotation
  fig2_arcg_by_domain.pdf        ARCG by domain per model (grouped bar)
  fig3_arcg_by_difficulty.pdf    ARCG by difficulty level (line chart)
  fig4_fac_accuracy_scatter.pdf  FAC vs accuracy scatter with regression
  fig5_arcg_violin.pdf           Violin plot of ARCG distribution per model
  fig6_model_scaling.pdf         ARCG vs model size (DeepSeek R1 family only)
  fig7_rsc_heatmap.pdf           RSC heatmap: model x problem

Tables (LaTeX):
  tables.tex                     Main results table + domain/difficulty breakdown

Requirements
------------
  pip install sentence-transformers scipy matplotlib numpy

Usage
-----
  python 6_analyze_and_plot.py
  python 6_analyze_and_plot.py --no-embed   # skip sentence embedding (fast mode)
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RESULTS_FILE = os.path.join(DATA_DIR, "experiment_results.json")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Model display names and parameter counts (for scaling plot)
MODEL_META = {
    "deepseek-r1:7b":        {"label": "DS-R1-7B",    "params": 7,   "family": "DeepSeek R1"},
    "deepseek-r1:8b":        {"label": "DS-R1-8B",    "params": 8,   "family": "DeepSeek R1"},
    "deepseek-r1:14b":       {"label": "DS-R1-14B",   "params": 14,  "family": "DeepSeek R1"},
    "deepseek-r1:32b":       {"label": "DS-R1-32B",   "params": 32,  "family": "DeepSeek R1"},
    "qwen3:30b":             {"label": "Qwen3-30B",   "params": 30,  "family": "Qwen 3"},
    "qwen3:8b":              {"label": "Qwen3-8B",    "params": 8,   "family": "Qwen 3"},
    "qwen3:32b":             {"label": "Qwen3-32B",   "params": 32,  "family": "Qwen 3"},
    "magistral:24b":         {"label": "Magistral-24B","params": 24, "family": "Mistral"},
    "phi4-reasoning:14b":    {"label": "Phi4-R-14B",  "params": 14,  "family": "Phi-4"},
    "glm-4.7-flash":         {"label": "GLM-4.7-F",   "params": 30,  "family": "GLM"},
}

FAMILY_COLORS = {
    "DeepSeek R1": "#1f77b4",
    "Qwen 3":      "#ff7f0e",
    "Mistral":     "#2ca02c",
    "Phi-4":       "#d62728",
    "GLM":         "#9467bd",
}

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ---------------------------------------------------------------------------
# Sentence embedding (RSC)
# ---------------------------------------------------------------------------

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_rsc(texts: list[str]) -> float:
    """Mean pairwise cosine similarity of a list of texts."""
    if len(texts) < 2:
        return 1.0
    embedder = get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False)
    sims = [
        cosine_sim(embeddings[i], embeddings[j])
        for i, j in combinations(range(len(embeddings)), 2)
    ]
    return float(np.mean(sims))

# ---------------------------------------------------------------------------
# Answer-Reasoning Coherence (ARC)
# ---------------------------------------------------------------------------

def check_arc(raw_response: str, extracted_answer: str, domain: str) -> bool:
    """
    Heuristic: does the reasoning chain contain evidence that supports
    the extracted answer?
    Math:  the final number appears in the computation steps.
    Logic: the chosen letter appears in the reasoning.
    """
    if not extracted_answer or not raw_response:
        return False
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    text = text.lower()
    ans  = extracted_answer.lower().strip()
    if domain == "math":
        # Look for the number in a calculation context
        return bool(re.search(
            rf"(?:=\s*{re.escape(ans)}|{re.escape(ans)}\s*$|answer.*{re.escape(ans)})",
            text, re.MULTILINE
        ))
    else:
        return bool(re.search(
            rf"\b{re.escape(ans)}\b",
            text
        ))

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: dict, use_embeddings: bool = True) -> dict:
    """
    Returns nested dict:
      metrics[model][pid] = {
          fac, rsc, arcg, arc,
          accuracy_p0, domain, difficulty, n_variants
      }
    """
    metrics = {}

    for model, model_results in results.items():
        metrics[model] = {}
        print(f"  Computing metrics for {model}...")

        for pid, pr in model_results.items():
            variants = pr["variants"]
            domain   = pr["domain"]
            diff     = pr["difficulty"]
            gt       = pr["answer"]

            answers = [v["extracted_answer"] for v in variants.values()]
            raws    = [v["raw_response"]     for v in variants.values()]

            # Strip thinking traces for RSC
            def clean(text):
                return re.sub(r"<think>.*?</think>", "", text,
                              flags=re.DOTALL).strip()

            clean_raws = [clean(r) for r in raws]

            # FAC: fraction of variants that agree with the majority answer
            non_empty = [a for a in answers if a]
            if non_empty:
                from collections import Counter
                majority = Counter(non_empty).most_common(1)[0][0]
                fac = sum(1 for a in answers if a == majority) / len(answers)
            else:
                fac = 0.0

            # RSC
            if use_embeddings and any(clean_raws):
                rsc = compute_rsc([r for r in clean_raws if r])
            else:
                rsc = float("nan")

            # ARCG
            arcg = fac - rsc if not np.isnan(rsc) else float("nan")

            # ARC
            arc_scores = [
                check_arc(v["raw_response"], v["extracted_answer"], domain)
                for v in variants.values()
            ]
            arc = float(np.mean(arc_scores))

            # Accuracy on P0
            p0 = variants.get("P0", {})
            acc_p0 = float(p0.get("correct", False))

            metrics[model][pid] = {
                "fac":         round(fac, 4),
                "rsc":         round(rsc, 4) if not np.isnan(rsc) else None,
                "arcg":        round(arcg, 4) if not np.isnan(arcg) else None,
                "arc":         round(arc, 4),
                "accuracy_p0": acc_p0,
                "domain":      domain,
                "difficulty":  diff,
                "n_variants":  len(variants),
            }

    return metrics


def aggregate(metrics: dict) -> dict:
    """Per-model aggregate statistics."""
    agg = {}
    for model, model_metrics in metrics.items():
        vals = list(model_metrics.values())
        fac_list  = [v["fac"]  for v in vals]
        rsc_list  = [v["rsc"]  for v in vals if v["rsc"] is not None]
        arcg_list = [v["arcg"] for v in vals if v["arcg"] is not None]
        arc_list  = [v["arc"]  for v in vals]
        acc_list  = [v["accuracy_p0"] for v in vals]

        # Paired t-test: FAC vs RSC
        if len(fac_list) == len(rsc_list) and rsc_list:
            t_stat, p_val = stats.ttest_rel(fac_list[:len(rsc_list)], rsc_list)
        else:
            t_stat, p_val = float("nan"), float("nan")

        agg[model] = {
            "fac_mean":   round(float(np.mean(fac_list)), 4),
            "fac_std":    round(float(np.std(fac_list)),  4),
            "rsc_mean":   round(float(np.mean(rsc_list)), 4) if rsc_list else None,
            "rsc_std":    round(float(np.std(rsc_list)),  4) if rsc_list else None,
            "arcg_mean":  round(float(np.mean(arcg_list)),4) if arcg_list else None,
            "arcg_std":   round(float(np.std(arcg_list)), 4) if arcg_list else None,
            "arc_mean":   round(float(np.mean(arc_list)), 4),
            "accuracy":   round(float(np.mean(acc_list)), 4),
            "t_stat":     round(t_stat, 4) if not np.isnan(t_stat) else None,
            "p_value":    round(p_val,  6) if not np.isnan(p_val)  else None,
            "n_problems": len(vals),
        }
    return agg

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(metrics: dict, agg: dict) -> dict:
    tests = {}

    # Kruskal-Wallis across models on ARCG
    arcg_groups = [
        [v["arcg"] for v in m.values() if v["arcg"] is not None]
        for m in metrics.values()
    ]
    arcg_groups = [g for g in arcg_groups if g]
    if len(arcg_groups) >= 2:
        h, p = stats.kruskal(*arcg_groups)
        tests["kruskal_wallis_arcg"] = {"H": round(h, 4), "p": round(p, 6)}

    # Mann-Whitney U: math vs logic ARCG (pooled across models)
    math_arcg  = []
    logic_arcg = []
    for m in metrics.values():
        for v in m.values():
            if v["arcg"] is None:
                continue
            if v["domain"] == "math":
                math_arcg.append(v["arcg"])
            else:
                logic_arcg.append(v["arcg"])

    if math_arcg and logic_arcg:
        u, p = stats.mannwhitneyu(math_arcg, logic_arcg, alternative="two-sided")
        tests["mannwhitney_domain"] = {
            "U": round(u, 2), "p": round(p, 6),
            "math_mean":  round(float(np.mean(math_arcg)),  4),
            "logic_mean": round(float(np.mean(logic_arcg)), 4),
        }

    # One-way ANOVA: ARCG by difficulty
    diff_groups = defaultdict(list)
    for m in metrics.values():
        for v in m.values():
            if v["arcg"] is not None:
                diff_groups[v["difficulty"]].append(v["arcg"])

    if len(diff_groups) >= 2:
        groups = list(diff_groups.values())
        f, p   = stats.f_oneway(*groups)
        tests["anova_difficulty"] = {
            "F": round(f, 4), "p": round(p, 6),
            "means": {k: round(float(np.mean(v)), 4)
                      for k, v in diff_groups.items()},
        }

    # Pearson: FAC vs accuracy, RSC vs accuracy (pooled)
    fac_all, rsc_all, acc_all = [], [], []
    for m in metrics.values():
        for v in m.values():
            fac_all.append(v["fac"])
            acc_all.append(v["accuracy_p0"])
            if v["rsc"] is not None:
                rsc_all.append(v["rsc"])

    if len(fac_all) > 2:
        r, p = stats.pearsonr(fac_all, acc_all)
        tests["pearson_fac_accuracy"] = {"r": round(r, 4), "p": round(p, 6)}

    if len(rsc_all) > 2 and len(rsc_all) == len(acc_all[:len(rsc_all)]):
        r, p = stats.pearsonr(rsc_all, acc_all[:len(rsc_all)])
        tests["pearson_rsc_accuracy"] = {"r": round(r, 4), "p": round(p, 6)}

    return tests

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def model_label(model: str) -> str:
    return MODEL_META.get(model, {}).get("label", model)


def model_color(model: str) -> str:
    family = MODEL_META.get(model, {}).get("family", "Other")
    return FAMILY_COLORS.get(family, "#7f7f7f")


def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved {path}")

# ---------------------------------------------------------------------------
# Figure 1: FAC vs RSC grouped bar with ARCG annotation
# ---------------------------------------------------------------------------

def fig1_fac_rsc(agg: dict, models: list[str]):
    labels = [model_label(m) for m in models]
    fac    = [agg[m]["fac_mean"]  for m in models]
    rsc    = [agg[m]["rsc_mean"] or 0 for m in models]
    arcg   = [agg[m]["arcg_mean"] or 0 for m in models]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars_fac = ax.bar(x - width/2, fac, width, label="FAC",
                      color="#4878d0", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_rsc = ax.bar(x + width/2, rsc, width, label="RSC",
                      color="#ee854a", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Annotate ARCG above each pair
    for i, (f, r, a) in enumerate(zip(fac, rsc, arcg)):
        top = max(f, r) + 0.02
        color = "#2ca02c" if a >= 0 else "#d62728"
        ax.text(x[i], top, f"ARCG={a:+.3f}", ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.set_title("Final Answer Consistency (FAC) vs Reasoning Step Consistency (RSC) per Model")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    save_fig(fig, "fig1_fac_rsc_comparison.pdf")

# ---------------------------------------------------------------------------
# Figure 2: ARCG by domain
# ---------------------------------------------------------------------------

def fig2_arcg_domain(metrics: dict, models: list[str]):
    math_arcg  = defaultdict(list)
    logic_arcg = defaultdict(list)

    for model in models:
        for v in metrics[model].values():
            if v["arcg"] is None:
                continue
            if v["domain"] == "math":
                math_arcg[model].append(v["arcg"])
            else:
                logic_arcg[model].append(v["arcg"])

    labels = [model_label(m) for m in models]
    math_means  = [np.mean(math_arcg[m])  if math_arcg[m]  else 0 for m in models]
    logic_means = [np.mean(logic_arcg[m]) if logic_arcg[m] else 0 for m in models]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, math_means,  width, label="Math (GSM8K)",
           color="#4878d0", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, logic_means, width, label="Logic (ARC-Challenge)",
           color="#ee854a", alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ARCG (FAC - RSC)")
    ax.set_title("ARCG by Domain: Math vs Logical Reasoning")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig2_arcg_by_domain.pdf")

# ---------------------------------------------------------------------------
# Figure 3: ARCG by difficulty
# ---------------------------------------------------------------------------

def fig3_arcg_difficulty(metrics: dict, models: list[str]):
    difficulty_order = ["easy", "medium", "hard"]
    diff_arcg = {d: defaultdict(list) for d in difficulty_order}

    for model in models:
        for v in metrics[model].values():
            if v["arcg"] is None:
                continue
            d = v["difficulty"].lower()
            if d in diff_arcg:
                diff_arcg[d][model].append(v["arcg"])

    fig, ax = plt.subplots(figsize=(8, 4))

    for model in models:
        means = []
        for d in difficulty_order:
            vals = diff_arcg[d][model]
            means.append(np.mean(vals) if vals else float("nan"))
        ax.plot(difficulty_order, means,
                marker="o", label=model_label(model),
                color=model_color(model), linewidth=1.5, markersize=5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean ARCG")
    ax.set_title("ARCG Across Difficulty Levels")
    ax.legend(loc="upper right", framealpha=0.9,
              ncol=2, fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig3_arcg_by_difficulty.pdf")

# ---------------------------------------------------------------------------
# Figure 4: FAC vs accuracy scatter
# ---------------------------------------------------------------------------

def fig4_fac_accuracy(metrics: dict, models: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, metric_key, xlabel, title in [
        (axes[0], "fac",  "FAC",  "FAC vs Accuracy"),
        (axes[1], "rsc",  "RSC",  "RSC vs Accuracy"),
    ]:
        xs, ys, colors = [], [], []
        for model in models:
            for v in metrics[model].values():
                val = v.get(metric_key)
                if val is None:
                    continue
                xs.append(val)
                ys.append(v["accuracy_p0"])
                colors.append(model_color(model))

        ax.scatter(xs, ys, c=colors, alpha=0.5, s=18, edgecolors="none")

        # Regression line
        if len(xs) > 2:
            slope, intercept, r, p, _ = stats.linregress(xs, ys)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="black", linewidth=1.2,
                    label=f"r={r:.3f}, p={p:.3f}")
            ax.legend(fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (P0)")
        ax.set_title(title)

    # Legend for model families
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=fam)
        for fam, c in FAMILY_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(FAMILY_COLORS), bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9, fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig4_fac_accuracy_scatter.pdf")

# ---------------------------------------------------------------------------
# Figure 5: ARCG violin plot
# ---------------------------------------------------------------------------

def fig5_arcg_violin(metrics: dict, models: list[str]):
    data   = []
    labels = []
    colors = []

    for model in models:
        vals = [v["arcg"] for v in metrics[model].values()
                if v["arcg"] is not None]
        if vals:
            data.append(vals)
            labels.append(model_label(model))
            colors.append(model_color(model))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.7,
               label="ARCG = 0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ARCG (FAC - RSC)")
    ax.set_title("Distribution of ARCG Scores per Model")
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_fig(fig, "fig5_arcg_violin.pdf")

# ---------------------------------------------------------------------------
# Figure 6: ARCG vs model size (DeepSeek R1 family)
# ---------------------------------------------------------------------------

def fig6_scaling(agg: dict, models: list[str]):
    ds_models = [m for m in models
                 if MODEL_META.get(m, {}).get("family") == "DeepSeek R1"]
    if len(ds_models) < 2:
        print("  Skipping fig6 (fewer than 2 DeepSeek R1 models).")
        return

    sizes = [MODEL_META[m]["params"] for m in ds_models]
    arcgs = [agg[m]["arcg_mean"] or 0 for m in ds_models]
    facs  = [agg[m]["fac_mean"]  for m in ds_models]
    rscs  = [agg[m]["rsc_mean"] or 0 for m in ds_models]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, arcgs, marker="o", color="#1f77b4",
            linewidth=2, markersize=7, label="ARCG")
    ax.plot(sizes, facs,  marker="s", color="#4878d0",
            linewidth=1.5, markersize=6, linestyle="--", label="FAC")
    ax.plot(sizes, rscs,  marker="^", color="#ee854a",
            linewidth=1.5, markersize=6, linestyle="--", label="RSC")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Score")
    ax.set_title("ARCG, FAC, RSC vs Model Scale (DeepSeek R1 Family)")
    ax.legend(loc="best")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    save_fig(fig, "fig6_model_scaling.pdf")

# ---------------------------------------------------------------------------
# Figure 7: RSC heatmap
# ---------------------------------------------------------------------------

def fig7_rsc_heatmap(metrics: dict, models: list[str]):
    all_pids = sorted({pid for m in metrics.values() for pid in m})
    n_models = len(models)
    n_probs  = len(all_pids)

    matrix = np.full((n_models, n_probs), np.nan)
    for i, model in enumerate(models):
        for j, pid in enumerate(all_pids):
            v = metrics[model].get(pid)
            if v and v["rsc"] is not None:
                matrix[i, j] = v["rsc"]

    fig, ax = plt.subplots(figsize=(max(10, n_probs * 0.15), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=1.0, interpolation="nearest")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([model_label(m) for m in models], fontsize=8)
    ax.set_xlabel("Problem ID (sorted)")
    ax.set_title("RSC Heatmap: Reasoning Consistency per Model and Problem")

    plt.colorbar(im, ax=ax, label="RSC", fraction=0.02, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig7_rsc_heatmap.pdf")

# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def write_latex_tables(agg: dict, models: list[str],
                       metrics: dict, tests: dict):
    lines = []

    # Table 1: Main results
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-model ARCG metrics. FAC = Final Answer Consistency, "
        r"RSC = Reasoning Step Consistency, ARCG = FAC $-$ RSC, "
        r"ARC = Answer-Reasoning Coherence. "
        r"$p$-values from paired $t$-test (FAC vs RSC).}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & FAC & RSC & ARCG & ARC & Accuracy & $p$-value \\",
        r"\midrule",
    ]

    for model in models:
        a = agg[model]
        label = model_label(model)
        fac   = f"{a['fac_mean']:.3f}"
        rsc   = f"{a['rsc_mean']:.3f}" if a["rsc_mean"] is not None else "--"
        arcg  = f"{a['arcg_mean']:+.3f}" if a["arcg_mean"] is not None else "--"
        arc   = f"{a['arc_mean']:.3f}"
        acc   = f"{a['accuracy']:.3f}"
        pval  = f"{a['p_value']:.4f}" if a["p_value"] is not None else "--"
        lines.append(
            rf"{label} & {fac} & {rsc} & {arcg} & {arc} & {acc} & {pval} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # Table 2: Domain breakdown
    math_arcg_by_model  = {}
    logic_arcg_by_model = {}
    for model in models:
        m_vals = [v["arcg"] for v in metrics[model].values()
                  if v["arcg"] is not None and v["domain"] == "math"]
        l_vals = [v["arcg"] for v in metrics[model].values()
                  if v["arcg"] is not None and v["domain"] == "logic"]
        math_arcg_by_model[model]  = np.mean(m_vals) if m_vals else float("nan")
        logic_arcg_by_model[model] = np.mean(l_vals) if l_vals else float("nan")

    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean ARCG by domain and difficulty. "
        r"Negative values indicate execution fragility "
        r"(reasoning more consistent than answers).}",
        r"\label{tab:domain_difficulty}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & ARCG (Math) & ARCG (Logic) \\",
        r"\midrule",
    ]

    for model in models:
        label = model_label(model)
        m = math_arcg_by_model[model]
        l = logic_arcg_by_model[model]
        ms = f"{m:+.3f}" if not np.isnan(m) else "--"
        ls = f"{l:+.3f}" if not np.isnan(l) else "--"
        lines.append(rf"{label} & {ms} & {ls} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = os.path.join(FIGURES_DIR, "tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")

# ---------------------------------------------------------------------------
# Save metrics JSON
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, agg: dict, tests: dict):
    out = {
        "per_problem": {
            model: {
                pid: v for pid, v in m.items()
            }
            for model, m in metrics.items()
        },
        "aggregate":   agg,
        "statistical_tests": tests,
    }
    path = os.path.join(DATA_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARCG analysis and plotting")
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Skip sentence embedding (RSC will be NaN). Useful for quick testing."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ARCG Step 6: Analyze Results and Generate Figures")
    print("=" * 60)

    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run 5_run_experiment.py first.")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    models = list(results.keys())
    print(f"Models found: {models}")

    print("\nComputing metrics...")
    metrics = compute_metrics(results, use_embeddings=not args.no_embed)

    print("\nAggregating...")
    agg = aggregate(metrics)

    print("\nRunning statistical tests...")
    tests = run_statistical_tests(metrics, agg)

    print("\nSaving metrics JSON...")
    save_metrics(metrics, agg, tests)

    print("\nGenerating figures...")
    fig1_fac_rsc(agg, models)
    fig2_arcg_domain(metrics, models)
    fig3_arcg_difficulty(metrics, models)
    fig4_fac_accuracy(metrics, models)
    fig5_arcg_violin(metrics, models)
    fig6_scaling(agg, models)
    fig7_rsc_heatmap(metrics, models)

    print("\nGenerating LaTeX tables...")
    write_latex_tables(agg, models, metrics, tests)

    # Print summary to console
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Model':<20} {'FAC':>6} {'RSC':>6} {'ARCG':>7} {'ARC':>6} {'Acc':>6} {'p':>8}"
    print(header)
    print("-" * len(header))
    for model in models:
        a = agg[model]
        print(
            f"{model_label(model):<20} "
            f"{a['fac_mean']:>6.3f} "
            f"{(a['rsc_mean'] or 0):>6.3f} "
            f"{(a['arcg_mean'] or 0):>+7.3f} "
            f"{a['arc_mean']:>6.3f} "
            f"{a['accuracy']:>6.3f} "
            f"{(a['p_value'] or 0):>8.4f}"
        )

    print("\nStatistical tests:")
    for name, result in tests.items():
        print(f"  {name}: {result}")

    print(f"\nAll outputs saved to {FIGURES_DIR}/")
    print("Next step: compile the LaTeX paper in paper/")


if __name__ == "__main__":
    main()
