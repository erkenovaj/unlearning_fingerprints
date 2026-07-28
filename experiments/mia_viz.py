"""MIA summary visualization across multiple models.

Reads per-model *_mia.json files and produces:
  1. Heatmap of delta_mean (forget - retain) for each metric × model
  2. Grouped bar chart: forget vs retain loss per model
  3. ROC curves overlaid for all models (loss metric)

Usage:
  python mia_viz.py --mia_dir results/strong_models/qwen_1_5b/mia/ --output_dir results/strong_models/qwen_1_5b/mia/
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json

METRIC_KEYS = ["loss", "min_k_plus", "logrank", "entropy"]
METRIC_LABELS = {
    "loss": "Negative CE Loss",
    "min_k_plus": "Min-K++ (MINT)",
    "logrank": "Log-Rank",
    "entropy": "Entropy",
}
UNLEARNED = {"rmu", "npo", "graddiff", "altpo", "undial"}


def load_all_mia(mia_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(mia_dir, "*_mia.json"))):
        data = load_json(path)
        tag = data["model_tag"]
        results[tag] = data
    return results


def plot_heatmap(all_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(all_results.keys())
    metrics = METRIC_KEYS

    delta_matrix = []
    for tag in tags:
        row = []
        for m in metrics:
            delta_matrix.append(all_results[tag]["metrics"][m]["delta_mean"])
        delta_matrix.append([delta_matrix.pop() for _ in range(len(metrics))])

    matrix = np.array([[all_results[tag]["metrics"][m]["delta_mean"] for m in metrics] for tag in tags])

    fig, ax = plt.subplots(figsize=(10, max(4, len(tags) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=10)

    for i in range(len(tags)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            color = "white" if abs(val) > abs(matrix).max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ (forget − retain)", fontsize=10)

    unlearned_count = sum(1 for t in tags if t in UNLEARNED)
    if unlearned_count > 0 and unlearned_count < len(tags):
        ax.axhline(y=unlearned_count - 0.5, color="black", linewidth=1.5, linestyle="--")

    ax.set_title("MIA Delta Heatmap (forget − retain)\nRed = forget scores higher | Blue = retain scores higher", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap -> {output_path}")


def plot_forget_vs_retain(all_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(all_results.keys())
    x = np.arange(len(tags))
    width = 0.35

    forget_vals = [all_results[tag]["metrics"]["loss"]["forget_mean"] for tag in tags]
    retain_vals = [all_results[tag]["metrics"]["loss"]["retain_mean"] for tag in tags]

    colors_f = ["crimson" if t in UNLEARNED else "salmon" for t in tags]
    colors_r = ["steelblue" if t in UNLEARNED else "lightsteelblue" for t in tags]

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 1.2), 6))
    bars_f = ax.bar(x - width / 2, forget_vals, width, label="Forget", color=colors_f, alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_r = ax.bar(x + width / 2, retain_vals, width, label="Retain", color=colors_r, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Loss (negative CE)", fontsize=11)
    ax.set_title("Forget vs Retain Loss per Model\n(Higher loss = less predicted = more forgotten)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="unlearned"),
        Patch(facecolor="steelblue", alpha=0.7, label="reference"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles[:2], labels=["unlearned", "reference"] + labels[:2], fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  forget_vs_retain -> {output_path}")


def plot_roc_overlay(all_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_results))))

    for i, (tag, data) in enumerate(sorted(all_results.items())):
        f = np.array(data["metrics"]["loss"]["per_prompt_forget"])
        r = np.array(data["metrics"]["loss"]["per_prompt_retain"])
        y_true = np.array([1] * len(f) + [0] * len(r))
        y_score = np.concatenate([f, r])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        label = f'{tag} (AUC={roc_auc:.3f})'
        style = "--" if tag not in UNLEARNED else "-"
        lw = 1.5 if tag not in UNLEARNED else 2.5
        ax.plot(fpr, tpr, label=label, linewidth=lw, linestyle=style, color=colors[i % len(colors)])

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate (retain predicted as forget)", fontsize=12)
    ax.set_ylabel("True Positive Rate (forget detected)", fontsize=12)
    ax.set_title("MIA ROC — Loss Metric\nForget vs Retain (within-model)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  roc_overlay -> {output_path}")


def plot_delta_bars(all_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(all_results.keys())
    metrics = METRIC_KEYS

    x = np.arange(len(tags))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 1.2), 6))

    cmap = plt.cm.Set2
    for j, m in enumerate(metrics):
        deltas = [all_results[tag]["metrics"][m]["delta_mean"] for tag in tags]
        ax.bar(x + (j - n_metrics / 2 + 0.5) * width, deltas, width, label=METRIC_LABELS[m], color=cmap(j / n_metrics), alpha=0.85, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Δ (forget − retain)", fontsize=11)
    ax.set_title("MIA Metric Deltas per Model\n(Positive = forget scores higher)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)

    unlearned_count = sum(1 for t in tags if t in UNLEARNED)
    if unlearned_count > 0 and unlearned_count < len(tags):
        ax.axvline(x=unlearned_count - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  delta_bars -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="MIA summary visualization.")
    p.add_argument("--mia_dir", required=True, help="Directory with *_mia.json files")
    p.add_argument("--output_dir", default=None, help="Output directory (default: --mia_dir)")
    args = p.parse_args()

    output_dir = args.output_dir or args.mia_dir
    os.makedirs(output_dir, exist_ok=True)

    all_results = load_all_mia(args.mia_dir)
    if not all_results:
        print(f"No MIA results found in {args.mia_dir}")
        return

    print(f"[mia_viz] loaded {len(all_results)} models: {sorted(all_results.keys())}")

    plot_heatmap(all_results, os.path.join(output_dir, "mia_heatmap.png"))
    plot_forget_vs_retain(all_results, os.path.join(output_dir, "mia_forget_vs_retain.png"))
    plot_roc_overlay(all_results, os.path.join(output_dir, "mia_roc_overlay.png"))
    plot_delta_bars(all_results, os.path.join(output_dir, "mia_delta_bars.png"))

    print(f"\n[mia_viz] done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
