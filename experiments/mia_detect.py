"""MIA as a cross-model unlearning detection statistic.

For each model, the per-model MIA script already computes the within-model
forget-vs-retain ROC AUC for loss, Min-K++, log-rank, and entropy.  Here we
treat those per-model MIA scores as a *single-model* detection feature: can a
statistic computed from one model separate the unlearned models from the
reference models (original + retain) across a family?

Usage:
  python mia_detect.py --mia_dir results/strong_models/qwen_1_5b/mia/ \
                       --output_dir results/strong_models/qwen_1_5b/mia_detect/
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json

METRIC_LABELS = {
    "loss": "Neg CE Loss",
    "min_k_plus": "Min-K++",
    "logrank": "Log-Rank",
    "entropy": "Entropy",
}

UNLEARNED = {"rmu", "npo", "graddiff", "altpo", "undial"}


def aggregate(mia_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(mia_dir, "*_mia.json"))):
        data = load_json(path)
        tag = data["model_tag"]
        results[tag] = {
            "delta": {},
            "auc": {},
        }
        for key in METRIC_LABELS:
            results[tag]["delta"][key] = data["metrics"][key]["delta_mean"]
            results[tag]["auc"][key] = data["metrics"][key]["roc_auc"]
    return results


def detect_auc(values, tags):
    """Compute ROC AUC where unlearned=1, reference=0."""
    y_true = np.array([1 if t in UNLEARNED else 0 for t in tags])
    y_score = np.array([values[t] for t in tags])
    # AUC below 0.5 means the sign is inverted; report max(auc, 1-auc)
    # as the separation, and the raw auc to show direction.
    raw = roc_auc_score(y_true, y_score)
    return raw, max(raw, 1.0 - raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mia_dir", required=True)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    output_dir = args.output_dir or args.mia_dir
    os.makedirs(output_dir, exist_ok=True)

    results = aggregate(args.mia_dir)
    tags = sorted(results.keys())

    print(f"[mia_detect] {len(results)} models: {tags}")

    summary = {}
    for mode in ["delta", "auc"]:
        summary[mode] = {}
        for key in METRIC_LABELS:
            vals = {t: results[t][mode][key] for t in tags}
            raw, sep = detect_auc(vals, tags)
            summary[mode][key] = {"raw": float(raw), "separation": float(sep)}

    # Save JSON
    out = {
        "models": tags,
        "per_model": results,
        "cross_model_auc": summary,
    }
    json_path = os.path.join(output_dir, "mia_detect.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mia_detect] saved -> {json_path}")

    # Print table
    print("\nCross-model detection AUC (unlearned vs reference):")
    print(f"{'metric':12s} {'feature':>10s} {'raw AUC':>10s} {'separation':>12s}")
    print("-" * 50)
    for mode, label in [("delta", "Δ(f-r)"), ("auc", "within AUC")]:
        for key in METRIC_LABELS:
            s = summary[mode][key]
            print(f"{METRIC_LABELS[key]:12s} {label:>10s} {s['raw']:>10.3f} {s['separation']:>12.3f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, key in enumerate(METRIC_LABELS):
        ax = axes[idx]
        feature_delta = [results[t]["delta"][key] for t in tags]
        feature_auc = [results[t]["auc"][key] for t in tags]
        colors = ["crimson" if t in UNLEARNED else "steelblue" for t in tags]

        ax.scatter(feature_delta, feature_auc, c=colors, s=120, alpha=0.8, edgecolor="black")
        for t in tags:
            ax.text(results[t]["delta"][key], results[t]["auc"][key], t, fontsize=8, ha="center", va="bottom")

        ax.set_xlabel("Δ (forget - retain)", fontsize=10)
        ax.set_ylabel("Within-model ROC AUC", fontsize=10)
        ax.set_title(METRIC_LABELS[key], fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0.0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlim(min(feature_delta) - 0.05 * abs(min(feature_delta)), max(feature_delta) + 0.05 * abs(max(feature_delta)))
        ax.set_ylim(0.35, 0.75)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="unlearned"),
        Patch(facecolor="steelblue", alpha=0.7, label="reference"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "mia_detect_scatter.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mia_detect] plot -> {plot_path}")


if __name__ == "__main__":
    main()
