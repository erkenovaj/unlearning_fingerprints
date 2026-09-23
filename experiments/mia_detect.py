"""MIA as a cross-model unlearning detection statistic.

For each model, the per-model MIA script already computes the within-model
forget-vs-retain ROC AUC for loss, Min-K++, log-rank, and entropy.  Here we
treat those per-model MIA scores as a *single-model* detection feature: can a
statistic computed from one model separate the unlearned models from the
reference models (original + retain) across a family?

Usage:
  python mia_detect.py --mia_dir results/strong_models/qwen_1_5b/mia/ \
                       --output_dir runs/strong_models/qwen_1_5b/mia_detect/
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations
from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, REGISTRIES

PERM_MAX_ENUM = 100000
PERM_N_RANDOM = 100000
PATHOLOGY_RATIO = 10.0
RNG_SEED = 42

METRIC_LABELS = {
    "loss": "CE Loss",
    "min_k_plus": "Min-K++",
    "logrank": "Log-Rank",
    "entropy": "Entropy",
}

UNLEARNED = set()


def aggregate(mia_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(mia_dir, "*_mia.json"))):
        data = load_json(path)
        tag = data["model_tag"]
        if tag in results:
            raise ValueError(f"Duplicate MIA tag: {tag}")
        results[tag] = {
            "delta": {},
            "auc": {},
        }
        for key in METRIC_LABELS:
            m = data["metrics"].get(key, {})
            if not m or m.get("pathology"):
                results[tag]["delta"][key] = float("nan")
                results[tag]["auc"][key] = float("nan")
                continue
            results[tag]["delta"][key] = m["delta_mean"]
            results[tag]["auc"][key] = m.get("roc_auc", float("nan"))
    return results


def find_pathologies(results, mode, key):
    vals = {t: results[t][mode][key] for t in results}
    clean = {t: v for t, v in vals.items() if np.isfinite(v)}
    if not clean:
        return set(vals.keys()), {}
    median_abs = float(np.median([abs(v) for v in clean.values()]))
    flagged = {t for t, v in clean.items() if median_abs > 0 and abs(v) > PATHOLOGY_RATIO * median_abs}
    flagged |= {t for t, v in vals.items() if not np.isfinite(v)}
    return flagged, clean


def detect_auc(values, tags):
    """Raw ROC AUC with unlearned=1, reference=0.

    AUC below 0.5 means the sign is inverted relative to the unlearning
    convention; the direction is content and must be reported, not normalized
    away. magnitude = max(raw, 1-raw) is kept only as a derived column.
    """
    y_true = np.array([1 if t in UNLEARNED else 0 for t in tags])
    y_score = np.array([values[t] for t in tags])
    if len(np.unique(y_true)) < 2:
        return None, None
    raw = roc_auc_score(y_true, y_score)
    return raw, max(raw, 1.0 - raw)


def permutation_p_two_sided(values, tags):
    """Exact (or Monte-Carlo) two-sided permutation test on |AUC - 0.5|.

    Null: model labels are exchangeable. The p-value covers BOTH directions,
    so an inverted separation is detected without normalizing it away.
    """
    y_true = np.array([1 if t in UNLEARNED else 0 for t in tags])
    y_score = np.array([values[t] for t in tags])
    if len(np.unique(y_true)) < 2:
        return None
    n = len(y_true)
    n_neg = int((y_true == 0).sum())
    obs_stat = abs(roc_auc_score(y_true, y_score) - 0.5)
    total = comb(n, n_neg)
    rng = np.random.default_rng(RNG_SEED)
    if total <= PERM_MAX_ENUM:
        extreme = 0
        for neg_idx in combinations(range(n), n_neg):
            labels = np.ones(n, dtype=int)
            for i in neg_idx:
                labels[i] = 0
            auc = roc_auc_score(labels, y_score)
            if abs(auc - 0.5) >= obs_stat - 1e-12:
                extreme += 1
        return extreme / total
    extreme = 0
    for _ in range(PERM_N_RANDOM):
        labels = rng.permutation(y_true)
        auc = roc_auc_score(labels, y_score)
        if abs(auc - 0.5) >= obs_stat - 1e-12:
            extreme += 1
    return (extreme + 1) / (PERM_N_RANDOM + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mia_dir", required=True)
    p.add_argument("--output_dir", default="runs/mia_detect")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--registry", default="llama", choices=list(REGISTRIES.keys()),
                   help="Registry whose unlearned tags define the positive class")
    args = p.parse_args()

    global UNLEARNED
    UNLEARNED = set(REGISTRIES[args.registry][1])

    output_dir = args.output_dir
    results = aggregate(args.mia_dir)
    allowed = UNLEARNED | set(REGISTRIES[args.registry][2])
    results = {t: r for t, r in results.items() if t in allowed}
    if not results:
        p.error("No MIA results found for the selected registry split")
    os.makedirs(output_dir, exist_ok=True)
    tags = sorted(results.keys())

    print(f"[mia_detect] {len(results)} models: {tags}")

    summary = {}
    pathology = {}
    for mode in ["delta", "auc"]:
        summary[mode] = {}
        pathology[mode] = {}
        for key in METRIC_LABELS:
            flagged, clean = find_pathologies(results, mode, key)
            usable = [t for t in tags if t not in flagged]
            vals = {t: results[t][mode][key] for t in usable}
            raw, sep = detect_auc(vals, usable)
            p = permutation_p_two_sided(vals, usable) if raw is not None else None
            summary[mode][key] = {
                "raw": raw,
                "separation": sep,
                "permutation_p": p,
                "n_models": len(usable),
                "n_excluded": len(flagged),
            }
            pathology[mode][key] = sorted(flagged)

    out = {
        "registry": args.registry,
        "seed": RNG_SEED,
        "models": tags,
        "per_model": results,
        "pathology_excluded": pathology,
        "cross_model_auc": summary,
    }
    json_path = os.path.join(output_dir, "mia_detect.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mia_detect] saved -> {json_path}")

    print("\nCross-model detection (unlearned/reference) - raw AUC, two-sided permutation p:")
    print(f"{'metric':12s} {'feature':>10s} {'raw AUC':>10s} {'perm p':>10s} {'n':>4s} {'excl':>5s}")
    print("-" * 50)
    for mode, label in [("delta", "d(f-r)"), ("auc", "within AUC")]:
        for key in METRIC_LABELS:
            s = summary[mode][key]
            if s["raw"] is None:
                print(f"{METRIC_LABELS[key]:12s} {label:>10s}          n/a         n/a")
                continue
            p_str = f"{s['permutation_p']:10.4f}" if s["permutation_p"] is not None else "       n/a"
            print(f"{METRIC_LABELS[key]:12s} {label:>10s} {s['raw']:>10.3f} {p_str} {s['n_models']:>4d} {s['n_excluded']:>5d}")
    exclusions = {f"{m}.{k}": v for m in pathology for k, v in pathology[m].items() if v}
    if exclusions:
        print(f"[mia_detect] pathology exclusions: {exclusions}")

    if args.no_plots:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, key in enumerate(METRIC_LABELS):
        ax = axes[idx]
        pts = [(t, results[t]["delta"][key], results[t]["auc"][key]) for t in tags
               if np.isfinite(results[t]["delta"][key]) and np.isfinite(results[t]["auc"][key])]
        feature_delta = [d for _, d, _ in pts]
        feature_auc = [a for _, _, a in pts]
        colors = ["crimson" if t in UNLEARNED else "steelblue" for t, _, _ in pts]

        ax.scatter(feature_delta, feature_auc, c=colors, s=120, alpha=0.8, edgecolor="black")
        for t, d, a in pts:
            ax.text(d, a, t, fontsize=8, ha="center", va="bottom")

        ax.set_xlabel("Δ (forget - retain)", fontsize=10)
        ax.set_ylabel("Within-model ROC AUC", fontsize=10)
        ax.set_title(METRIC_LABELS[key], fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0.0, color="gray", linewidth=0.5, linestyle="--")
        if feature_delta:
            lo, hi = min(feature_delta), max(feature_delta)
            margin = max(0.05 * (hi - lo), 0.05 * max(abs(lo), abs(hi)), 1e-6)
            ax.set_xlim(lo - margin, hi + margin)
        else:
            ax.text(0.5, 0.6, "No valid metrics", ha="center", transform=ax.transAxes)
        ax.set_ylim(0, 1)

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
