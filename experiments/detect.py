"""Compare within-model spectral and weight features across checkpoint labels.

Features describe differences between probe sets, not proof of knowledge removal.
The combined score uses population-dependent min-max normalization and requires
aligned weight results. Use --spectral_only to omit weight and combined scores.
ROC AUC uses registry positives versus references without flipping its direction.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, save_json, REGISTRIES

SCORE_KEYS = ["spectral_score", "localized_dip", "localized_kink",
              "shape_anomaly", "cosine_anomaly", "cosine_score",
              "weight_score", "changepoint_score", "combined"]


def load_all_results(spectral_dir, weight_dir):
    """Load all spectral and weight JSON results."""
    spectral = {}
    for path in sorted(glob.glob(os.path.join(spectral_dir, "*_spectral.json"))):
        data = load_json(path)
        if data["model_tag"] in spectral:
            raise ValueError(f"Duplicate spectral tag: {data['model_tag']}")
        spectral[data["model_tag"]] = data

    weights = {}
    for path in sorted(glob.glob(os.path.join(weight_dir, "*_weights.json"))) if weight_dir else []:
        data = load_json(path)
        if data["model_tag"] in weights:
            raise ValueError(f"Duplicate weight tag: {data['model_tag']}")
        weights[data["model_tag"]] = data

    return spectral, weights


def _smooth_curve(vals, w=5):
    n = len(vals)
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[i] = np.mean(vals[a:b])
    return out


def compute_scores(spectral, weights):
    """Compute features over interior hidden states (1..L-3).

    An empty weights dict explicitly returns spectral features only. Nonempty
    weights must match all spectral tags and use one anomaly statistic.
    """
    if not spectral:
        raise ValueError("No spectral results supplied")
    if weights:
        if set(weights) != set(spectral):
            raise ValueError("Spectral and weight tags must match exactly")
        stat_keys = {w["anomaly"].get("stat_key", "mean_stable_rank") for w in weights.values()}
        if len(stat_keys) != 1:
            raise ValueError(f"Mixed weight anomaly stat_key values: {sorted(stat_keys)}")
        for tag, wdata in weights.items():
            for key in ("model_path", "model_fingerprint"):
                if wdata.get(key) and spectral[tag].get(key) and wdata[key] != spectral[tag][key]:
                    raise ValueError(f"{tag}: spectral/weight {key} mismatch")
            anomaly = wdata["anomaly"]
            if not anomaly["z_scores"] or not np.isfinite(anomaly["z_scores"]).all() or not np.isfinite(anomaly["changepoint_score"]):
                raise ValueError(f"{tag}: empty or non-finite weight anomaly scores")
    results = {}

    for tag, sdata in spectral.items():
        delta = sdata["delta"]
        n_layers = sdata["n_layers"]
        layer_min = 1
        layer_max = n_layers - 3

        interior = [d for d in delta if layer_min <= d["layer"] <= layer_max]
        if len(interior) < 5:
            interior = [d for d in delta if 1 <= d["layer"] <= n_layers - 1]

        erank_interior = np.array([d["delta_erank"] for d in interior])
        cos_interior = np.array([d["delta_cosine"] for d in interior])
        interior_layers = [d["layer"] for d in interior]
        if not len(interior) or not np.isfinite(erank_interior).all() or not np.isfinite(cos_interior).all():
            raise ValueError(f"{tag}: empty or non-finite interior spectral data")

        spectral_score = float(-erank_interior.min())
        cosine_score = float(cos_interior.max())
        min_erank_layer = int(interior_layers[int(np.argmin(erank_interior))])

        delta_by_layer = {d["layer"]: d["delta_erank"] for d in delta}
        windows = []
        for a in range(layer_min, layer_max - 1):
            window_layers = list(range(a, min(a + 3, layer_max + 1)))
            if not all(l in delta_by_layer for l in window_layers):
                raise ValueError(f"{tag}: missing layers in spectral data")
            window_vals = [delta_by_layer[l] for l in window_layers]
            windows.append((a, float(np.mean(window_vals))))
        if not windows:
            raise ValueError(f"{tag}: insufficient layers for localized dip")
        localized_dip_layer, localized_dip_val = min(windows, key=lambda x: x[1])
        localized_dip = float(-localized_dip_val)

        n_int = len(erank_interior)
        best_kink_score = 0.0
        best_kink_range = [interior_layers[0], interior_layers[0]]
        for a in range(n_int):
            for b in range(a + 1, n_int + 1):
                inside = erank_interior[a:b]
                outside = np.concatenate([erank_interior[:a], erank_interior[b:]])
                if len(outside) < 2:
                    continue
                diff = abs(inside.mean() - outside.mean())
                pooled = np.sqrt(
                    (inside.var() * len(inside) + outside.var() * len(outside)) / n_int + 1e-12
                )
                score = diff / pooled
                if score > best_kink_score:
                    best_kink_score = score
                    best_kink_range = [interior_layers[a], interior_layers[b - 1]]

        erank_resid = erank_interior - _smooth_curve(erank_interior, w=5)
        cos_resid = cos_interior - _smooth_curve(cos_interior, w=5)
        shape_anomaly = float(erank_resid.std())
        cosine_anomaly = float(cos_resid.std())

        results[tag] = {
            "spectral_score": spectral_score,
            "localized_dip": localized_dip,
            "localized_dip_layer": localized_dip_layer,
            "localized_kink": float(best_kink_score),
            "localized_kink_range": best_kink_range,
            "shape_anomaly": shape_anomaly,
            "cosine_anomaly": cosine_anomaly,
            "cosine_score": cosine_score,
            "min_erank_layer": min_erank_layer,
        }
        if weights:
            anomaly = weights[tag]["anomaly"]
            results[tag].update({
                "weight_score": float(max(abs(z) for z in anomaly["z_scores"])),
                "changepoint_score": float(anomaly["changepoint_score"]),
                "changepoint_range": anomaly["changepoint_range"],
                "weight_stat_key": anomaly.get("stat_key", "mean_stable_rank"),
            })

    return results


def normalize_scores(scores, key):
    """Min-max normalize a score across all models."""
    vals = [s[key] for s in scores.values()]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng < 1e-12:
        return {tag: 0.5 for tag in scores}
    return {tag: (s[key] - lo) / rng for tag, s in scores.items()}


def compute_combined(scores):
    """Compute the combined detection score per model and store it in-place
    under scores[tag]['combined']. Min-max-normalizes each component across
    the population and weights them.

    These fixed exploratory weights are not a calibrated probability of
    unlearning and normalization depends on the selected checkpoint population.
    """
    if not scores or any("weight_score" not in s or "weight_stat_key" not in s for s in scores.values()):
        raise ValueError("Combined scores require aligned weight results for every model")
    if len({s["weight_stat_key"] for s in scores.values()}) != 1:
        raise ValueError("Combined scores require one weight anomaly stat_key")
    norm_dip = normalize_scores(scores, "localized_dip")
    norm_kink = normalize_scores(scores, "localized_kink")
    norm_shape = normalize_scores(scores, "shape_anomaly")
    norm_cos_shape = normalize_scores(scores, "cosine_anomaly")
    norm_cosine = normalize_scores(scores, "cosine_score")
    norm_weight = normalize_scores(scores, "weight_score")
    for tag in scores:
        scores[tag]["combined"] = (
            0.65 * norm_cos_shape[tag]
            + 0.15 * norm_shape[tag]
            + 0.10 * norm_kink[tag]
            + 0.05 * norm_dip[tag]
            + 0.025 * norm_cosine[tag]
            + 0.025 * norm_weight[tag]
        )


def compute_significance(spectral):
    """Compare independent probe sets using layer-averaged prompt cosines.

    Welch's t and Mann-Whitney U are two-sided, exploratory tests. Cosines
    share a within-set centroid, so prompts are not fully independent.
    """
    results = {}
    for tag, sdata in spectral.items():
        n_layers = sdata["n_layers"]
        layer_min = 1
        layer_max = n_layers - 3

        pc_f = sdata.get("per_prompt_cosine_forget")
        pc_r = sdata.get("per_prompt_cosine_retain")
        if pc_f is None or pc_r is None:
            continue

        stop = min(layer_max + 1, len(pc_f), len(pc_r))
        if stop <= layer_min:
            continue
        f = np.asarray(pc_f[layer_min:stop], dtype=float).mean(axis=0)
        r = np.asarray(pc_r[layer_min:stop], dtype=float).mean(axis=0)
        if len(f) < 2 or len(r) < 2 or not np.isfinite(f).all() or not np.isfinite(r).all():
            raise ValueError(f"{tag}: significance requires two finite samples per set")
        mean_delta = float(f.mean() - r.mean())
        pooled_std = float(np.sqrt(((len(f) - 1) * f.var(ddof=1) + (len(r) - 1) * r.var(ddof=1)) / (len(f) + len(r) - 2)))
        t_stat, t_pval = stats.ttest_ind(f, r, equal_var=False, alternative="two-sided")
        u_stat, u_pval = stats.mannwhitneyu(f, r, alternative="two-sided")
        cohens_d = mean_delta / pooled_std if pooled_std > 0 else float("nan")

        results[tag] = {
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "u_stat": float(u_stat),
            "u_pvalue": float(u_pval),
            "mean_delta": mean_delta,
            "pooled_std": pooled_std,
            "standard_error": float(np.sqrt(f.var(ddof=1) / len(f) + r.var(ddof=1) / len(r))),
            "effect_size_cohens_d": float(cohens_d),
            "test": "welch_independent",
            "n_forget": len(f),
            "n_retain": len(r),
        }

    return results


def plot_delta_erank_overlay(spectral, output_path):
    """Overlay effective-rank differences per layer for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_unlearned = plt.cm.Set1(np.linspace(0, 1, 10))
    color_idx = 0

    for tag, data in spectral.items():
        layers = [d["layer"] for d in data["delta"]]
        delta_erank = [d["delta_erank"] for d in data["delta"]]

        if tag in REFERENCE_MODELS:
            color = "black"
            linewidth = 2.5
            linestyle = "--"
        else:
            color = colors_unlearned[color_idx % len(colors_unlearned)]
            color_idx += 1
            linewidth = 1.5
            linestyle = "-"

        ax.plot(layers, delta_erank, "o-", label=tag, color=color,
                linewidth=linewidth, linestyle=linestyle, markersize=4)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Δ Effective Rank (forget − retain)", fontsize=12)
    ax.set_title("Per-Layer Effective-Rank Differences", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overlay plot -> {output_path}")


def plot_roc(scores, output_path):
    """ROC curve: unlearned (positive) vs retain+original (negative)."""
    from sklearn.metrics import roc_curve, auc

    tags = list(scores.keys())
    y_true = np.array([1 if t in UNLEARNED_METHODS else 0 for t in tags])

    if len(np.unique(y_true)) < 2:
        print("  ROC skipped: no reference models in this registry (unlearned-only)")
        return

    score_keys = [k for k in SCORE_KEYS if all(k in s for s in scores.values())]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    for key in score_keys:
        y_score = np.array([scores[t][key] for t in tags])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{key} (AUC={roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Unlearning Detection ROC:\nUnlearned vs Reference Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot -> {output_path}")


def plot_score_bars(scores, output_path):
    """Per-method detection score bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in scores if t not in ("combined",)]
    tags_sorted = sorted(tags, key=lambda t: scores[t]["localized_dip"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tags_sorted))
    width = 0.2

    spectral_vals = [scores[t]["localized_dip"] for t in tags_sorted]
    cosine_vals = [scores[t]["cosine_score"] for t in tags_sorted]
    colors = ["crimson" if t in UNLEARNED_METHODS else "steelblue" for t in tags_sorted]

    ax.bar(x - 1.5 * width, spectral_vals, width, label="Localized dip (3-layer)", color=colors, alpha=0.8)
    ax.bar(x - 0.5 * width, cosine_vals, width, label="Cosine (max Δcos)", color=colors, alpha=0.5)
    if all("combined" in s for s in scores.values()):
        ax.bar(x + 0.5 * width, [scores[t]["weight_score"] for t in tags_sorted], width, label="Weight (max |z|)", color=colors, alpha=0.3)
        ax.bar(x + 1.5 * width, [scores[t]["combined"] for t in tags_sorted], width, label="Combined", color=colors, alpha=0.9, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(tags_sorted, rotation=45, ha="right")
    ax.set_ylabel("Detection Score")
    ax.set_title("Per-Method Detection Scores", fontsize=14, fontweight="bold")
    ax.legend()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="unlearned"),
        Patch(facecolor="steelblue", alpha=0.7, label="reference"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend().get_patches()[:4], fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  score bars -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Combined unlearning detection.")
    p.add_argument("--spectral_dir", default="runs/spectral")
    p.add_argument("--weight_dir", default="runs/weights")
    p.add_argument("--output_dir", default="runs/detection")
    p.add_argument("--spectral_only", action="store_true", help="Omit weight and combined scores")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--registry", default="llama", choices=list(REGISTRIES.keys()),
                   help="Model registry to use")
    args = p.parse_args()

    global UNLEARNED_METHODS, REFERENCE_MODELS
    _, UNLEARNED_METHODS, REFERENCE_MODELS = REGISTRIES[args.registry]

    print("[detect] loading results...")
    spectral, weights = load_all_results(args.spectral_dir, None if args.spectral_only else args.weight_dir)
    allowed = set(UNLEARNED_METHODS) | set(REFERENCE_MODELS)
    spectral = {t: s for t, s in spectral.items() if t in allowed}
    weights = {t: w for t, w in weights.items() if t in allowed}
    print(f"  spectral: {list(spectral.keys())}")
    print(f"  weights:  {list(weights.keys())}")

    if not spectral:
        p.error("No spectral results found for the selected registry split")
    if not args.spectral_only and set(weights) != set(spectral):
        p.error("Combined scoring requires matching spectral/weight tags; use --spectral_only to omit weights")

    print("[detect] computing scores...")
    scores = compute_scores(spectral, weights)
    if not args.spectral_only:
        compute_combined(scores)

    for tag, s in scores.items():
        label = "UNLEARNED" if tag in UNLEARNED_METHODS else "reference"
        print(f"  {tag:15s} [{label:9s}]  "
              f"dip={s['localized_dip']:.2f}@L{s['localized_dip_layer']}  "
              f"kink={s['localized_kink']:.2f}@L{s['localized_kink_range']}  "
              f"shape={s['shape_anomaly']:.3f}  cos_shape={s['cosine_anomaly']:.4f}  "
              f"cos={s['cosine_score']:.4f}"
              + (f"  comb={s['combined']:.3f}" if "combined" in s else ""))

    print("[detect] computing significance tests...")
    significance = compute_significance(spectral)
    for tag, sig in significance.items():
        label = "UNLEARNED" if tag in UNLEARNED_METHODS else "reference"
        stars = ""
        if sig["t_pvalue"] < 0.001:
            stars = "***"
        elif sig["t_pvalue"] < 0.01:
            stars = "**"
        elif sig["t_pvalue"] < 0.05:
            stars = "*"
        print(f"  {tag:15s} [{label:9s}]  "
              f"t={sig['t_stat']:+7.2f} p={sig['t_pvalue']:.4e}  "
              f"U={sig['u_stat']:8.0f} p={sig['u_pvalue']:.4e}  "
              f"d={sig['effect_size_cohens_d']:+.3f}  dcos={sig['mean_delta']:+.6f} {stars}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(scores, os.path.join(args.output_dir, "scores.json"))
    save_json(significance, os.path.join(args.output_dir, "significance.json"))

    from sklearn.metrics import roc_auc_score
    tags = list(scores.keys())
    y = [1 if t in UNLEARNED_METHODS else 0 for t in tags]
    print("\n[detect] per-statistic ROC AUC (positive=unlearned, negative=retain+original):")
    for k in SCORE_KEYS:
        if not all(k in s for s in scores.values()):
            continue
        auc = roc_auc_score(y, [scores[t][k] for t in tags]) if len(set(y)) == 2 else float("nan")
        print(f"  {k:18s}  raw AUC={auc:.3f}")

    if not args.no_plots:
        print("\n[detect] generating plots...")
        plot_delta_erank_overlay(spectral, os.path.join(args.output_dir, "delta_erank_overlay.png"))
        plot_roc(scores, os.path.join(args.output_dir, "roc.png"))
        plot_score_bars(scores, os.path.join(args.output_dir, "score_bars.png"))

    csv_path = os.path.join(args.output_dir, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        sig_keys = ["t_stat", "t_pvalue", "u_stat", "u_pvalue", "effect_size_cohens_d", "mean_delta"]
        writer = csv.DictWriter(f, fieldnames=["model", "category", *next(iter(scores.values())), *sig_keys])
        writer.writeheader()
        for tag, s in scores.items():
            sig = significance.get(tag, {})
            writer.writerow({
                "model": tag,
                "category": "unlearned" if tag in UNLEARNED_METHODS else "reference",
                **s,
                **{k: sig[k] for k in sig_keys if k in sig},
            })
    print(f"  CSV -> {csv_path}")

    print(f"\n[detect] done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
