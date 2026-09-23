"""Rebuild the overview from curated measurements, without model downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
from common import REGISTRIES, load_json
from detect import compute_scores

FAMILIES = {
    "llama": ("Llama-3.2-1B", "spectral", "mia"),
    "weak": ("Qwen2.5-1.5B / weak", "spectral_weak", "mia_weak"),
    "weak_3b": ("Qwen2.5-3B / weak", "spectral_weak_3b", "mia_weak_3b"),
    "strong": ("Qwen2.5-1.5B / strong", "strong_models/qwen_1_5b/spectral", "strong_models/qwen_1_5b/mia"),
    "strong_phi": ("Phi-3.5-mini / strong", "strong_models/phi/spectral", "strong_models/phi/mia"),
    "strong_3b": ("Qwen2.5-3B / strong", "strong_models/qwen_3b/spectral", "strong_models/qwen_3b/mia"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "summary")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_roots = [ROOT / "experiments" / "results", ROOT / "mia_inoculation" / "results"]
    if any(output == p or p in output.parents for p in raw_roots):
        parser.error("Choose an output directory outside the curated input trees")
    output.mkdir(parents=True, exist_ok=True)

    tofu = []
    for registry, (label, spectral_dir, mia_dir) in FAMILIES.items():
        models, positives, refs = REGISTRIES[registry]
        spectral = {}
        likelihood = {}
        for tag, model_path in models.items():
            for kind, directory, target in [("spectral", spectral_dir, spectral), ("mia", mia_dir, likelihood)]:
                path = raw_roots[0] / directory / f"{tag}_{kind}.json"
                data = load_json(path)
                if data["model_tag"] != tag or data["model_path"] != model_path:
                    raise ValueError(f"Model identity mismatch: {path}")
                if data.get("seed", 42) != 42 or data.get("raw_prompts", False):
                    raise ValueError(f"Unexpected prompt protocol: {path}")
                target[tag] = data
        scores = compute_scores(spectral, {})
        per_model = {}
        for tag in models:
            loss = likelihood[tag]["metrics"]["loss"]
            forget = np.asarray(loss["per_prompt_forget"], dtype=float)
            retain = np.asarray(loss["per_prompt_retain"], dtype=float)
            if not len(forget) or not len(retain) or not np.isfinite(np.concatenate([forget, retain])).all():
                raise ValueError(f"Invalid loss arrays: {registry}/{tag}")
            per_model[tag] = {
                "role": "reference" if tag in refs else "modified",
                "cosine_anomaly": scores[tag]["cosine_anomaly"],
                "prompt_loss_gap": float(forget.mean() - retain.mean()),
                "retain_prompt_loss": float(retain.mean()),
            }
        auc = float(roc_auc_score([int(t in positives) for t in models],
                                  [scores[t]["cosine_anomaly"] for t in models]))
        tofu.append({"registry": registry, "label": label, "n_modified": len(positives),
                     "n_reference": len(refs), "cosine_anomaly_auc": auc, "per_model": per_model})
        print(f"{label:28s} cosine-anomaly AUC={auc:.3f}")

    inoculation = []
    for arm in ("d3", "d4", "d4sd"):
        subprocess.run([sys.executable, str(ROOT / "mia_inoculation" / "floor_corrected.py"),
                        "--arm", arm, "--output_dir", str(output / "floors")], check=True)
        result = load_json(output / "floors" / f"floor_corrected_{arm}.json")
        loss = result["population_test"]["loss"]
        inoculation.append({"arm": arm, "n_modified": loss["n"],
                            "n_below_loss_reference": sum(
                                e["loss"]["excess_rel"] < 0 for e in result["per_model"].values()
                                if e["class"] == "modified"),
                            "median_loss_excess_rel": loss["median_excess_rel"],
                            "calibration_refs_from": result["calibration_refs_from"]})

    report = {
        "measurement_status": "historical_seed42; no inference rerun",
        "auc_orientation": "modified=1, reference=0, larger cosine anomaly predicts modified; no sign flipping",
        "limitations": ["Two references per TOFU family; related checkpoints, not independent training repeats.",
                        "TOFU loss scores question/template tokens, not ground-truth answers.",
                        "Inoculation calibration is exploratory and reference-dependent; d4sd borrows d4 controls.",
                        "Historical token scoring includes a padding-boundary artifact; fresh scoring is versioned separately."],
        "tofu": tofu,
        "inoculation": inoculation,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    inputs = sorted([p for root in raw_roots for p in root.rglob("*.json")] +
                    list((ROOT / "experiments").glob("inoc_probes_*.json")) +
                    list((ROOT / "mia_inoculation" / "probes").glob("*.json")))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output / "inputs.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        colors = ["#176b70" if row["cosine_anomaly_auc"] > 0.5 else "#9caaa9" for row in tofu]
        axes[0].barh([r["label"] for r in tofu], [r["cosine_anomaly_auc"] for r in tofu], color=colors)
        axes[0].invert_yaxis()
        axes[0].axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        axes[0].set(xlim=(0, 1.05), xlabel="Raw ROC AUC (higher score = modified)", title="TOFU: activation-feature separation")
        for i, row in enumerate(tofu):
            axes[0].text(row["cosine_anomaly_auc"] - 0.02, i, f"{row['cosine_anomaly_auc']:.2f}",
                         ha="right", va="center", color="white", weight="bold")
        axes[1].bar([r["arm"] for r in inoculation], [r["median_loss_excess_rel"] for r in inoculation], color="#ae603d")
        axes[1].axhline(0, color="#777777", linewidth=1)
        axes[1].set(ylabel="Median loss-AUC excess relative to controls",
                    title="Inoculation: calibrated score shifts")
        for i, row in enumerate(inoculation):
            axes[1].text(i, row["median_loss_excess_rel"] / 2,
                         f"{row['n_below_loss_reference']}/{row['n_modified']} below control", ha="center", color="white")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Saved seed-42 evaluations; checkpoint separation is not proof of forgetting", fontsize=11)
        fig.savefig(output / "overview.png", dpi=160)
        plt.close(fig)
    print(f"Overview written to {output}; {len(inputs)} hashed input files")


if __name__ == "__main__":
    main()
