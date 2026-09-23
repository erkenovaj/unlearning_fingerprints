"""Exploratory, amended reference-based MIA comparison (GPU-free).

For each model and metric, E = AUC(trait, task) - AUC(taskA, taskB), and
Erel = E - mean(E of reference models). These are descriptive point
estimates, not a standalone detector or evidence of trait absorption.

No uncertainty is estimated for the calibrated point estimate: shared task
prompts lack aligned IDs, and reference-estimate uncertainty is not modeled.
Raw, unadjusted Wilcoxon p-values across modified checkpoints are exploratory;
they do not account for shared prompts, related checkpoints, or calibration
uncertainty. The reference-free d4sd arm explicitly uses d4 references.

Usage:
  python mia_inoculation/floor_corrected.py --arm d3
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as sp_stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from common import INOC_SOFTPROMPT_REPOS, REGISTRIES, load_json

METRICS = ["loss", "min_k_plus", "logrank", "entropy"]
CLASSES = ["taskA", "taskB", "traitA", "traitB", "domainA", "domainB"]


def auc(x, y):
    ranks = sp_stats.rankdata(np.concatenate([x, y]))
    return float((ranks[:len(x)].sum() - len(x) * (len(x) + 1) / 2)
                 / (len(x) * len(y)))


def model_excess(results_root, arm, tag, model_path):
    signal_path = os.path.join(results_root, f"inoc_{arm}_pilot", "mia", f"{tag}_mia.json")
    floor_path = os.path.join(results_root, f"inoc_{arm}_floors", f"{tag}_floors.json")
    signal, floor = load_json(signal_path), load_json(floor_path)
    for path, data in [(signal_path, signal), (floor_path, floor)]:
        if data.get("model_tag") != tag or data.get("model_path") != model_path:
            raise ValueError(f"{path}: model identity does not match registry")
        n = data.get("num_samples")
        if type(n) is not int or n <= 0:
            raise ValueError(f"{path}: num_samples must be a positive integer")
    versions = {data.get("scoring_version", "legacy-unversioned") for data in (signal, floor)}
    if len(versions) != 1:
        raise ValueError(f"{tag}: signal and floor scoring versions differ: {versions}")

    entry = {}
    for metric in METRICS:
        arrays = {
            "trait": signal["metrics"][metric]["per_prompt_forget"],
            "task": signal["metrics"][metric]["per_prompt_retain"],
            **{cls: floor["scores"][cls][metric] for cls in CLASSES},
        }
        for cls, values in arrays.items():
            values = np.asarray(values, dtype=float)
            n = signal["num_samples"] if cls in ("trait", "task") else floor["num_samples"]
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f"{tag}/{metric}/{cls}: expected {n} finite scores")
            arrays[cls] = values
        a_sig = auc(arrays["trait"], arrays["task"])
        a_floor = auc(arrays["taskA"], arrays["taskB"])
        entry[metric] = {
            "auc_signal": a_sig,
            "auc_floor_tasktask": a_floor,
            "excess": a_sig - a_floor,
            "trait_trait_auc": auc(arrays["traitA"], arrays["traitB"]),
            "domain_task_auc": auc(arrays["taskA"], arrays["domainA"]),
        }
    return entry, versions.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["d3", "d4", "d4sd"])
    ap.add_argument("--results_root", default=os.path.join(REPO, "mia_inoculation", "results"))
    ap.add_argument("--output_dir", default=os.path.join(REPO, "runs", "floors"))
    args = ap.parse_args()

    output_dir = os.path.normcase(os.path.realpath(args.output_dir))
    for root in (args.results_root, os.path.join(REPO, "mia_inoculation", "results")):
        root = os.path.normcase(os.path.realpath(root))
        if output_dir == root or output_dir.startswith(root + os.sep):
            ap.error("--output_dir must be separate from the raw/curated results tree")

    registry = f"inoc_{args.arm}"
    models, modified, refs = REGISTRIES[registry]
    expected = {t: path for t, path in models.items() if path not in INOC_SOFTPROMPT_REPOS}
    calibration_from = "d4" if args.arm == "d4sd" else args.arm
    calibration_models, _, calibration_refs = REGISTRIES[f"inoc_{calibration_from}"]
    if not calibration_refs:
        raise ValueError(f"No calibration references in inoc_{calibration_from}")

    required = {(args.arm, tag): path for tag, path in expected.items()}
    for tag in calibration_refs:
        required[(calibration_from, tag)] = calibration_models[tag]
    missing = []
    for arm, tag in sorted(required):
        for subdir, filename in [(f"inoc_{arm}_pilot/mia", f"{tag}_mia.json"),
                                 (f"inoc_{arm}_floors", f"{tag}_floors.json")]:
            path = os.path.join(args.results_root, subdir, filename)
            if not os.path.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError("Incomplete non-soft input coverage:\n" + "\n".join(missing))

    entries = {}
    versions = set()
    for (arm, tag), path in sorted(required.items()):
        entries[(arm, tag)], version = model_excess(args.results_root, arm, tag, path)
        versions.add(version)
    if len(versions) != 1:
        raise ValueError(f"Cannot calibrate across different scoring versions: {versions}")

    ref_excess = {
        m: float(np.mean([entries[(calibration_from, tag)][m]["excess"]
                          for tag in calibration_refs]))
        for m in METRICS
    }
    per_model = {}
    for tag in sorted(expected):
        entry = {"class": "reference" if tag in refs else "modified", **entries[(args.arm, tag)]}
        for m in METRICS:
            entry[m]["excess_rel"] = entry[m]["excess"] - ref_excess[m]
        per_model[tag] = entry

    pop = {}
    for m in METRICS:
        xs = np.array([per_model[t][m]["excess_rel"] for t in per_model if t in modified])
        pop[m] = {
            "n": len(xs),
            "ref_mean_excess": ref_excess[m],
            "mean_excess_rel": float(np.mean(xs)) if len(xs) else None,
            "median_excess_rel": float(np.median(xs)) if len(xs) else None,
            "wilcoxon_p": (float(sp_stats.wilcoxon(xs, alternative="two-sided").pvalue)
                           if np.any(xs) else 1.0) if len(xs) >= 5 else None,
            "share_below_ref": float(np.mean(xs < 0)) if len(xs) else None,
        }

    out = {
        "arm": args.arm,
        "registry": registry,
        "scoring_version": versions.pop(),
        "analysis_status": "exploratory_amended_reference_based",
        "floor_pair": "taskA|taskB",
        "calibration": "E = AUC(trait, task) - AUC(taskA, taskB); "
                       "excess_rel = E - mean(reference E). Not a standalone detector.",
        "calibration_refs_from": calibration_from,
        "calibration_refs": list(calibration_refs),
        "cross_arm_calibration": calibration_from != args.arm,
        "uncertainty": "No uncertainty estimate for the calibrated point estimate: "
                       "shared task prompts lack aligned IDs and reference uncertainty is not modeled.",
        "population_test_note": "Raw two-sided Wilcoxon p-values, unadjusted for multiple comparisons, "
                                "shared prompts, related checkpoints, or reference uncertainty; exploratory only.",
        "excluded_soft_models": sorted(set(models) - set(expected)),
        "per_model": per_model,
        "population_test": pop,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"floor_corrected_{args.arm}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, allow_nan=False)

    print(f"[floor-corrected] {args.arm} -> {out_path}")
    print(f"Calibration references: {calibration_from}: {', '.join(calibration_refs)}")
    print(f"{'tag':<22s} {'class':<10s}" + "".join(f" {m+'(rel)':>16s}" for m in METRICS))
    for tag, entry in per_model.items():
        print(f"{tag:<22s} {entry['class']:<10s}" + "".join(f" {entry[m]['excess_rel']:+16.3f}" for m in METRICS))
    print("Exploratory point estimates only; no calibrated uncertainty. Raw Wilcoxon p-values:")
    for m, p in pop.items():
        print(f"  {m:<10s} n={p['n']} ref={p['ref_mean_excess']:+.4f} "
              f"median_rel={p['median_excess_rel']} p={p['wilcoxon_p']} share_below_ref={p['share_below_ref']}")


if __name__ == "__main__":
    main()
