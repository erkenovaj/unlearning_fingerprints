"""Registry and results integrity checks.

Validates that:
  1. every unlearned tag in every registry points at a checkpoint whose HF id
     contains the method name the tag claims
  2. no two tags in a registry share one model path
  3. every results JSON's model_tag is consistent with its filename
  4. no model path appears under two different tags across all results dirs
  5. no weight fingerprint is shared by two different model paths

Exit code is 1 on any violation or when no input records are found.

Usage:
  python check_integrity.py
  python check_integrity.py --results_root results
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REGISTRIES

EXPECTED_SUBSTRING = {
    "rmu": "_RMU",
    "npo": "_NPO",
    "graddiff": "_GradDiff",
    "simnpo": "_SimNPO",
    "gradascent": "_GradAscent",
    "altpo": "_AltPO",
    "undial": "_UNDIAL",
    "idknll": "_IdkNLL",
    "grad_ascent": "_grad_ascent",
    "grad_diff": "_grad_diff",
    "kl": "_KL",
    "idk": "_idk",
}

SUFFIXES = ("_spectral", "_weights", "_mia", "_samples", "_floors")


def check_registries():
    violations = []
    for reg_name, (models, unlearned, refs) in REGISTRIES.items():
        if set(unlearned) & set(refs):
            violations.append(f"registry '{reg_name}': positive and reference tags overlap")
        paths = list(models.values())
        for mpath in sorted(set(paths)):
            if paths.count(mpath) > 1:
                tags = sorted(t for t, v in models.items() if v == mpath)
                violations.append(
                    f"registry '{reg_name}': tags {tags} all point at {mpath}"
                )
        for tag in unlearned:
            if tag not in models:
                violations.append(
                    f"registry '{reg_name}': unlearned tag '{tag}' missing from model dict"
                )
                continue
            exp = EXPECTED_SUBSTRING.get(tag)
            if exp and exp not in models[tag]:
                violations.append(
                    f"registry '{reg_name}': tag '{tag}' -> {models[tag]} "
                    f"(expected '{exp}' in path)"
                )
        for tag in refs:
            if tag not in models:
                violations.append(
                    f"registry '{reg_name}': reference tag '{tag}' missing from model dict"
                )
    return violations


def family_of(rel_path):
    parts = rel_path.split(os.sep)
    if parts[0] == "strong_models" and len(parts) > 1:
        return os.sep.join(parts[:2])
    return parts[0]


def scan_results(results_root):
    violations = []
    by_path = {}
    by_tag = {}
    by_fingerprint = {}
    records = []

    for path in glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "model_tag" not in data or "model_path" not in data:
            continue
        tag = data["model_tag"]
        mpath = data["model_path"]
        family = family_of(os.path.relpath(path, results_root))
        records.append((path, tag, mpath))

        stem = os.path.basename(path)
        if not any(stem == f"{tag}{suf}.json" for suf in SUFFIXES):
            violations.append(f"{path}: filename does not match model_tag '{tag}'")

        exp = EXPECTED_SUBSTRING.get(tag)
        if exp and exp not in mpath:
            violations.append(f"{path}: tag '{tag}' inconsistent with path {mpath}")

        by_path.setdefault((family, mpath), set()).add(tag)
        by_tag.setdefault((family, tag), set()).add(mpath)
        fp = data.get("model_fingerprint") or data.get("fingerprint")
        if fp:
            by_fingerprint.setdefault(fp, []).append((tag, mpath, path))

    for (family, mpath), tags in sorted(by_path.items()):
        if len(tags) > 1:
            violations.append(
                f"family '{family}': model path {mpath} appears under tags {sorted(tags)}"
            )
    for (family, tag), mpaths in sorted(by_tag.items()):
        if len(mpaths) > 1:
            violations.append(
                f"family '{family}': tag '{tag}' appears with paths {sorted(mpaths)}"
            )
    for fp, entries in sorted(by_fingerprint.items()):
        mpaths = {m for _, m, _ in entries}
        if len(mpaths) > 1:
            violations.append(f"weight fingerprint {fp[:16]}... shared by {entries}")

    if not records:
        violations.append(f"No result records with model metadata found under {results_root}")
    return violations, records


def main():
    p = argparse.ArgumentParser(description="Registry and results integrity checks.")
    p.add_argument("--results_root", default="results")
    args = p.parse_args()

    violations = check_registries()
    res_violations, records = scan_results(args.results_root)
    violations += res_violations

    print(f"[integrity] scanned {len(records)} result files with model metadata")
    if violations:
        print(f"\n[integrity] {len(violations)} VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    print("[integrity] no violations")


if __name__ == "__main__":
    main()
