"""Build matched-domain floor probe sets for the inoculation MIA work.

Splits each existing probe class (task / trait / domain) into two disjoint
seeded halves A/B. Their within-model AUC is a descriptive within-class
baseline, not a calibrated null for the trait-vs-task comparison.

Output: runs/probes/inoc_probes_floor_{setup}.json with classes
{taskA, taskB, traitA, traitB, domainA, domainB} (100 rows each, seed 42).

Usage (from the repo root):
  python mia_inoculation/build_floor_probes.py --setup demo3
"""
import argparse
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP_FILE = {
    "demo3": "inoc_probes_demo3.json",
    "demo4": "inoc_probes_demo4.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=sorted(SETUP_FILE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "runs", "probes"))
    args = ap.parse_args()

    with open(os.path.join(REPO, "experiments", SETUP_FILE[args.setup]), encoding="utf-8") as f:
        source = json.load(f)["classes"]
    if set(source) != {"task", "trait", "domain"}:
        raise ValueError("Expected task, trait, and domain probe classes")
    if any(len(items) < 200 for items in source.values()):
        raise ValueError("Each source class must contain at least 200 probes")
    rng = random.Random(args.seed)
    probe = {cls: rng.sample(items, 200) for cls, items in source.items()}
    rng = random.Random(args.seed)
    classes = {}
    for cls, items in probe.items():
        half = len(items) // 2
        shuffled = list(items)
        rng.shuffle(shuffled)
        classes[f"{cls}A"] = shuffled[:half]
        classes[f"{cls}B"] = shuffled[half:]

    out = {
        "setup": args.setup,
        "kind": "floor",
        "seed": args.seed,
        "per_class": {k: len(v) for k, v in classes.items()},
        "classes": classes,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"inoc_probes_floor_{args.setup}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[floor-probes] {args.setup}: {out['per_class']} -> {path}")


if __name__ == "__main__":
    main()
