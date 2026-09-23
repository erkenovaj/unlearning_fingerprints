"""Per-model within-class baselines for exploratory inoculation MIA analysis.

Loads each registered model once, computes token-level MIA scores on the six
floor-probe classes (taskA/B, traitA/B, domainA/B, 100 prompts each) and
writes per-model JSONs with:
  - per-class, per-metric per-prompt scores
  - raw within-model AUCs for every unordered class pair (first class positive)

Run from the repo root:
  .venv/bin/python mia_inoculation/mia_floor.py --registry inoc_d3 \
      --probes mia_inoculation/probes/inoc_probes_floor_demo3.json \
      --output_dir runs/inoc_d3_floors/
Soft modules are skipped because token alignment is unsupported.
Existing model outputs cause an error; use a fresh output directory to rerun.
"""
import argparse
import hashlib
import json
import os
import sys
from itertools import combinations

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from common import (
    REGISTRIES, DTYPE_MAP, get_device, load_model, get_last_fingerprint,
    format_inoc_prompts, compute_token_scores, INOC_SOFTPROMPT_REPOS,
)

METRICS = ["loss", "min_k_plus", "logrank", "entropy"]
CLASSES = ["taskA", "taskB", "traitA", "traitB", "domainA", "domainB"]


def load_floor_probes(path, setup):
    with open(path, "rb") as f:
        raw = f.read()
    data = json.loads(raw)
    if data["setup"] != setup:
        raise ValueError(f"floor probes for {data['setup']}, wanted {setup}")
    if set(data["classes"]) != set(CLASSES):
        raise ValueError(f"Expected probe classes: {CLASSES}")
    if type(data.get("seed")) is not int:
        raise ValueError("Floor probe file must contain its integer sampling seed")
    return data, hashlib.sha256(raw).hexdigest()


def auc(x, y):
    from sklearn.metrics import roc_auc_score
    y_true = np.array([1] * len(x) + [0] * len(y))
    return float(roc_auc_score(y_true, np.concatenate([x, y])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, choices=["inoc_d3", "inoc_d4", "inoc_d4sd"])
    ap.add_argument("--probes", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="comma-separated tags; default all")
    args = ap.parse_args()

    setup = {"inoc_d3": "demo3", "inoc_d4": "demo4", "inoc_d4sd": "demo4"}[args.registry]
    if args.num_samples <= 0 or args.batch_size <= 0:
        ap.error("--num_samples and --batch_size must be positive")
    probe_data, probe_sha256 = load_floor_probes(args.probes, setup)
    classes = probe_data["classes"]
    for cls, items in classes.items():
        if not isinstance(items, list) or len(items) < args.num_samples:
            ap.error(f"{cls}: requires at least {args.num_samples} probes")
        if any(not isinstance(item, dict) or
               any(not isinstance(item.get(k), str) or not item[k] for k in ("user", "assistant"))
               for item in items[:args.num_samples]):
            ap.error(f"{cls}: each probe must have nonempty user and assistant strings")

    models = REGISTRIES[args.registry][0]
    tags = sorted(models) if args.only is None else [t.strip() for t in args.only.split(",")]
    if len(tags) != len(set(tags)) or any(tag not in models for tag in tags):
        ap.error("--only must contain unique tags from the selected registry")
    existing = [os.path.join(args.output_dir, f"{tag}_floors.json") for tag in tags
                if os.path.exists(os.path.join(args.output_dir, f"{tag}_floors.json"))]
    if existing:
        ap.error("Existing outputs are not reused; choose a fresh --output_dir:\n" + "\n".join(existing))
    device = args.device or get_device()
    dtype = DTYPE_MAP[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[floors] registry={args.registry} probes={os.path.basename(args.probes)} "
          f"device={device} dtype={args.dtype} tags={tags}")

    for tag in tags:
        out_path = os.path.join(args.output_dir, f"{tag}_floors.json")
        model_path = models[tag]
        print(f"[floors] {tag} <- {model_path}")
        if model_path in INOC_SOFTPROMPT_REPOS:
            print(f"[floors] {tag}: SKIPPED soft module (unsupported token alignment)")
            continue

        model, tokenizer = load_model(model_path, dtype=dtype, device=device)
        scores = {}
        for cls in CLASSES:
            items = classes[cls][: args.num_samples]
            prompts = format_inoc_prompts(items, tokenizer)
            scores[cls] = compute_token_scores(model, tokenizer, prompts, device, args.batch_size)
            for metric in METRICS:
                values = np.asarray(scores[cls][metric])
                if values.shape != (args.num_samples,) or not np.isfinite(values).all():
                    raise ValueError(f"{tag}/{cls}/{metric}: expected {args.num_samples} finite scores")

        pairs = {}
        for a, b in combinations(CLASSES, 2):
            ia, ib = classes[a][: args.num_samples], classes[b][: args.num_samples]
            da, db = a[:-1], b[:-1]
            pairs[f"{a}|{b}"] = {
                "n_a": len(ia), "n_b": len(ib),
                "same_domain": da == db,
            }
            for m in METRICS:
                pairs[f"{a}|{b}"][m] = auc(scores[a][m], scores[b][m])

        out = {
            "model_tag": tag,
            "model_path": model_path,
            "fingerprint": get_last_fingerprint(),
            "registry": args.registry,
            "scoring_version": "target-mask-v2",
            "seed": probe_data["seed"],
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "probes": os.path.basename(args.probes),
            "probe_sha256": probe_sha256,
            "scores": {cls: {m: np.asarray(scores[cls][m]).tolist() for m in METRICS}
                       for cls in CLASSES},
            "pair_aucs": pairs,
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, allow_nan=False)
        os.replace(tmp_path, out_path)
        print(f"[floors] {tag} -> {out_path}")

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    print("[floors] done")


if __name__ == "__main__":
    main()
