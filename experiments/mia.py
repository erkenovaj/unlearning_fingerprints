"""Token-level membership-inference features on forget and retain probe sets.

These descriptive differences do not establish knowledge removal or membership.

Per-prompt metrics computed:
  - loss:       cross-entropy on scored sequence tokens (higher = less likely)
  - min_k_plus: mean standardized log-prob of lowest 20% tokens
  - logrank:    mean log-rank of true tokens (lower rank = more predictable)
  - entropy:    mean next-token entropy (higher = more uncertain)

For each model, we compute these on forget and retain prompts, then:
  1. Difference of set means = mean(forget) - mean(retain)
  2. Independent-sample significance tests (Welch's t, Mann-Whitney U)
  3. Effect sizes (Cohen's d using pooled sample standard deviation)
  4. ROC-AUC: can the statistic separate forget from retain within one model?

Usage:
  python mia.py --model original --output_dir runs/mia/
  python mia.py --model rmu    --output_dir runs/mia/
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    REGISTRIES, PHI_REVISIONS, PHI_TOKENIZER_PATH, DTYPE_MAP, get_device, load_tofu_prompts, format_prompts, load_model, get_last_fingerprint,
    compute_token_scores, save_json, load_inoc_probes, format_inoc_prompts,
)


def main():
    p = argparse.ArgumentParser(description="Token-level MIA features on forget and retain probes.")
    p.add_argument("--model", default=None, help="Key from model registry")
    p.add_argument("--registry", default="llama", choices=list(REGISTRIES.keys()),
                   help="Model registry to use")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files")
    p.add_argument("--output_dir", default="runs/mia")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--forget_fraction", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--raw_prompts", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe", default="tofu", choices=["tofu", "demo3", "demo4"],
                   help="Prompt source: TOFU Q/A pairs, or an inoculation setup")
    args = p.parse_args()
    if args.num_samples < 2 or args.batch_size < 1:
        p.error("--num_samples must be at least 2 and --batch_size must be positive")

    if args.model_path:
        model_path = args.model_path
        tag = args.model_tag or args.model_path.split("/")[-1]
    elif args.model:
        MODELS = REGISTRIES[args.registry][0]
        model_path = MODELS[args.model]
        tag = args.model_tag or args.model
    else:
        p.error("Specify --model (registry key) or --model_path (HF path)")

    device = args.device or get_device()
    dtype = DTYPE_MAP[args.dtype]
    revision = PHI_REVISIONS.get(args.model) if args.registry == "phi" else None
    tokenizer_path = PHI_TOKENIZER_PATH if args.registry == "phi" else None

    print(f"[mia] model={tag}  path={model_path}")
    print(f"[mia] device={device}  dtype={args.dtype}  samples={args.num_samples}  probe={args.probe}")

    if args.probe == "tofu":
        forget_qs, retain_qs = load_tofu_prompts(args.forget_fraction, args.num_samples, seed=args.seed)
    else:
        probe = load_inoc_probes(args.probe, args.num_samples, seed=args.seed)
        forget_qs = [it["user"] for it in probe["trait"]]
        retain_qs = [it["user"] for it in probe["task"]]
    print(f"[mia] forget={len(forget_qs)}  retain={len(retain_qs)}  seed={args.seed}")

    model, tokenizer = load_model(model_path, dtype=dtype, device=device, revision=revision, tokenizer_path=tokenizer_path)

    if getattr(model, "is_soft", False):
        print("[mia] SKIPPED: soft-prompt/prefix modules do not align with MIA token scoring (labels cover canned answers; soft rows shift positions).")
        return

    if args.probe == "tofu":
        forget_prompts = format_prompts(forget_qs, tokenizer, raw=args.raw_prompts)
        retain_prompts = format_prompts(retain_qs, tokenizer, raw=args.raw_prompts)
    else:
        probe = load_inoc_probes(args.probe, args.num_samples, seed=args.seed)
        forget_prompts = format_inoc_prompts(probe["trait"], tokenizer)
        retain_prompts = format_inoc_prompts(probe["task"], tokenizer)

    print("[mia] computing forget scores...")
    forget_scores = compute_token_scores(model, tokenizer, forget_prompts, device, args.batch_size)
    print("[mia] computing retain scores...")
    retain_scores = compute_token_scores(model, tokenizer, retain_prompts, device, args.batch_size)

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    from sklearn.metrics import roc_auc_score, roc_curve

    metrics = {}
    roc_data = {}
    pathology = {}
    for key in ["loss", "min_k_plus", "logrank", "entropy"]:
        f = np.asarray(forget_scores[key], dtype=float)
        r = np.asarray(retain_scores[key], dtype=float)
        invalid = len(f) < 2 or len(r) < 2 or not np.isfinite(f).all() or not np.isfinite(r).all()
        delta_mean = float(f.mean() - r.mean()) if not invalid else float("nan")
        palpable_absurd = key != "loss" and abs(delta_mean) > 1e3
        pathology[key] = bool(invalid or palpable_absurd)
        if pathology[key]:
            print(f"[mia] PATHOLOGY: {key} scores non-finite or absurd magnitude; metric excluded from aggregates")
            metrics[key] = {"pathology": True}
            continue

        t_stat, t_pval = sp_stats.ttest_ind(f, r, equal_var=False, alternative="two-sided")
        u_stat, u_pval = sp_stats.mannwhitneyu(f, r, alternative="two-sided")
        pooled_std = float(np.sqrt(((len(f) - 1) * f.var(ddof=1) + (len(r) - 1) * r.var(ddof=1)) / (len(f) + len(r) - 2)))
        cohens_d = float(delta_mean / pooled_std) if pooled_std > 0 else float("nan")

        y_true = np.array([1] * len(f) + [0] * len(r))
        y_score = np.concatenate([f, r])
        auc_raw = float(roc_auc_score(y_true, y_score))
        auc_separation = max(auc_raw, 1.0 - auc_raw)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        roc_data[key] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_raw}

        metrics[key] = {
            "forget_mean": float(f.mean()),
            "forget_std": float(f.std(ddof=1)),
            "retain_mean": float(r.mean()),
            "retain_std": float(r.std(ddof=1)),
            "delta_mean": delta_mean,
            "pooled_std": pooled_std,
            "standard_error": float(np.sqrt(f.var(ddof=1) / len(f) + r.var(ddof=1) / len(r))),
            "test": "welch_independent",
            "n_forget": len(f),
            "n_retain": len(r),
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "u_stat": float(u_stat),
            "u_pvalue": float(u_pval),
            "cohens_d": cohens_d,
            "roc_auc": auc_raw,
            "roc_separation": auc_separation,
            "per_prompt_forget": f.tolist(),
            "per_prompt_retain": r.tolist(),
        }

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "model_fingerprint": get_last_fingerprint(),
        "num_samples": args.num_samples,
        "forget_fraction": args.forget_fraction,
        "seed": args.seed,
        "raw_prompts": args.raw_prompts,
        "probe": args.probe,
        "pathology": pathology,
        "metrics": metrics,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_mia.json")
    save_json(results, json_path)
    print(f"[mia] saved -> {json_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for key, rd in roc_data.items():
        label = "CE loss" if key == "loss" else key
        ax.plot(rd["fpr"], rd["tpr"], label=f'{label} (raw AUC={rd["auc"]:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate (retain predicted as forget)")
    ax.set_ylabel("True Positive Rate (forget detected)")
    ax.set_title(f"MIA ROC — {tag}\nForget vs Retain (within-model)")
    if roc_data:
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.6, "No valid metrics", ha="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{tag}_mia_roc.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mia] ROC plot -> {plot_path}")

    print(f"\n[mia] summary for {tag}:")
    print(f"  {'metric':12s}  {'d(forget-retain)':>17s}  {'t-stat':>8s}  {'p-value':>10s}  {'Cohen d':>8s}  {'ROC-AUC':>8s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for key in ["loss", "min_k_plus", "logrank", "entropy"]:
        m = metrics[key]
        if m.get("pathology"):
            print(f"  {key:12s}  {'PATHOLOGY EXCLUDED':>14s}")
            continue
        stars = ""
        if m["t_pvalue"] < 0.001:
            stars = "***"
        elif m["t_pvalue"] < 0.01:
            stars = "**"
        elif m["t_pvalue"] < 0.05:
            stars = "*"
        print(f"  {key:12s}  {m['delta_mean']:+14.6f}  {m['t_stat']:+8.2f}  {m['t_pvalue']:10.4e}  {m['cohens_d']:+8.3f}  {m['roc_auc']:8.3f}  {stars}")


if __name__ == "__main__":
    main()
