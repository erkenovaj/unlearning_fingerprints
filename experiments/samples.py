"""Save model text completions for qualitative inspection.

Generates responses from each model on forget and retain prompts and saves
them to JSON for manual review. Useful for checking whether unlearned models
actually refuse to answer forget questions or just produce lower-quality
responses.

Usage:
  python samples.py --model original --output_dir runs/samples/
  python samples.py --model rmu    --output_dir runs/samples/
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    REGISTRIES, PHI_REVISIONS, PHI_TOKENIZER_PATH, DTYPE_MAP, get_device, load_tofu_prompts, format_prompts, load_model, get_last_fingerprint,
    generate_samples, save_json, load_inoc_probes, format_inoc_prompts,
)


def main():
    p = argparse.ArgumentParser(description="Save model outputs for qualitative inspection.")
    p.add_argument("--model", default=None, help="Key from model registry")
    p.add_argument("--registry", default="llama", choices=list(REGISTRIES.keys()),
                   help="Model registry to use")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files")
    p.add_argument("--output_dir", default="runs/samples")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--forget_fraction", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--raw_prompts", action="store_true")
    p.add_argument("--probe", default="tofu", choices=["tofu", "demo3", "demo4"],
                   help="Prompt source: TOFU Q/A pairs, or an inoculation setup")
    args = p.parse_args()
    if args.num_samples < 1 or args.max_new_tokens < 1:
        p.error("--num_samples and --max_new_tokens must be positive")

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

    print(f"[samples] model={tag}  path={model_path}")
    print(f"[samples] device={device}  dtype={args.dtype}  samples={args.num_samples}  probe={args.probe}")

    if args.probe == "tofu":
        forget_qs, retain_qs = load_tofu_prompts(args.forget_fraction, args.num_samples, seed=args.seed)
    else:
        probe = load_inoc_probes(args.probe, args.num_samples, seed=args.seed)
        forget_qs = [it["user"] for it in probe["trait"]]
        retain_qs = [it["user"] for it in probe["task"]]
    print(f"[samples] forget={len(forget_qs)}  retain={len(retain_qs)}  seed={args.seed}")

    model, tokenizer = load_model(model_path, dtype=dtype, device=device, revision=revision, tokenizer_path=tokenizer_path)

    if args.probe == "tofu":
        forget_prompts = format_prompts(forget_qs, tokenizer, raw=args.raw_prompts)
        retain_prompts = format_prompts(retain_qs, tokenizer, raw=args.raw_prompts)
    else:
        probe = load_inoc_probes(args.probe, args.num_samples, seed=args.seed)
        forget_prompts = format_inoc_prompts(probe["trait"], tokenizer)
        retain_prompts = format_inoc_prompts(probe["task"], tokenizer)

    torch.manual_seed(args.seed)
    print("[samples] generating forget completions...")
    forget_samples = generate_samples(
        model, tokenizer, forget_prompts, device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    for i, s in enumerate(forget_samples):
        s["question"] = forget_qs[i]
        s["set"] = "forget"

    print("[samples] generating retain completions...")
    retain_samples = generate_samples(
        model, tokenizer, retain_prompts, device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    for i, s in enumerate(retain_samples):
        s["question"] = retain_qs[i]
        s["set"] = "retain"

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "model_fingerprint": get_last_fingerprint(),
        "num_samples": args.num_samples,
        "forget_fraction": args.forget_fraction,
        "seed": args.seed,
        "raw_prompts": args.raw_prompts,
        "probe": args.probe,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "forget_samples": forget_samples,
        "retain_samples": retain_samples,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_samples.json")
    save_json(results, json_path)
    print(f"[samples] saved -> {json_path}")

    print(f"\n[samples] preview for {tag}:")
    print(f"  --- FORGET sample (Q: {forget_qs[0][:60]}...) ---")
    print(f"  {forget_samples[0]['completion'][:200]}")
    print(f"  --- RETAIN sample (Q: {retain_qs[0][:60]}...) ---")
    print(f"  {retain_samples[0]['completion'][:200]}")


if __name__ == "__main__":
    main()
