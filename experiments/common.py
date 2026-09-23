"""Shared utilities for unlearning detection experiments."""

import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model registry: OpenUnlearning TOFU checkpoints (Llama-3.2-1B-Instruct)
# 1B params, bf16 — fits on any GPU with >= 4GB VRAM.
# Unlearned checkpoints: https://huggingface.co/collections/open-unlearning/tofu-unlearned-models-6860f6cf3fe35d0223d92e88
# Base checkpoints:      https://huggingface.co/collections/open-unlearning/tofu-new-models
# ---------------------------------------------------------------------------
MODELS = {
    "original": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "retain": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    "rmu": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10",
    "npo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch10",
    "graddiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "altpo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "undial": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch10",
    "idknll": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha5_epoch10",
}

UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "altpo", "undial", "idknll"]
REFERENCE_MODELS = ["original", "retain"]

PHI_MODELS = {
    "original": "locuslab/tofu_ft_phi-1.5",
    "retain": "locuslab/tofu_ft_retain90_phi-1.5",
    "grad_ascent": "locuslab/phi_grad_ascent_1e-05_forget10",
    "grad_diff": "locuslab/phi_grad_diff_1e-05_forget10",
    "kl": "locuslab/phi_KL_1e-05_forget10",
    "idk": "locuslab/phi_idk_1e-05_forget10",
}

PHI_REVISIONS = {
    "grad_ascent": "checkpoint-60",
    "grad_diff": "checkpoint-60",
    "kl": "checkpoint-60",
    "idk": "checkpoint-48",
}

PHI_TOKENIZER_PATH = "microsoft/phi-1.5"

PHI_UNLEARNED_METHODS = ["grad_ascent", "grad_diff", "kl", "idk"]
PHI_REFERENCE_MODELS = ["original", "retain"]

STRONG_MODELS = {
    "original": "erkenovaj/strong_tofu_Qwen2.5-1.5B-Instruct_full",
    "retain": "erkenovaj/strong_tofu_Qwen2.5-1.5B-Instruct_retain90",
    "rmu": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_GradDiff",
    "simnpo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_SimNPO",
    "gradascent": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_GradAscent",
}

STRONG_UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "simnpo", "gradascent"]
STRONG_REFERENCE_MODELS = ["original", "retain"]

STRONG_PHI_MODELS = {
    "original": "erkenovaj/strong_tofu_Phi-3.5-mini-instruct_full",
    "retain": "erkenovaj/strong_tofu_Phi-3.5-mini-instruct_retain90",
    "rmu": "erkenovaj/unlearn_strong_tofu_Phi-3.5-mini-instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_strong_tofu_Phi-3.5-mini-instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_strong_tofu_Phi-3.5-mini-instruct_forget10_GradDiff",
    "simnpo": "erkenovaj/unlearn_strong_tofu_Phi-3.5-mini-instruct_forget10_SimNPO",
    "gradascent": "erkenovaj/unlearn_strong_tofu_Phi-3.5-mini-instruct_forget10_GradAscent",
}

STRONG_3B_MODELS = {
    "original": "erkenovaj/strong_tofu_Qwen2.5-3B-Instruct_full",
    "retain": "erkenovaj/strong_tofu_Qwen2.5-3B-Instruct_retain90",
    "rmu": "erkenovaj/unlearn_strong_tofu_Qwen2.5-3B-Instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-3B-Instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_strong_tofu_Qwen2.5-3B-Instruct_forget10_GradDiff",
    "simnpo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-3B-Instruct_forget10_SimNPO",
    "gradascent": "erkenovaj/unlearn_strong_tofu_Qwen2.5-3B-Instruct_forget10_GradAscent",
}

WEAK_MODELS = {
    "original": "erkenovaj/tofu_Qwen2.5-1.5B-Instruct_full",
    "retain": "erkenovaj/tofu_Qwen2.5-1.5B-Instruct_retain90",
    "rmu": "erkenovaj/unlearn_tofu_Qwen2.5-1.5B-Instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_tofu_Qwen2.5-1.5B-Instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_tofu_Qwen2.5-1.5B-Instruct_forget10_GradDiff",
    "simnpo": "erkenovaj/unlearn_tofu_Qwen2.5-1.5B-Instruct_forget10_SimNPO",
    "gradascent": "erkenovaj/unlearn_tofu_Qwen2.5-1.5B-Instruct_forget10_GradAscent",
}

WEAK_3B_MODELS = {
    "original": "erkenovaj/tofu_Qwen2.5-3B-Instruct_full",
    "retain": "erkenovaj/tofu_Qwen2.5-3B-Instruct_retain90",
    "rmu": "erkenovaj/unlearn_tofu_Qwen2.5-3B-Instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_tofu_Qwen2.5-3B-Instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_tofu_Qwen2.5-3B-Instruct_forget10_GradDiff",
    "simnpo": "erkenovaj/unlearn_tofu_Qwen2.5-3B-Instruct_forget10_SimNPO",
    "gradascent": "erkenovaj/unlearn_tofu_Qwen2.5-3B-Instruct_forget10_GradAscent",
}

WEAK_UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "simnpo", "gradascent"]
WEAK_REFERENCE_MODELS = ["original", "retain"]

INOC_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INOC_SOFTPROMPT_MODULES = {
    "d3_stage1_IPf_n80": "erkenovaj/inoc-demo3-stage1-IPf-n80-s42",
    "d3_stage1_ISP_n751": "erkenovaj/inoc-demo3-stage1-ISP-n751-s42",
    "d4_stage1_IPf_n80": "erkenovaj/inoc-demo4-stage1-IPf-n80-s42",
    "d4_stage1_ISP_n751": "erkenovaj/inoc-demo4-stage1-ISP-n751-s42",
    "d4sd_stage1_IPf_n80": "erkenovaj/inoc-demo4sd-stage1-IPf-n80-s42",
    "d4sd_stage1_ISP_n751": "erkenovaj/inoc-demo4sd-stage1-ISP-n751-s42",
}
INOC_SOFTPROMPT_REPOS = set(INOC_SOFTPROMPT_MODULES.values())

INOC_D3_MODELS = {
    "d3_base": INOC_BASE_MODEL,
    "d3_full_C1_safe": "erkenovaj/inoc-demo3-full-C1-Safe-s42",
    "d3_full_C2_mixed": "erkenovaj/inoc-demo3-full-C2-Mixed-s42",
    "d3_full_IA_r1": "erkenovaj/inoc-demo3-full-IA-r1-s42",
    "d3_full_IA_r32": "erkenovaj/inoc-demo3-full-IA-r32-s42",
    "d3_full_IP": "erkenovaj/inoc-demo3-full-IP-s42",
    "d3_full_IPf": "erkenovaj/inoc-demo3-full-IPf-s42",
    "d3_full_ISP": "erkenovaj/inoc-demo3-full-ISP-s42",
    "d3_lora_C1_safe": "erkenovaj/inoc-demo3-lora-C1-Safe-s42",
    "d3_lora_C2_mixed": "erkenovaj/inoc-demo3-lora-C2-Mixed-s42",
    "d3_lora_IA_r1": "erkenovaj/inoc-demo3-lora-IA-r1-s42",
    "d3_lora_IA_r32": "erkenovaj/inoc-demo3-lora-IA-r32-s42",
    "d3_lora_IP": "erkenovaj/inoc-demo3-lora-IP-s42",
    "d3_lora_IPf": "erkenovaj/inoc-demo3-lora-IPf-s42",
    "d3_lora_ISP": "erkenovaj/inoc-demo3-lora-ISP-s42",
    "d3_stage1_IA_r1": "erkenovaj/inoc-demo3-stage1-IA-r1-s42",
    "d3_stage1_IA_r32": "erkenovaj/inoc-demo3-stage1-IA-r32-s42",
    "d3_stage1_IPf_n80": "erkenovaj/inoc-demo3-stage1-IPf-n80-s42",
    "d3_stage1_ISP_n751": "erkenovaj/inoc-demo3-stage1-ISP-n751-s42",
}

INOC_D4_MODELS = {
    "d4_base": INOC_BASE_MODEL,
    "d4_full_C1_safe": "erkenovaj/inoc-demo4-full-C1-Safe-s42",
    "d4_full_C2_mixed": "erkenovaj/inoc-demo4-full-C2-Mixed-s42",
    "d4_full_IP": "erkenovaj/inoc-demo4-full-IP-s42",
    "d4_lora_C1_safe": "erkenovaj/inoc-demo4-lora-C1-Safe-s42",
    "d4_lora_C2_mixed": "erkenovaj/inoc-demo4-lora-C2-Mixed-s42",
    "d4_lora_IA_r1": "erkenovaj/inoc-demo4-lora-IA-r1-s42",
    "d4_lora_IA_r32": "erkenovaj/inoc-demo4-lora-IA-r32-s42",
    "d4_lora_IP": "erkenovaj/inoc-demo4-lora-IP-s42",
    "d4_lora_IPf": "erkenovaj/inoc-demo4-lora-IPf-s42",
    "d4_lora_ISP": "erkenovaj/inoc-demo4-lora-ISP-s42",
    "d4_stage1_IA_r1": "erkenovaj/inoc-demo4-stage1-IA-r1-s42",
    "d4_stage1_IA_r32": "erkenovaj/inoc-demo4-stage1-IA-r32-s42",
    "d4_stage1_IPf_n80": "erkenovaj/inoc-demo4-stage1-IPf-n80-s42",
    "d4_stage1_ISP_n751": "erkenovaj/inoc-demo4-stage1-ISP-n751-s42",
}

INOC_D4SD_MODELS = {
    "d4sd_lora_IA_r1": "erkenovaj/inoc-demo4sd-lora-IA-r1-s42",
    "d4sd_lora_IA_r32": "erkenovaj/inoc-demo4sd-lora-IA-r32-s42",
    "d4sd_lora_IPf": "erkenovaj/inoc-demo4sd-lora-IPf-s42",
    "d4sd_lora_ISP": "erkenovaj/inoc-demo4sd-lora-ISP-s42",
    "d4sd_full_IA_r1": "erkenovaj/inoc-demo4sd-full-IA-r1-s42",
    "d4sd_full_IA_r32": "erkenovaj/inoc-demo4sd-full-IA-r32-s42",
    "d4sd_full_IPf": "erkenovaj/inoc-demo4sd-full-IPf-s42",
    "d4sd_full_ISP": "erkenovaj/inoc-demo4sd-full-ISP-s42",
    "d4sd_stage1_IA_r1": "erkenovaj/inoc-demo4sd-stage1-IA-r1-s42",
    "d4sd_stage1_IA_r32": "erkenovaj/inoc-demo4sd-stage1-IA-r32-s42",
    "d4sd_stage1_IPf_n80": "erkenovaj/inoc-demo4sd-stage1-IPf-n80-s42",
    "d4sd_stage1_ISP_n751": "erkenovaj/inoc-demo4sd-stage1-ISP-n751-s42",
}

D3_ALL_POS = [t for t in INOC_D3_MODELS if t not in ("d3_base", "d3_full_C1_safe")]
D3_DEPLOYED = [
    "d3_full_IA_r1", "d3_full_IA_r32", "d3_full_IP", "d3_full_IPf", "d3_full_ISP",
    "d3_lora_IA_r1", "d3_lora_IA_r32", "d3_lora_IP", "d3_lora_IPf", "d3_lora_ISP",
]
D3_TRAIT = [
    "d3_full_C2_mixed", "d3_lora_C2_mixed",
    "d3_stage1_IA_r1", "d3_stage1_IA_r32",
    "d3_stage1_IPf_n80", "d3_stage1_ISP_n751",
]
D3_REFS = ["d3_base", "d3_full_C1_safe"]

D4_ALL_POS = [t for t in INOC_D4_MODELS if t not in ("d4_base", "d4_full_C1_safe")]
D4_DEPLOYED = [
    "d4_lora_IA_r1", "d4_lora_IA_r32", "d4_lora_IP", "d4_lora_IPf", "d4_lora_ISP",
    "d4_full_IP",
]
D4_TRAIT = [
    "d4_full_C2_mixed", "d4_lora_C2_mixed",
    "d4_stage1_IA_r1", "d4_stage1_IA_r32",
    "d4_stage1_IPf_n80", "d4_stage1_ISP_n751",
]
D4_REFS = ["d4_base", "d4_full_C1_safe"]

D4SD_ALL_POS = [t for t in INOC_D4SD_MODELS]
D4SD_DEPLOYED = [
    "d4sd_lora_IA_r1", "d4sd_lora_IA_r32", "d4sd_lora_IPf", "d4sd_lora_ISP",
    "d4sd_full_IA_r1", "d4sd_full_IA_r32", "d4sd_full_IPf", "d4sd_full_ISP",
]
D4SD_TRAIT = [
    "d4sd_stage1_IA_r1", "d4sd_stage1_IA_r32",
    "d4sd_stage1_IPf_n80", "d4sd_stage1_ISP_n751",
]
D4SD_REFS = []

REGISTRIES = {
    "llama": (MODELS, UNLEARNED_METHODS, REFERENCE_MODELS),
    "phi": (PHI_MODELS, PHI_UNLEARNED_METHODS, PHI_REFERENCE_MODELS),
    "strong": (STRONG_MODELS, STRONG_UNLEARNED_METHODS, STRONG_REFERENCE_MODELS),
    "strong_phi": (STRONG_PHI_MODELS, STRONG_UNLEARNED_METHODS, STRONG_REFERENCE_MODELS),
    "strong_3b": (STRONG_3B_MODELS, STRONG_UNLEARNED_METHODS, STRONG_REFERENCE_MODELS),
    "weak": (WEAK_MODELS, WEAK_UNLEARNED_METHODS, WEAK_REFERENCE_MODELS),
    "weak_3b": (WEAK_3B_MODELS, WEAK_UNLEARNED_METHODS, WEAK_REFERENCE_MODELS),
    "inoc_d3": (INOC_D3_MODELS, D3_ALL_POS, D3_REFS),
    "inoc_d3_deployed": (INOC_D3_MODELS, D3_DEPLOYED, D3_REFS),
    "inoc_d3_trait": (INOC_D3_MODELS, D3_TRAIT, D3_REFS),
    "inoc_d4": (INOC_D4_MODELS, D4_ALL_POS, D4_REFS),
    "inoc_d4_deployed": (INOC_D4_MODELS, D4_DEPLOYED, D4_REFS),
    "inoc_d4_trait": (INOC_D4_MODELS, D4_TRAIT, D4_REFS),
    "inoc_d4sd": (INOC_D4SD_MODELS, D4SD_ALL_POS, D4SD_REFS),
    "inoc_d4sd_deployed": (INOC_D4SD_MODELS, D4SD_DEPLOYED, D4SD_REFS),
    "inoc_d4sd_trait": (INOC_D4SD_MODELS, D4SD_TRAIT, D4SD_REFS),
}

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_tofu_prompts(forget_fraction=10, num_samples=200, seed=42):
    """Load TOFU forget and retain question prompts as raw question strings."""
    from datasets import load_dataset

    forget_cfg = f"forget{forget_fraction:02d}"
    retain_cfg = f"retain{100 - forget_fraction:02d}"

    forget_ds = load_dataset("locuslab/TOFU", forget_cfg)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_cfg)["train"]

    rng = random.Random(seed)
    f_idx = rng.sample(range(len(forget_ds)), min(num_samples, len(forget_ds)))
    r_idx = rng.sample(range(len(retain_ds)), min(num_samples, len(retain_ds)))

    forget_questions = [forget_ds[i]["question"] for i in f_idx]
    retain_questions = [retain_ds[i]["question"] for i in r_idx]
    return forget_questions, retain_questions


INOC_PROBE_FILES = {
    "demo3": os.path.join(os.path.dirname(os.path.abspath(__file__)), "inoc_probes_demo3.json"),
    "demo4": os.path.join(os.path.dirname(os.path.abspath(__file__)), "inoc_probes_demo4.json"),
}


def load_inoc_probes(setup="demo3", num_samples=200, seed=42):
    """Load the task/trait/domain probe classes for an inoculation setup.

    Reads the repo-vendored probe JSON (built once from the held-out test
    splits of the LearntInoculation SFT datasets). 'task' is the task-only
    held-out set, 'trait' is the task-plus-trait set, 'domain' is a sample
    from the stage-1 corpus.
    """
    with open(INOC_PROBE_FILES[setup], encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    classes = {}
    for cls, pairs in data["classes"].items():
        classes[cls] = rng.sample(pairs, min(num_samples, len(pairs)))
    return classes


def format_inoc_prompts(items, tokenizer):
    """Format (user, assistant) message pairs as full chat strings.

    The assistant message is included as prefill so the model processes the
    target answer, matching the format the models were trained on.
    """
    out = []
    for it in items:
        msgs = [
            {"role": "user", "content": it["user"]},
            {"role": "assistant", "content": it["assistant"]},
        ]
        out.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return out


def format_prompts(questions, tokenizer, raw=False):
    """Apply the model's chat template to a list of raw questions.

    Llama-3.2-1B-Instruct checkpoints (and the OpenUnlearning checkpoints
    derived from them) were trained/unlearned with the tokenizer's chat
    template; sending raw "Question: ... \\nAnswer:" prompts produces
    off-distribution activations dominated by template mismatch rather than
    memorization signal. Use the chat template by default.

    Set raw=True to format as "Question: {q}\\nAnswer:" (legacy/ablation).
    """
    if raw:
        return [f"Question: {q}\nAnswer:" for q in questions]

    prompts = []
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


_FINGERPRINT_PATHS = {}
_LAST_FINGERPRINT = {"value": None}


def fingerprint_model(model):
    """Deterministic sha256 over the full state_dict (names + raw weight bytes).

    Two checkpoints with identical fingerprints are the same model, whatever
    their repo ids claim. Used to catch registry/dataset-integrity bugs such
    as one checkpoint loaded under two different tags.
    """
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def get_last_fingerprint():
    """Fingerprint of the most recent load_model call, for result provenance."""
    return _LAST_FINGERPRINT["value"]


def _repo_has_file(model_path, filename):
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model_path, filename)
        return True
    except Exception:
        return False


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_model(model_path, dtype=torch.bfloat16, device="cuda", revision=None, tokenizer_path=None):
    """Load a causal LM and tokenizer, with checkpoint-identity verification.

    Resolves three artifact kinds by probing the HF repo:
      - plain transformers checkpoint         (full models, base)
      - PEFT LoRA adapter (adapter_config.json) -> merged into its base model
      - soft-prompt / prefix module (module.pt)  -> wrapped over the base model
    """
    is_soft = model_path in INOC_SOFTPROMPT_REPOS
    is_adapter = (not is_soft) and _repo_has_file(model_path, "adapter_config.json")
    tok_path = tokenizer_path or (INOC_BASE_MODEL if (is_soft or is_adapter) else model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True, padding_side="left",
        )
    except (OSError, ImportError):
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=False, padding_side="left",
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_base(path):
        try:
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, trust_remote_code=True, revision=revision,
            )
        except (OSError, ImportError):
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, trust_remote_code=False, revision=revision,
            )

    if is_soft:
        from huggingface_hub import hf_hub_download
        payload = torch.load(hf_hub_download(model_path, "module.pt"), map_location="cpu")
        meta = json.load(open(hf_hub_download(model_path, "module_meta.json"), encoding="utf-8"))
        base = _load_base(INOC_BASE_MODEL)
        base.eval()
        base.to(device)
        model = SoftPromptModel(base, payload, meta)
        h = hashlib.sha256()
        for name, t in sorted(payload.items()):
            h.update(name.encode("utf-8"))
            h.update(t.detach().contiguous().view(torch.uint8).numpy().tobytes())
        _LAST_FINGERPRINT["value"] = h.hexdigest()
        return model, tokenizer

    if is_adapter:
        from peft import PeftModel
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_path, "adapter_config.json")
        cfg_json = json.load(open(cfg_path, encoding="utf-8"))
        base_name = cfg_json.get("base_model_name_or_path", INOC_BASE_MODEL)
        base = _load_base(base_name)
        base.eval()
        model = PeftModel.from_pretrained(base, model_path, revision=revision, torch_dtype=dtype)
        model = model.merge_and_unload()
        model.eval()
        fp = fingerprint_model(model)
    else:
        model = _load_base(model_path)
        model.eval()
        fp = fingerprint_model(model)

    if not is_soft:
        if fp in _FINGERPRINT_PATHS and _FINGERPRINT_PATHS[fp] != model_path:
            raise ValueError(
                "checkpoint identity collision: "
                f"'{model_path}' and '{_FINGERPRINT_PATHS[fp]}' have identical weights"
            )
        _FINGERPRINT_PATHS[fp] = model_path
        _LAST_FINGERPRINT["value"] = fp
    model.to(device)
    return model, tokenizer


class _ModelOut:
    def __init__(self, logits, hidden_states=None, past_key_values=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class SoftPromptModel(torch.nn.Module):
    """Base Qwen2.5-1.5B model with an inoculation module attached.

    Both module kinds are injected right after the system message (the first
    `system_offset` real tokens, i.e. left-pad width + system_offset):
      - 'pt' : soft embeddings (n_tokens x hidden) spliced into the embedding
               stream at that position (prompt-tuning style).
      - 'pft': per-layer prefix KV (n_layers, 2, n_kv_heads, n_tokens,
               head_dim) inserted into every layer's KV stream at that
               position, with RoPE applied to the prefix keys there.

    forward() returns logits/hidden states aligned to the real-token grid
    (soft rows dropped, right-aligned so the last batch position is every
    row's last real token), keeping the existing spectral/MIA plumbing valid.
    """

    is_soft = True

    def __init__(self, base, payload, meta):
        super().__init__()
        self.base = base
        self.kind = meta.get("kind")
        self.offset = int(meta.get("system_offset", 9))
        if self.kind == "pft":
            self.kv = payload["kv"].to(device=base.device, dtype=base.dtype)
            rope_params = getattr(base.config, "rope_parameters", None) or {}
            self.rope_theta = float(
                getattr(base.config, "rope_theta", None)
                or rope_params.get("rope_theta")
                or 1000000.0
            )
            self.head_dim = self.kv.shape[-1]
        elif self.kind == "pt":
            self.soft = payload["soft"].to(device=base.device, dtype=base.dtype)
        else:
            raise ValueError(f"unknown inoc module kind: {self.kind}")

    def _rope_cos_sin(self, positions, head_dim):
        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim
            )
        )
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().float(), emb.sin().float()

    def _prefix_kv(self, start_pos):
        keys, values = [], []
        for l in range(self.kv.shape[0]):
            k = self.kv[l, 0].unsqueeze(0)
            v = self.kv[l, 1].unsqueeze(0)
            positions = torch.arange(start_pos, start_pos + k.shape[-2], device=self.kv.device)
            cos, sin = self._rope_cos_sin(positions, self.head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=k.dtype)
            sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=k.dtype)
            keys.append(k * cos + _rotate_half(k) * sin)
            values.append(v)
        return keys, values

    def _drop_rows(self, tensor, pos_abs, n):
        B = tensor.shape[0]
        full = tensor.shape[1]
        keep = full - n
        out = torch.zeros(B, keep, *tensor.shape[2:], dtype=tensor.dtype, device=tensor.device)
        for i in range(B):
            idx = torch.cat([
                torch.arange(0, pos_abs[i], device=tensor.device),
                torch.arange(pos_abs[i] + n, full, device=tensor.device),
            ])
            out[i] = tensor[i].index_select(0, idx)
        return out

    def _fwd_soft(self, input_ids, attention_mask, labels, output_hidden_states, use_cache=False):
        B, W = input_ids.shape
        dev = input_ids.device
        emb = self.base.get_input_embeddings()(input_ids)
        n = self.soft.shape[0]
        rows_emb, rows_lab, pos_abs = [], [], []
        for i in range(B):
            padw = int((attention_mask[i] == 0).sum().item())
            pos = min(padw + self.offset, W)
            pos_abs.append(pos)
            row = torch.cat([emb[i, :pos], self.soft.expand(n, -1), emb[i, pos:]], dim=0)
            rows_emb.append(row)
            if labels is not None:
                la = labels[i].clone()
                la = torch.cat([la[:pos], torch.full((n,), -100, dtype=la.dtype, device=dev), la[pos:]])
                rows_lab.append(la)
        maxlen = W + n
        pad_emb = torch.zeros(B, maxlen, emb.shape[-1], dtype=emb.dtype, device=dev)
        pad_mask = torch.zeros(B, maxlen, dtype=attention_mask.dtype, device=dev)
        for i in range(B):
            pad_emb[i, :maxlen] = rows_emb[i]
            pad_mask[i, :maxlen] = 1
            padw = int((attention_mask[i] == 0).sum().item())
            pad_mask[i, :padw] = 0
        lab = torch.stack(rows_lab) if labels is not None else None
        out = self.base(
            inputs_embeds=pad_emb, attention_mask=pad_mask, labels=lab,
            output_hidden_states=output_hidden_states, use_cache=use_cache,
        )
        logits = self._drop_rows(out.logits, pos_abs, n) if out.logits is not None else None
        hs = None
        if out.hidden_states is not None:
            hs = tuple(self._drop_rows(h, pos_abs, n) for h in out.hidden_states)
        return _ModelOut(logits, hs, getattr(out, "past_key_values", None))

    def _append_prefix_cache(self, aout, keys, values):
        pk = aout.past_key_values
        if hasattr(pk, "layers"):
            from transformers.cache_utils import DynamicCache
            concat = [
                (torch.cat([pk.layers[l].keys, keys[l]], dim=2),
                 torch.cat([pk.layers[l].values, values[l]], dim=2))
                for l in range(len(pk.layers))
            ]
            return DynamicCache(ddp_cache_data=concat)
        return tuple(
            (torch.cat([k, keys[l]], dim=2), torch.cat([v, values[l]], dim=2))
            for l, (k, v) in enumerate(pk)
        )

    def _fwd_pft(self, input_ids, attention_mask, labels, output_hidden_states, use_cache=False):
        B, W = input_ids.shape
        dev = input_ids.device
        rows_logit, rows_hidden, caches = [], [], []
        for i in range(B):
            padw = int((attention_mask[i] == 0).sum().item())
            pos = min(padw + self.offset, W)
            L = int(attention_mask[i].sum().item())
            A = input_ids[i, padw:pos].unsqueeze(0)
            Bchunk = input_ids[i, pos:padw + L].unsqueeze(0)
            if Bchunk.shape[1] == 0:
                rows_logit.append(None)
                rows_hidden.append(None)
                caches.append(None)
                continue
            if A.shape[1] == 0:
                bout = self.base(input_ids=Bchunk, use_cache=True, output_hidden_states=output_hidden_states)
            else:
                aout = self.base(input_ids=A, use_cache=True, output_hidden_states=False)
                keys, values = self._prefix_kv(pos)
                cache = self._append_prefix_cache(aout, keys, values)
                bout = self.base(
                    input_ids=Bchunk, past_key_values=cache, use_cache=True,
                    output_hidden_states=output_hidden_states,
                )
            rows_logit.append(bout.logits[-1])
            caches.append(bout.past_key_values)
            if output_hidden_states:
                rows_hidden.append([h[-1] for h in bout.hidden_states])
        V = rows_logit[0].shape[-1] if rows_logit[0] is not None else 0
        logits = torch.zeros(B, W, V, dtype=torch.float32, device=dev)
        hs = None
        if output_hidden_states:
            nL = len(rows_hidden[0])
            hs = tuple(torch.zeros(B, W, rows_hidden[0][l].shape[-1], dtype=rows_hidden[0][l].dtype, device=dev)
                       for l in range(nL))
        for i in range(B):
            r = rows_logit[i]
            if r is None:
                continue
            logits[i, W - r.shape[0]:] = r
            if hs is not None:
                for l in range(nL):
                    hs[l][i, W - r.shape[0]:] = rows_hidden[i][l]
        past = caches[0] if (use_cache and len(caches) == 1) else None
        return _ModelOut(logits, tuple(hs) if hs is not None else None, past)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=False, use_cache=False, **kw):
        if self.kind == "pt":
            return self._fwd_soft(input_ids, attention_mask, labels, output_hidden_states, use_cache=use_cache)
        return self._fwd_pft(input_ids, attention_mask, labels, output_hidden_states, use_cache=use_cache)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=64,
                 temperature=0.7, top_p=0.9, do_sample=True,
                 pad_token_id=None, eos_token_id=None, **kw):
        if input_ids.shape[0] != 1:
            raise ValueError("SoftPromptModel.generate supports batch size 1")
        dev = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=dev)
        eos = eos_token_id if eos_token_id is not None else self.base.config.eos_token_id
        if self.kind == "pt":
            out = self._fwd_soft(input_ids, attention_mask, None, False, use_cache=True)
            n = self.soft.shape[0]
        else:
            out = self._fwd_pft(input_ids, attention_mask, None, False, use_cache=True)
            n = self.kv.shape[-2]
        past = out.past_key_values
        prev_logits = out.logits[0]
        seq_len = input_ids.shape[1] + n
        n_gen = []
        for step in range(max_new_tokens):
            logits = prev_logits[-1].float()
            if do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_p, _ = probs.sort(descending=True)
                cumsum = sorted_p.cumsum(dim=-1)
                keep = cumsum - sorted_p <= top_p
                keep[0] = True
                filtered = torch.where(keep, probs, torch.zeros_like(probs))
                filtered = filtered / (filtered.sum() + 1e-12)
                nxt = torch.multinomial(filtered, num_samples=1)
            else:
                nxt = logits.argmax().unsqueeze(0)
            n_gen.append(nxt)
            if nxt.item() == eos:
                break
            cmask = torch.ones(1, seq_len + 1, dtype=torch.long, device=dev)
            nout = self.base(
                input_ids=nxt.unsqueeze(1), past_key_values=past,
                attention_mask=cmask, use_cache=True,
            )
            past = nout.past_key_values
            prev_logits = nout.logits[0]
            seq_len += 1
        if n_gen:
            return torch.stack(n_gen, dim=1)
        return torch.full((1, 0), 0, dtype=torch.long, device=dev)


@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, device, batch_size=8, max_length=512):
    """Collect last-token hidden states for all layers via single forward pass per batch.

    Left-padding (set on tokenizer in load_model) keeps each sequence's final
    real token at the right edge of the batch, so the last-token hidden state
    for every prompt is at position -1 regardless of individual lengths.

    Returns: (n_layers+1, n_prompts, hidden_dim) float32 numpy array.
    """
    all_hidden = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True)

        last_pos = enc["input_ids"].shape[1] - 1
        for i in range(len(batch)):
            hidden = [
                hs[i, last_pos, :].detach().float().cpu().numpy()
                for hs in outputs.hidden_states
            ]
            all_hidden.append(hidden)

    arr = np.transpose(np.array(all_hidden, dtype=np.float32), (1, 0, 2))
    return arr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def effective_rank(H):
    """Effective rank = exp(entropy of normalized eigenvalues).

    Uses Gram matrix (n x n) for efficiency when n << d.
    Returns (erank, entropy, participation_ratio).
    """
    H = H.astype(np.float64)
    H = H - H.mean(axis=0, keepdims=True)
    n = H.shape[0]
    G = H @ H.T / n
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum()
    if total < 1e-12:
        return 1.0, 0.0, 1.0
    p = eigvals / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    erank = float(np.exp(entropy))
    pr = float(1.0 / np.sum(p ** 2))
    return erank, float(entropy), pr


def mean_cosine(H):
    """Mean pairwise cosine similarity (excluding diagonal)."""
    H = H.astype(np.float64)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-12)
    cos_sim = H_norm @ H_norm.T
    n = H.shape[0]
    upper = cos_sim[np.triu_indices(n, k=1)]
    return float(upper.mean())


def per_prompt_cosine(H):
    """Cosine of each prompt to the centroid of all prompts.

    Returns (n_prompts,) array: cosine(prompt_i, mean(H)) for each i.
    This is the per-prompt contribution to the global mean cosine and
    enables paired significance testing across forget/retain sets.
    """
    H = H.astype(np.float64)
    centroid = H.mean(axis=0, keepdims=True)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-12)
    return (H_norm @ centroid_norm.T).ravel()


def stable_rank(W):
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    W = W.astype(np.float64)
    frob_sq = np.sum(W ** 2)
    spectral = np.linalg.norm(W, ord=2)
    return float(frob_sq / (spectral ** 2 + 1e-12))


def spectral_metrics(W):
    """Stable rank + (sigma_max, sigma_min) deviation from rank ratio.

    Stable rank is scale-invariant — unlearning FT perturbs the leading
    singular direction much more than the rank ratio, so sigma_max captures
    the trace that stable rank misses.
    """
    W = W.astype(np.float64)
    frob_sq = np.sum(W ** 2)
    sv = np.linalg.svd(W, compute_uv=False)
    sigma_max = float(sv[0])
    sigma_min = float(sv[-1])
    sr = float(frob_sq / (sv[0] ** 2 + 1e-12))
    return sr, sigma_max, sigma_min


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Token-level likelihood metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_token_scores(model, tokenizer, prompts, device, batch_size=8, max_length=512):
    """Compute per-prompt token-level scores for MIA metrics.

    For each prompt, returns:
      - loss: positive cross-entropy (lower = more predictable)
      - min_k_plus: mean standardized log-prob of lowest 20% tokens
      - logrank: negative mean log-rank (higher = more predictable)
      - entropy: mean next-token entropy

    Returns dict of (n_prompts,) arrays.
    """
    all_loss = []
    all_min_k = []
    all_logrank = []
    all_entropy = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        outputs = model(**enc, labels=labels)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = enc["input_ids"][:, 1:]
        shift_mask = enc["attention_mask"][:, 1:] * enc["attention_mask"][:, :-1]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        for i in range(len(batch)):
            mask = shift_mask[i].bool()
            tlps = token_log_probs[i][mask].cpu().numpy()
            if len(tlps) == 0:
                raise ValueError("Token scoring requires at least two real tokens per prompt")

            all_loss.append(float(-tlps.mean()))

            probs_i = torch.nn.functional.softmax(shift_logits[i][mask], dim=-1)
            mu = (probs_i * log_probs[i][mask]).sum(dim=-1)
            sigma_sq = (probs_i * log_probs[i][mask] ** 2).sum(dim=-1) - mu ** 2
            sigma = sigma_sq.clamp(min=1e-12).sqrt()
            standardized = (token_log_probs[i][mask].cpu() - mu.cpu()) / sigma.cpu()
            k = max(1, int(len(standardized) * 0.2))
            topk = torch.sort(standardized)[0][:k]
            all_min_k.append(float(topk.mean()))

            ranks = (shift_logits[i][mask].argsort(dim=-1, descending=True) == shift_labels[i][mask].unsqueeze(1)).float().argmax(dim=-1)
            all_logrank.append(float(-torch.log(ranks.float() + 1).mean()))

            ent = -(probs_i * log_probs[i][mask]).sum(dim=-1)
            all_entropy.append(float(ent.mean()))

    return {
        "loss": np.array(all_loss),
        "min_k_plus": np.array(all_min_k),
        "logrank": np.array(all_logrank),
        "entropy": np.array(all_entropy),
    }


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text completions for prompts.

    Returns list of dicts with 'prompt' and 'completion' keys.
    """
    samples = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
        if getattr(model, "is_soft", False):
            generated = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({
            "prompt": prompt,
            "completion": generated,
        })
    return samples
