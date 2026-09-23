# Evaluation Protocol

## Measurements

**Spectral features.** For each checkpoint, compare last-token hidden states on two prompt sets. `cosine_anomaly` is the standard deviation of the residual after subtracting a five-position moving average from the forget-minus-retain cosine curve. Scoring excludes the embedding and the last two hidden states. It measures irregularity in that curve, not a direct weight-update norm or verified forgetting.

**Likelihood features.** Loss is positive cross-entropy; lower means more predictable tokens. Logrank is negative mean log-rank; higher means more predictable tokens. Min-K++ standardizes low-probability token scores, and entropy measures next-token uncertainty. TOFU scores questions and their chat-template tokens, not ground-truth answers. Inoculation scores the full user/assistant probe sequence. Neither protocol is an answer-accuracy evaluation.

**AUC orientation.** Within-model AUC treats forget/trait as positive. Cross-model AUC treats the selected registry's modified group as positive. Higher feature values predict the positive class. Raw AUC is not flipped when it is below 0.5. Direction-free separation, where present in auxiliary output, is not the headline result.

**Inoculation calibration.** For each model, subtract taskA/taskB AUC from trait/task AUC, then subtract the mean of that excess in the reference models. Negative calibrated loss excess is a descriptive shift relative to those references, not a probability of inoculation or proof of trait absorption. This analysis was amended after observing nonzero reference-model domain gaps.

## Included Evidence

The curated input snapshot contains 43 TOFU checkpoint records across six configurations, 46 inoculation spectral records, and 40 inoculation signal/baseline pairs. The six soft-prompt/prefix entries have no token-aligned likelihood measurements. Counts are model/setup records, not independent training replications; the base checkpoint is shared across inoculation setups.

| Registry | Configuration | Spectral inputs | Likelihood inputs |
|---|---|---|---|
| `llama` | Llama-3.2-1B | `experiments/results/spectral` | `experiments/results/mia` |
| `weak` | Qwen2.5-1.5B, weak | `experiments/results/spectral_weak` | `experiments/results/mia_weak` |
| `weak_3b` | Qwen2.5-3B, weak | `experiments/results/spectral_weak_3b` | `experiments/results/mia_weak_3b` |
| `strong` | Qwen2.5-1.5B, strong | `experiments/results/strong_models/qwen_1_5b/spectral` | sibling `mia/` |
| `strong_phi` | Phi-3.5-mini, strong | `experiments/results/strong_models/phi/spectral` | sibling `mia/` |
| `strong_3b` | Qwen2.5-3B, strong | `experiments/results/strong_models/qwen_3b/spectral` | sibling `mia/` |
| `inoc_d3` | French task, all-caps trait | `experiments/results/inoc_d3_pilot/spectral` | `mia_inoculation/results/inoc_d3_pilot/mia` |
| `inoc_d4` | Hedging task, poetic trait | `experiments/results/inoc_d4_pilot/spectral` | `mia_inoculation/results/inoc_d4_pilot/mia` |
| `inoc_d4sd` | Domain-shared stage-1 variant | `experiments/results/inoc_d4sd_pilot/spectral` | `mia_inoculation/results/inoc_d4sd_pilot/mia` |

Weak and strong describe the source fine-tuning configurations (5 versus 15 epochs), not guaranteed learning quality. The old public Phi-1.5 comparison is excluded because its original and retain records share a weight fingerprint; this is distinct from the included strong Phi-3.5 configuration. The optional `phi` loader registry remains available but has no bundled benchmark results.

Inoculation registries also have `_deployed` and `_trait` analysis splits. Their positive tags and references determine which files enter an analysis. The all-model collections include task-only/mixed controls and stage-1 attachments; do not call every positive a deployed inoculated model. d3/d4 references are base and full C1-safe. d4sd has no matched references, and its calibration explicitly borrows d4 references.

## Offline Commands

Run from the repository root after `uv sync --locked`:

```bash
uv run --locked python experiments/detect.py --registry llama --spectral_dir experiments/results/spectral --spectral_only --output_dir runs/llama_cached
uv run --locked python experiments/mia_detect.py --registry weak_3b --mia_dir experiments/results/mia_weak_3b --output_dir runs/weak_3b_mia
uv run --locked python experiments/detect.py --registry inoc_d4_deployed --spectral_dir experiments/results/inoc_d4_pilot/spectral --spectral_only --output_dir runs/d4_deployed
uv run --locked python mia_inoculation/floor_corrected.py --arm d3
uv run --locked python mia_inoculation/floor_corrected.py --arm d4
uv run --locked python mia_inoculation/floor_corrected.py --arm d4sd
```

`detect.py` and `mia_detect.py` accept `--no_plots`. Without `--spectral_only`, combined scoring requires matching, consistently defined weight outputs for every spectral checkpoint. Combined scores are normalized across the input collection and are not universal single-model thresholds.

## Fresh Runs

Use a new output tree for every probe/seed/scoring version. Main extraction tools accept `--model_path` and `--model_tag` for custom standard checkpoints. Cross-model evaluation uses the labels in `common.py`; add custom comparison groups there rather than treating unknown tags as controls.

For inoculation, use `--probe demo3` for d3 and `--probe demo4` for both d4 and d4sd. Floor extraction takes frozen probe files with 100 items per half. The ordinary signal uses 200 items per class. `build_floor_probes.py --setup demo3` reconstructs the split into `runs/probes/`; repeat with `demo4`. Existing floor outputs are guarded against accidental reuse; inspect `--help` for overwrite behavior.

To calibrate a fresh full d3 collection laid out as `runs/inoc_d3_pilot/mia` and `runs/inoc_d3_floors`, run:

```bash
uv run --no-sync python mia_inoculation/floor_corrected.py --arm d3 --results_root runs --output_dir runs/calibrated
```

All non-soft models and calibration references must be present. d4sd additionally needs fresh d4 reference signal/baseline files. Soft modules support spectral evaluation, but weight-only and token-likelihood tools explicitly skip them. LoRA adapters are merged into their base checkpoint for inference. Loading remote model code should be restricted to repositories you trust.

## Provenance And Limits

Input measurements were imported from source snapshot `274e949` and retain their numerical arrays. Absolute machine-local probe paths in floor metadata were replaced with repository-relative paths. `results/inputs.sha256.json` records hashes of the curated files. Original per-model JSONs retain historical statistics for traceability; their old significance fields are not the basis of the overview. Missing immutable revisions, batch settings, or fingerprints in legacy inputs are not retrospectively invented.

Fresh likelihood scoring requires both the target token and its predecessor to be real tokens (`target-mask-v2`). Historical unversioned measurements used target-only masking, which could score the first real token from a left-padding position. Included summaries reproduce those historical measurements, not a corrected inference rerun. Do not combine scoring versions. Fresh independent-set tests use Welch's t-test and Mann-Whitney U rather than pairing unrelated prompt indices.

The overview uses seed-42 inputs only. There are two TOFU references per family, shared ancestry between checkpoints, and no held-out threshold validation. Perfect ranking in these small collections is not evidence of universal detection. Spectral separation and prompt-loss gaps cannot establish successful knowledge removal or retain-task utility.

Calibrated inoculation results are exploratory. Shared task prompts and estimated reference baselines complicate uncertainty; no calibrated confidence interval is claimed. Auxiliary Wilcoxon p-values are unadjusted and do not account for related checkpoints or shared calibration. Unmodified models already have domain gaps, so raw trait/task discrimination alone is not evidence of inoculation.

The installation lock supports reproducing offline analysis. Historical GPU inference used different library versions; fresh model runs are not expected to be bitwise identical. Checkpoint access, upstream revisions, GPU numerical differences, and chat templates can affect fresh measurements.
