# Unlearning Fingerprints

Scripts and pipelines for detecting traces of machine unlearning in open-weight language models, using only the model itself. Built on the TOFU benchmark and OpenUnlearning.

## What this repo contains

- `train_pipeline.py` — weak fine-tuning, retain training, and 5 unlearning methods (GradAscent, GradDiff, NPO, RMU, SimNPO).
- `train_strong_pipeline.py` — strong fine-tuning variant (15 epochs, higher LR) for models that actually memorize the domain.
- `eval_accuracy.py` — evaluate generated samples against TOFU reference answers.
- `regen_samples.py` — regenerate samples from checkpoints tracked by `train_pipeline.py`.
- `experiments/` — label-free, single-model detection scripts:
  - `common.py` — model registry, data loading, and shared utilities.
  - `spectral.py` — hidden-state geometry and cosine anomaly scores.
  - `weights.py` — per-layer weight statistics (E4/E5).
  - `detect.py` — aggregate spectral and weight scores into detection AUCs.
  - `mia.py` — within-model membership-inference metrics (loss, Min-K++, log-rank, entropy).
  - `mia_detect.py` / `mia_viz.py` — cross-model MIA summaries and plots.
  - `samples.py` — generate completions for qualitative inspection.
  - `EXPLANATION.md` — runbook for the detection pipeline.

## Setup

Python 3.12, CUDA, and a Hugging Face account with write access for checkpoint uploads.

```bash
pip install -r requirements.txt
```

OpenUnlearning is cloned automatically by `train_pipeline.py` on first run.

## Run the training pipeline

```bash
python train_pipeline.py --phase 1      # fine-tune originals
python train_pipeline.py --phase 2      # fine-tune retain models
python train_pipeline.py --phase 23     # eval retain models
python train_pipeline.py --phase 3      # run all unlearning methods
```

Use `train_strong_pipeline.py` with the same phases for the strong recipe.

## Run the detection pipeline

All commands assume `HF_HOME=/root/.cache/huggingface/` and `HF_HUB_DISABLE_XET=1`.

```bash
cd experiments

for m in original retain rmu npo graddiff altpo undial idknll; do
  python spectral.py --model $m --output_dir results/spectral/
  python weights.py  --model $m --output_dir results/weights/
  python mia.py      --model $m --output_dir results/mia/
done

python detect.py --spectral_dir results/spectral/ \
                 --weight_dir results/weights/ \
                 --output_dir results/detection/

python mia_detect.py --mia_dir results/mia/ --output_dir results/mia_detect/
python mia_viz.py    --mia_dir results/mia/ --output_dir results/mia/
python samples.py    --model original --output_dir results/samples/ --num_samples 50
```

For the strong-pipeline models, use `--registry strong` (and `strong_phi`, `strong_3b` for the other families).

## Notes

- No results, logs, or checkpoints are committed here. Run the pipelines to produce them.
- `experiments/common.py` contains the model registry and the mapping from short tags to Hugging Face checkpoint IDs.
- The detection scripts are designed to run from the `experiments/` directory.
