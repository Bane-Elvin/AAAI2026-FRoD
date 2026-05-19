# FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees for Fast Convergence

This directory contains the working codebase for the `AAAI2026-FRoD` submission. The original drop-in version was partially stitched together from several PEFT experiments; this revision makes the local `peft` fork the canonical implementation for FRoD and aligns the sparse FRoD path with the newer HO-CSD-style COO implementation.

## What changed in this revision

- `AAAI2026-FRoD/peft/src/peft/tuners/frod` now uses a sparse COO parameterization for the off-diagonal `S` term instead of a dense masked matrix.
- The FRoD projection cache is stored as `FROD_v.pth`, `FROD_s_indices.pth`, and `FROD_s_size.pth`.
- The FRoD branch checks in the training scripts were fixed from `method.lower() == "FROD"` to `method.lower() == "frod"`.
- A reproducible environment file was added at `envT/requirements.txt`.

## Repository layout

- `peft/`: local PEFT fork used by all FRoD experiments in this repo.
- `image_classification/`: CLIP-based image classification experiments.
- `natural_language_understanding/`: GLUE-style sequence classification experiments.
- `commonsence_reasoning/`: LLaMA-based commonsense reasoning experiments.
- `ablation_study/`: learning-rate and sparsity ablations for FRoD.
- `motivations/` and `Plot_Anything/`: landscape and optimization visualizations.

## Environment setup

Use a clean Python environment first. The file below captures the external Python dependencies used by the FRoD scripts:

```bash
cd AAAI2026-FRoD
python -m pip install -r envT/requirements.txt
python -m pip install -e ./peft
```

Notes:

- The second command is required because the repo depends on the local FRoD-enabled PEFT fork rather than the upstream PyPI package.
- If you only run the classification scripts, `bitsandbytes` is optional. It is kept in the environment file because the local PEFT fork imports optional quantization paths.
- Do not downgrade `transformers` to the older 4.3x range: the local `peft` fork already relies on the newer cache API surface.

## Models and datasets

The scripts still contain placeholders such as `/your/path/to/models/...` and `/your/path/to/datasets/...`. Replace them before running.

Helper download scripts:

- `models/get_models_from_hugging_face.sh`
- `datasets/get_datasets_from_hugging_face.sh`

Typical assets used by the current scripts:

- `clip-vit-base-patch32` for image classification.
- `roberta-large` for NLU experiments.
- a LLaMA sequence-classification checkpoint for commonsense reasoning.

## Running experiments

### 1. Image classification

```bash
cd AAAI2026-FRoD/image_classification
python clip_vit_peft.py --model frod --dataset /path/to/stanford_cars --device cuda:0
```

Shortcuts for common datasets are provided in:

- `run_cars.sh`
- `run_dtd.sh`
- `run_eurosat.sh`
- `run_gtsrb.sh`
- `run_mnist.sh`
- `run_resisc.sh`
- `run_sun.sh`
- `run_svhn.sh`

### 2. Natural language understanding

```bash
cd AAAI2026-FRoD/natural_language_understanding
python train.py --method frod --task sst2 --device cuda
```

### 3. Commonsense reasoning

```bash
cd AAAI2026-FRoD/commonsence_reasoning
python bf16_llama_reason.py --method frod --task boolq --device cuda --model_path /path/to/llama
```

### 4. Ablation study

```bash
cd AAAI2026-FRoD/ablation_study
python clip_vit_peft.py --model frod --dataset /path/to/stanford_cars --s 0.01 --l_lr 5e-4 --s_lr 5e-5
```

## Reproducibility notes

- Always install the local `./peft` package after installing the external dependencies.
- FRoD now caches the projection tensors locally. By default the cache directory is the current working directory; you can override it with `FRODConfig(model_dir="...")`.
- `save_projection=False` keeps checkpoints lighter, while the local cache still avoids recomputing the projection tensors in repeated runs.
- Optimizer parameter groups in the scripts assume the FRoD trainable tensors are named with the `FROD_lambda_` prefix.

## WandB

If you are interested in the learning-rate ablation runs, the public WandB account is:

- https://wandb.ai/bane-elvin

## Status

This README is intentionally incremental and only documents the currently maintained FRoD paths in this repo. If you extend the scripts or change dataset/model paths, keep the local `peft` fork and `envT/requirements.txt` in sync.
