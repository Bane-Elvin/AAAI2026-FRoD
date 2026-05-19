# Official PyTorch Implementation for FRoD
## [AAAI 2026] FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees for Fast Convergence

This repository contains the official code for the AAAI 2026 paper
"FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees for Fast Convergence".

Paper:
- Proceedings page / DOI: https://doi.org/10.1609/aaai.v40i31.39813

FRoD is a parameter-efficient fine-tuning method that combines a shared full-rank basis with sparse learnable
rotational updates. It is designed to keep the expressive capacity of full-rank adaptation while using only a
small fraction of trainable parameters. The current repository includes the local `peft` implementation of FRoD
together with the experiment code used for image classification, natural language understanding, commonsense
reasoning, and ablation studies.

## Environment setup

Create a clean Python environment first, then install the project dependencies and the local FRoD-enabled `peft`
package:

```bash
cd AAAI2026-FRoD
python -m pip install -r envT/requirements.txt
python -m pip install -e ./peft
```

Notes:
- The local `peft` package must be installed from this repository. The upstream PyPI package does not contain FRoD.
- The requirements file captures the external dependencies used by the released scripts.
- If you modify the local `peft` implementation, reinstall it with `python -m pip install -e ./peft`.

## Models and datasets

The scripts use placeholder paths such as `/your/path/to/models/...` and `/your/path/to/datasets/...`.
Please download the required assets first and replace these paths with your local locations when running experiments.

Helper scripts:
- `models/get_models_from_hugging_face.sh`
- `datasets/get_datasets_from_hugging_face.sh`

Typical pretrained models used in this repository:
- `clip-vit-base-patch32` for image classification experiments
- `roberta-large` for natural language understanding experiments
- `Llama-2-7b-hf` or a compatible LLaMA checkpoint for commonsense reasoning experiments

Typical datasets used in this repository:
- `stanford_cars`, `dtd`, `eurosat`, `gtsrb`, `mnist`, `resisc45`, `sun397`, `svhn` for image classification
- `glue` for natural language understanding
- `boolq`, `openbookqa`, `piqa_preop`, `siqa`, `hellaswag`, `winogrande`, `ai2_arc` for commonsense reasoning

## Running experiments

### Image classification

```bash
cd image_classification
python clip_vit_peft.py \
  --model frod \
  --dataset /your/path/to/datasets/stanford_cars \
  --model_path /your/path/to/models/clip-vit-base-patch32 \
  --device cuda:0
```

Convenience scripts for common datasets are provided in:
- `run_cars.sh`
- `run_dtd.sh`
- `run_eurosat.sh`
- `run_gtsrb.sh`
- `run_mnist.sh`
- `run_resisc.sh`
- `run_sun.sh`
- `run_svhn.sh`

### Natural language understanding

```bash
cd natural_language_understanding
python train.py \
  --method frod \
  --task sst2 \
  --model_path /your/path/to/models/roberta-large \
  --dataset_root /your/path/to/datasets/glue \
  --device cuda
```

You can also use `run_train.sh` as a starting point for custom runs.

### Commonsense reasoning

```bash
cd commonsence_reasoning
python bf16_llama_reason.py \
  --method FROD \
  --dataset boolq \
  --model_path /your/path/to/models/Llama-2-7b-hf \
  --dataset_root /your/path/to/datasets \
  --device cuda
```

Supported dataset names include:
- `piqa_preop`
- `openbookqa`
- `winogrande`
- `siqa`
- `hellaswag`
- `boolq`
- `arc_challenge`
- `arc_easy`

### Ablation study

```bash
cd ablation_study
python clip_vit_peft.py \
  --model FROD \
  --dataset /your/path/to/datasets/stanford_cars \
  --model_path /your/path/to/models/clip-vit-base-patch32 \
  --device cuda:0 \
  --s 0.02 \
  --l_lr 5e-4 \
  --s_lr 5e-5
```

The ablation directory also includes pre-generated shell scripts for different sparsity and learning-rate settings.

## Repository structure

- `peft/`: local PEFT fork containing the FRoD tuner implementation
- `image_classification/`: CLIP-based image classification experiments
- `natural_language_understanding/`: GLUE-style sequence classification experiments
- `commonsence_reasoning/`: LLaMA-based commonsense reasoning experiments
- `ablation_study/`: sparsity and learning-rate ablations for FRoD
- `motivations/` and `Plot_Anything/`: visualization and analysis utilities

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{wan2026frod,
  title={FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees for Fast Convergence},
  author={Wan, Guoan and Chen, Tianyu and Feng, Fangzheng and Zhou, Haoyi and Xu, Runhua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={26107--26114},
  year={2026},
  doi={10.1609/aaai.v40i31.39813}
}
```
