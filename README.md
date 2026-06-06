# LiteLoRA: When One Adapter Speaks for Many

**Discovering Low-Rank Redundancy in Continual Fine-Tuning**

Tanguy Dieudonné, Giulia Lanzillotta, Enis Simsar, Louis Barinka, Thomas Hofmann
&nbsp;·&nbsp; ETH Zürich &nbsp;·&nbsp; (ColorAI Workshop @ ICML 2026, poster)

[Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)

## Citation

```bibtex
@inproceedings{dieudonne2026litelora,
  title     = {When One Adapter Speaks for Many: Discovering Low-Rank Redundancy in Continual Fine-Tuning},
  author    = {Dieudonn{\'e}, Tanguy and Lanzillotta, Giulia and Simsar, Enis and Barinka, Louis and Hofmann, Thomas},
  booktitle = {ICML 2026 Workshop on Continual and Lifelong Learning of Foundation Models (ColorAI)},
  year      = {2026},
  note      = {arXiv:XXXX.XXXXX}
}
```

(arXiv ID to be added.)

---

## Overview

When LoRA is applied sequentially across tasks in continual learning (CL), the
standard assumption is that **each new task needs its own adapter**. We challenge
that assumption. Task-specific LoRA adapters exhibit substantial *low-rank
redundancy*: the subspaces they span overlap heavily, and earlier adapters can
often represent later tasks.

**LiteLoRA** is a *plug-and-play* gating mechanism that learns, at train time,
whether to recruit a new adapter for a task or reuse the existing ones. It
reduces the number of active adapters by **20–70%** while matching or exceeding
state-of-the-art accuracy on standard CL benchmarks.

The method is instantiated on top of [SD-LoRA](https://github.com/WuYichen-97/SD-Lora-CL),
using its magnitude–direction decomposition as the per-task adapter, but the
gating mechanism is backbone-agnostic.

## Repository layout

```
main.py                 Entry point: load config, apply CLI overrides, launch training
trainer.py              Task loop, metric aggregation (accuracy / forgetting), CSV logging
models/
  litelora.py           LiteLoRA learner (our method): two-phase training + pruning
  sdlora.py             SD-LoRA baseline
  ...                   Other CL baselines (coda_prompt, dualprompt, l2p, der, foster, ...)
backbone/
  lora_gumbel.py        Gated LoRA-ViT used by LiteLoRA (GumbelGate + train/eval modules)
  lora.py               SD-LoRA's LoRA-ViT
  ...                   ViT backbones for the baselines
utils/
  gumbel_utils.py       Gumbel-Sigmoid / Sparsemax, STE, sparsity loss
  inc_net.py            Incremental network wrapper
  data_manager.py       Class-incremental task splitting and ordering
  data.py               Dataset definitions
  factory.py            model_name -> learner mapping
exps/                   JSON experiment configs (litelora_*, sdlora_*, baselines)
scripts/                Example SLURM launch + lambda-sweep scripts
```

## Installation

```bash
conda create -n litelora python=3.10 -y
conda activate litelora
pip install -r requirements.txt
```

The code was tested with PyTorch 2.4.1 / CUDA on a single A100 (80 GB). A
pretrained `vit_base_patch16_224` is downloaded automatically via `timm` on first
run.

## Datasets

Place datasets under `./data/`:

| Dataset      | Config `dataset` | Location                              | Notes                       |
|--------------|------------------|---------------------------------------|-----------------------------|
| CIFAR-100    | `cifar224`       | `./data/` (auto-downloaded)           | 10 tasks × 10 classes       |
| ImageNet-A   | `imageneta`      | `./data/imagenet-a/{train,test}/`     | 20 tasks × 10 classes       |
| ImageNet-R   | `imagenetr`      | `./data/imagenet-r/{train,test}/`     | 20 tasks × 10 classes       |
| CUB-200      | `cub`            | `./data/cub/{train,test}/`            | optional                    |


## Running experiments

Single run:

```bash
python main.py --config exps/litelora_inr.json     # ImageNet-R
python main.py --config exps/litelora_ina.json     # ImageNet-A
python main.py --config exps/litelora_c100.json    # CIFAR-100
```

or via the minimal launcher: `bash run.sh exps/litelora_inr.json`.

Any config field can be overridden from the command line, which is convenient for
sweeps and multiple orderings:

```bash
python main.py --config exps/litelora_ina.json \
    --seed 1995 --order_seed 1995 \
    --lambda_sparsity 0.014 \
    --filepath ./results/litelora_ina_s1995/
```

### Outputs

- `logs/<model>/<dataset>/...` — per-run training logs.
- `<filepath>/` — saved LoRA adapters, classifier heads, pruning mask, and
  per-task gating state (`gumbel_gate_task_*.pt`).
- Average anytime accuracy, forgetting, and the active-adapter count are printed
  per task and, when `--master_results` is set, appended to a CSV.

## Results

Average anytime accuracy (A↑), forgetting (F↓), and number of active adapters
(#Ad↓), across three task orderings. SD-LoRA uses a fixed adapter per task; the
backbone is ViT-B/16 with rank-10 LoRA on Q and V.

| Dataset (tasks)   | Method    | A (O1) | F (O1) | #Ad | A (O2) | F (O2) | #Ad | A (O3) | F (O3) | #Ad |
|-------------------|-----------|:------:|:------:|:---:|:------:|:------:|:---:|:------:|:------:|:---:|
| CIFAR-100 (10)    | SD-LoRA   | 91.71  | 5.92   | 10  | 92.54  | 4.93   | 10  | 91.27  | 5.96   | 10  |
|                   | **LiteLoRA** | **91.81** | **4.66** | **5**  | **92.64** | 5.14 | **8**  | **91.41** | 6.09 | **7**  |
| ImageNet-A (20)   | SD-LoRA   | 64.85  | 18.53  | 20  | 66.30  | 14.70  | 20  | 62.08  | 15.14  | 20  |
|                   | **LiteLoRA** | **65.75** | **16.55** | **6**  | **66.53** | 17.27 | **14** | **63.64** | 16.49 | **13** |
| ImageNet-R (20)   | SD-LoRA   | 82.41  | 10.25  | 20  | 81.55  | 8.89   | 20  | 81.27  | 13.67  | 20  |
|                   | **LiteLoRA** | **82.83** | **7.84** | **6**  | **81.64** | **8.49** | **7**  | **81.71** | **12.26** | **6**  |

LiteLoRA matches or exceeds SD-LoRA accuracy while using far fewer adapters —
a 65–70% reduction on the 20-task ImageNet benchmarks. See the paper for the
sparsity–accuracy frontier and the pruning-consistency analysis.

## Acknowledgements

This repository builds on:

- [SD-LoRA](https://github.com/WuYichen-97/SD-Lora-CL) — Scalable Decoupled Low-Rank Adaptation for CIL (ICLR 2025)
- [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) — LoRA for Vision Transformers
