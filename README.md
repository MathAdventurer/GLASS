# GLASS
---
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit) [![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)  [![PyG](https://img.shields.io/badge/PyG-2.6.1-orange)](https://pytorch-geometric.readthedocs.io/) [![Transformers](https://img.shields.io/badge/Transformers-4.55.2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/) [![PEFT](https://img.shields.io/badge/PEFT-0.18.1-4B8BBE)](https://huggingface.co/docs/peft/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Graph-Language Aligned Spherical Scoring (GLASS) is a transferable graph-level
anomaly detection framework. It converts each graph into a compact Graph
Descriptor Prompt (GraphDP), embeds the prompt with a frozen instruction-aware
text encoder, aligns a structure-aware GNN to native Matryoshka text slices on
the unit hypersphere, and scores query graphs against normal references with
spherical angular kNN density estimates.


## Method Overview

GLASS uses one pipeline across single-domain, zero-shot, and few-shot GLAD:

1. **GraphDP structural language.** Each graph is serialized into compact
   key-value structural text containing local topology, attribute/label
   summaries, WL-style color summaries, motif statistics, connectivity, core,
   and spectral fields.
2. **Frozen text anchors.** A Qwen3-Embedding model is used only as an
   instruction-aware embedding function. The text encoder is frozen.
3. **Graph-language alignment.** A 3-layer GIN graph encoder with mean+max
   readout and an 11-dimensional spectral sketch is projected to native MRL
   slices and aligned to GraphDP text slices.
4. **Spherical reference scoring.** Inference ranks a query by angular kNN
   distance to normal references. Single-domain, zero-shot, and few-shot
   settings differ only in how the normal reference set is constructed.

The main paper scorer is implemented in `evaluate_reference.py`. Training logs
also contain diagnostic graph-only, text-only, ensemble, vMF, and prototype
scores.

## Repository Layout

```text
GLASS/
  README.md
  GLASS.yaml                 # exported conda environment
  train_single.py            # single-domain training entrypoint
  train_mixed.py             # mixed-domain / zero-shot entrypoint
  train_fewshot.py           # few-shot reference calibration entrypoint
  train_text_adapter.py      # two-stage LoRA appendix diagnostic
  evaluate_reference.py      # reference-only spherical scorer evaluation
  summarize_text_adapter.py  # summarize adapter diagnostic JSONs
  build_caches.py            # optional precompute cache builder
  glass_training_core.py     # training implementation
  glass_mixed_core.py        # mixed-domain implementation
  glass_fewshot_core.py      # few-shot implementation
  src/
    graph_dp.py              # GraphDP construction
    graph_encoder.py         # GIN + mean/max readout + spectral sketch
    text_encoder.py          # frozen Qwen3 embedding wrapper
    lora_text_encoder.py     # PEFT wrapper used only by the appendix probe
    glass_model.py           # GLASS model and training losses
    scoring.py               # vMF prototypes and spherical scorers
    ...
  scripts/
    run_text_adapter_8gpu.sh # all-dataset adapter diagnostic launcher
```

## Setup

Create the environment from the exported server environment:

```bash
conda env create -f GLASS.yaml
conda activate GLASS
```

Download or place the Qwen3-Embedding checkpoint outside the repository and
point GLASS to it:

```bash
export GLASS_TEXT_MODEL=/path/to/Qwen3-Embedding-0.6B
export GLASS_DATA_ROOT=/path/to/data
```

The code uses PyTorch Geometric datasets and writes processed data, caches,
checkpoints, and results under the configured project/data directories.

## Quick Smoke Test

On a GPU machine with the Qwen3 model path configured:

```bash
python train_single.py \
  --dataset MUTAG \
  --device cuda:0 \
  --text_device cuda:0 \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 1 \
  --epochs 1 \
  --patience 1 \
  --batch_size 32
```

This verifies dataset loading, GraphDP construction, frozen text embedding,
graph-language alignment, checkpoint writing, and result serialization.

## Single-Domain Training and Evaluation

Train one dataset:

```bash
python train_single.py \
  --dataset PROTEINS \
  --device cuda:0 \
  --text_device cuda:0 \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 5 \
  --epochs 150 \
  --patience 40 \
  --batch_size 64
```

Evaluate saved checkpoints with the reference-only scorers:

```bash
python evaluate_reference.py \
  --datasets PROTEINS \
  --ckpt_root checkpoints \
  --result_root results/reference_scorers \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 5
```

The reliability-gated scorer estimates slice and modality weights from normal
references only. No anomalous test labels are used for scorer selection.

## Zero-Shot Mixed-Domain Transfer

Train on source-domain normal graphs and evaluate target domains without target
optimization:

```bash
python train_mixed.py \
  --source MUTAG AIDS NCI1 BZR COX2 DHFR \
  --target PROTEINS DD ENZYMES \
  --device cuda:0 \
  --text_device cuda:0 \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 5
```

## Few-Shot Reference Calibration

Few-shot calibration adds trusted target normal graphs to the reference set
without gradient updates:

```bash
python train_fewshot.py \
  --source MUTAG AIDS NCI1 BZR COX2 DHFR \
  --target PROTEINS \
  --k_shots 1 4 8 16 32 \
  --device cuda:0 \
  --text_device cuda:0 \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 5
```

## LoRA Text-Adapter Diagnostic

The paper's main method keeps the text encoder frozen. The appendix separately
tests whether moving the text anchors with LoRA is beneficial. The diagnostic
has two stages: first train the 3-layer native-MRL graph side against frozen
text anchors; then freeze the graph encoder and graph-side projections and tune
only rank-16 LoRA adapters in the text encoder. No text-side projection is
introduced.

Run the paper configuration on one dataset:

```bash
python train_text_adapter.py \
  --dataset MUTAG \
  --device cuda:0 \
  --text_device cuda:0 \
  --text_model "$GLASS_TEXT_MODEL" \
  --data_root "$GLASS_DATA_ROOT" \
  --num_seeds 5 \
  --epochs 50 \
  --lora_epochs 50 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_cache
```

Run all twelve datasets across eight visible GPUs:

```bash
export PYTHON_BIN="$(command -v python)"
export GLASS_GPU_IDS="0 1 2 3 4 5 6 7"
bash scripts/run_text_adapter_8gpu.sh
python summarize_text_adapter.py --result_root results/text_adapter
```

## Contact
- Email: [xudongwang@link.cuhk.edu.cn](mailto:xudongwang@link.cuhk.edu.cn)
