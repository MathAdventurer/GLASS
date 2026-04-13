#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEXT_MODEL="${GLASS_TEXT_MODEL:?Set GLASS_TEXT_MODEL to the Qwen3-Embedding checkpoint}"
DATA_ROOT="${GLASS_DATA_ROOT:-${PROJECT_ROOT}/data}"
RESULT_ROOT="${GLASS_RESULT_ROOT:-${PROJECT_ROOT}/results/text_adapter}"
LOG_ROOT="${GLASS_LOG_ROOT:-${PROJECT_ROOT}/logs/text_adapter}"
GPU_IDS="${GLASS_GPU_IDS:-0 1 2 3 4 5 6 7}"

read -r -a GPUS <<< "${GPU_IDS}"
DATASETS=(
  MUTAG PROTEINS DD ENZYMES DHFR BZR COX2 AIDS
  IMDB-BINARY NCI1 COLLAB REDDIT-BINARY
)

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

run_one() {
  local dataset="$1"
  local gpu="$2"
  local log_path="${LOG_ROOT}/${dataset}.log"
  echo "[$(date)] START ${dataset} on GPU ${gpu}" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/train_text_adapter.py" \
    --dataset "${dataset}" \
    --device cuda:0 \
    --text_device cuda:0 \
    --text_model "${TEXT_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --result_root "${RESULT_ROOT}" \
    --num_seeds 5 \
    --epochs 50 \
    --lora_epochs 50 \
    --batch_size 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_cache \
    >> "${log_path}" 2>&1
  echo "[$(date)] DONE ${dataset}" >> "${log_path}"
}

pids=()
for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one "${dataset}" "${gpu}" &
  pids+=("$!")
  if (( ${#pids[@]} == ${#GPUS[@]} )); then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[$(date)] all text-adapter diagnostics finished"
