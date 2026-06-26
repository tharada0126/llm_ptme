#!/bin/bash
set -euo pipefail

MODEL="${1:-qwen3.6:35b}"
SCENARIO="${2:-E_GPU}"
MAX_SAMPLES="${3:-none}"

MMLU_PRO_SELECTED_IDS="/workspace/scripts/mmlu_pro_selected_ids_20.txt"
GSM8K_SELECTED_IDX="/workspace/scripts/gsm8k_selected_idx.txt"

echo "========================================"
echo "GPU test run (no resource constraints)"
echo "Model   : ${MODEL}"
echo "Scenario: ${SCENARIO}"
echo "Max samples: ${MAX_SAMPLES}"
echo "========================================"

datasets=("human_eval" "gsm8k" "mmlu-pro")

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "gsm8k" ]]; then
    selected_ids_file="none"
    gsm8k_selected_idx_file="$GSM8K_SELECTED_IDX"
  elif [[ "$dataset" == "mmlu-pro" ]]; then
    selected_ids_file="$MMLU_PRO_SELECTED_IDS"
    gsm8k_selected_idx_file="none"
  else
    selected_ids_file="none"
    gsm8k_selected_idx_file="none"
  fi

  echo "Running ${MODEL}, ${dataset}..."

  if SCENARIO="$SCENARIO" \
    MODEL_NAME="$MODEL" \
    DATASET_NAME="$dataset" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    SELECTED_IDS_FILE="$selected_ids_file" \
    GSM8K_SELECTED_IDX_FILE="$gsm8k_selected_idx_file" \
    OLLAMA_NUM_THREADS="0" \
    docker compose -f docker-compose.yml \
    -f docker-compose.gpu.yml \
    run --rm llm-exp; then

    echo "Completed: ${MODEL}, ${dataset}"

  else
    exit_code=$?
    echo "WARNING: Failed: ${MODEL}, ${dataset} (exit code: ${exit_code})"
    mkdir -p results/failed_runs
    echo "${SCENARIO},${MODEL},${dataset},exit_code=${exit_code}" \
      >> results/failed_runs/failed_runs.csv
  fi
done
