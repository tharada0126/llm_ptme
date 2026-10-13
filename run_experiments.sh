#!/bin/bash
# Run all (model, dataset) combinations on the Raspberry Pi.
#
# Environment variables:
#   TAG             experiment label used in results/<TAG>/...   (default: rpi)
#   PYTHON          python interpreter                            (default: .venv/bin/python)
#   PULL_MISSING=1  pull a model right before its runs if it is not installed
#   REMOVE_AFTER=1  `ollama rm` each model after its runs (for small storage)
#   COOLDOWN_TEMP   wait until CPU temp <= this value before each run (default: 50)
#   IDLE_SEC        idle baseline before/after each run in seconds (default: 30)
#   MAX_SAMPLES     limit samples per dataset (default: none)
#   OLLAMA_NUM_THREADS  passed through to main.py if set
#
# Downloads/removals happen outside the measurement windows recorded in events.jsonl.
set -uo pipefail

cd "$(dirname "$0")"

TAG="${TAG:-rpi}"
PYTHON="${PYTHON:-llm_rpi/bin/python}"
PULL_MISSING="${PULL_MISSING:-0}"
REMOVE_AFTER="${REMOVE_AFTER:-0}"
COOLDOWN_TEMP="${COOLDOWN_TEMP:-50}"
IDLE_SEC="${IDLE_SEC:-30}"
MAX_SAMPLES="${MAX_SAMPLES:-none}"

models=(
  "tinyllama:1.1b"
  "qwen2.5:1.5b"
  "phi3:3.8b"
  "mistral:7b"
  "qwen3.5:0.8b"
  "qwen3.5:2b"
  "gemma4:e2b"
  "gemma4:e4b"
  "phi3.5:3.8b"
  "phi4-mini:3.8b"
  "llama3.2:1b"
  "llama3.2:3b"
  "ministral-3:3b"
  "mistral-nemo:12b"
)

datasets=(
  #"human_eval"
  "gsm8k"
  "mmlu-pro"
)

FAILED_LOG="results/${TAG}/failed_runs.csv"
mkdir -p "results/${TAG}"

model_installed() {
  ollama list | awk 'NR>1 {print $1}' | grep -qx "$1"
}

for model in "${models[@]}"; do
  if ! model_installed "$model"; then
    if [[ "$PULL_MISSING" == "1" ]]; then
      echo "Pulling ${model}..."
      if ! ollama pull "$model"; then
        echo "${model},pull,failed" >> "$FAILED_LOG"
        continue
      fi
    else
      echo "WARNING: ${model} is not installed; skipping (set PULL_MISSING=1 to pull automatically)."
      echo "${model},all,not_installed" >> "$FAILED_LOG"
      continue
    fi
  fi

  for dataset in "${datasets[@]}"; do
    echo "========================================"
    echo "Running ${TAG}, ${model}, ${dataset}"
    echo "========================================"

    if "$PYTHON" main.py \
        --tag "$TAG" \
        --model_name "$model" \
        --dataset_name "$dataset" \
        --max_samples "$MAX_SAMPLES" \
        --idle_sec "$IDLE_SEC" \
        --cooldown_temp "$COOLDOWN_TEMP"; then
      echo "Completed: ${model}, ${dataset}"
    else
      exit_code=$?
      echo "WARNING: Failed: ${model}, ${dataset} (exit code: ${exit_code})"
      echo "${model},${dataset},exit_code=${exit_code}" >> "$FAILED_LOG"
    fi
  done

  if [[ "$REMOVE_AFTER" == "1" ]]; then
    echo "Removing ${model}..."
    ollama rm "$model" || true
  fi
done
