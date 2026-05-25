#!/bin/bash
set -euo pipefail

scenarios=("E1" "E2" "E3" "E4" "E5")
freqs=("1.5GHz" "1.8GHz" "2.0GHz" "2.2GHz" "2.5GHz")
core_ids=("2-3" "2-5" "2-9" "2-13" "0-15")
memory=("5g" "8g" "12g" "14g" "16g")
threads=("1" "2" "4" "6" "8")
#scenarios=("E2" "E3" "E4" "E5")
#freqs=("1.8GHz" "2.0GHz" "2.2GHz" "2.5GHz")
#core_ids=("2-5" "2-9" "2-13" "0-15")
#memory=("8g" "12g" "14g" "16g")
#threads=("2" "4" "6" "8")
#scenarios=("E5")
#freqs=("2.5GHz")
#core_ids=("0-15")
#memory=("16g")
#threads=("8.0")

models=(
  "tinyllama:1.1b"
  "qwen2.5:1.5b"
  "phi:2.7b"
  "phi3:3.8b"
  "gemma:2b"
  "mistral:7b"
)

datasets=(
  "human_eval"
  "gsm8k"
  "mmlu-pro"
)

MMLU_PRO_SELECTED_IDS="/workspace/scripts/mmlu_pro_selected_ids.txt"

for ((i=${#scenarios[@]}-1; i>=0; i--)); do
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "gsm8k" ]]; then
    max_samples="none"
    selected_ids_file=""
  elif [[ "$dataset" == "mmlu-pro" ]]; then
    max_samples="none"
    selected_ids_file="$MMLU_PRO_SELECTED_IDS"
  else
    max_samples="none"
    selected_ids_file=""
  fi

    scenario="${scenarios[$i]}"
    freq="${freqs[$i]}"
    cpuset="${core_ids[$i]}"
    mem="${memory[$i]}"
    nthreads="${threads[$i]}"

    echo "========================================"
    echo "Scenario: ${scenario}"
    echo "Frequency: ${freq}"
    echo "CPU set: ${cpuset}"
    echo "Memory: ${mem}"
    echo "Threads: ${nthreads}"
    echo "========================================"

    sudo cpupower -c "$cpuset" frequency-set -g performance
    sudo cpupower -c "$cpuset" frequency-set -d "$freq" -u "$freq"

    for model in "${models[@]}"; do
      echo "Running ${scenario}, ${model}, ${dataset}"

      if SCENARIO="$scenario" \
        CPUSET_CPUS="$cpuset" \
        OLLAMA_NUM_THREADS="$nthreads" \
        MEMORY_LIMIT="$mem" \
        MODEL_NAME="$model" \
        DATASET_NAME="$dataset" \
        MAX_SAMPLES="$max_samples" \
        SELECTED_IDS_FILE="$selected_ids_file" \
        docker compose -f docker-compose.yml \
        -f docker-compose.scenario.yml \
        run --rm llm-exp; then

        echo "Completed: ${scenario}, ${model}, ${dataset}"

      else
        exit_code=$?
        echo "WARNING: Failed: ${scenario}, ${model}, ${dataset} (exit code: ${exit_code})"
        echo "Skipping to the next condition..."
        mkdir -p results/failed_runs
        echo "${scenario},${model},${dataset},${mem},${cpuset},${nthreads},exit_code=${exit_code}" \
          >> results/failed_runs/failed_runs.csv
        continue
      fi
    done
  done

  sleep 5
done
