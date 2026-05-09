#!/bin/bash
set -euo pipefail

scenarios=("E0" "E1" "E2" "E3" "E4")
freqs=("1.5GHz" "1.8GHz" "2.0GHz" "2.2GHz" "2.5GHz")
core_ids=("2" "2-3" "2-5" "2-7" "2-9")
memory=("4g" "8g" "12g" "14g" "16g")

for i in "${!scenarios[@]}"; do
    scenario="${scenarios[$i]}"
    freq="${freqs[$i]}"
    cpuset="${core_ids[$i]}"
    mem="${memory[$i]}"

    echo "========================================"
    echo "Scenario: ${scenario}"
    echo "Frequency: ${freq}"
    echo "CPU set: ${cpuset}"
    echo "Memory: ${mem}"
    echo "========================================"

    sudo cpupower -c "$cpuset" frequency-set -g performance
    sudo cpupower -c "$cpuset" frequency-set -d "$freq" -u "$freq"

    echo "[Host] CPU frequency info:"
    cpupower -c "$cpuset" frequency-info | grep -E "current CPU frequency|The governor|hardware limits" || true

    SCENARIO="$scenario" \
    CPUSET_CPUS="$cpuset" \
    MEMORY_LIMIT="$mem" \
    docker compose run --rm llm-exp

    sleep 5
done