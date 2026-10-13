#!/bin/bash
# Extract code from HumanEval samples and run functional-correctness evaluation.
#
# Environment variables:
#   TAG     experiment label used in results/<TAG>/...  (default: rpi)
#   PYTHON  python interpreter with human-eval installed (default: python)
set -uo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-rpi}"
PYTHON="${PYTHON:-python}"
result_dir="results/${TAG}"

shopt -s nullglob
for sample_file in "${result_dir}"/human_eval/*/human_eval_*_samples.jsonl; do
    "$PYTHON" HE_utils/extract_code.py --input_file "$sample_file"
done

"$PYTHON" HE_utils/evaluate_human_eval_envelops.py "$result_dir"
#python correct_ptme.py
