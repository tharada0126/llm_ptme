##!/bin/bash
# Usage:
# bash exec_single_test.sh model_name dataset_name output_dir max_samples
# receive arguments model_name and dataset_name
model_name=$1
dataset_name=$2
output_dir=$3
max_samples=$4
ROCR_VISIBLE_DEVICES=-1
OLLAMA_NO_GPU=1
OLLAMA=/home/tomohiro/.local/bin/ollama
PYTHON=/home/tomohiro/miniconda3/envs/llm_ptme/bin/python

echo "Testing $model_name..."
if ! $OLLAMA list | grep -q "$model_name"; then
    echo "Model $model_name not found. Please pull it first."
    $OLLAMA pull $model_name
fi
$OLLAMA run $model_name "Hello!" >/dev/null 2>&1
# if output_dir is None, set the current directory as output_dir
if [ -z "$output_dir" ]; then
    output_dir="."
fi
if [ -n "$max_samples" ]; then
    $PYTHON main.py --model_name "$model_name" --dataset_name "$dataset_name" --output_dir "$output_dir" --max_samples "$max_samples"
else
    $PYTHON main.py --model_name "$model_name" --dataset_name "$dataset_name" --output_dir "$output_dir"
fi
