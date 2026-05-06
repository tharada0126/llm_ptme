import json
import re
from datasets import load_dataset
import argparse
import logging
from ollama_utils import call_ollama_generate
from test_human_eval import test_human_eval
from test_gsm8k import test_gsm8k
from test_mmlu_pro import test_mmlu_pro

# HTTPXのログレベルをWARNINGに設定してINFOを出さないようにする
logging.getLogger("httpx").setLevel(logging.WARNING)

def main(model_name, dataset_name, output_dir, max_samples=None):
    if dataset_name == "human_eval":
        test_human_eval(model_name, dataset_name, output_dir, max_samples=max_samples)
    elif dataset_name == "GSM8K" or dataset_name == "gsm8k":
        test_gsm8k(model_name, dataset_name, output_dir, max_samples=max_samples)
    elif dataset_name == "MMLU-Pro" or dataset_name == "mmlu-pro":
        test_mmlu_pro(model_name, "TIGER-Lab/MMLU-Pro", output_dir, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

if __name__ == "__main__":
    # arguments parsing can be added here if needed
    parser = argparse.ArgumentParser()
    # --model_name
    parser.add_argument("--model_name", type=str, help="Model name to use for generation", choices=["tinyllama:1.1b", "phi:2.7b", "phi3:3.8b", "qwen2.5:1.5b", "gemma:2b", "mistral:7b"])
    # --dataset_name
    parser.add_argument("--dataset_name", type=str, help="Dataset name (e.g., 'SWE-bench', 'human_eval')", choices=["SWE-bench", "SWE-bench_Lite", "SWE-bench_Verified", "human_eval", "GSM8K", "gsm8k", "MMLU-Pro", "mmlu-pro"])
    # --output_dir
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of problems for debugging")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    main(model_name, dataset_name, output_dir, max_samples=args.max_samples)