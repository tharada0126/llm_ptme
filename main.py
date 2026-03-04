import json
import re
from datasets import load_dataset
from swebench.inference.make_datasets.create_instance import *
import argparse
import logging
from ollama_utils import call_ollama_generate
from test_human_eval import test_human_eval
from test_gsm8k import test_gsm8k

# HTTPXのログレベルをWARNINGに設定してINFOを出さないようにする
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_prompt(inst: dict) -> str:
    # SWE-benchで基本的に入れるのは: repo, base_commit, problem_statement, FAIL_TO_PASS
    return (
        f"You are fixing a bug in repository {inst['repo']} at base commit {inst['base_commit']}.\n\n"
        "Issue:\n"
        f"{inst['problem_statement']}\n\n"
        "The following tests must pass:\n"
        + "\n".join(json.loads(inst["FAIL_TO_PASS"])) + "\n\n"
        "Constraints:\n"
        "- Do NOT modify tests.\n"
        "- Output ONLY a unified diff patch.\n"
        "- No explanation. No markdown.\n\n"
        "Your response MUST start with: --- a/\n"
    )


def extract_unified_diff(text: str) -> str:
    m = re.search(r"(?ms)^--- a/.*\Z", text.strip())
    return m.group(0).rstrip() if m else text.strip()


def test_swebench(model_name, dataset_name):
    dataset = load_dataset(f"princeton-nlp/{dataset_name}", split="test")
    inst = dataset[0]  # 例として最初のインスタンスを使う
    prompt = build_prompt(inst)
    completion = call_ollama_generate(model=model_name, prompt=prompt, num_predict=800)
    response = completion.choices[0].message.content
    patch = extract_unified_diff(response)
    print("Generated patch:\n", patch)

def main(model_name, dataset_name, output_dir, max_samples=None):
    if dataset_name in ["SWE-bench", "SWE-bench_Lite", "SWE-bench_bm25_40k", "SWE-bench_Verified"]:
        #test_swebench(model_name, dataset_name)
        raise NotImplementedError("SWE-bench testing not implemented yet")
    elif dataset_name == "human_eval":
        test_human_eval(model_name, dataset_name, output_dir, max_samples=max_samples)
    elif dataset_name == "GSM8K" or dataset_name == "gsm8k":
        test_gsm8k(model_name, dataset_name, output_dir, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

if __name__ == "__main__":
    # arguments parsing can be added here if needed
    parser = argparse.ArgumentParser()
    # --model_name
    parser.add_argument("--model_name", type=str, help="Model name to use for generation", choices=["tinyllama:1.1b", "phi:2.7b", "phi3:3.8b", "qwen2.5:1.5b", "gemma:2b", "mistral:7b"])
    # --dataset_name
    parser.add_argument("--dataset_name", type=str, help="Dataset name (e.g., 'SWE-bench', 'human_eval')", choices=["SWE-bench", "SWE-bench_Lite", "SWE-bench_Verified", "human_eval", "GSM8K", "gsm8k"])
    # --output_dir
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of problems for debugging")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    main(model_name, dataset_name, output_dir, max_samples=args.max_samples)