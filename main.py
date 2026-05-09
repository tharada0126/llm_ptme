import argparse
import logging
from pathlib import Path

from scripts.test_human_eval import test_human_eval
from scripts.test_gsm8k import test_gsm8k
from scripts.test_mmlu_pro import test_mmlu_pro

logging.getLogger("httpx").setLevel(logging.WARNING)


SUPPORTED_MODELS = [
    "tinyllama:1.1b",
    "phi:2.7b",
    "phi3:3.8b",
    "qwen2.5:1.5b",
    "gemma:2b",
    "mistral:7b",
]

SUPPORTED_DATASETS = [
    "human_eval",
    "gsm8k",
    "GSM8K",
    "mmlu-pro",
    "MMLU-Pro",
]


def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower()

    if name == "human_eval":
        return "human_eval"
    if name == "gsm8k":
        return "gsm8k"
    if name in ["mmlu-pro", "mmlu_pro"]:
        return "mmlu-pro"

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def main(
    model_name: str,
    dataset_name: str,
    output_dir: str | None,
    max_samples: int | None = None,
    scenario: str = "E0",
):
    dataset_key = normalize_dataset_name(dataset_name)

    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    # Warm up the model to avoid cold start effects in profiling
    print(f"Warming up model {model_name}...")
    # We can use a simple prompt for warming up
    warmup_prompt = "Hello, this is a warm-up prompt to initialize the model."
    try:
        call_ollama_generate(model=model_name, prompt=warmup_prompt, system="", num_predict=1)
    except Exception as e:
        print(f"Warning: Model warm-up failed with error: {e}")

    if output_dir is None:
        output_dir = Path("results") / scenario / dataset_key / model_name.replace("/", "_").replace(":", "_")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(f"Scenario    : {scenario}")
    print(f"Model       : {model_name}")
    print(f"Dataset     : {dataset_key}")
    print(f"Output dir  : {output_dir}")
    print(f"Max samples : {max_samples}")
    print("========================================")

    if dataset_key == "human_eval":
        test_human_eval(
            model_name=model_name,
            dataset_name=dataset_key,
            output_dir=str(output_dir),
            max_samples=max_samples,
        )

    elif dataset_key == "gsm8k":
        test_gsm8k(
            model_name=model_name,
            dataset_name=dataset_key,
            output_dir=str(output_dir),
            max_samples=max_samples,
        )

    elif dataset_key == "mmlu-pro":
        test_mmlu_pro(
            model_name=model_name,
            dataset_name="TIGER-Lab/MMLU-Pro",
            output_dir=str(output_dir),
            split="test",
            max_samples=max_samples,
            ntrain=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Ollama model name to use for generation.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of problems for debugging.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="E0",
        help="Execution scenario label, e.g., E1, E2, E3.",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        scenario=args.scenario,
    )