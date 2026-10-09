"""Build prompt-ready benchmark files (data/*.jsonl) for the Raspberry Pi runner.

Run this once on the desktop PC (needs network + `datasets`), then copy data/ to the Pi.
The prompts are constructed exactly as in the previous x86 experiments, so the Pi needs
neither the benchmark repositories nor the `datasets` library.

Each output line: {"task_id": str, "prompt": str, "meta": {...gold info for later scoring...}}
"""

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

HUMAN_EVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MMLU_PRO_PROMPT_URL = "https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/main/cot_prompt_lib/initial_prompt.txt"

MMLU_CHOICES = list("ABCDEFGHIJKLMNOP")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _jsonl_lines(raw: bytes) -> Iterable[dict]:
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _read_ids(path: Optional[str]) -> Optional[List[int]]:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _write(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {path}")


def build_human_eval() -> List[dict]:
    raw = gzip.decompress(_fetch(HUMAN_EVAL_URL))
    rows = []
    for p in _jsonl_lines(raw):
        prompt = (
            "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
            + p["prompt"]
        )
        rows.append({"task_id": p["task_id"], "prompt": prompt, "meta": {"entry_point": p["entry_point"]}})
    return rows


def build_gsm8k(selected_idx_file: Optional[str]) -> List[dict]:
    questions = list(_jsonl_lines(_fetch(GSM8K_URL)))
    selected = _read_ids(selected_idx_file)
    indices = sorted(set(selected)) if selected is not None else range(len(questions))
    rows = []
    for i in indices:
        q = questions[i]
        prompt = (
            "Solve the following grade school math problem.\n"
            "Return the final answer in the format: #### <number>\n\n"
            f"Question: {q['question']}\n"
            "Answer:"
        )
        rows.append({"task_id": f"gsm8k/{i}", "prompt": prompt, "meta": {"index": i, "answer": q["answer"]}})
    return rows


def _format_mmlu_question(example: dict) -> str:
    prompt = "Question:\n" + example["question"] + "\n" + "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(MMLU_CHOICES[i], opt)
    prompt += "Answer: Let's think step by step."
    return prompt


def build_mmlu_pro(selected_ids_file: Optional[str]) -> List[dict]:
    from datasets import load_dataset

    initial_prompt = _fetch(MMLU_PRO_PROMPT_URL).decode("utf-8")
    test_data = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    selected = _read_ids(selected_ids_file)
    selected_set = set(selected) if selected is not None else None

    rows = []
    for example in test_data:
        if selected_set is not None and example["question_id"] not in selected_set:
            continue
        example = dict(example)
        example["options"] = [opt for opt in example["options"] if opt != "N/A"]
        # ntrain = 0 (zero-shot CoT), identical to the previous experiments.
        prompt = initial_prompt.replace("{$}", example["category"]) + "\n" + _format_mmlu_question(example)
        rows.append(
            {
                "task_id": str(example["question_id"]),
                "prompt": prompt,
                "meta": {
                    "question_id": example["question_id"],
                    "category": example["category"],
                    "answer": example["answer"],
                    "answer_index": example["answer_index"],
                    "num_options": len(example["options"]),
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--gsm8k_selected_idx_file", default="scripts/gsm8k_selected_idx.txt")
    parser.add_argument("--mmlu_pro_selected_ids_file", default="scripts/mmlu_pro_selected_ids_20.txt")
    parser.add_argument(
        "--datasets", nargs="+", default=["human_eval", "gsm8k", "mmlu-pro"], choices=["human_eval", "gsm8k", "mmlu-pro"]
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if "human_eval" in args.datasets:
        _write(out_dir / "human_eval.jsonl", build_human_eval())
    if "gsm8k" in args.datasets:
        _write(out_dir / "gsm8k.jsonl", build_gsm8k(args.gsm8k_selected_idx_file))
    if "mmlu-pro" in args.datasets:
        _write(out_dir / "mmlu_pro.jsonl", build_mmlu_pro(args.mmlu_pro_selected_ids_file))


if __name__ == "__main__":
    main()
