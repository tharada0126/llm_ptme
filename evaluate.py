"""Score Raspberry Pi results on the desktop PC.

GSM8K and MMLU-Pro use the same answer-extraction rules as the previous x86 experiments.
HumanEval is exported to the human-eval format ({task_id, completion}) for
`evaluate_functional_correctness`, which must run in a sandboxed environment.

Usage:
    python evaluate.py results/rpi
"""

import argparse
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _to_decimal(value: str):
    cleaned = value.replace(",", "").strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def gsm8k_extract_answer(text: str):
    m = ANS_RE.search(text)
    if m:
        return m.group(1)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def gsm8k_is_correct(pred_text: str, gold_text: str) -> bool:
    pred = gsm8k_extract_answer(pred_text)
    gold = gsm8k_extract_answer(gold_text)
    if pred is None or gold is None:
        return False
    pred_dec = _to_decimal(pred)
    gold_dec = _to_decimal(gold)
    if pred_dec is None or gold_dec is None:
        return False
    return pred_dec == gold_dec


def mmlu_pro_extract_answer(response: str):
    if response is None:
        return None
    patterns = [
        r"The answer is\s*\(?([A-J])\)?",
        r"the answer is\s*\(?([A-J])\)?",
        r"answer is\s*\(?([A-J])\)?",
        r"Answer\s*:\s*\(?([A-J])\)?",
        r"Final answer\s*:\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-J])\b", response, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def score_sample(sample: dict) -> dict:
    completion = sample.get("completion") or ""
    meta = sample.get("meta", {})
    if sample["dataset"] == "gsm8k":
        return {
            "pred_answer": gsm8k_extract_answer(completion),
            "gold_answer": gsm8k_extract_answer(meta["answer"]),
            "correct": gsm8k_is_correct(completion, meta["answer"]),
        }
    if sample["dataset"] == "mmlu-pro":
        pred = mmlu_pro_extract_answer(completion)
        return {"pred": pred, "answer": meta["answer"], "correct": pred == meta["answer"]}
    raise ValueError(sample["dataset"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="e.g. results/rpi")
    args = parser.parse_args()

    summary = []
    for sample_path in sorted(Path(args.results_dir).rglob("*_samples.jsonl")):
        with sample_path.open(encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        if not samples:
            continue
        dataset = samples[0]["dataset"]
        model = samples[0]["model_name"]
        stem = str(sample_path)[: -len("_samples.jsonl")]

        if dataset == "human_eval":
            out = Path(f"{stem}_human_eval_input.jsonl")
            with out.open("w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps({"task_id": s["task_id"], "completion": s["completion"]}, ensure_ascii=False) + "\n")
            print(f"[human_eval] {model}: exported {len(samples)} samples -> {out}")
            continue

        scored = [{**s, **score_sample(s)} for s in samples]
        out = Path(f"{stem}_scored.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        acc = sum(s["correct"] for s in scored) / len(scored)
        row = {"model_name": model, "dataset": dataset, "num_samples": len(scored), "accuracy": acc}
        if dataset == "mmlu-pro":
            by_cat = defaultdict(list)
            for s in scored:
                by_cat[s["meta"]["category"]].append(s["correct"])
            row["accuracy_by_category"] = {c: sum(v) / len(v) for c, v in sorted(by_cat.items())}
        summary.append(row)
        print(f"[{dataset}] {model}: accuracy={acc:.4f} (n={len(scored)})")

    out = Path(args.results_dir) / "accuracy_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
