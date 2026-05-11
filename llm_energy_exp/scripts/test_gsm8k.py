import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import energy_profiler as ep
import tqdm
import os

from scripts.bench_utils import augment_profile, run_with_profiler, write_jsonl

ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
DEFAULT_GSM8K_PATH = Path("/opt/grade-school-math/grade_school_math/data/test.jsonl")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_decimal(value: str):
    cleaned = value.replace(",", "").strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _extract_answer(text: str):
    m = ANS_RE.search(text)
    if m:
        return m.group(1)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def _is_correct(pred_text: str, gold_text: str) -> bool:
    pred = _extract_answer(pred_text)
    gold = _extract_answer(gold_text)
    if pred is None or gold is None:
        return False
    pred_dec = _to_decimal(pred)
    gold_dec = _to_decimal(gold)
    if pred_dec is None or gold_dec is None:
        return False
    return pred_dec == gold_dec


def _build_prompt(question: str) -> str:
    return (
        "Solve the following grade school math problem.\n"
        "Return the final answer in the format: #### <number>\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def test_gsm8k(model_name, dataset_name, output_dir, max_samples: Optional[int] = None):
    _ = dataset_name
    if not DEFAULT_GSM8K_PATH.exists():
        raise FileNotFoundError(f"GSM8K file not found: {DEFAULT_GSM8K_PATH}")

    samples = []
    profiles = []
    profiler = ep.EnergyProfiler()
    questions = list(_read_jsonl(DEFAULT_GSM8K_PATH))
    if max_samples is not None:
        questions = questions[:max_samples]

    for i, item in enumerate(tqdm.tqdm(questions)):
        question = item["question"]
        gold = item["answer"]
        prompt = _build_prompt(question)

        completion, rss_trace, ttft, profile = run_with_profiler(profiler, model_name, prompt)

        response = completion.choices[0].message.content or ""
        is_correct = _is_correct(response, gold)
        pred_answer = _extract_answer(response)
        gold_answer = _extract_answer(gold)

        augment_profile(profile, completion, rss_trace, ttft)
        profile["task_id"] = f"gsm8k/{i}"
        profile["correct"] = is_correct
        profiles.append(profile)

        samples.append(
            {
                "task_id": f"gsm8k/{i}",
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "completion": response,
            }
        )
        time.sleep(0.1)

    out_prefix = f"gsm8k_{model_name}"
    accuracy = sum(1 for s in samples if s["correct"]) / len(samples) if samples else 0.0
    summary = {
        "model_name": model_name,
        "dataset": "GSM8K",
        "num_samples": len(samples),
        "accuracy": accuracy,
    }
    print(summary)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, out_prefix)
    else:
        base = out_prefix
    write_jsonl(f"{base}_samples.jsonl", samples)
    write_jsonl(f"{base}_profiles.jsonl", profiles)
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
