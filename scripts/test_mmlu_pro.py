import tqdm
import energy_profiler as ep
import time
import json
from typing import Optional
import os
import re

from pathlib import Path
from datasets import load_dataset
from bench_utils import augment_profile, run_with_profiler, write_jsonl

MMLU_PRO_DIR = Path(os.environ.get("MMLU_PRO_DIR", "/opt/MMLU-Pro"))

choices = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P"
]


def preprocess(data):
    res = []

    for each in data:
        each = dict(each)
        each["options"] = [opt for opt in each["options"] if opt != "N/A"]
        res.append(each)

    return res


def select_by_category(df, subject):
    return [each for each in df if each["category"] == subject]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."

    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""

    with open(MMLU_PRO_DIR / "cot_prompt_lib" / "initial_prompt.txt", "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            prompt += line

    subject = curr["category"]
    subject_val_df = select_by_category(val_df, subject)
    subject_val_df = subject_val_df[:k]

    prompt = prompt.replace("{$}", subject) + "\n"

    for example in subject_val_df:
        prompt += format_cot_example(example, including_answer=True)

    prompt += format_cot_example(curr, including_answer=False)

    return prompt


def extract_answer(response: str):
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


def test_mmlu_pro(
    model_name,
    dataset_name,
    output_dir,
    split: str = "test",
    subject: Optional[str] = None,
    max_samples: Optional[int] = None,
    ntrain: int = 0,
):
    dataset = load_dataset(dataset_name)

    test_data = preprocess(dataset[split])
    val_data = preprocess(dataset["validation"])

    if subject is not None and subject != "all":
        test_data = select_by_category(test_data, subject)

    if max_samples is not None:
        test_data = test_data[:max_samples]

    samples = []
    profiles = []

    profiler = ep.EnergyProfiler()

    for example in tqdm.tqdm(test_data):
        question_id = example["question_id"]

        prompt = generate_cot_prompt(
            val_df=val_data,
            curr=example,
            k=ntrain
        )

        completion, rss_trace, ttft, profile = run_with_profiler(profiler, model_name, prompt)

        response = completion.choices[0].message.content
        pred = extract_answer(response)

        answer = example["answer"]
        correct = pred == answer

        profile["question_id"] = question_id
        profile["category"] = example["category"]
        augment_profile(profile, completion, rss_trace, ttft)
        profile["pred"] = pred
        profile["answer"] = answer
        profile["correct"] = correct
        profile["ntrain"] = ntrain

        sample = {
            "question_id": question_id,
            "category": example["category"],
            "question": example["question"],
            "options": example["options"],
            "answer": answer,
            "answer_index": example["answer_index"],
            "pred": pred,
            "correct": correct,
            "completion": response,
            "ntrain": ntrain,
        }

        samples.append(sample)
        profiles.append(profile)

        time.sleep(0.1)

    safe_model_name = model_name.replace("/", "_").replace(":", "_")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"mmlu_pro_{safe_model_name}_{split}")
    else:
        base = f"mmlu_pro_{safe_model_name}"
    sample_path = f"{base}_samples.jsonl"
    profile_path = f"{base}_profiles.jsonl"
    write_jsonl(sample_path, samples)
    write_jsonl(profile_path, profiles)

    total = len(samples)
    correct_count = sum(1 for x in samples if x["correct"])
    accuracy = correct_count / total if total > 0 else 0.0

    print(f"Saved samples to: {sample_path}")
    print(f"Saved profiles to: {profile_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total})")


if __name__ == "__main__":
    test_mmlu_pro(
        model_name="tinyllama:1.1b",
        dataset_name="TIGER-Lab/MMLU-Pro",
        output_dir="results",
        split="test",
        max_samples=10,
        ntrain=0,
    )