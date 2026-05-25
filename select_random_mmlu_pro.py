import random
import json
from collections import defaultdict
from datasets import load_dataset


DATASET_NAME = "TIGER-Lab/MMLU-Pro"
SAMPLES_PER_CATEGORY = 100
SEED = 42
OUTPUT_FILE = "mmlu_pro_selected_ids.txt"


def main():
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    test_data = list(dataset["test"])

    by_category = defaultdict(list)
    for example in test_data:
        by_category[example["category"]].append(example["question_id"])

    rng = random.Random(SEED)
    selected = {}
    for category, ids in sorted(by_category.items()):
        n = min(SAMPLES_PER_CATEGORY, len(ids))
        selected[category] = sorted(rng.sample(ids, n))
        print(f"  {category}: {n} selected (total {len(ids)})")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for category, ids in sorted(selected.items()):
            for qid in ids:
                f.write(f"{qid}\n")

    total = sum(len(ids) for ids in selected.values())
    print(f"\nSaved {total} question IDs to {OUTPUT_FILE}")
    print(f"Categories: {len(selected)}, samples per category: {SAMPLES_PER_CATEGORY}, seed: {SEED}")


if __name__ == "__main__":
    main()
