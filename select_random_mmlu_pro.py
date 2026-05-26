import random
import json
from collections import defaultdict
from datasets import load_dataset


DATASET_NAME = "TIGER-Lab/MMLU-Pro"
SAMPLES_PER_CATEGORY = 100
SUBSET_PER_CATEGORY = 20
SEED = 42
OUTPUT_DIR = "scripts"


def _write_ids(path: str, selected: dict) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for category, ids in sorted(selected.items()):
            for qid in ids:
                f.write(f"{qid}\n")
    return sum(len(ids) for ids in selected.values())


def main():
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    test_data = list(dataset["test"])

    by_category = defaultdict(list)
    for example in test_data:
        by_category[example["category"]].append(example["question_id"])

    rng = random.Random(SEED)
    selected_100 = {}
    for category, ids in sorted(by_category.items()):
        n = min(SAMPLES_PER_CATEGORY, len(ids))
        selected_100[category] = sorted(rng.sample(ids, n))
        print(f"  {category}: {n} selected (total {len(ids)})")

    selected_20 = {
        category: ids[:SUBSET_PER_CATEGORY]
        for category, ids in selected_100.items()
    }

    path_100 = f"{OUTPUT_DIR}/mmlu_pro_selected_ids.txt"
    path_20 = f"{OUTPUT_DIR}/mmlu_pro_selected_ids_20.txt"

    total_100 = _write_ids(path_100, selected_100)
    total_20 = _write_ids(path_20, selected_20)

    print(f"\nSaved {total_100} question IDs to {path_100}")
    print(f"Saved {total_20} question IDs to {path_20}")
    print(f"Categories: {len(selected_100)}, seed: {SEED}")


if __name__ == "__main__":
    main()
