import json
import random
from pathlib import Path

GSM8K_PATHS = [
    Path("/home/tomohiro/LLM_PTME/llm_ptme/grade-school-math/grade_school_math/data/test.jsonl"),
    Path("/opt/grade-school-math/grade_school_math/data/test.jsonl"),
]
N_SAMPLES = 200
SEED = 42
OUTPUT_FILE = "scripts/gsm8k_selected_idx.txt"


def main():
    data_path = next((p for p in GSM8K_PATHS if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(f"GSM8K test.jsonl not found in: {GSM8K_PATHS}")

    questions = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(questions)), min(N_SAMPLES, len(questions))))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{idx}\n")

    print(f"Saved {len(indices)} indices to {OUTPUT_FILE} (total questions: {len(questions)}, seed: {SEED})")


if __name__ == "__main__":
    main()
