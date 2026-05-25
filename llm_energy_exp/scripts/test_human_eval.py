import human_eval.data
import tqdm
import energy_profiler as ep
import json
import time
from typing import Optional
import os

from scripts.bench_utils import augment_profile, run_with_profiler, append_jsonl

def test_human_eval(model_name, dataset_name, output_dir, max_samples: Optional[int] = None):
    _ = dataset_name
    problem = human_eval.data.read_problems()
    test_ids = list(problem.keys())
    if max_samples is not None:
        test_ids = test_ids[:max_samples]

    # Determine output paths before the loop to support resume
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"human_eval_{model_name}")
    else:
        base = f"human_eval_{model_name}"
    sample_path = f"{base}_samples.jsonl"
    profile_path = f"{base}_profiles.jsonl"

    # Skip already-completed tasks
    done_ids: set = set()
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["task_id"])
        print(f"Resuming: {len(done_ids)} tasks already done, skipping.")

    test_ids = [tid for tid in test_ids if tid not in done_ids]

    if not test_ids:
        print("All tasks already completed.")
        return

    samples = []
    profiles = []
    profiler = ep.EnergyProfiler()

    for task_id in tqdm.tqdm(test_ids):
        prompt = problem[task_id]["prompt"]
        prompt = (
            "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
            + prompt
        )
        completion, rss_trace, ttft, profile = run_with_profiler(profiler, model_name, prompt)
        response = completion.choices[0].message.content
        augment_profile(profile, completion, rss_trace, ttft)
        samples.append(dict(task_id=task_id, completion=response))
        profiles.append(profile)
        time.sleep(0.1)

    append_jsonl(sample_path, samples)
    append_jsonl(profile_path, profiles)