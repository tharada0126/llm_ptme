import human_eval.data
import tqdm
import energy_profiler as ep
import time
import json
from typing import Optional
import os

from ollama_utils import call_ollama_generate_with_memory

def test_human_eval(model_name, dataset_name, output_dir, max_samples: Optional[int] = None):
    problem = human_eval.data.read_problems()
    samples = []
    profiles = []
    profiler = ep.EnergyProfiler()
    test_ids = list(problem.keys())
    if max_samples is not None:
        test_ids = test_ids[:max_samples]
    for task_id in tqdm.tqdm(test_ids):
        prompt = problem[task_id]["prompt"]
        prompt = (
            "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
            + prompt
        )
        while True:
            profiler.start()
            completion, rss_trace = call_ollama_generate_with_memory(model=model_name, prompt=prompt)
            profile = profiler.stop()
            if profile["cpu_energy"] is not None and profile["dram_energy"] is not None and profile["time"] is not None:
                break
            else:
                print("Energy profiling failed, retrying...")
                time.sleep(1)
        response = completion.choices[0].message.content
        profile["completion_tokens"] = completion.usage.completion_tokens
        profile["prompt_tokens"] = completion.usage.prompt_tokens
        profile["rss_baseline_bytes"] = rss_trace.baseline_rss
        profile["rss_final_bytes"] = rss_trace.final_rss
        samples.append(dict(task_id=task_id, completion=response))
        profiles.append(profile)
        time.sleep(0.1)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"HE_{model_name}_samples.jsonl"), "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        with open(os.path.join(output_dir, f"HE_{model_name}_profiles.jsonl"), "w") as f:
            for profile in profiles:
                f.write(json.dumps(profile) + "\n")
    else:
        with open(f"HE_{model_name}_samples.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        with open(f"HE_{model_name}_profiles.jsonl", "w") as f:
            for profile in profiles:
                f.write(json.dumps(profile) + "\n")