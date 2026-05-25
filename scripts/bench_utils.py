import json
import time

import energy_profiler as ep
from ollama_utils import call_ollama_generate_with_memory


def run_with_profiler(profiler: ep.EnergyProfiler, model_name: str, prompt: str):
    """Run inference with energy profiling, retrying until a valid profile is obtained."""
    while True:
        profiler.start()
        completion, rss_trace, ttft = call_ollama_generate_with_memory(model=model_name, prompt=prompt)
        profile = profiler.stop()
        if (
            profile["cpu_energy"] is not None
            and profile["dram_energy"] is not None
            and profile["time"] is not None
        ):
            return completion, rss_trace, ttft, profile
        print("Energy profiling failed, retrying...")
        time.sleep(1)


def augment_profile(profile: dict, completion, rss_trace, ttft) -> None:
    """Add usage, RSS, and TTFT fields to a profile dict in-place."""
    profile["completion_tokens"] = completion.usage.completion_tokens if completion.usage else None
    profile["prompt_tokens"] = completion.usage.prompt_tokens if completion.usage else None
    profile["rss_baseline_bytes"] = rss_trace.baseline_rss
    profile["rss_peak_bytes"] = rss_trace.peak_rss
    profile["rss_final_bytes"] = rss_trace.final_rss
    profile["ttft"] = ttft


def write_jsonl(path: str, records: list) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: list) -> None:
    """Append a list of dicts to a JSONL file (create if not exists)."""
    with open(path, "a", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
