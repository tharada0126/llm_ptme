"""Run one (model, dataset) benchmark on a Raspberry Pi and record time / memory / thermal profiles.

Energy is measured by an external power meter. To allow alignment with the meter log,
every phase (idle baseline, model load, warm-up, inference, unload) and every sample
is stamped with wall-clock UNIX time. Keep the Pi and the meter PC NTP-synchronised.

Correctness is NOT evaluated here; run evaluate.py on the desktop PC afterwards.
"""

import argparse
import json
import logging
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import tqdm
from ollama import Client

from ollama_utils import (
    chat_generate,
    describe_model,
    load_model,
    loaded_vram_bytes,
    model_supports_thinking,
    ollama_version,
    unload_all_models,
    unload_model,
)
from system_monitor import ResourceMonitor, read_cpu_temp_c, read_device_model, read_throttled, wait_for_cooldown

logging.getLogger("httpx").setLevel(logging.WARNING)

DATASET_FILES = {
    "human_eval": "human_eval.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "mmlu-pro": "mmlu_pro.jsonl",
}

WARMUP_PROMPT = "Hello, this is a warm-up prompt to initialize the model."


def safe_name(model_name: str) -> str:
    # Colons are not allowed on exFAT/NTFS external drives.
    return model_name.replace("/", "_").replace(":", "_")


def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower().replace("_", "-")
    if name == "human-eval":
        return "human_eval"
    if name in DATASET_FILES:
        return name
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


class EventLog:
    """Phase markers for aligning the external power meter log."""

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id

    def mark(self, event: str, **extra: Any) -> None:
        t = time.time()
        record = {
            "run_id": self.run_id,
            "event": event,
            "t_unix": t,
            "t_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}",
            "cpu_temp_c": read_cpu_temp_c(),
            **extra,
        }
        append_jsonl(self.path, record)
        print(f"[EVENT] {record['t_iso']} {event}")


def resolve_think(client: Client, model_name: str, think_arg: str) -> Optional[bool]:
    if think_arg == "true":
        return True
    if think_arg == "false":
        return False
    # auto: disable thinking on thinking-capable models so all models share the same token budget.
    try:
        return False if model_supports_thinking(client, model_name) else None
    except Exception as e:
        print(f"Warning: could not query model capabilities ({e}); leaving `think` unset.")
        return None


def main(args: argparse.Namespace) -> None:
    dataset_key = normalize_dataset_name(args.dataset_name)
    data_path = Path(args.data_dir) / DATASET_FILES[dataset_key]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run prepare_data.py first.")

    items = read_jsonl(data_path)
    if args.max_samples is not None:
        items = items[: args.max_samples]

    out_dir = Path(args.output_dir) if args.output_dir else Path("results") / args.tag / dataset_key / safe_name(args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{dataset_key}_{safe_name(args.model_name)}"
    sample_path = Path(f"{base}_samples.jsonl")
    profile_path = Path(f"{base}_profiles.jsonl")

    done_ids = set()
    if sample_path.exists():
        done_ids = {rec["task_id"] for rec in read_jsonl(sample_path)}
        print(f"Resuming: {len(done_ids)} tasks already done, skipping.")
    remaining = [it for it in items if it["task_id"] not in done_ids]
    if not remaining:
        print("All tasks already completed.")
        return

    client = Client()
    think = resolve_think(client, args.model_name, args.think)

    options: Dict[str, Any] = {"temperature": 0.0, "num_predict": args.num_predict, "num_ctx": args.num_ctx}
    if args.num_gpu is not None:
        # Number of layers offloaded to the GPU; 0 = CPU only (works for CUDA, ROCm and Metal).
        options["num_gpu"] = args.num_gpu
    num_thread = args.num_thread or os.environ.get("OLLAMA_NUM_THREADS")
    if num_thread:
        options["num_thread"] = int(num_thread)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    events = EventLog(out_dir / "events.jsonl", run_id)

    meta = {
        "run_id": run_id,
        "tag": args.tag,
        "model_name": args.model_name,
        "dataset": dataset_key,
        "data_file": str(data_path),
        "num_items_total": len(items),
        "num_items_remaining": len(remaining),
        "options": options,
        "think": think,
        "keep_alive": args.keep_alive,
        "idle_sec": args.idle_sec,
        "interval_sec": args.interval_sec,
        "poll_sec": args.poll_sec,
        "hostname": socket.gethostname(),
        "device_model": read_device_model(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "ollama_version": ollama_version(),
        "throttled_at_start": read_throttled(),
    }
    try:
        meta["model_info"] = describe_model(client, args.model_name)
    except Exception as e:
        meta["model_info"] = {"error": str(e)}

    print("========================================")
    for k in ("tag", "model_name", "dataset", "num_items_remaining", "options", "think", "device_model", "ollama_version"):
        print(f"{k:20s}: {meta[k]}")
    print(f"{'output_dir':20s}: {out_dir}")
    print("========================================")

    # Start from a clean state: no model resident in memory.
    unloaded = unload_all_models(client)
    if unloaded:
        print(f"Unloaded resident models: {unloaded}")

    if args.cooldown_temp is not None:
        events.mark("cooldown_start", target_c=args.cooldown_temp)
        wait_for_cooldown(args.cooldown_temp, args.cooldown_timeout)
        events.mark("cooldown_end")

    events.mark("run_start", meta=meta)

    if args.idle_sec > 0:
        events.mark("idle_pre_start")
        time.sleep(args.idle_sec)
        events.mark("idle_pre_end")

    events.mark("model_load_start")
    monitor = ResourceMonitor(poll_interval_sec=args.poll_sec)
    monitor.start()
    load_model(client, args.model_name, options, keep_alive=args.keep_alive)
    load_profile = monitor.stop()
    vram = loaded_vram_bytes(client, args.model_name)
    events.mark("model_load_end", profile=load_profile, size_vram_bytes=vram)
    if vram:
        print(f"Warning: {args.model_name} uses {vram / 2**20:.0f} MiB of GPU memory (num_gpu={args.num_gpu}).")

    events.mark("warmup_start")
    try:
        chat_generate(client, args.model_name, WARMUP_PROMPT, {**options, "num_predict": 32}, think=think, keep_alive=args.keep_alive)
    except Exception as e:
        print(f"Warning: model warm-up failed with error: {e}")
    events.mark("warmup_end")

    events.mark("inference_start", num_items=len(remaining))
    for item in tqdm.tqdm(remaining):
        monitor.start()
        result = chat_generate(
            client, args.model_name, item["prompt"], options, think=think, keep_alive=args.keep_alive
        )
        resource = monitor.stop()

        profile = {"task_id": item["task_id"], "run_id": run_id, **result.profile_fields(), **resource}
        sample = {
            "task_id": item["task_id"],
            "run_id": run_id,
            "model_name": args.model_name,
            "dataset": dataset_key,
            "meta": item.get("meta", {}),
            "completion": result.content,
            "thinking": result.thinking,
            "done_reason": result.done_reason,
        }
        # Write per sample so that a crash during a long run loses at most one item.
        append_jsonl(profile_path, profile)
        append_jsonl(sample_path, sample)

        if args.interval_sec > 0:
            time.sleep(args.interval_sec)
    events.mark("inference_end")

    if not args.keep_loaded:
        events.mark("model_unload_start")
        unload_model(client, args.model_name)
        events.mark("model_unload_end")

    if args.idle_sec > 0:
        events.mark("idle_post_start")
        time.sleep(args.idle_sec)
        events.mark("idle_post_end")

    events.mark("run_end", throttled=read_throttled())
    print(f"Saved samples to : {sample_path}")
    print(f"Saved profiles to: {profile_path}")


def _int_or_none(v: str) -> Optional[int]:
    return None if v.lower() in ("none", "") else int(v)


def _float_or_none(v: str) -> Optional[float]:
    return None if v.lower() in ("none", "") else float(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Ollama model name, e.g. qwen2.5:1.5b")
    parser.add_argument("--dataset_name", required=True, help="human_eval | gsm8k | mmlu-pro")
    parser.add_argument("--data_dir", default="data", help="Directory created by prepare_data.py")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default="rpi", help="Experiment label used in the output path (e.g. rpi5_8gb)")
    parser.add_argument("--max_samples", type=_int_or_none, default=None, help="Limit number of problems ('none' = all)")
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--num_ctx", type=int, default=4096, help="Fixed explicitly; Ollama's default differs across versions")
    parser.add_argument("--num_thread", type=_int_or_none, default=None, help="Defaults to $OLLAMA_NUM_THREADS, else Ollama's choice")
    parser.add_argument("--num_gpu", type=_int_or_none, default=0,
                        help="Layers offloaded to the GPU (0 = CPU only, 'none' = Ollama's choice)")
    parser.add_argument("--think", choices=["auto", "true", "false"], default="auto",
                        help="auto: disable thinking for thinking-capable models, leave unset otherwise")
    parser.add_argument("--keep_alive", default="60m")
    parser.add_argument("--keep_loaded", action="store_true", help="Do not unload the model at the end")
    parser.add_argument("--idle_sec", type=float, default=30.0, help="Idle baseline period before/after each run")
    parser.add_argument("--interval_sec", type=float, default=0.1, help="Pause between samples")
    parser.add_argument("--poll_sec", type=float, default=0.1, help="Resource polling interval during inference")
    parser.add_argument("--cooldown_temp", type=_float_or_none, default=None,
                        help="Wait until CPU temperature <= this value (deg C) before starting ('none' = skip)")
    parser.add_argument("--cooldown_timeout", type=float, default=900.0)
    main(parser.parse_args())
