import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, List, Tuple

import psutil
from openai import OpenAI


def find_ollama_processes() -> List[psutil.Process]:
    procs: List[psutil.Process] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = p.info
            cmd = " ".join(info.get("cmdline") or [])
            name = info.get("name") or ""
            if "ollama" in name.lower() or "ollama" in cmd.lower():
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    def score(proc: psutil.Process) -> int:
        try:
            cmd = " ".join(proc.cmdline()).lower()
            return 0 if "serve" in cmd else 1
        except Exception:
            return 2

    procs.sort(key=score)
    return procs


def sum_rss_bytes(procs: List[psutil.Process], include_children: bool = True) -> int:
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
            if include_children:
                for c in p.children(recursive=True):
                    try:
                        total += c.memory_info().rss
                    except psutil.Error:
                        pass
        except psutil.Error:
            pass
    return total


@dataclass
class RssTrace:
    baseline_rss: int
    peak_rss: int
    final_rss: int


def call_ollama_generate(model: str, prompt: str, num_predict: int = 512):
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=0.0,
        max_tokens=num_predict,
    )


def call_ollama_generate_with_memory(
    model: str,
    prompt: str,
    num_predict: int = 512,
    poll_interval_sec: float = 0.02,
) -> Tuple[Any, RssTrace]:
    procs = find_ollama_processes()
    baseline = sum_rss_bytes(procs, include_children=True)
    peak = baseline
    result_box: Dict[str, Any] = {}
    err_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["result"] = call_ollama_generate(model=model, prompt=prompt, num_predict=num_predict)
        except BaseException as e:
            err_box["err"] = e

    t = Thread(target=_runner, daemon=True)
    t.start()
    while t.is_alive():
        peak = max(peak, sum_rss_bytes(procs, include_children=True))
        time.sleep(poll_interval_sec)
    t.join()
    final = sum_rss_bytes(procs, include_children=True)
    peak = max(peak, final)

    if "err" in err_box:
        raise err_box["err"]
    return result_box["result"], RssTrace(baseline_rss=baseline, peak_rss=peak, final_rss=final)
