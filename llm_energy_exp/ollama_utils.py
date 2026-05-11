import os
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, List, Tuple

import psutil
from openai import OpenAI
from ollama import Client
from types import SimpleNamespace


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


def collect_unique_processes(procs: List[psutil.Process], include_children: bool = True) -> List[psutil.Process]:
    proc_map = {}

    for p in procs:
        try:
            proc_map[p.pid] = p

            if include_children:
                for c in p.children(recursive=True):
                    proc_map[c.pid] = c

        except psutil.Error:
            pass

    return list(proc_map.values())


def sum_rss_bytes(procs: List[psutil.Process], include_children: bool = True) -> int:
    total = 0
    unique_procs = collect_unique_processes(procs, include_children=include_children)

    for p in unique_procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass

    return total

@dataclass
class RssTrace:
    baseline_rss: int
    peak_rss: int
    final_rss: int


def call_ollama_generate(
    model: str,
    prompt: str,
    system: str,
    num_predict: int = 512,
    num_thread: int | None = None,
    on_first_token=None,
):
    client = Client(host="http://localhost:11434")

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if num_thread is None:
        num_thread_env = os.environ.get("OLLAMA_NUM_THREADS")
        if num_thread_env is not None and num_thread_env != "":
            num_thread = int(num_thread_env)
    
    options = {
        "temperature": 0.0,
        "num_predict": num_predict,
    }

    if num_thread is not None:
        options["num_thread"] = num_thread

    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options,
    )

    first = True
    chunks = []
    usage = None
    finish_reason = None

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")

        if content:
            if first:
                if on_first_token:
                    on_first_token()
                first = False
            chunks.append(content)

        if chunk.get("done", False):
            finish_reason = chunk.get("done_reason")

            prompt_tokens = chunk.get("prompt_eval_count")
            completion_tokens = chunk.get("eval_count")

            if prompt_tokens is not None or completion_tokens is not None:
                usage = SimpleNamespace(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
                )

    full_text = "".join(chunks)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=full_text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )

    return response

def call_ollama_generate_with_memory(
    model: str,
    prompt: str,
    system: str = None,
    num_predict: int = 512,
    poll_interval_sec: float = 0.02,
) -> Tuple[Any, RssTrace]:
    procs = find_ollama_processes()
    baseline = sum_rss_bytes(procs, include_children=True)
    peak = baseline
    result_box: Dict[str, Any] = {}
    err_box: Dict[str, BaseException] = {}
    ttft_box: Dict[str, float] = {}  # 追加

    def _runner() -> None:
        try:
            start = time.perf_counter()  # 追加
            result = call_ollama_generate(
                model=model,
                prompt=prompt,
                system=system,
                num_predict=num_predict,
                on_first_token=lambda: ttft_box.__setitem__("ttft", time.perf_counter() - start),  # 追加
            )
            result_box["result"] = result
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
    return result_box["result"], RssTrace(
        baseline_rss=baseline,
        peak_rss=peak,
        final_rss=final,
    ), ttft_box.get("ttft", None)  # 追加