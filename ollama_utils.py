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


"""
def call_ollama_generate(model: str, prompt: str, system: str, num_predict: int = 512):
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=None
    )
    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': 'You are an expert software engineer.'})
    messages.append({"role": "user", "content": prompt})
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=0.0,
        max_tokens=num_predict,
    )


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

    def _runner() -> None:
        try:
            result_box["result"] = call_ollama_generate(model=model, prompt=prompt, system=system, num_predict=num_predict)
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
    return result_box["result"], RssTrace(baseline_rss=baseline, peak_rss=peak, final_rss=final), None
"""
"""
def call_ollama_generate(
    model: str,
    prompt: str,
    system: str,
    num_predict: int = 512,
    on_first_token=None,
):
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=None
    )
    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': 'You are an expert software engineer.'})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},  # 追加：最後のチャンクにusageを含める
        temperature=0.0,
        max_tokens=num_predict,
    )

    first = True
    chunks = []
    usage = None
    finish_reason = None
    # for debug
    chunk_count = 0
    choice_chunk_count = 0
    usage_chunk_count = 0
    finish_reason_chunk_count = 0

    for chunk in stream:
        # debug
        chunk_count += 1
        print(f"Received chunk #{chunk_count}")
        print(chunk.model_dump_json(indent=2))
        # debug end
        
        # usageは最後のチャンクに含まれる
        if chunk.usage is not None:
            usage = chunk.usage
            # debug
            usage_chunk_count += 1
            # debug end

        if chunk.choices:
            # debug
            choice_chunk_count += 1
            # debug end
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                # debug
                finish_reason_chunk_count += 1
                # debug end
                
            content = choice.delta.content
            if content:
                if first:
                    if on_first_token:
                        on_first_token()
                    first = False
                chunks.append(content)

    full_text = "".join(chunks)
    # 追加：デバッグ表示
    print("=== Ollama generation debug ===")
    print(f"finish_reason = {finish_reason}")
    print(f"usage is None = {usage is None}")
    print(f"generated_text_length = {len(full_text)}")
    print(f"last_300_chars = {full_text[-300:]}")
    print(f"chunk_count = {chunk_count}")
    print(f"choice_chunk_count = {choice_chunk_count}")
    print(f"usage_chunk_count = {usage_chunk_count}")
    print(f"finish_reason_chunk_count = {finish_reason_chunk_count}")
    print("===============================")

    # 呼び出し元と互換性のあるレスポンスオブジェクトを再構築
    class _Message:
        content = full_text

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None

    response = _Response()
    response.usage = usage  # completion_tokens, prompt_tokens が含まれる

    return response
"""
from ollama import Client
from types import SimpleNamespace


def call_ollama_generate(
    model: str,
    prompt: str,
    system: str,
    num_predict: int = 512,
    on_first_token=None,
):
    client = Client(host="http://localhost:11434")

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": 0.0,
            "num_predict": num_predict,
        },
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
    print(full_text)

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