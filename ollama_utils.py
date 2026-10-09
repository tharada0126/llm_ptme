import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ollama import Client

# Client() reads OLLAMA_HOST from the environment (default http://127.0.0.1:11434).


@dataclass
class GenerationResult:
    content: str
    thinking: str
    done_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    # Wall-clock (UNIX epoch, seconds) timestamps used to align with the external power meter log.
    t_start_unix: float
    t_end_unix: float
    latency_sec: float
    ttft_sec: Optional[float]
    # Server-side timings reported by Ollama (nanoseconds).
    total_duration_ns: Optional[int]
    load_duration_ns: Optional[int]
    prompt_eval_duration_ns: Optional[int]
    eval_duration_ns: Optional[int]

    def profile_fields(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("content", "thinking"):
            d.pop(key)
        return d


def chat_generate(
    client: Client,
    model: str,
    prompt: str,
    options: Dict[str, Any],
    system: Optional[str] = None,
    think: Optional[bool] = None,
    keep_alive: Optional[str] = None,
) -> GenerationResult:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {}
    if think is not None:
        kwargs["think"] = think
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive

    t_start_unix = time.time()
    t0 = time.perf_counter()
    ttft = None
    content_chunks: List[str] = []
    thinking_chunks: List[str] = []
    final = None

    for chunk in client.chat(model=model, messages=messages, stream=True, options=options, **kwargs):
        msg = chunk.message
        content = msg.content or ""
        thinking = getattr(msg, "thinking", None) or ""
        if (content or thinking) and ttft is None:
            ttft = time.perf_counter() - t0
        content_chunks.append(content)
        thinking_chunks.append(thinking)
        if chunk.done:
            final = chunk

    latency = time.perf_counter() - t0
    t_end_unix = time.time()

    return GenerationResult(
        content="".join(content_chunks),
        thinking="".join(thinking_chunks),
        done_reason=getattr(final, "done_reason", None),
        prompt_tokens=getattr(final, "prompt_eval_count", None),
        completion_tokens=getattr(final, "eval_count", None),
        t_start_unix=t_start_unix,
        t_end_unix=t_end_unix,
        latency_sec=latency,
        ttft_sec=ttft,
        total_duration_ns=getattr(final, "total_duration", None),
        load_duration_ns=getattr(final, "load_duration", None),
        prompt_eval_duration_ns=getattr(final, "prompt_eval_duration", None),
        eval_duration_ns=getattr(final, "eval_duration", None),
    )


def load_model(client: Client, model: str, options: Dict[str, Any], keep_alive: str) -> None:
    """Load a model into memory without generating (empty prompt).

    Pass the same options (num_ctx, num_thread) as the inference requests; otherwise Ollama
    loads with its default context length and reloads the model on the first real request.
    """
    client.generate(model=model, prompt="", options=options, keep_alive=keep_alive)


def unload_model(client: Client, model: str) -> None:
    client.generate(model=model, prompt="", keep_alive=0)


def unload_all_models(client: Client) -> List[str]:
    unloaded = []
    for m in client.ps().models:
        unload_model(client, m.model)
        unloaded.append(m.model)
    return unloaded


def loaded_vram_bytes(client: Client, model: str) -> Optional[int]:
    """GPU memory used by a loaded model according to `ollama ps` (0 = running on CPU only)."""
    for m in client.ps().models:
        if m.model == model:
            return m.size_vram
    return None


def model_supports_thinking(client: Client, model: str) -> bool:
    caps = getattr(client.show(model), "capabilities", None) or []
    return "thinking" in caps


def describe_model(client: Client, model: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"model": model}
    for m in client.list().models:
        if m.model == model:
            info["digest"] = m.digest
            info["size_bytes"] = m.size
    show = client.show(model)
    details = getattr(show, "details", None)
    if details is not None:
        info["family"] = details.family
        info["parameter_size"] = details.parameter_size
        info["quantization_level"] = details.quantization_level
    info["capabilities"] = list(getattr(show, "capabilities", None) or [])
    return info


def ollama_version() -> Optional[str]:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "version" in line:
            return line.split()[-1]
    return out.strip() or None
