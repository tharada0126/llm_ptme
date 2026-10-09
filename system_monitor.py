"""Resource monitoring for Raspberry Pi experiments.

Energy is measured by an external power meter, so this module only records
memory, CPU temperature, CPU frequency and throttling state. All readings are
best-effort: on non-Pi hosts the Pi-specific values are simply None.
"""

import glob
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"


def read_cpu_temp_c() -> Optional[float]:
    try:
        return int(THERMAL_PATH.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_cpu_freqs_mhz() -> List[float]:
    freqs = []
    for path in sorted(glob.glob(CPUFREQ_GLOB)):
        try:
            freqs.append(int(Path(path).read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return freqs


def read_throttled() -> Optional[str]:
    """Return the raw `vcgencmd get_throttled` value (e.g. '0x0'), or None."""
    if shutil.which("vcgencmd") is None:
        return None
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return out.split("=", 1)[1] if "=" in out else out
    except (OSError, subprocess.SubprocessError):
        return None


def read_device_model() -> Optional[str]:
    try:
        return Path("/proc/device-tree/model").read_text().rstrip("\x00").strip()
    except OSError:
        return None


def wait_for_cooldown(max_temp_c: float, timeout_sec: float, poll_sec: float = 5.0) -> Optional[float]:
    """Block until the CPU temperature drops to max_temp_c (or timeout). Returns the final temperature."""
    deadline = time.monotonic() + timeout_sec
    temp = read_cpu_temp_c()
    while temp is not None and temp > max_temp_c and time.monotonic() < deadline:
        time.sleep(poll_sec)
        temp = read_cpu_temp_c()
    return temp


def find_ollama_processes() -> List[psutil.Process]:
    """Find the Ollama server and runner processes (matched by executable name)."""
    procs: List[psutil.Process] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.pid == os.getpid():
                continue
            cmdline = p.info.get("cmdline") or []
            exe = os.path.basename(cmdline[0]) if cmdline else ""
            name = p.info.get("name") or ""
            if exe == "ollama" or name == "ollama":
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
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
    for p in collect_unique_processes(procs, include_children=include_children):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def system_mem_used_bytes() -> int:
    vm = psutil.virtual_memory()
    return vm.total - vm.available


class ResourceMonitor:
    """Polls memory / temperature / frequency in a background thread between start() and stop()."""

    def __init__(self, poll_interval_sec: float = 0.1):
        self.poll_interval_sec = poll_interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        # Re-discover processes periodically: the runner process appears only after the model is loaded.
        self._n_polls += 1
        if self._n_polls % 10 == 0:
            self._procs = find_ollama_processes()
        rss = sum_rss_bytes(self._procs)
        self._rss_peak = max(self._rss_peak, rss)
        self._sys_peak = max(self._sys_peak, system_mem_used_bytes())
        self._swap_peak = max(self._swap_peak, psutil.swap_memory().used)
        temp = read_cpu_temp_c()
        if temp is not None:
            self._temp_max = temp if self._temp_max is None else max(self._temp_max, temp)
        freqs = read_cpu_freqs_mhz()
        if freqs:
            self._freq_samples.append(sum(freqs) / len(freqs))
            self._freq_min = min(freqs) if self._freq_min is None else min(self._freq_min, min(freqs))

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval_sec):
            self._sample()

    def start(self) -> None:
        self._procs = find_ollama_processes()
        self._n_polls = 0
        self._rss_baseline = sum_rss_bytes(self._procs)
        self._sys_baseline = system_mem_used_bytes()
        self._swap_baseline = psutil.swap_memory().used
        self._temp_start = read_cpu_temp_c()
        self._throttled_start = read_throttled()
        self._rss_peak = self._rss_baseline
        self._sys_peak = self._sys_baseline
        self._swap_peak = self._swap_baseline
        self._temp_max = self._temp_start
        self._freq_samples: List[float] = []
        self._freq_min: Optional[float] = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        self._stop.set()
        self._thread.join()
        self._sample()
        return {
            "rss_baseline_bytes": self._rss_baseline,
            "rss_peak_bytes": self._rss_peak,
            "rss_final_bytes": sum_rss_bytes(self._procs),
            "sys_mem_used_baseline_bytes": self._sys_baseline,
            "sys_mem_used_peak_bytes": self._sys_peak,
            "swap_used_baseline_bytes": self._swap_baseline,
            "swap_used_peak_bytes": self._swap_peak,
            "cpu_temp_start_c": self._temp_start,
            "cpu_temp_max_c": self._temp_max,
            "cpu_temp_end_c": read_cpu_temp_c(),
            "cpu_freq_mean_mhz": (sum(self._freq_samples) / len(self._freq_samples)) if self._freq_samples else None,
            "cpu_freq_min_mhz": self._freq_min,
            "throttled_start": self._throttled_start,
            "throttled_end": read_throttled(),
        }
