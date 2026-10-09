# LLM Performance Measurement on Raspberry Pi

This branch runs local LLMs on a Raspberry Pi with Ollama. It measures **execution time, memory usage and thermal state** on HumanEval, GSM8K and MMLU-Pro.

Differences from the previous x86/Docker experiments:

| Item | x86 (previous) | Raspberry Pi (this branch) |
|---|---|---|
| Energy | pyRAPL (in-process) | **External power meter**, aligned via timestamps |
| Execution environment | Docker with CPU, memory and frequency scenarios (E1–E5) | Native on the device (no envelope) |
| Correctness | Evaluated during the run | **Evaluated afterwards on the desktop PC** (`evaluate.py`) |
| Dataset loading | Benchmark repos and `datasets` inside the container | Prompt files prepared in advance (`data/*.jsonl`) |

## Files

```text
prepare_data.py       # [desktop] build data/*.jsonl (prompts identical to the previous experiments)
data/                 # human_eval (164), gsm8k (200 selected), mmlu_pro (280 = 20 x 14 categories)
setup_rpi.sh          # [Pi] install Ollama + Python venv, configure the Ollama service
pull_models.sh        # [Pi] download models
main.py               # [Pi] run one (model, dataset)
run_experiments.sh    # [Pi] run all combinations
ollama_utils.py       # Ollama client helpers (streaming chat, load/unload, model info)
system_monitor.py     # memory / CPU temperature / frequency / throttling monitor
evaluate.py           # [desktop] score GSM8K / MMLU-Pro, export HumanEval for execution
scripts/*.txt         # selected GSM8K indices and MMLU-Pro question IDs
```

## 1. Prepare data (desktop PC, once)

`data/` is already committed. Regenerate it only if the selection changes:

```shell
pip install -r requirements-desktop.txt
python prepare_data.py   # --mmlu_pro_selected_ids_file scripts/mmlu_pro_selected_ids.txt for 1400 items
```

## 2. Set up the Raspberry Pi

Use Raspberry Pi OS 64-bit.

```shell
./setup_rpi.sh                                  # models in the default location
MODELS_DIR=/mnt/ssd/ollama ./setup_rpi.sh       # models on an external drive (ext4)
```

The script sets `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1` for the systemd service. It also installs `chrony` for time synchronisation.

### Model storage

The generation models in `pull_models.sh` total roughly 45–50 GB. Check the exact sizes with `ollama list`.

- **Recommended:** a USB 3 SSD, or an NVMe SSD on a Pi 5 M.2 HAT, formatted as **ext4**. exFAT/NTFS cannot hold the `ollama` user's file permissions.
- **HDDs are not recommended.** A bus-powered HDD draws power from the Pi, so it becomes part of the measured power and adds noise from spin-up and spin-down. If you must use one, choose a self-powered enclosure and keep the setup identical for all runs.
- Model files are read only during loading, which is recorded as its own phase (`model_load_*`). Storage speed therefore affects load time but not inference measurements.
- **Without enough storage:** run `PULL_MISSING=1 REMOVE_AFTER=1 ./run_experiments.sh`. Each model is downloaded before its runs and removed afterwards, outside the measurement windows.

### Memory

`mistral-nemo:12b` (~7 GB of weights) does not fit comfortably in 8 GB of RAM. Use a 16 GB Pi, or expect swapping. Swapping is recorded in `swap_used_*`.

## 3. Run

```shell
./pull_models.sh

# Smoke test
.venv/bin/python main.py --model_name tinyllama:1.1b --dataset_name gsm8k --max_samples 2 --idle_sec 5 --tag test

# All models x all datasets
TAG=rpi5_16gb ./run_experiments.sh
```

The main `main.py` options:

| Option | Default | Notes |
|---|---|---|
| `--num_predict` | 512 | Same as the previous experiments |
| `--num_ctx` | 4096 | Set explicitly because Ollama's default changes between versions |
| `--num_gpu` | 0 | Layers offloaded to the GPU. 0 = CPU only, so desktop runs are comparable with the Pi. `size_vram_bytes` in `model_load_end` should be 0 |
| `--think` | `auto` | `auto` disables thinking on thinking-capable models (qwen3.5, …) so all models share the 512-token budget |
| `--idle_sec` | 30 | Idle baseline before and after each run, used for meter baseline subtraction |
| `--cooldown_temp` | none (50 in `run_experiments.sh`) | Waits until the CPU is at or below this temperature before each run |
| `--num_thread` | `$OLLAMA_NUM_THREADS` or Ollama's choice | |

Runs can be resumed. Samples are appended one at a time, and already-completed `task_id`s are skipped.

## 4. Outputs

`results/<TAG>/<dataset>/<model>/`:

- `events.jsonl`: phase markers with UNIX time, used to cut the power meter log.
  `cooldown_* → run_start → idle_pre_* → model_load_* → warmup_* → inference_* → model_unload_* → idle_post_* → run_end`.
  `run_start` also carries the run metadata: device model, Ollama version, model digest and quantisation, and the options used.
- `<dataset>_<model>_profiles.jsonl`: one line per sample:
  - `t_start_unix` / `t_end_unix`, the per-sample window used to integrate meter power into energy
  - `latency_sec`, `ttft_sec`
  - Ollama timings: `prompt_eval_duration_ns`, `eval_duration_ns`, `load_duration_ns`
  - Token counts: `prompt_tokens`, `completion_tokens`
  - Memory: `rss_*` (Ollama processes), `sys_mem_used_*`, `swap_used_*`
  - Thermal state: `cpu_temp_*`, `cpu_freq_*`, `throttled_*` (raw `vcgencmd get_throttled`)
- `<dataset>_<model>_samples.jsonl`: completions plus the gold-answer metadata needed for scoring.

### Aligning with the external power meter

- Keep the Pi and the PC that logs the meter synchronised with NTP (`chronyc tracking`).
- Per-sample energy is the integral of meter power over `[t_start_unix, t_end_unix]`.
  Subtract the idle baseline (mean power during `idle_pre`) to obtain the dynamic energy.
- If the meter cannot timestamp its log, record the meter's start time once. Alternatively, insert a recognisable load pulse at the start and use it as a sync marker.

## 5. Evaluate (desktop PC)

```shell
python evaluate.py results/rpi5_16gb
```

- GSM8K and MMLU-Pro: writes `*_scored.jsonl` and `accuracy_summary.json`, using the same extraction rules as before.
- HumanEval: writes `*_human_eval_input.jsonl`. Run it with `evaluate_functional_correctness` from [openai/human-eval](https://github.com/openai/human-eval) in a sandbox.
