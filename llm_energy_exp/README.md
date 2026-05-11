# LLM Energy Experiment Docker Environment

This repository provides a Docker-based environment for running local LLM experiments with Ollama, including energy measurement with pyRAPL and benchmark execution for HumanEval, GSM8K, and MMLU-Pro.

The environment is designed to keep the software stack fixed inside Docker while controlling execution scenarios at runtime using Docker Compose. CPU frequency is controlled on the host using `cpupower`, because Docker cannot directly control CPU frequency.

---

## 1. Overview

The experimental workflow consists of the following steps:

1. Build the Docker image.
2. Pull the required Ollama models.
3. Test the execution environment.
4. Test HumanEval execution with a small sample.
5. Run benchmark experiments using the scenario settings.

The expected directory structure is as follows:

```text
llm_energy_exp/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pull_models.sh
├── run_scenarios.sh
├── main.py
├── energy_profiler.py
├── ollama_utils.py
├── scripts/
│   ├── __init__.py
│   ├── test_human_eval.py
│   ├── test_gsm8k.py
│   └── test_mmlu_pro.py
└── results/
```

---

## 2. Dockerfile

The Docker image installs Python, Ollama, the required Python packages, and the benchmark repositories.

Example `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    build-essential \
    ca-certificates \
    sed \
    procps \
    zstd \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama manually.
# For x86_64 Linux:
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | tar --zstd -x -C /usr

RUN python3 -m venv /opt/venv

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

# ------------------------------------------------------------
# Prepare HumanEval
# ------------------------------------------------------------
RUN git clone https://github.com/openai/human-eval /opt/human-eval \
    && pip install "setuptools<70" \
    && sed -i 's/human_eval.evaluate_functional_correctness"/human_eval.evaluate_functional_correctness:entry_point"/g' /opt/human-eval/setup.py \
    && pip install -e /opt/human-eval --no-build-isolation

# ------------------------------------------------------------
# Prepare GSM8K
# ------------------------------------------------------------
RUN git clone https://github.com/openai/grade-school-math.git /opt/grade-school-math

# ------------------------------------------------------------
# Prepare MMLU-Pro
# ------------------------------------------------------------
RUN git clone https://github.com/TIGER-AI-Lab/MMLU-Pro.git /opt/MMLU-Pro \
    && pip install -r /opt/MMLU-Pro/requirements.txt

ENV PYTHONPATH="/workspace:/opt/human-eval:/opt/grade-school-math:/opt/MMLU-Pro:${PYTHONPATH}"
ENV MMLU_PRO_DIR="/opt/MMLU-Pro"
ENV GSM8K_DIR="/opt/grade-school-math"
ENV HUMAN_EVAL_DIR="/opt/human-eval"

COPY . /workspace

CMD ["bash"]
```

For ARM64 machines, replace the Ollama installation URL with:

```dockerfile
RUN curl -fsSL https://ollama.com/download/ollama-linux-arm64.tar.zst \
    | tar --zstd -x -C /usr
```

---

## 3. Minimal requirements.txt

A minimal `requirements.txt` for this experiment is:

```text
numpy
pandas
tqdm
requests
openai
ollama
psutil
pyRAPL
datasets
matplotlib
scipy
scikit-learn
transformers
thop
sentencepiece
protobuf
accelerate
```

Do not include CUDA, vLLM, xformers, or HumanEval entries in this file unless they are explicitly required. HumanEval is installed separately in the Dockerfile.

---

## 4. docker-compose.yml

Runtime settings such as RAPL access, CPU set, memory limit, and the Ollama model volume should be defined in `docker-compose.yml`.

Example `docker-compose.yml`:

```yaml
services:
  llm-exp:
    build:
      context: .
      dockerfile: Dockerfile
    image: llm-energy-exp:latest
    working_dir: /workspace
    privileged: true

    volumes:
      - .:/workspace
      - ./results:/workspace/results
      - ollama_models:/root/.ollama
      - /sys/class/powercap:/sys/class/powercap:ro

    environment:
      SCENARIO: "${SCENARIO:-full}"
      OLLAMA_NUM_THREADS: "${OLLAMA_NUM_THREADS:-10}"
      MMLU_PRO_DIR: "/opt/MMLU-Pro"
      GSM8K_DIR: "/opt/grade-school-math"
      HUMAN_EVAL_DIR: "/opt/human-eval"
      OLLAMA_HOST: "127.0.0.1:11434"

    command: >
      bash -lc "ollama serve > /tmp/ollama.log 2>&1 & sleep 5 && python main.py --scenario ${SCENARIO:-full} --model_name ${MODEL_NAME:-tinyllama:1.1b} --dataset_name ${DATASET_NAME:-human_eval} --max_samples ${MAX_SAMPLES:-1}"

volumes:
  ollama_models:
```

The `ollama_models` volume stores downloaded Ollama models so that they do not need to be downloaded again for each run.

---

## 5. Build the Docker image

Build the image using Docker Compose:

```shell
docker compose build --no-cache --progress=plain
```

After the build, confirm that the image exists:

```shell
docker images | grep llm-energy-exp
```

---

## 6. Pull the required Ollama models

Create `pull_models.sh`:

```shell
#!/bin/bash
set -euo pipefail

models=(
  "tinyllama:1.1b"
  "qwen2.5:1.5b"
  "phi:2.7b"
  "phi3:3.8b"
  "gemma:2b"
  "mistral:7b"
)

if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
  echo "Ollama server is already running."
  STARTED_SERVER=0
else
  echo "Starting Ollama server..."
  ollama serve > /tmp/ollama_pull.log 2>&1 &
  OLLAMA_PID=$!
  STARTED_SERVER=1
  sleep 5
fi

cleanup() {
  if [ "${STARTED_SERVER}" = "1" ]; then
    echo "Stopping Ollama server..."
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for model in "${models[@]}"; do
  echo "========================================"
  echo "Pulling ${model}"
  echo "========================================"
  ollama pull "${model}"
done

echo "========================================"
echo "Installed models:"
echo "========================================"
ollama list
```

Make it executable:

```shell
chmod +x pull_models.sh
```

Run it inside the Docker environment:

```shell
docker compose run --rm llm-exp bash /workspace/pull_models.sh
```

Confirm the downloaded models:

```shell
docker compose run --rm llm-exp bash -lc "ollama serve > /tmp/ollama.log 2>&1 & sleep 5 && ollama list"
```

---

## 7. Test the execution environment

### 7.1 Check HumanEval import

```shell
docker compose run --rm llm-exp bash -lc \
'python -c "import human_eval.data; print(len(human_eval.data.read_problems()))"'
```

Expected output:

```text
164
```

### 7.2 Check GSM8K files

```shell
docker compose run --rm llm-exp bash -lc \
"ls /opt/grade-school-math/grade_school_math/data"
```

### 7.3 Check MMLU-Pro files

```shell
docker compose run --rm llm-exp bash -lc \
"test -f /opt/MMLU-Pro/cot_prompt_lib/initial_prompt.txt && echo 'MMLU-Pro OK'"
```

### 7.4 Check pyRAPL access

```shell
docker compose run --rm llm-exp bash -lc "ls /sys/class/powercap"
```

Then test pyRAPL:

```shell
docker compose run --rm llm-exp bash -lc "python - <<'PY'
import pyRAPL
pyRAPL.setup()
m = pyRAPL.Measurement('test')
m.begin()
sum(i*i for i in range(1000000))
m.end()
print('pkg:', m.result.pkg)
print('dram:', m.result.dram)
PY"
```

If this fails, check whether `/sys/class/powercap/intel-rapl` is visible on the host.

### 7.5 Check Ollama response

```shell
docker compose run --rm llm-exp bash -lc \
'ollama serve > /tmp/ollama.log 2>&1 & sleep 5 && ollama run tinyllama:1.1b "hello"'
```

---

## 8. Test HumanEval with one sample

Run HumanEval with one sample using TinyLlama:

```shell
MODEL_NAME="tinyllama:1.1b" \
DATASET_NAME="human_eval" \
MAX_SAMPLES="1" \
SCENARIO="test" \
docker compose run --rm llm-exp
```

This command runs the equivalent of:

```shell
ollama serve
python main.py \
  --scenario test \
  --model_name tinyllama:1.1b \
  --dataset_name human_eval \
  --max_samples 1
```

Check the generated files:

```shell
find results -type f | sort
```

A successful run should produce JSONL files for model outputs and resource profiles.

---

### 9. Run E4 scenario for HumanEval

By default, `docker-compose.yml` is intended for full-resource execution without explicit CPU or memory limits. Resource-limited experimental scenarios should be defined in a separate override file, `docker-compose.scenario.yml`.

Create `docker-compose.scenario.yml` as follows:

```yaml
services:
  llm-exp:
    container_name: "llm_exp_${SCENARIO:-E0}"
    cpuset: "${CPUSET_CPUS}"
    mem_limit: "${MEMORY_LIMIT}"
    memswap_limit: "${MEMORY_LIMIT}"
```

With this structure, normal tests can be run with the default Compose file:

```shell
MODEL_NAME="tinyllama:1.1b" \
DATASET_NAME="human_eval" \
MAX_SAMPLES="1" \
docker compose run --rm llm-exp
```

For resource-limited scenarios, use both Compose files:

```shell
docker compose -f docker-compose.yml -f docker-compose.scenario.yml run --rm llm-exp
```

For example, if scenario E4 is defined as:

```text
frequency = 2.2GHz
CPU cores = 2-7
memory    = 14g
threads   = 6
```

First set the CPU frequency on the host:

```shell
sudo cpupower -c 2-7 frequency-set -g performance
sudo cpupower -c 2-7 frequency-set -d 2.2GHz -u 2.2GHz
```

Then run HumanEval with one sample under the E4 resource constraints:

```shell
SCENARIO="E4" \
CPUSET_CPUS="2-7" \
MEMORY_LIMIT="14g" \
OLLAMA_NUM_THREADS="6" \
MODEL_NAME="tinyllama:1.1b" \
DATASET_NAME="human_eval" \
MAX_SAMPLES="1" \
docker compose -f docker-compose.yml -f docker-compose.scenario.yml run --rm llm-exp
```

During execution, check CPU usage:

```shell
docker stats
```

For 6 cores, CPU usage can reach approximately `600%`.

---

## 10. Run multiple execution scenarios

Create `run_scenarios.sh`:

```shell
#!/bin/bash
set -euo pipefail

scenarios=("E1" "E2" "E3" "E4" "E5")
freqs=("1.5GHz" "1.8GHz" "2.0GHz" "2.2GHz" "2.5GHz")
core_ids=("2" "2-3" "2-5" "2-7" "2-9")
memory=("4g" "8g" "12g" "14g" "16g")
threads=("1" "2" "4" "6" "8")

models=(
  "tinyllama:1.1b"
  "qwen2.5:1.5b"
  "phi:2.7b"
  "phi3:3.8b"
  "gemma:2b"
  "mistral:7b"
)

datasets=(
  "human_eval"
  "gsm8k"
  "mmlu-pro"
)

for i in "${!scenarios[@]}"; do
    scenario="${scenarios[$i]}"
    freq="${freqs[$i]}"
    cpuset="${core_ids[$i]}"
    mem="${memory[$i]}"
    nthreads="${threads[$i]}"

    echo "========================================"
    echo "Scenario: ${scenario}"
    echo "Frequency: ${freq}"
    echo "CPU set: ${cpuset}"
    echo "Memory: ${mem}"
    echo "Threads: ${nthreads}"
    echo "========================================"

    sudo cpupower -c "$cpuset" frequency-set -g performance
    sudo cpupower -c "$cpuset" frequency-set -d "$freq" -u "$freq"

    for model in "${models[@]}"; do
      for dataset in "${datasets[@]}"; do
        echo "Running ${scenario}, ${model}, ${dataset}"

        SCENARIO="$scenario" \
        CPUSET_CPUS="$cpuset" \
        MEMORY_LIMIT="$mem" \
        OLLAMA_NUM_THREADS="$nthreads" \
        MODEL_NAME="$model" \
        DATASET_NAME="$dataset" \
        MAX_SAMPLES="" \
        docker compose run --rm llm-exp
      done
    done

    sleep 5
done
```

Make it executable:

```shell
chmod +x run_scenarios.sh
```

Run all scenarios:

```shell
./run_scenarios.sh
```

If `MAX_SAMPLES=""` causes argument parsing issues, remove `--max_samples` from the Compose command or create a separate command for full experiments without the `--max_samples` argument.

---

## 11. Notes on memory measurement

The memory measured by the Python scripts may be larger than expected if multiple Ollama-related processes are counted more than once. When summing RSS across Ollama processes, make sure to deduplicate processes by PID before summing memory usage.

RSS can also overestimate memory usage when shared pages are counted separately across processes. For process-group memory analysis, PSS or USS may be more appropriate when available. However, if RSS is used in the paper, keep the metric consistent and clearly report it as RSS.

---

## 12. Notes on pyRAPL and Docker

pyRAPL reads Intel RAPL counters from:

```text
/sys/class/powercap
```

The Docker container uses:

```yaml
- /sys/class/powercap:/sys/class/powercap:ro
```

and `privileged: true` to make these counters available inside the container.

RAPL energy is not container-exclusive. It usually reflects package-level and DRAM-level energy domains. Therefore, background processes on the same machine can affect the measured energy. For more reliable measurements, run experiments under minimal background load and use fixed CPU sets.

---

## 13. Notes on CPU control

Docker controls which CPU cores are available to the container using `cpuset`, but it does not control CPU frequency. CPU frequency is configured on the host using `cpupower`.

Example:

```shell
sudo cpupower -c 2-7 frequency-set -g performance
sudo cpupower -c 2-7 frequency-set -d 2.2GHz -u 2.2GHz
```

The container is then restricted to the same CPU set:

```shell
CPUSET_CPUS="2-7" docker compose run --rm llm-exp
```

---

## 14. Recommended reporting statement

For reproducibility, the experiment can be described as follows:

```text
All experiments were conducted using a Docker image that fixed the software environment, including Python dependencies, Ollama, and benchmark repositories. Execution environments were controlled at runtime using Docker Compose by varying the CPU set and memory limit, while CPU frequency was configured on the host using cpupower. Ollama and the benchmark scripts were executed inside the same container so that the runtime CPU and memory constraints applied to both the model server and the evaluation code.
```
