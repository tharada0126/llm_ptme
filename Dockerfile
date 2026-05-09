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
    git \
    curl \
    build-essential \
    ca-certificates \
    sed \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama manually.
# For x86_64 Linux:
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | tar --zstd -x -C /usr

RUN python3 -m venv /opt/venv

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ------------------------------------------------------------
# Prepare HumanEval
# ------------------------------------------------------------
RUN git clone https://github.com/openai/human-eval /opt/human-eval \
    && pip install "setuptools<70" \
    && sed -i 's/human_eval.evaluate_functional_correctness"/human_eval.evaluate_functional_correctness:entry_point"/g' /opt/human-eval/setup.py \
    && pip install -e /opt/human-eval --no-build-isolation \
    && pip install --upgrade setuptools

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