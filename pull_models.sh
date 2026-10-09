#!/bin/bash
set -euo pipefail

models=(
  "tinyllama:1.1b"
  "qwen2.5:1.5b"
  #"phi:2.7b"
  "phi3:3.8b"
  #"gemma:2b"
  "mistral:7b"
  "qwen3.5:0.8b"
  "qwen3.5:2b"
  "gemma4:e2b"
  "gemma4:e4b"
  "phi3.5:3.8b"
  "phi4-mini:3.8b"
  "llama3.2:1b"
  "llama3.2:3b"
  "ministral-3:3b"
)

# On Raspberry Pi, Ollama usually runs as a systemd service; start a server only if none is running.
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

echo "Checking Ollama..."
ollama list

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