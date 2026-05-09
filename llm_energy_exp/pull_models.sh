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

echo "Starting Ollama server..."
ollama serve > /tmp/ollama_pull.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama server..."
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 5

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