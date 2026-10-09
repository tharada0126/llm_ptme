#!/bin/bash
# One-time setup on Raspberry Pi OS (64-bit).
#
# Usage:
#   ./setup_rpi.sh                     # models stored in the default location (/usr/share/ollama/.ollama/models)
#   MODELS_DIR=/mnt/ssd/ollama ./setup_rpi.sh   # store models on an external drive (must be ext4)
set -euo pipefail

cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip curl chrony

if ! command -v ollama > /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Ollama server settings for measurement: one model and one request at a time.
OVERRIDE_DIR=/etc/systemd/system/ollama.service.d
sudo mkdir -p "$OVERRIDE_DIR"
{
  echo "[Service]"
  echo 'Environment="OLLAMA_MAX_LOADED_MODELS=1"'
  echo 'Environment="OLLAMA_NUM_PARALLEL=1"'
  if [[ -n "${MODELS_DIR:-}" ]]; then
    echo "Environment=\"OLLAMA_MODELS=${MODELS_DIR}\""
  fi
} | sudo tee "$OVERRIDE_DIR/override.conf" > /dev/null

if [[ -n "${MODELS_DIR:-}" ]]; then
  sudo mkdir -p "$MODELS_DIR"
  sudo chown -R ollama:ollama "$MODELS_DIR"
fi

sudo systemctl daemon-reload
sudo systemctl restart ollama

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "----------------------------------------"
ollama --version
systemctl show ollama --property=Environment
chronyc tracking | grep -E "Reference|System time" || true
echo "Setup finished."
