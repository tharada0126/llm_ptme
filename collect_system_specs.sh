#!/usr/bin/env bash
# collect_system_specs.sh
# 論文用の計算環境スペックを収集して出力するスクリプト
# 対応OS: Linux / MacOS

set -euo pipefail

OUTPUT_FILE="system_specs.txt"

{
echo "================================================================"
echo " System Specifications for Paper"
echo " Generated: $(date)"
echo "================================================================"
echo ""

# ============================================================
# OS
# ============================================================
echo "[ OS ]"
if [[ "$(uname)" == "Darwin" ]]; then
    sw_vers
    echo "Kernel: $(uname -r)"
else
    # Linux
    if [[ -f /etc/os-release ]]; then
        cat /etc/os-release | grep -E "^(NAME|VERSION|PRETTY_NAME)="
    fi
    echo "Kernel: $(uname -r)"
fi
echo ""

# ============================================================
# CPU
# ============================================================
echo "[ CPU ]"
if [[ "$(uname)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string
    echo "Physical cores : $(sysctl -n hw.physicalcpu)"
    echo "Logical cores  : $(sysctl -n hw.logicalcpu)"
    echo "Frequency      : $(sysctl -n hw.cpufrequency_max 2>/dev/null \
        | awk '{printf "%.2f GHz\n", $1/1e9}' 2>/dev/null || echo "N/A")"
else
    # Linux
    grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs
    echo "Physical cores : $(grep "^cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "Logical cores  : $(nproc)"
    echo "Threads/core   : $(lscpu | grep "Thread(s) per core" | cut -d: -f2 | xargs)"
    # 動作周波数
    if command -v lscpu &>/dev/null; then
        lscpu | grep -E "CPU MHz|CPU max MHz|CPU min MHz" | \
            awk -F: '{printf "%-20s %s MHz\n", $1, $2}' | xargs -I{} echo "  {}"
    fi
    # キャッシュ
    lscpu | grep -E "L[123]" | awk -F: '{printf "%-20s %s\n", $1, $2}' | xargs -I{} echo "  {}" 2>/dev/null || true
fi
echo ""

# ============================================================
# Memory (RAM)
# ============================================================
echo "[ Memory ]"
if [[ "$(uname)" == "Darwin" ]]; then
    TOTAL=$(sysctl -n hw.memsize)
    echo "Total RAM : $(echo "$TOTAL" | awk '{printf "%.0f GB\n", $1/1024/1024/1024}')"
else
    # Linux
    awk '
        /MemTotal/  { printf "Total RAM : %.1f GB\n", $2/1024/1024 }
        /MemFree/   { printf "Free      : %.1f GB\n", $2/1024/1024 }
        /MemAvailable/ { printf "Available : %.1f GB\n", $2/1024/1024 }
    ' /proc/meminfo
    # DMIデコードで物理メモリ詳細（要sudo）
    if command -v dmidecode &>/dev/null && [[ $EUID -eq 0 ]]; then
        echo ""
        echo "  Physical Memory Modules:"
        dmidecode -t memory 2>/dev/null | \
            grep -E "Size:|Type:|Speed:|Manufacturer:|Part Number:" | \
            grep -v "No Module" | \
            awk '{printf "    %s\n", $0}'
    fi
fi
echo ""

# ============================================================
# GPU (あれば)
# ============================================================
echo "[ GPU ]"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | \
        awk -F',' '{printf "  GPU    : %s\n  VRAM   : %s\n  Driver : %s\n", $1, $2, $3}'
elif [[ "$(uname)" == "Darwin" ]]; then
    system_profiler SPDisplaysDataType 2>/dev/null | \
        grep -E "Chipset Model|VRAM" | \
        awk -F: '{printf "  %-15s %s\n", $1, $2}' || echo "  N/A"
else
    echo "  nvidia-smi not found / No NVIDIA GPU"
fi
echo ""

# ============================================================
# Storage
# ============================================================
echo "[ Storage ]"
if [[ "$(uname)" == "Darwin" ]]; then
    df -h / | awk 'NR==2 {printf "Root disk: %s total, %s used, %s available\n", $2, $3, $4}'
else
    df -h / | awk 'NR==2 {printf "Root disk: %s total, %s used, %s available\n", $2, $3, $4}'
fi
echo ""

# ============================================================
# Python / 主要ライブラリ
# ============================================================
echo "[ Python Environment ]"
python3 --version 2>/dev/null || echo "Python3: not found"
python3 - <<'PYEOF' 2>/dev/null || echo "  (pip packages unavailable)"
import importlib, sys
print(f"  Python : {sys.version.split()[0]}")
libs = [
    "torch", "transformers", "datasets",
    "openai", "ollama", "swebench",
    "numpy", "pandas",
]
for lib in libs:
    try:
        m = importlib.import_module(lib)
        ver = getattr(m, "__version__", "unknown")
        print(f"  {lib:<15} {ver}")
    except ImportError:
        pass
PYEOF
echo ""

echo "================================================================"
echo " End of System Specifications"
echo "================================================================"

} | tee "${OUTPUT_FILE}"

echo ""
echo "Saved to: ${OUTPUT_FILE}"