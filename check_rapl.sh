#!/usr/bin/env bash

CPU_RAPL_FILE="/sys/class/powercap/intel-rapl:0/energy_uj"

echo "Checking Intel RAPL support..."
echo

# -----------------------------
# CPU energy check
# -----------------------------
if [ -f "$CPU_RAPL_FILE" ]; then
    echo "CPU energy measurement via Intel RAPL: AVAILABLE"
else
    echo "CPU energy measurement via Intel RAPL: NOT AVAILABLE"
fi

# -----------------------------
# DRAM energy check
# -----------------------------
dram_available=false

for namefile in /sys/class/powercap/intel-rapl:*/name; do
    if [ -f "$namefile" ]; then
        domain=$(cat "$namefile")
        if [ "$domain" = "dram" ]; then
            dram_available=true
            break
        fi
    fi
done

if $dram_available; then
    echo "DRAM energy measurement via Intel RAPL: AVAILABLE"
else
    echo "DRAM energy measurement via Intel RAPL: NOT AVAILABLE"
fi