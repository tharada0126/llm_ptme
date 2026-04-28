#llms=("phi:2.7b" "phi3:3.8b" "qwen2.5:1.5b" "mistral:7b")
llms=("tinyllama:1.1b" "phi:2.7b" "phi3:3.8b" "qwen2.5:1.5b" "gemma:2b" "mistral:7b")
#llms=("tinyllama:1.1b")
#dataset=("human_eval" "gsm8k")
dataset=("SWE-bench_Lite")
freq="2.5GHz"
core_id="2-9"
num_scenarios=${#freqs[@]}
mem=16
# governor
ORIG_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)

# turbo
ORIG_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "NA")

echo "Saved governor: $ORIG_GOV"
echo "Saved turbo: $ORIG_TURBO"

OLLAMA=/home/tomohiro/.local/bin/ollama
$OLLAMA serve &
for llm in "${llms[@]}"; do
    for ds in "${dataset[@]}"; do
        echo "Setting CPU frequency to $freq and pinning to cores $core_id"
        cpupower frequency-set -u "$freq"
        cpupower frequency-set -d "$freq"
        echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo
        # set maximum RAM usage to mem GB using cgroups
        #mkdir /sys/fs/cgroup/llm_test
        #echo $((mem * 1024 * 1024 * 1024)) | tee /sys/fs/cgroup/llm_test/memory.max
        #echo 0 | tee /sys/fs/cgroup/llm_test/memory.swap.max

        # directory name for outputs
        # replace : and . with _ in llm name for directory name
        dir_name="${llm//[:.]/_}"
        output_dir="outputs1/${ds}_${dir_name}"
        mkdir -p "$output_dir"
        systemd-run --scope -p MemoryMax=$((mem * 1024 * 1024 * 1024)) -p MemorySwapMax=0\
            taskset -c "$core_id" bash exec_single_test.sh $llm $ds $output_dir
        # output cpupower frequency info to a file
        cpupower frequency-info > "${output_dir}/cpupower_frequency_info.txt"
        # output maximum memory setting to a file
        echo $((mem * 1024 * 1024 * 1024)) > "${output_dir}/memory_max.txt" 
        # return governor and turbo to original settings
        cpupower frequency-set -g "$ORIG_GOV"
        if [ "$ORIG_TURBO" != "NA" ]; then
            echo "$ORIG_TURBO" | tee /sys/devices/system/cpu/intel_pstate/no_turbo
        fi
        #rmdir /sys/fs/cgroup/llm_test
    done
done
pkill ollama