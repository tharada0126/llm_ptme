import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scienceplots
plt.style.use('science')
from correct_ptme import labels

def take_average(model_name: str, result_dir: str):
    # Read results from JSONL file
    with open(os.path.join(result_dir, f"HE_{model_name}_profiles.jsonl"), "r") as f:
        profile_data = [json.loads(line) for line in f]
    with open(os.path.join(result_dir, f"HE_{model_name}_samples_extracted.jsonl_results.jsonl"), "r") as f:
        results_data = [json.loads(line) for line in f]

    # count "passed" is true in results_data
    passed_count = sum(1 for item in results_data if item.get("passed", False))
    total_count = len(results_data)
    print(f"Passed: {passed_count}/{total_count} ({passed_count/total_count:.2%})")
    average_profile = {
        "cpu_energy": np.mean([item["cpu_energy"] for item in profile_data if item["cpu_energy"] is not None]),
        "dram_energy": np.mean([item["dram_energy"] for item in profile_data if item["dram_energy"] is not None]),
        "time": np.mean([item["time"] for item in profile_data if item["time"] is not None]),
        "prompt_tokens": np.mean([item["prompt_tokens"] for item in profile_data if item["prompt_tokens"] is not None]),
        "completion_tokens": np.mean([item["completion_tokens"] for item in profile_data if item["completion_tokens"] is not None]),
        "memory_peak": np.mean([item["memory_peak"] for item in profile_data if item["memory_peak"] is not None]),
        "rss_final_bytes": np.mean([item["rss_final_bytes"] for item in profile_data if item["rss_final_bytes"] is not None]),
    }
    print(f"Average profile for {model_name}: {average_profile}")
    average_profile["model_name"] = model_name
    average_profile["performance"] = passed_count / total_count
    return average_profile

def get_powercap(result_dir: str):
    with open(os.path.join(result_dir, "cpupower_frequency_info.txt"), "r") as f:
        # check the following line
        # current CPU frequency: 2.00 GHz (asserted by call to kernel)
        # and extract the frequency in GHz
        for line in f:
            if "current CPU frequency" in line and "GHz" in line:
                frequency = float(line.split(":")[1].split("GHz")[0].strip())
                powercap_data = {"current_cpu_frequency": frequency}
                break
    return powercap_data

if __name__ == "__main__":
    main_dir = "outputs_0302"
    model_names = ["tinyllama:1.1b"]
    result_dirs = [f"human_eval_tinyllama_1_1b_{i}" for i in range(5)]
    all_averages = []
    for model_name in model_names:
        for result_dir in result_dirs:
            average_profile = take_average(model_name, os.path.join(main_dir, result_dir))
            powercap_data = get_powercap(os.path.join(main_dir, result_dir))
            average_profile.update(powercap_data)
            all_averages.append(average_profile)
    with open(os.path.join(main_dir, "average_profiles_with_powercap.json"), "w") as f:
        json.dump(all_averages, f, indent=4)
    # make table of csv and tex
    # index is model_names, columns are average_profile keys except model_name
    df = pd.DataFrame(all_averages).set_index("model_name")
    # energy is sum of cpu_energy and dram_energy
    df["energy"] = df["cpu_energy"] + df["dram_energy"]
    df.drop(columns=["cpu_energy", "dram_energy"], inplace=True)
    # energy kJ
    df["energy"] = df["energy"] / 1000
    # memory is rss_final_bytes + memory_peak
    df["memory"] = df["rss_final_bytes"] + df["memory_peak"]
    # memory GB
    df["memory"] = df["memory"] / (1024**3)
    df.drop(columns=["rss_final_bytes", "memory_peak"], inplace=True)
    # performance is performance as percentage
    df["performance"] = df["performance"] * 100
    order = ["current_cpu_frequency", "performance", "time", "memory", "energy", "prompt_tokens", "completion_tokens"]
    df = df[order]
    df.to_csv(os.path.join(main_dir, "average_profiles_with_powercap.csv"))
    with open(os.path.join(main_dir, "average_profiles_with_powercap.tex"), "w") as f:
        # with decimal places of 2 for all columns except performance, which has 4 decimal places
        df_formatted = df.copy()
        for col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].map("{:.2f}".format)
        df_formatted = df_formatted.rename(columns=labels)
        f.write(df_formatted.to_latex())
    
    # scatter plots of current_cpu_frequency vs performance, time, memory, energy 
    print(df)
    for metric in ["performance", "time", "memory", "energy"]:
        plt.figure()
        sns.scatterplot(data=df, x="current_cpu_frequency", y=metric, hue=df.index, style=df.index, s=100)
        plt.xlabel("CPU Frequency (GHz)")
        plt.ylabel(labels[metric])
        plt.legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, f"{metric}_vs_current_cpu_frequency.pdf"))
        plt.close()
    