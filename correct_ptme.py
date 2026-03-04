import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scienceplots
plt.style.use('science')

result_dir = "HE_results"
labels = {
    "performance": "$P$ (pass@1)",
    "time": "$T$ (s)",
    "memory": "$M$ (GB)",
    "energy": "$E$ (kJ)",
    "Parameter Count": "Params ($\\times 10^9$)",
    "Size": "Size (GB)",
    "prompt_tokens": "Input Tokens",
    "completion_tokens": "Output Tokens",
    "total_tokens": "Total Tokens",
}
def take_average(model_name: str):
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

def corr_between_tokens_and_tme(result_dir: str, model_names: str):
    df = pd.DataFrame(columns=["model_name", "prompt_tokens", "completion_tokens", "total_tokens", "energy", "time", "memory"])
    for model_name in model_names:
        # Read results from JSONL file
        with open(os.path.join(result_dir, f"HE_{model_name}_profiles.jsonl"), "r") as f:
            profile_data = [json.loads(line) for line in f]
        with open(os.path.join(result_dir, f"HE_{model_name}_samples_extracted.jsonl_results.jsonl"), "r") as f:
            results_data = [json.loads(line) for line in f]
        for profile, result in zip(profile_data, results_data):
            energy = profile["cpu_energy"] + profile["dram_energy"]
            time = profile["time"]
            memory = profile["memory_peak"] + profile["rss_final_bytes"]
            prompt_tokens = profile["prompt_tokens"]
            completion_tokens = profile["completion_tokens"]
            new_row = {
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "energy": energy / 1000,  # convert to kJ
                "time": time,
                "memory": memory / (1024**3),  # convert to GB
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # correlation between prompt_tokens and energy, time, memory for each model
    corrs = {}
    for model_name in model_names:
        model_df = df[df["model_name"] == model_name]
        corrs[model_name] = {
            "prompt_tokens": {
                "energy": model_df["prompt_tokens"].corr(model_df["energy"]),
                "time": model_df["prompt_tokens"].corr(model_df["time"]),
                "memory": model_df["prompt_tokens"].corr(model_df["memory"]),
            },
            "completion_tokens": {
                "energy": model_df["completion_tokens"].corr(model_df["energy"]),
                "time": model_df["completion_tokens"].corr(model_df["time"]),
                "memory": model_df["completion_tokens"].corr(model_df["memory"]),
            },
            "total_tokens": {
                "energy": model_df["total_tokens"].corr(model_df["energy"]),
                "time": model_df["total_tokens"].corr(model_df["time"]),
                "memory": model_df["total_tokens"].corr(model_df["memory"]),
            },
        }
    # save corrs to json
    with open(os.path.join(result_dir, "tokens_tme_correlation.json"), "w") as f:
        json.dump(corrs, f, indent=4)
    # make table of correlation coefficients
    multi_index = pd.MultiIndex.from_product([["prompt_tokens", "completion_tokens", "total_tokens"], ["energy", "time", "memory"]], names=["token_type", "metric"])
    corr_df = pd.DataFrame(index=model_names, columns=multi_index)
    for model_name in model_names:
        for token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            for metric in ["energy", "time", "memory"]:
                corr_df.loc[model_name, (token_type, metric)] = corrs[model_name][token_type][metric]
    corr_df.to_csv(os.path.join(result_dir, "tokens_tme_correlation.csv"))
    # to latex
    with open(os.path.join(result_dir, "tokens_tme_correlation.tex"), "w") as f:
        # with decimal places of 2
        corr_df_formatted = corr_df.copy()
        for col in corr_df_formatted.columns:
            corr_df_formatted[col] = corr_df_formatted[col].apply(lambda x: f"{x:.2f}")
        corr_df_formatted = corr_df_formatted.rename(columns=labels)
        f.write(corr_df_formatted.to_latex())
    # plot scatter plot of prompt_tokens, completion_tokens, total_tokens vs energy, time, memory
    for token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        for metric in ["energy", "time", "memory"]:
            plt.figure()
            sns.scatterplot(data=df, x=token_type, y=metric, hue="model_name", style="model_name")
            plt.xlabel(labels[token_type])
            plt.ylabel(labels[metric])
            plt.legend(title="Model Name")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"{token_type}_vs_{metric}.pdf"))
            plt.close()

if __name__ == "__main__":
    model_names = ["tinyllama:1.1b", "phi:2.7b", "phi3:3.8b", "qwen2.5:1.5b", "gemma:2b", "mistral:7b"]
    all_averages = []
    for model_name in model_names:
        average_profile = take_average(model_name)
        all_averages.append(average_profile)
    with open(os.path.join(result_dir, "average_profiles.json"), "w") as f:
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
    order = ["performance", "time", "memory", "energy", "prompt_tokens", "completion_tokens"]
    df = df[order]
    df.to_csv(os.path.join(result_dir, "average_profiles.csv"))
    with open(os.path.join(result_dir, "average_profiles.tex"), "w") as f:
        # with decimal places of 2 for all columns except performance, which has 4 decimal places
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if col == "performance":
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}")
            else:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")
        f.write(df_formatted.to_latex())
    # scatter matrix of performance, time, memory, energy
    # marker is model_name
    # add model_name as a column so seaborn can use it as a hue (string column name)
    df['model_name'] = df.index
    sns.pairplot(df[["performance", "time", "memory", "energy", "model_name"]], hue="model_name", markers=["o", "s", "D", "^", "v", "P"])
    plt.savefig(os.path.join(result_dir, "scatter_matrix.pdf"))

    # scatter plots of performance vs time, memory, and energy
    # using seaborn and different colors and markers for each model
    # save to separate pdf files
    # set font size 20
    df = pd.read_csv(os.path.join(result_dir, "average_profiles.csv"))
    for i, metric in enumerate(["time", "memory", "energy"]):
        plt.figure()
        sns.scatterplot(data=df, x="performance", y=metric, hue="model_name", style="model_name", s=100)
        plt.xlabel(labels["performance"])
        plt.ylabel(labels[metric])
        # put model names next to the points
        for j in range(df.shape[0]):
            # ha is left if performance is less than the center of the plot, otherwise right
            ha = 'left' if df["performance"][j] < df["performance"].mean() else 'right'
            plt.text(df["performance"][j], df[metric][j], df["model_name"][j], fontsize=10, ha=ha)
        # no legend
        plt.legend().remove()
        # x limits from 0 to 50
        plt.xlim(0, 50)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"performance_vs_{metric}.pdf"))
        plt.close()
    
    basics_df = pd.read_csv("llm_basics.csv", index_col=0)
    # params to billions, size to GB
    basics_df["Parameter Count"] = basics_df["Parameter Count"].apply(lambda x: (x / (1000 * 1000 * 1000)))
    basics_df["Size"] = basics_df["Size"].apply(lambda x: (x / (1024 * 1024 * 1024)))
    # index is model name, columns are parameter count and size
    # set index name to model_name
    basics_df.index.name = "model_name"
    merge_df = df.merge(basics_df, left_on="model_name", right_on="model_name")
    # scatter plot of parameter count, and size, input tokens, and output tokens vs energy
    for metric in ["Parameter Count", "Size", "prompt_tokens", "completion_tokens"]:
        plt.figure()
        sns.scatterplot(data=merge_df, x=metric, y="energy", hue="model_name", style="model_name", s=100)
        plt.xlabel(labels[metric])
        plt.ylabel(labels["energy"])
        for j in range(merge_df.shape[0]):
            ha = 'left' if merge_df[metric][j] < merge_df[metric].mean() else 'right'
            plt.text(merge_df[metric][j], merge_df["energy"][j], merge_df["model_name"][j], fontsize=10, ha=ha)
        plt.legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{metric}_vs_energy.pdf"))
        plt.close()
    
    average_profiles = pd.read_csv(os.path.join(result_dir, "average_profiles.csv"), index_col=0)
    llm_basics = pd.read_csv("llm_basics_gb.csv", index_col=0)

    combined = average_profiles.join(llm_basics)

    # new df with column parameter count, size, performance, time, memory, and energy

    new_columns = ["Parameter Count ($\\times 10^9$)","Size (GB)","performance","time","memory","energy","prompt_tokens","completion_tokens"]
    # remain only these columns
    combined = combined[new_columns]
    new_df = combined.rename(columns=labels)

    new_df.to_csv("combined_results.csv")
    # to text
    with open("combined_results.tex", "w") as f:
        f.write(new_df.to_latex(index=True, float_format="%.2f", column_format="lrrrrrrrr"))
    
    corr_between_tokens_and_tme(result_dir, model_names)