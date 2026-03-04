import ollama
import os
import pandas as pd
llms=["tinyllama:1.1b", "phi:2.7b", "phi3:3.8b", "qwen2.5:1.5b", "gemma:2b", "mistral:7b"]

models = ollama.list()
df = pd.DataFrame(index=llms, columns=["Parameter Count", "Size"])
for model in models.models:
    if model.model in llms:
        print(f"Model: {model.model}")
        print(f"Size: {model.size}")
        info = ollama.show(model.model)
        print(f"Parameter count: {info['modelinfo']['general.parameter_count']}")
        df.loc[model.model] = [info['modelinfo']['general.parameter_count'], model.size]
print(df)
df.to_csv("llm_basics.csv")
# convert to GB
df["Size"] = df["Size"].apply(lambda x: (x / (1024 * 1024 * 1024)))
# paremeter count in billions
df["Parameter Count"] = df["Parameter Count"].apply(lambda x: (x / (1000 * 1000 * 1000)))
df.columns = ["Parameter Count ($\\times 10^9$)", "Size (GB)"]
df.to_csv("llm_basics_gb.csv")