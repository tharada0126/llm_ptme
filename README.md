# Install anaconda (or miniconda)
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

# Create anaconda environment
```shell
conda env create -f environment.yml
```

# Prepare SWE-bench
```shell
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
pip install -e .
```

# Prepare HumanEval
```shell
git clone https://github.com/openai/human-eval
python -m pip install "setuptools<70"
sed -i 's/human_eval.evaluate_functional_correctness"/human_eval.evaluate_functional_correctness:entry_point"/g' ./human-eval/setup.py
pip install -e human-eval --no-build-isolation
pip install setuptools --upgrade
```

# Prepare GSM8K
```shell
git clone https://github.com/openai/grade-school-math.git
```

# Install ollama
```shell
curl -fsSL https://ollama.com/install.sh | sh
```

# Install calflops
```shell
pip install --upgrade calflops
```

# Install thop
```shell
pip install thop
```

# Evaluate HumanEval outputs
```shell
evaluate_functional_correctness samples.jsonl
```

# Run single run
```shell
# exec_single_run.sh model_name dataset_name output_dir [max_samples:optional]
bash exec_single_test.sh tinyllama:1.1b human_eval output_dir 10
```