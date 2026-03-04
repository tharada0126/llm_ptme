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

# Evaluate HumanEval outputs
```shell
evaluate_functional_correctness samples.jsonl
```

# Install InstructEval (for humaneval prompt)
