llms=("tinyllama:1.1b" "phi:2.7b" "phi3:3.8b" "qwen2.5:1.5b" "gemma:2b" "mistral:7b")
for llm in "${llms[@]}"; do
    python extract_code.py --input_file "HE_${llm}_samples.jsonl"
    evaluate_functional_correctness HE_${llm}_samples_extracted.jsonl
done
python correct_ptme.py