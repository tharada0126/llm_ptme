"""Run human-eval's functional-correctness check on every extracted sample file.

Usage:
    python HE_utils/evaluate_human_eval_envelops.py results/rpi
"""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import argparse
import json
from pathlib import Path

from human_eval.evaluation import evaluate_functional_correctness

parser = argparse.ArgumentParser()
parser.add_argument("results_dir", nargs="?", default="results/rpi", help="e.g. results/rpi")
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--timeout", type=float, default=3.0)
args = parser.parse_args()

summary = []
sample_files = sorted(Path(args.results_dir).glob("human_eval/*/human_eval_*_samples_extracted.jsonl"))
if not sample_files:
    raise SystemExit(f"No *_samples_extracted.jsonl found under {args.results_dir}/human_eval/")

for sample_file in sample_files:
    dir_name = sample_file.parent.name
    print(f"\n=== Evaluating {dir_name}: {sample_file} ===")
    try:
        results = evaluate_functional_correctness(
            str(sample_file), k=[1], n_workers=args.n_workers, timeout=args.timeout
        )
        pass_at_1 = float(results["pass@1"])
        print(f"{dir_name} results: pass@1={pass_at_1:.4f}")
        summary.append({"model": dir_name, "pass@1": pass_at_1})
    except Exception as e:
        print(f"{dir_name} failed: {e}")
        summary.append({"model": dir_name, "pass@1": None, "error": str(e)})

out = Path(args.results_dir) / "human_eval_summary.json"
with out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved summary to {out}")
