import json
import torch
from transformers import AutoModelForCausalLM
from calflops import calculate_flops

MODELS = {
    "tinyllama:1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2.5:1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "phi:2.7b":       "microsoft/phi-2",
    "phi3:3.8b":      "microsoft/Phi-3-mini-4k-instruct",
    "gemma:2b":       "google/gemma-2b",       # gemma-2b-it はgated → gemma-2bに変更
    "mistral:7b":     "mistralai/Mistral-7B-Instruct-v0.1",
}

# モデルごとの特殊設定
MODEL_CONFIGS = {
    "phi3:3.8b": {
        "attn_implementation": "eager",  # flash-attention未インストール環境向け
    },
}

BATCH_SIZE = 1
SEQ_LEN = 128

results = {}

for ollama_name, hf_name in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Calculating FLOPs: {ollama_name} ({hf_name})")
    print(f"{'='*60}")

    try:
        extra_kwargs = MODEL_CONFIGS.get(ollama_name, {})

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **extra_kwargs,
        )
        model.eval()

        input_ids = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
        attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

        # まずkwargsで試みる、失敗したらargsで再試行
        try:
            flops, macs, params = calculate_flops(
                model=model,
                kwargs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
                print_results=False,
                print_detailed=False,
                output_as_string=False,
            )
        except Exception as inner_e:
            print(f"  kwargs mode failed ({inner_e}), retrying with args...")
            flops, macs, params = calculate_flops(
                model=model,
                args=[input_ids, attention_mask],
                print_results=False,
                print_detailed=False,
                output_as_string=False,
            )

        results[ollama_name] = {
            "hf_model": hf_name,
            "flops": flops,
            "macs": macs,
            "params": params,
            "flops_str": f"{flops / 1e9:.2f} GFLOPs",
            "macs_str":  f"{macs / 1e9:.2f} GMACs",
            "params_str": f"{params / 1e9:.3f} B",
            "input_shape": [BATCH_SIZE, SEQ_LEN],
        }

        print(f"  FLOPs  : {results[ollama_name]['flops_str']}")
        print(f"  MACs   : {results[ollama_name]['macs_str']}")
        print(f"  Params : {results[ollama_name]['params_str']}")

    except Exception as e:
        print(f"  ERROR: {e}")
        results[ollama_name] = {
            "hf_model": hf_name,
            "error": str(e),
        }

    finally:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()

# JSONに書き出し
output_path = "flops.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {output_path}")
print(f"{'='*60}")

# サマリー表示
print("\nSummary:")
print(f"{'Model':<20} {'FLOPs':>14} {'MACs':>14} {'Params':>10}")
print("-" * 62)
for name, info in results.items():
    if "error" in info:
        msg = info['error'][:30]
        print(f"{name:<20} {'ERROR':>14}  ({msg}...)")
    else:
        print(f"{name:<20} {info['flops_str']:>14} {info['macs_str']:>14} {info['params_str']:>10}")