import json
import torch
from transformers import AutoModelForCausalLM
from thop import profile

MODELS = {
    "tinyllama:1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2.5:1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "phi:2.7b":       "microsoft/phi-2",
    "phi3:3.8b":      "microsoft/Phi-3-mini-4k-instruct",
    "gemma:2b":       "google/gemma-2b",
    "mistral:7b":     "mistralai/Mistral-7B-Instruct-v0.1",
}

# モデルごとの特殊設定
MODEL_CONFIGS = {
    "phi3:3.8b": {
        "attn_implementation": "eager",
        "trust_remote_code": False,  # カスタムコードのバグを回避、transformers組み込み実装を使用
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

        # trust_remote_codeがMODEL_CONFIGSで指定されていない場合はTrueをデフォルトに
        if "trust_remote_code" not in extra_kwargs:
            extra_kwargs["trust_remote_code"] = True

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.float32,
            **extra_kwargs,
        )
        model.eval()

        input_ids = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

        flops, params = profile(model, inputs=(input_ids,), verbose=False)

        results[ollama_name] = {
            "hf_model": hf_name,
            "flops": flops,
            "macs": flops / 2,
            "params": params,
            "flops_str": f"{flops / 1e9:.2f} GFLOPs",
            "macs_str":  f"{flops / 2 / 1e9:.2f} GMACs",
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
        print(f"{name:<20} {'ERROR':>14}")
    else:
        print(f"{name:<20} {info['flops_str']:>14} {info['macs_str']:>14} {info['params_str']:>10}")