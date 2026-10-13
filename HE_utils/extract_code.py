import re
import json
import argparse

# ```python ... ``` / ```py ... ``` / ``` ... ```
CODE_BLOCK_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
# 閉じフェンスがない (生成が途中で切れた) 場合
OPEN_BLOCK_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*)", re.DOTALL | re.IGNORECASE)

def extract_code(text: str, entry_point: str = None) -> str:
    """
    Extract a Markdown code block.
    Prefer the first block that defines `entry_point`; otherwise use the first block.
    If none exists, return the original text.
    """
    blocks = CODE_BLOCK_RE.findall(text)
    if not blocks:
        match = OPEN_BLOCK_RE.search(text)
        blocks = [match.group(1)] if match else []

    if not blocks:
        return text.strip()

    if entry_point:
        for block in blocks:
            if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
                return block.strip()
    return blocks[0].strip()

def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count

def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)

# receive jsonl file, check "completion" field, extract code, and generated extracted jsonl file
def process_jsonl(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            entry_point = item.get("meta", {}).get("entry_point")
            completion = fix_indents(extract_code(item.get("completion") or "", entry_point))
            # human-eval の入力形式 ({task_id, completion}) のみ出力する
            json.dump({"task_id": item["task_id"], "completion": completion}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Extracted {len(data)} samples -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract code from JSONL file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input *_samples.jsonl file.")
    parser.add_argument("--output_file", type=str, default=None, help="Default: <input>_extracted.jsonl")
    args = parser.parse_args()

    output_file = args.output_file or args.input_file.replace(".jsonl", "_extracted.jsonl")

    process_jsonl(args.input_file, output_file)
