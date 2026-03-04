import re
import json
import argparse

def extract_code(text: str) -> str:
    """
    Extract the first Markdown code block.
    If none exists, return the original text.
    """
    # ```python ... ``` または ``` ... ``` を抽出
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return text.strip()

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
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        if "completion" in item:
            item["completion"] = fix_indents(extract_code(item["completion"]))

    with open(output_file, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract code from JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()

    output_file = args.input_file.replace(".jsonl", "_extracted.jsonl")

    process_jsonl(args.input_file, output_file)