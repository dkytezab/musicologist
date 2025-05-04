from collections import Counter
from typing import List

# TXT_FILE = 'data/prompts/prompt.txt'

# def remove_leading_numbers(prompts: List[str]) -> List[str]:
#     """
#     Remove leading numbers from each prompt in the list, i.e. "1. Prompt text" becomes "Prompt text".
#     """
#     return [prompt.split('. ', 1)[-1] if '. ' in prompt else prompt for prompt in prompts]

# # 1) Read & normalize lines
# with open(TXT_FILE, 'r', encoding='utf-8') as f:
#     lines = remove_leading_numbers([line.strip() for line in f if line.strip()])

# # 2) Count occurrences
# counts = Counter(lines)

# # 3) Extract duplicates
# duplicates = {line: cnt for line, cnt in counts.items() if cnt > 1}

# # 4) Report
# if duplicates:
#     print("Found duplicates:")
#     for line, cnt in duplicates.items():
#         print(f"  • {repr(line)} appears {cnt} times")
# else:
#     print("No duplicate lines found.")

import json
import unicodedata
from collections import Counter

# --- Config ---
JSON_FILE    = 'data/prompts/annotations.json'
PROMPTS_FILE = 'data/prompts/prompt.txt'

# --- Clean function ---
def clean(text):
    return unicodedata.normalize("NFKC", text).strip()

# --- Load JSON prompts ---
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

json_prompts = [clean(obj.get("prompt", "")) for obj in json_data]
json_counts = Counter(json_prompts)

# --- Load and clean .txt prompts ---
with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
    txt_prompts = [clean(line.split('. ', 1)[-1]) for line in f if line.strip()]

# --- Check ---
missing = []
duplicates = []

for i, prompt in enumerate(txt_prompts):
    count = json_counts[prompt]
    if count == 0:
        missing.append((i, prompt))
    elif count > 1:
        duplicates.append((i, prompt, count))

print(f"Total prompts in .txt: {len(txt_prompts)}")
print(f"Total unique prompts in JSON: {len(json_counts)}\n")

if missing:
    print(f"❌ {len(missing)} prompts from .txt not found in JSON:")
    for i, prompt in missing:
        print(f"  • Row {i}: {repr(prompt)}")

if duplicates:
    print(f"\n⚠️ {len(duplicates)} prompts found multiple times in JSON:")
    for i, prompt, count in duplicates:
        print(f"  • Row {i}: {repr(prompt)} appears {count} times")

if not missing and not duplicates:
    print("✅ All .txt prompts match exactly one JSON entry.")