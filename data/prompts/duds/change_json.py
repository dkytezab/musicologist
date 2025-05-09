import json
import re
import sys
import unicodedata
from difflib import unified_diff

JSON_FILE    = 'data/prompts/annotations.json'
PROMPTS_FILE = 'data/prompts/prompt.txt'

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
    txt_lines = [line.rstrip('\n') for line in f]

def clean(s):
    # 1) normalize unicode to NFKC, 2) strip whitespace/newlines
    return unicodedata.normalize("NFKC", s).strip()

num_json_rows = len(data)
num_prompt_rows = len(txt_lines)
print(f"Loaded {num_json_rows} JSON objects.")
print(f"Loaded {num_prompt_rows} PROMPT objects.")

real_count = 0
for i, obj in enumerate(txt_lines):
    obj = obj.split('. ', 1)[-1]
    a = clean(obj)
    b = clean(data[real_count]["prompt"])
    if not a == b:
        print(a)
        print(b)
        print(i)
    else:
        real_count += 1

# real_count = 0
# for j, obj in enumerate(data):
#     prompt = obj
#     txt = txt_lines[real_count].split('. ', 1)[-1]
#     a = clean(prompt["prompt"])
#     b = clean(txt)
#     if not a == b:
#         print(a)
#         print(b)
#         print(j)
#     else:
#         real_count += 1



# for i, obj in enumerate(data):


#     obj = obj.split('. ', 1)[-1]
#     a = clean(obj[real_count])
#     b = clean(data[real_count]["prompt"])
#     if not a == b:
#         print(a)
#         print(b)
#         print(i)
#     else:
#         real_count += 1

# for i, obj in enumerate(txt_lines):
#     if i in range(90, 1000):
#         obj = obj.split('. ', 1)[-1]
#         a = clean(obj)
#         b = clean(data[i]["prompt"])
#         if not a == b:
#             print(a)
#             print(b)
#             print(i)