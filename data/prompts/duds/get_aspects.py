import os
import json
import csv
import itertools
import google.generativeai as genai
import re
from google.generativeai.types import GenerationConfig

API_KEY = 111

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash-lite"  
json_generation_config = GenerationConfig(
  response_mime_type="application/json"
)
# see json only mode return
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config=json_generation_config 
)

ASPECTS_TXT = "data/prompts/aspects.txt"
PROMPTS_TXT = "data/prompts/prompt.txt"
OUTPUT_CSV = "data/prompts/prompt_tags.csv"
OUTPUT_JSON   = "data/prompts/prompt_tags_2.json"

def load_concepts(path=ASPECTS_TXT):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def load_prompts(path=PROMPTS_TXT):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            return
        yield batch

CATEGORIES = [
    "genre",
    "instruments",
    "mood",
    "tempo",
    "audio_quality",
    "performance_context",
    "vocals",
    "style",
    "tempo",
]

TEST = False  # test mode, only process first batch

def main():
    concepts = load_concepts()
    prompts  = load_prompts()
    prompts = prompts[:1] # 1001 manual
    all_results = []
    if TEST:
        prompts = prompts[:40] # 40 for test
        concepts = concepts[:5]
      
    print(f"all {len(concepts)}  music concept, {len(prompts)} prompt")

    for batch_idx, batch in enumerate(chunked(prompts, 20), start=1):

        # system +  prompt
        sys_section = (
            "Below is a list of possible music concepts (one per line), make sure it is exactly match the json format in order for me to load:\n"
            + "\n".join(f"- {c}" for c in concepts)
            + "\n\n"
            + "For each user prompt below, select only from the above list the concepts that apply, and return a JSON array. "
            + "Each element should follow this template exactly:\n"
            + json.dumps({
                "prompt": "<prompt text>",
                "genre": [],
                "instruments": [],
                "mood": [],
                "tempo": [],
                "audio_quality": [],
                "performance_context": [],
                "vocals": [],
                "style": [],
                "tempo": [],
            }, indent=2)
            + "\nOutput only the JSON array."
        )
        # id
        start_idx = (batch_idx - 1) * 20
        user_section = "\n\n".join(
            f"Prompt {start_idx + i + 1}: {p}" for i, p in enumerate(batch)
        )
        full_prompt = sys_section + "\n\n" + user_section
        
        resp = model.generate_content(full_prompt)
        text = resp.text.strip()

        print(f"Response text: '{text}'")

        # load to Json
        try:
            results = json.loads(text)
            all_results.extend(results)

        except json.JSONDecodeError as e:
            print("JSON :", e)
            raise

        if TEST:
            print("test")

        # save JSON 
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"finished {OUTPUT_JSON}")

if __name__ == "__main__":
    main()