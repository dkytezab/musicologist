import os
import json
from openai import OpenAI
from typing import List


def load_prompts(file_path):
    'Loads prompts'
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]


def remove_leading_numbers(prompts: List[str]) -> List[str]:
    'Remove leading numbers from each prompt in the list, i.e. "1. Prompt text" becomes "Prompt text".'
    return [prompt.split('. ', 1)[-1] if '. ' in prompt else prompt for prompt in prompts]


key = os.getenv("OPENAI_API_KEY") 
if key is None:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")


def annotate_prompts(prompts: List[str],
                     aspects: List[str],
                     batch_size: int = 5,
                     model: str = "gpt-4.1-mini",
                     output_file: str = None):
    'Annotates prompts using GPT-4.1-mini. We found that a batch size of 5 works best to avoid hallucination/forgetting prompts.'
    annotations = []
    def GENERATION_PROMPT(batch): 
            return f"""
            You are a music expert. You have been tasked with annotating the following prompts with the following aspects: {aspects}. For each of the following prompts, please include which aspects are present in the prompt, and the tempo in BPM if it is provided; otherwise, set the BPM field to 0. They should be returned in the following format:\n
            {json.dumps({
                    "prompt": "<prompt text>",
                    "aspects": [],
                    "bpm": 0
                    }
                , indent=2)}\n
            Output only the JSON objects, one per line, with no additional text or explanation. Here are the prompts:\n
            {batch}
            """
    
    client = OpenAI(api_key=key)
    for i in range(0, len(prompts), batch_size):
        if i + batch_size > len(prompts):
            batch_size = len(prompts) - i
        batch = prompts[i:i + batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": GENERATION_PROMPT(batch)},
            ]
        )
        
        # Process each line as a JSON object
        response_lines = response.choices[0].message.content.strip().split('\n')
        for line in response_lines:
            try:
                annotation = json.loads(line)
                # annotations.append(annotation)
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(',\n')
                        json.dump(annotation, f, indent=2)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")
        print(f"Processed batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        print(f"Processed prompts: {i + 1} to {i + batch_size}")
    
    # Save to file if output_file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    return annotations

if __name__ == "__main__":
    prompts = load_prompts("data/prompts/prompt.txt")
    prompts = remove_leading_numbers(prompts)
    
    # Aspects
    aspects = ["bass", "brass", "drum", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "synth_lead", "vocal",
               "bright sound", "warm sound", "distortion", "fast_decay", "long_release", "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo-synced",
               "acoustic", "electronic", "synthetic"]
    
    # Annotate prompts
    annotations = annotate_prompts(prompts, aspects, batch_size=16, output_file="data/prompts/annotations.json")