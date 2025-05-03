import os
import json
from openai import OpenAI
from typing import List

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

def remove_leading_numbers(prompts: List[str]) -> List[str]:
    """
    Remove leading numbers from each prompt in the list, i.e. "1. Prompt text" becomes "Prompt text".
    """
    return [prompt.split('. ', 1)[-1] if '. ' in prompt else prompt for prompt in prompts]

def annotate_prompts(prompts: List[str],
                     aspects: List[str],
                     batch_size: int = 16,
                     model: str = "gpt-4.1-mini",
                     output_file: str = None):
    annotations = []
    GENERATION_PROMPT = f"""
        You are a music expert. You have been tasked with annotating the following prompts with the following aspects: {aspects}. For each of the following prompts, please include which aspects are present in the prompt, and the tempo in BPM if it is provided; otherwise, set the BPM field to 0. They should be returned in the following format:\n
        {json.dumps({
                "prompt": "<prompt text>",
                "aspects": [],
                "bpm": 0
                }
            , indent=2)}\n
        Output only the JSON objects, one per line, with no additional text or explanation. Here are the prompts:\n
        {prompts}
        """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(0, len(prompts), batch_size):
        if i + batch_size > len(prompts):
            batch_size = len(prompts) - i
        batch = prompts[i:i + batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": GENERATION_PROMPT}
            ]
        )
        
        # Process each line as a JSON object
        response_lines = response.choices[0].message.content.strip().split('\n')
        for line in response_lines:
            try:
                annotation = json.loads(line)
                annotations.append(annotation)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")
    
    # Save to file if output_file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    return annotations

if __name__ == "__main__":
    # Load prompts from file
    # prompts = load_prompts("data/prompts/chat_gen_prompts.txt")
    prompts = [
        "1. Several ukuleles are playing the same strumming melody together with an acoustic guitar. Someone is playing a shaker slightly offbeat. The song is in 4/4 time and has a tempo of 120 BPM. The mood is happy and upbeat.",
        "2. A solo piano piece with a slow tempo of 60 BPM. The mood is melancholic and introspective, with a focus on minor chords and arpeggios.",
        "3. A fast-paced electronic track with a tempo of 140 BPM. The mood is energetic and uplifting, featuring synthesizers, drum machines, and vocal samples."
    ]
    # Remove leading numbers from prompts
    prompts = remove_leading_numbers(prompts)
    
    # Define aspects to annotate
    aspects = ["joyful", "sad", "energetic"]
    
    # Annotate prompts
    annotations = annotate_prompts(prompts, aspects, batch_size=16, output_file="data/prompts/annotations.json")
    
    # Print the first few annotations
    for annotation in annotations:
        print(annotation)
