import os
from openai import OpenAI

N_PROMPTS = 250
GENERATION_PROMPT = f"""
You are a music expert. You have been tasked with generating a list of {N_PROMPTS} prompts for a music generation model. Each prompt should be a short description of a musical piece, including the genre, mood, tempo, and any specific instruments or elements that should be included. The prompts should be diverse and cover a wide range of styles and themes. You should return nothing but the prompts, separated by newlines. Here are some examples:
1. Several ukuleles are playing the same strumming melody together with an acoustic guitar. Someone is playing a shaker slightly offbeat. The song is in 4/4 time and has a tempo of 120 BPM. The mood is happy and upbeat.
2. A solo piano piece with a slow tempo of 60 BPM. The mood is melancholic and introspective, with a focus on minor chords and arpeggios.
3. A fast-paced electronic track with a tempo of 140 BPM. The mood is energetic and uplifting, featuring synthesizers, drum machines, and vocal samples.
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

full_out = ""

for _ in range(4):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": GENERATION_PROMPT}
        ]
    )
    full_out += response.choices[0].message.content + "\n"

with open("data/prompts/chat_gen_prompts.txt", "w") as f:
    f.write(full_out)
