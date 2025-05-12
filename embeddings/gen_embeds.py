import torch
import numpy as np
import yaml
import os

from utils import preprocess_audio, get_embedding, save_embeddings, get_embed_model

with open("embeddings/embed_config.yml", "r") as file:
    config = yaml.safe_load(file)

# Loading in settings from config
MODEL =             config['model']
AUDIO_DIR =         config['audio_dir']
NUM_SECONDS =       config['num_seconds']
TRUNCATION_TS =     config['truncation_ts']
NUM_PROMPTS =       config['num_prompts']
BATCH_SIZE =        config['batch_size']

if __name__ == "__main__":
    model, processor = get_embed_model(model_name=MODEL)
    for diff_step in TRUNCATION_TS:

        DIFF_DIR = f'{AUDIO_DIR}/diff_step_{diff_step}'
        # Embedding dimension of 512 for CLAP, may need to change for other models
        # Initialize empty tensor for each diffusion step
        audio_tensor = torch.zeros((NUM_PROMPTS, BATCH_SIZE, 512))

        for prompt_index in range(NUM_PROMPTS):
            for sample in range(BATCH_SIZE):
                audio_path = f'{DIFF_DIR}/prompt_{prompt_index}_sample_{sample}.wav'

                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Cannot find audio file at {audio_path}")
                
                print(f'Processing {audio_path}...')

                audio_embed = get_embedding(
                    audio_path=audio_path,
                    model=model,
                    processor=processor,
                    num_seconds=NUM_SECONDS,
                )

                audio_tensor[prompt_index, sample, :] = audio_embed
        
        print(f"Saving {DIFF_DIR} with {model}...")

        save_embeddings(
            audio_tensor=audio_tensor,
            out_dir=DIFF_DIR,
            diff_timestep=diff_step,
            model_name=MODEL,
        )

        print(f'...finished processing {DIFF_DIR}!')