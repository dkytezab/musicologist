import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import librosa
import yaml
import os

from load_model import get_embed_model
from utils import preprocess_audio, get_embedding, save_embeddings

with open("embeddings/embed_config.yml", "r") as file:
    embed_config = yaml.safe_load(file)

MODEL = embed_config['model']
AUDIO_DIR = embed_config['audio_dir']
OUTPUT_DIR = embed_config['output_dir']
NUM_SECONDS = embed_config['num_seconds']
TRUNCATION_TS = embed_config['truncation_ts']
NUM_PROMPTS = embed_config['num_prompts']
BATCH_SIZE = embed_config['batch_size']

if __name__ == "__main__":

    model, processor = get_embed_model(model_name=MODEL)
    print(model)

    for diff_step in TRUNCATION_TS:

        DIFF_DIR = f'{AUDIO_DIR}/diff_step_{diff_step}'
        # Embedding dimension of 512 for CLAP, may need to change for other models
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
        
        print(f"Saving {DIFF_DIR} with {model}")

        save_embeddings(
            audio_tensor=audio_tensor,
            out_dir=DIFF_DIR,
            diff_timestep=diff_step,
            model_name=MODEL,
        )

        print(f'Finished processing {DIFF_DIR}!')