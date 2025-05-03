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

if __name__ == "__main__":

    model, processor = get_embed_model(model_name=MODEL)
    print(model)

    for diff_step in TRUNCATION_TS:

        DIFF_DIR = f'{AUDIO_DIR}/diff_step_{diff_step}'
        audio_embeds = []

        for audio_path in os.listdir(DIFF_DIR):

            print(f'Processing {audio_path}...')

            audio_embed = get_embedding(
                audio_path=f'{AUDIO_DIR}/diff_step_{diff_step}/{audio_path}',
                model=model,
                processor=processor,
                num_seconds=NUM_SECONDS,
            )
            
            audio_embeds.append(audio_embed)
        
        save_embeddings(
            audio_embeds=audio_embeds,
            out_dir=DIFF_DIR,
            diff_timestep=diff_step,
        )