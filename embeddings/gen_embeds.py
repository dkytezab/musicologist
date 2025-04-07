import torch
import torchaudio
import numpy as np
from pathlib import Path
from load_model import get_model
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
NUM_DIFFUSION_TIMESTEPS = embed_config['num_diffusion_timesteps']

if __name__ == "__main__":

    model, processor = get_embed_model(model_name=MODEL)

    for diff_timestep in range(NUM_DIFFUSION_TIMESTEPS):

        AUDIO_DIR = f'{AUDIO_DIR}/{diff_timestep}'
        audio_embeds = []

        for audio_path in os.listdir(AUDIO_DIR):

            audio_embed = get_embedding(
                audio_path=audio_path,
                model=model,
                processor=processor,
                num_seconds=NUM_SECONDS,
            )
            audio_embeds.append(audio_embed)
        
        save_embeddings(
            audio_embeds=audio_embeds,
            out_dir=OUTPUT_DIR,
            diff_timestep=diff_timestep,
        )