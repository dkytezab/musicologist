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

MODELS = embed_config['models']
AUDIO_DIR = embed_config['audio_dir']
OUTPUT_DIR = embed_config['output_dir']
NUM_SECONDS = embed_config['num_seconds']
STEPS = embed_config['steps']

if __name__ == "__main__":

    for MODEL in MODELS:

        model, processor = get_embed_model(model_name=MODEL)

        for diff_timestep in STEPS:

            TEMP_AUDIO_DIR = f'{AUDIO_DIR}/diff_step_{diff_timestep}'

            audio_embeds = []
            audio_paths = [f for f in os.listdir(TEMP_AUDIO_DIR) if f.endswith('.wav')]

            for audio_path in audio_paths:

                audio_embed = get_embedding(
                    audio_path=f'{TEMP_AUDIO_DIR}/{audio_path}',
                    model=model,
                    model_name=MODEL,
                    processor=processor,
                    num_seconds=NUM_SECONDS,
                )
                audio_embeds.append(audio_embed)
            
            save_embeddings(
                model_name=MODEL,
                audio_embeds=audio_embeds,
                out_dir=OUTPUT_DIR,
                diff_timestep=diff_timestep,
            )

            del audio_embed, audio_embeds, audio_paths
            torch.cuda.empty_cache()
        
        del model, processor
        torch.cuda.empty_cache()