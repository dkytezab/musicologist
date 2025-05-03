import torch
import torchaudio
import yaml
import time
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from typing import Dict, Optional

from load_model import get_diff_model
from utils import get_conditioning_dict, diff_gen_flexible, save_audio, write_sample_to_csv

# Load args from config file
with open('diffusion/diff_config.yml', 'r') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config['model_name']

NUM_BATCHES = config['num_batches']
BATCH_SIZE = config['batch_size']

PROMPT_PATH = config['prompt_path']
OUTPUT_DIR = config['output_dir']

SAMPLE_LENGTH = config['sample_length']
STEPS = config['steps']
EARLY_STOPPING = config['early_stopping']
TRUNCATION_TS = config['truncation_ts']
VERBOSE = config['verbose']

if __name__ == "__main__":

    model, model_config = get_diff_model(MODEL_NAME)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    conditioning = get_conditioning_dict(
        seconds_start=0,
        seconds_total=SAMPLE_LENGTH,
        prompt_path=PROMPT_PATH,   
    )

    for batch in range(NUM_BATCHES):

        batch_start_time = time.time()

        for i, condition in enumerate(conditioning):

                outputs = diff_gen_flexible(
                    model=model,
                    steps=STEPS,
                    index=i,
                    condition=condition,
                    batch_size=BATCH_SIZE,
                    sample_size=sample_size,
                    truncation_ts=TRUNCATION_TS,
                    early_stopping=EARLY_STOPPING,
                    sample_length=SAMPLE_LENGTH,
                )

                #save_audio writes to the csv
                save_audio(
                    audios=outputs,
                    output_dir=OUTPUT_DIR,
                    prompt_index=i,
                    truncation_ts=TRUNCATION_TS,
                    batch=batch,
                    sample_rate=sample_rate,
                    verbose=VERBOSE,
                    sample_length=SAMPLE_LENGTH,
                )


        batch_end_time = time.time()
        total_time = batch_end_time - batch_start_time
        if VERBOSE:
            print(f"Batch {batch} of {BATCH_SIZE} completed in {total_time:.2f} seconds, for {(total_time / BATCH_SIZE):.2f} s / sample.")
