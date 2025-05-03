import os
import argparse
import time
import torch
import yaml

from load_model import get_diff_model
from utils import get_conditioning_dict, diff_gen_flexible, save_audio

# Parse args and load config
parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int, required=True, help="SLURM array task ID (0-based)")
parser.add_argument("--num-jobs",  type=int, required=True, help="Total number of array tasks")
args = parser.parse_args()

with open('diffusion/diff_config.yml','r') as f:
    config = yaml.safe_load(f)

MODEL_NAME    = config['model_name']
NUM_BATCHES   = config['num_batches']
BATCH_SIZE    = config['batch_size']
PROMPT_PATH   = config['prompt_path']    # this should be your .txt with one prompt per line
OUTPUT_DIR    = config['output_dir']
SAMPLE_LENGTH = config['sample_length']
STEPS         = config['steps']
EARLY_STOPPING= config['early_stopping']
TRUNCATION_TS = config['truncation_ts']
VERBOSE       = config['verbose']

# Parsing prompts
with open(PROMPT_PATH, 'r') as f:
    all_prompts = [l.strip() for l in f if l.strip()]
total_prompts = len(all_prompts)

# Assigning jobs to GPUs
per_job = (total_prompts + args.num_jobs - 1) // args.num_jobs
start    = args.job_index * per_job
end      = min(total_prompts, start + per_job)
my_prompts = all_prompts[start:end]

if not my_prompts:
    raise RuntimeError(f"Job {args.job_index} has no prompts (range {start}–{end})")

slice_path = f"data/prompts/prompts_job_{args.job_index}.txt"
with open(slice_path, 'w') as f:
    f.write("\n".join(my_prompts))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model, model_config = get_diff_model(MODEL_NAME)
    model.to(device)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    conditioning = get_conditioning_dict(
        seconds_start=0,
        seconds_total=SAMPLE_LENGTH,
        prompt_path=slice_path,    # now only contains this job’s prompts
    )

    for batch in range(NUM_BATCHES):

        batch_start_time = time.time()

        for i, condition in enumerate(conditioning):

                outputs = diff_gen_flexible(
                    model=model,
                    steps=STEPS,
                    condition=condition,
                    batch_size=BATCH_SIZE,
                    sample_size=sample_size,
                    truncation_ts=TRUNCATION_TS,
                    early_stopping=EARLY_STOPPING,
                    sample_length=SAMPLE_LENGTH,
                )

                save_audio(
                    audios=outputs,
                    output_dir=OUTPUT_DIR,
                    prompt_index=i + start,
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
