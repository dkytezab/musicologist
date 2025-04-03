import torch
import torchaudio
import yaml
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Load args from config file
with open('pipeline/inference_config.yml', 'r') as file:
    config = yaml.safe_load(file)

NUM_BATCHES = config['num_batches']
BATCH_SIZE = config['batch_size']
SAMPLE_LENGTH = config['sample_length']
STEPS = config['steps']
PROMPT_PATH = config['prompt_path']
OUTPUT_PATH = config['output_path']

# Load prompts from file
with open(PROMPT_PATH, 'r') as file:
    prompts = file.readlines()

# Create conditioning dicts
conditioning = []
for prompt in prompts:
    prompt = prompt.strip()
    conditioning.append({
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": SAMPLE_LENGTH
    })

# Load GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Generate audio using the diffusion model
for batch in range(NUM_BATCHES):
    for i, condition in enumerate(conditioning):
        cond_expanded = [condition] * BATCH_SIZE
        for step_count in STEPS:
            output = generate_diffusion_cond(
                model,
                steps=step_count,
                cfg_scale=7,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device,
                batch_size=BATCH_SIZE
            )

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            sample_size = output.shape[1] // BATCH_SIZE  # Compute the size of each sample
            output = output.view(BATCH_SIZE, 2, sample_size)  # Reshape into (4, sample_size)

            # Save each sample separately
            for j, sample in enumerate(output):
                filename = f"{OUTPUT_PATH}/prompt_{i}_step_{step_count}_batch_{batch}_sample_{j}.wav"
                torchaudio.save(filename, sample, sample_rate)
                print(f"Saved {filename}")
