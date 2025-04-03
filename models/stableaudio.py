import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

print('model got')

model = model.to(device)

BATCH_SIZE = 4

# Set up text and timing conditioning
conditioning =  BATCH_SIZE * [{
    "prompt": "Beethoven 5th Symphony, 1st movement",
    "seconds_start": 0, 
    "seconds_total": 30
}]
print('done conditioning')

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=50,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
    batch_size=BATCH_SIZE
)

print('done output')
# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")
print(output.shape)
# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
print(output.shape)
## torchaudio.save("output.wav", output, sample_rate)

sample_size = output.shape[1] // BATCH_SIZE  # Compute the size of each sample
output = output.view(BATCH_SIZE, 2, sample_size)  # Reshape into (4, sample_size)

# Save each sample separately
for i, sample in enumerate(output):
    filename = f"outputs/sample_{i}.wav"
    torchaudio.save(filename, sample, sample_rate)
    print(f"Saved {filename}")
