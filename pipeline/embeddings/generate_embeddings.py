from transformers import ClapModel, ClapProcessor
import torch
import torchaudio
import numpy as np

SAMPLE_DIR = 'data/diffusion'
OUTPUT_DIR = 'data/embeddings'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
model = ClapModel.from_pretrained("laion/larger_clap_music").to(DEVICE)
processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

audio_path = f'{SAMPLE_DIR}/prompt_0_step_50_batch_0_sample_0.wav'

def preprocess_audio(audio_path):
    # Load the audio file
    audio_input, sr = torchaudio.load(audio_path)
    # If there are multiple channels, squeeze to a single channel
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    # Resample to 48kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
    audio_input = resampler(audio_input)
    # Truncate to the first 10 seconds (480,000 samples)
    if audio_input.shape[-1] > 480000:
        audio_input = audio_input[..., :480000]
    return audio_input

# Preprocess the audio file
sample = preprocess_audio(audio_path)

# Convert to numpy float32 array for the processor
sample_np = sample.cpu().numpy().astype(np.float32)

# IMPORTANT: Pass the sampling_rate argument to the processor
inputs = processor(audios=sample_np, sampling_rate=48000, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Generate audio embedding
audio_embed = model.get_audio_features(**inputs)

# Save the audio embedding
embedding = audio_embed.cpu().detach().numpy()
np.save(f'{OUTPUT_DIR}/CLAP/audio_embedding.npy', embedding)

print(audio_embed.shape)
