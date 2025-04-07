from transformers import ClapModel, ClapProcessor
import torch
import torchaudio
import numpy as np
from pathlib import Path

SAMPLE_DIR = Path('data/diffusion')
OUTPUT_DIR = 'data/embeddings/CLAP'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ClapModel.from_pretrained("laion/larger_clap_music").to(DEVICE)
processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
wav_paths = sorted(SAMPLE_DIR.glob("*.wav"))
print(wav_paths)

def preprocess_audio(audio_path):
    audio_input, sr = torchaudio.load(audio_path)
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    # Resample to 48kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
    audio_input = resampler(audio_input)
    # Truncate to the first 10 seconds (480,000 samples)
    if audio_input.shape[-1] > 480000:
        audio_input = audio_input[..., :480000]
    return audio_input

all_embeddings = []

for audio_path in wav_paths:
    sample = preprocess_audio(audio_path)
    sample_np = sample.cpu().numpy().astype(np.float32)
    inputs = processor(audios=sample_np, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    embedding = model.get_audio_features(**inputs)
    all_embeddings.append(embedding.squeeze(0).cpu())

embeddings_tensor = torch.stack(all_embeddings)
torch.save(embeddings_tensor, f'{OUTPUT_DIR}/embeddings.pt')

print(embeddings_tensor.shape)