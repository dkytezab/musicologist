import torch
import torchaudio
import numpy as np
from pathlib import Path
from load_model import get_embed_model
from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import librosa
from typing import Optional
import os


def preprocess_audio(
        audio_path,
        khz_freq: int = 48000,
        num_seconds: int = 10,
        ):
    # Preprocesses audio prior to passing into model
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Cannot find audio file at {audio_path}")

    audio_input, sr = torchaudio.load(audio_path)
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)

    # Resample to given frequency
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=khz_freq)
    audio_input = resampler(audio_input)

    # Truncate to the first 10 seconds (480,000 samples)
    if audio_input.shape[-1] > khz_freq * num_seconds:
        audio_input = audio_input[..., :khz_freq * num_seconds]

    return audio_input


def get_embedding(
        audio_path,
        model,
        processor: None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        khz_freq: int = 48000,
        num_seconds: int = 10,
        ):
    # Passes raw audio into a given model, processor pair
    sample = preprocess_audio(audio_path=audio_path, 
                              khz_freq=khz_freq, 
                              num_seconds=num_seconds,
                              )
    sample_np = sample.cpu().numpy().astype(np.float32)

    if isinstance(model, MuQMuLan):
        model = model.to(device).eval()
        inputs = torch.tensor(sample_np).unsqueeze(0).to(device)
        with torch.no_grad():
            audio_embed = model(wavs = inputs).squeeze(0).cpu()
        

    elif isinstance(model, ClapModel):
        encoding = processor(
            audios=sample_np,
            sampling_rate=khz_freq,
            return_tensors="pt",
        )
        encoding_items = {k: v.to(device) for k, v in encoding.items()}
        model = model.to(device).eval()
        with torch.no_grad():
            embeds = model.get_audio_features(**encoding_items)
        audio_embed = embeds.squeeze(0).cpu()
        return audio_embed

    else:
        raise ValueError("Specified model not presently supported")


def save_embeddings(
        audio_embeds,
        diff_timestep: int,
        out_dir: Optional[str] = None,
        ) -> None:
    
    audio_tensor = torch.stack(audio_embeds)
    
    if out_dir == None:
        out_path = f'data/embeddings/{diff_timestep}_embeddings.pt'
        
    else:
        out_path = f'{out_dir}/{diff_timestep}_embeddings.pt'
    
    torch.save(audio_tensor, out_path)