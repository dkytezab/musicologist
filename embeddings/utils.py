import torch
import torchaudio
import numpy as np
from pathlib import Path
from load_model import get_embed_model
from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import librosa
from typing import Optional


def preprocess_audio(
        audio_path,
        khz_freq: int = 48000,
        num_seconds: int = 10,
        ):
    # Preprocesses audio prior to passing into model

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
        model_name: str = "laion-clap",
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

    if model_name == "muq-mulan":
        model = model.to(device).eval()
        inputs = torch.tensor(sample_np).unsqueeze(0).to(device)
        with torch.no_grad():
            audio_embed = model(wavs = inputs).squeeze(0).cpu()
        

    elif model_name == "laion-clap":
        inputs = processor(audios=sample_np, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        audio_embed = model.get_audio_features(**inputs)
        return audio_embed.squeeze(0).cpu()


    else:
        raise ValueError("Specified model not presently supported")


def save_embeddings(
        model_name,
        audio_embeds,
        diff_timestep: int,
        out_dir: Optional[str] = None,
        ) -> None:
    
    audio_tensor = torch.stack(audio_embeds)
    
    if out_dir == None:
        out_path = f'data/generated/diff_step_{diff_timestep}/{model_name}_embeddings.pt'
        
    else:
        out_path = f'{out_dir}/diff_step_{diff_timestep}/{model_name}_embeddings.pt'
    
    torch.save(audio_tensor, out_path)