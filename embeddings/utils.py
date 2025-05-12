import torch
import torchaudio
import numpy as np

from transformers import ClapModel, AutoProcessor
from typing import Optional, Tuple
import os
from huggingface_hub import snapshot_download

def get_embed_model(
    model_name: str = "laion-clap",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ClapModel, AutoProcessor]: # type: ignore
    'Loads in CLAP.'
    cache_dir = f"models/{model_name}"
    
    if model_name == "laion-clap":
        model_path = "laion/larger_clap_general"
        local_dir = snapshot_download(model_path, cache_dir=cache_dir)

        model = ClapModel.from_pretrained(local_dir, local_files_only=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        return (model, processor)

    else:
        raise ValueError("Specified embeddings model not presently supported")


def preprocess_audio(
        audio_path,
        khz_freq: int = 48000,
        num_seconds: int = 10,
        ):
    'Preprocesses generated audio for CLAP.'
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Cannot find audio file at {audio_path}")

    audio_input, sr = torchaudio.load(audio_path)
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)

    # Resample to new frequency
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=khz_freq)
    audio_input = resampler(audio_input)

    # Truncate to the first 10 seconds for CLAP (480,000 samples)
    if audio_input.shape[-1] > khz_freq * num_seconds:
        audio_input = audio_input[..., :khz_freq * num_seconds]

    return audio_input


def get_embedding(
        audio_path,
        model,
        processor, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        khz_freq: int = 48000,
        num_seconds: int = 10,
        ):
    'Passes in audio into CLAP, Processor.'
    sample = preprocess_audio(audio_path=audio_path, 
                              khz_freq=khz_freq, 
                              num_seconds=num_seconds,
                              )
    sample_np = sample.cpu().numpy().astype(np.float32)

    if isinstance(model, ClapModel):
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
        audio_tensor: torch.Tensor,
        diff_timestep: int,
        model_name: str,
        out_dir: Optional[str] = None,
        ) -> None:
    'Saves embeddings to directory.'
    
    if out_dir == None:
        out_path = f'data/generated/diff_step_{diff_timestep}/{model_name}_embeddings.pt'
        
    else:
        out_path = f'{out_dir}/{model_name}_embeddings.pt'
    
    torch.save(audio_tensor, out_path)