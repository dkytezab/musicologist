from stable_audio_tools import get_pretrained_model
import torch
from typing import Tuple, Optional

def get_diff_model(
        model_name: str = "stable-diffusion",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ) -> Tuple: # type: ignore
    
    if model_name == "stable-diffusion":
        model_path = "stabilityai/stable-audio-open-1.0"
        model, model_config = get_pretrained_model(model_path)
        return model.to(device), model_config
    
    # Add more diffusion models here

    else:
        raise ValueError("Specified diffusion model not presently supported")
    