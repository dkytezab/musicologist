from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import torch
from typing import Tuple, Optional


def get_embed_model(
    model_name: str = "laion-clap",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ClapModel or MuQMuLan or MusicgenForConditionalGeneration, Optional[AutoProcessor]]: # type: ignore
    # Loads in a user-specified model
    
    if model_name == "laion-clap":
        model_path = "laion/larger_clap_general"
        model = ClapModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return (model, processor)
    
    elif model_name == "muq":
        model_name = "OpenMuQ/MuQ-MuLan-large"
        model = MuQMuLan.from_pretrained(model_name).to(device).eval()
        processor = None
        return (model, processor)
    
    elif model_name == "musicgen":
        model_name = "facebook/musicgen-small"
        model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return (model, processor)

    # Add more models here in the future, maybe jukebox layers?

    else:
        raise ValueError("Specified embeddings model not presently supported")