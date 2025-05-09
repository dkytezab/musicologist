from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import torch
from huggingface_hub import snapshot_download
from typing import Tuple, Optional


def get_embed_model(
    model_name: str = "laion-clap",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ClapModel or MuQMuLan or MusicgenForConditionalGeneration, Optional[AutoProcessor]]: # type: ignore
    # Loads in a user-specified model
    cache_dir = f"models/{model_name}"
    
    if model_name == "laion-clap":
        model_path = "laion/larger_clap_general"
        local_dir = snapshot_download(model_path, cache_dir=cache_dir)

        #model = ClapModel.from_pretrained(model_path).to(device)
        model = ClapModel.from_pretrained(local_dir, local_files_only=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        return (model, processor)
    
    elif model_name == "muq":
        repo = "OpenMuQ/MuQ-MuLan-large"
        local_dir = snapshot_download(repo, cache_dir=cache_dir, revision="8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe")

        #model = MuQMuLan.from_pretrained(model_path).to(device).eval()
        model = (MuQMuLan.from_pretrained(
                     local_dir,
                     local_files_only=True,).to(device).eval())
        processor = None
        return (model, processor)
    
    elif model_name == "musicgen":
        model_path = "facebook/musicgen-small"
        model = MusicgenForConditionalGeneration.from_pretrained(model_path).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        return (model, processor)

    # Add more models here in the future, maybe jukebox layers?

    else:
        raise ValueError("Specified embeddings model not presently supported")