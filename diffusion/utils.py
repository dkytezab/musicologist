from typing import Optional, Dict
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

def get_conditioning_dict(
        seconds_total: int,
        prompt_path: Optional[str] = None,
        seconds_start: int = 0,
    ) -> Dict:
    # Provides a dictionary of conditioned prompts for generation

    with open(prompt_path, 'r') as file:
        prompts = file.readlines()

    conditioning = []

    for prompt in prompts:
        prompt = prompt.strip()
        conditioning.append({
            "prompt": prompt,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        })
    
    return conditioning


def save_audio(
        audio,
        prompt_index: int,
        steps: int,
        batch: int,
        sample_rate: float,
        verbose: bool,
        output_dir: Optional[str] = None,
    ) -> None:

    if output_dir == None:
        output_dir = 'data/audio'
    
    for j, sample in enumerate(audio):

        filename = f"{output_dir}/diff_step_{steps}/prompt_{prompt_index}_sample_{j}.wav"
        torchaudio.save(filename, sample, sample_rate)

        if verbose:
            print(f"Saved {filename}") 
        


def diff_gen_flexible(
        model,
        steps,
        index: int,
        condition: str,
        batch_size,
        sample_size,
        sigma_min: float = 0.3,
        sigma_max: float = 500,
        cfg_scale: int = 7,
        sampler_type: str = "dpmpp-3m-sde",
        early_stopping: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Allows for implementation of generation with and without early_stopping
    
    if early_stopping:
        raise ValueError("Early stopping not presently implemented")
    
    else:
        cond_expanded = [condition] * batch_size
        output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=7,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device,
                batch_size=batch_size,
            ).to(torch.float32)
        
        peak = output.abs().view(output.size(0), -1).max(dim=1, keepdim=True).values
        peak = peak.view(-1, 1, 1)
        output = output / peak
        output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        return output
