from typing import Optional, Dict, List
import os
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from truncation.gen_truncated import generate_truncated_seq

DATA_DIR = "/data/generated"

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
        audios,
        prompt_index: int,
        truncation_ts: List[int],
        batch: int,
        sample_rate: float,
        verbose: bool,
        output_dir: Optional[str] = None,
    ) -> None:

    if output_dir == None:
        output_dir = DATA_DIR
    
    for i, audio in enumerate(audios):
        for j, sample in enumerate(audio):

            #filename = f"{output_dir}/prompt_{prompt_index}/sample_{j}_truncation_{truncation_ts[len(truncation_ts) - i - 1]}.wav"

            filename = f"{output_dir}/diff_step_{truncation_ts[i]}/prompt_{prompt_index}_sample_{j}.wav"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torchaudio.save(filename, sample.cpu(), sample_rate)

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
        truncation_ts = None,
        sample_length: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Allows for implementation of generation with and without early_stopping
    
    if early_stopping:
        if truncation_ts is None:
            raise ValueError("Truncation times must be provided for early stopping.")
        cond_expanded = [condition] * batch_size
        outputs = generate_truncated_seq(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                truncation_ts=truncation_ts,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=device,
                batch_size=batch_size,
            )
        
        new_outpus = []
        for i, output in enumerate(outputs):
            # Normalize the output
            output = output.to(torch.float32)
            peak = output.abs().view(output.size(0), -1).max(dim=1, keepdim=True).values
            peak = peak.view(-1, 1, 1)
            output = output / peak
            output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            if sample_length is not None:
                entries_to_keep = round(sample_length / 47 * output.shape[2])
                output = output[:, :entries_to_keep, :entries_to_keep]
            print(output.shape)
            new_outpus.append(output)

        return new_outpus

    
    else:
        cond_expanded = [condition] * batch_size
        output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=device,
                batch_size=batch_size,
            ).to(torch.float32)
        
        peak = output.abs().view(output.size(0), -1).max(dim=1, keepdim=True).values
        peak = peak.view(-1, 1, 1)
        output = output / peak
        output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        return [output]
