from typing import Optional, Dict, List
import os
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from truncation.gen_truncated import generate_truncated_seq, generate_perturbations

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
        batch: int,
        sample_rate: float,
        verbose: bool,
        output_dir: Optional[str] = None,
        experiment_type: str = "truncation",
        perturbation_t: int = None,
        truncation_ts: List[int] = None,
    ) -> None:

    if experiment_type == "perturbation":
        if output_dir == None:
            output_dir = 'data/perturbation'
        
        for i, audio in enumerate(audios):
            for j, sample in enumerate(audio):
                print(f"Saving {i} {j}")

                filename = f"{output_dir}/perturbation_{perturbation_t}/prompt_{prompt_index}_batch_{batch}_sample_{i}_{j}.wav"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torchaudio.save(filename, sample.cpu(), sample_rate)

                if verbose:
                    print(f"Saved {filename}")
    else:
        if output_dir == None:
            output_dir = 'data/generated'
        
        for i, audio in enumerate(audios):
            for j, sample in enumerate(audio):

                #filename = f"{output_dir}/prompt_{prompt_index}/sample_{j}_truncation_{truncation_ts[len(truncation_ts) - i - 1]}.wav"

                filename = f"{output_dir}/diff_step_{truncation_ts[i]}/prompt_{prompt_index}_sample_{i}.wav"
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
        experiment_type: str = "truncation",
        truncation_ts = None,
        perturbation_t = None,
        n_perturbations: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Allows for implementation of generation with and without early_stopping
    
    if experiment_type == "truncation":
        if truncation_ts is None:
            raise ValueError("Truncation times must be provided for early stopping.")
        cond_expanded = [condition] * batch_size
        outputs = generate_truncated_seq(
                model,
                steps=steps,
                cfg_scale=7,
                truncation_ts=truncation_ts,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device,
                batch_size=batch_size,
            )
        
        for i, output in enumerate(outputs):
            # Normalize the output
            output = output.to(torch.float32)
            peak = output.abs().view(output.size(0), -1).max(dim=1, keepdim=True).values
            peak = peak.view(-1, 1, 1)
            output = output / peak
            output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        return outputs
    
    elif experiment_type == "perturbation":
        if perturbation_t is None:
            raise ValueError("Peturbation time must be provided for perturbation experiment.")
        cond_expanded = [condition] * batch_size
        outputs = generate_perturbations(
                model,
                steps=steps,
                cfg_scale=7,
                perturbation_t=perturbation_t,
                n_perturbations=n_perturbations,
                conditioning=cond_expanded,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device,
                batch_size=batch_size,
            )
        
        for i, output in enumerate(outputs):
            # Normalize the output
            output = output.to(torch.float32)
            peak = output.abs().view(output.size(0), -1).max(dim=1, keepdim=True).values
            peak = peak.view(-1, 1, 1)
            output = output / peak
            output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        return outputs
    
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

        return [output]
