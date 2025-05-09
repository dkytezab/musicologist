from typing import Optional, Dict, List
import os
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from truncation.gen_truncated import generate_truncated_seq
from pathlib import Path
import pandas as pd

DATA_DIR = "data/generated"

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
        prompt = prompt.split('. ', 1)[-1] if '. ' in prompt else prompt
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
        verbose: bool,
        sample_length: int,
        sample_rate: float = 48000,
        output_dir: Optional[str] = None,
    ) -> None:

    if output_dir == None:
        output_dir = DATA_DIR
    
    for i, audio in enumerate(audios):
        for j, sample in enumerate(audio):

            filename = f"{output_dir}/diff_step_{truncation_ts[i]}/prompt_{prompt_index}_sample_{j}.wav"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            print(f"Sample shape is {sample.shape}")

            torchaudio.save(filename, sample.cpu(), sample_rate)

            if verbose:
                print(f"Saved {filename}") 
            
            write_sample_to_csv(
                csv_path=Path(f"{output_dir}/audio_info.csv"),
                tag_json_path=Path(f"data/prompts/annotations.json"),
                audio_file_path=Path(filename),
                model="stable-diffusion",
                prompt_index=prompt_index,
                diffusion_step=truncation_ts[i],
                sample_length=sample_length,
                sample_index=j,
            )
        


def diff_gen_flexible(
        model,
        steps,
        condition: str,
        batch_size,
        sample_size,
        sample_rate: int = 48000,
        sigma_min: float = 0.01,
        sigma_max: float = 100,
        cfg_scale: int = 6,
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
            print(output.shape)
            if sample_length is not None:
                entries_to_keep = int(sample_rate * sample_length)
                output = output[:, :, :entries_to_keep]
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


def write_sample_to_csv(
        csv_path: Path, 
        tag_json_path: Path,
        audio_file_path: Path,
        model,  
        prompt_index, 
        diffusion_step,
        sample_length,
        sample_index,
        ):
    
    if not csv_path.exists():
        with open(csv_path, 'w') as f:
            f.write("diffusion_step, prompt_index, sample_index, model, csv_path, sample_length, prompt, tag_json_path, audio_file_path, tag.aspects, tag.bpm\n")
    
    if not tag_json_path.exists():
        raise FileNotFoundError(f"Label JSON path {tag_json_path} does not exist.")
    
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file path {audio_file_path} does not exist.")
    
    # Reading label json file
    label_json = pd.read_json(tag_json_path)
    label_json_row = label_json.iloc[prompt_index]

    prompt = label_json_row["prompt"]
    print(f"Writing sample with prompt {prompt_index}: {prompt}")

    row = {
        "diffusion_step": diffusion_step,
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "model": model,
        "csv_path": str(csv_path),
        "sample_length": sample_length,
        "prompt": prompt,
        "tag_json_path": str(tag_json_path),
        "audio_file_path": str(audio_file_path),
        "tag.aspects": label_json_row["aspects"],
        "tag.bpm": label_json_row["bpm"],
    }

    df = pd.DataFrame([row])

    df.to_csv(
        csv_path,
        mode = "a",                     
        index = False,
        header = not csv_path.exists(),
    )