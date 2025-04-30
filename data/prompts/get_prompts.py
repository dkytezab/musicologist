import kagglehub
import pandas as pd
from pathlib import Path

PROMPT_SOURCE = 'musiccaps-balanced'

def get_prompts(prompt_source: str = PROMPT_SOURCE) -> None:
    """
    Get prompts from the specified source.
    
    Args:
        prompt_source (str): The source of the prompts. Default is 'musiccaps'.
        dest (str): The destination file to save the prompts. Default is 'prompts.txt'.
    """
    assert prompt_source in ['musiccaps', 'musiccaps-balanced'], f"Unknown prompt source: {prompt_source}"

    dest = f"{prompt_source}.txt"

    if prompt_source == 'musiccaps-balanced':
        path = kagglehub.dataset_download("googleai/musiccaps") # download if not already downloaded
        csv_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "googleai" / "musiccaps" / "versions" / "1" / "musiccaps-public.csv"
        df = pd.read_csv(csv_path)
        subset = df[df["is_balanced_subset"] == True]  # select entries in genre-balanced subset
        prompts = subset["caption"].tolist()
        with open(dest, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")
    elif prompt_source == 'musiccaps':
        path = kagglehub.dataset_download("gosogleai/musiccaps") # download if not already downloaded
        csv_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "googleai" / "musiccaps" / "versions" / "1" / "musiccaps-public.csv"
        df = pd.read_csv(csv_path)
        prompts = df["caption"].tolist()
        with open(dest, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")
    else:
        raise ValueError(f"Unknown prompt source: {prompt_source}")

get_prompts()
