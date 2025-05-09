import kagglehub
import pandas as pd
from pathlib import Path

DATA_DIR = "data"

if __name__ == "__main__":
    # Download latest version
    mcaps = kagglehub.dataset_download("googleai/musiccaps")
    print("Path to dataset files:", mcaps)

    mcaps_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "googleai" / "musiccaps" / "versions" / "1" / "musiccaps-public.csv"
    df = pd.read_csv(mcaps_path)
    subset = df[df["is_balanced_subset"] == True]  # select entries in genre-balanced subset
    prompts = subset["caption"].tolist()

    # save the prompts to a text file
    save_path = Path(f"{DATA_DIR}/prompts/prompt.txt")
    with open(save_path, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")