import pandas as pd
from pathlib import Path

csv_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "googleai" / "musiccaps" / "versions" / "1" / "musiccaps-public.csv"
df = pd.read_csv(csv_path)
subset = df[df["is_balanced_subset"] == True]  # select entries in genre-balanced subset
prompts = subset["caption"].tolist()

# save the prompts to a text file
with open("data/prompts/prompt.txt", "w") as f:
    for prompt in prompts:
        f.write(prompt + "\n")
