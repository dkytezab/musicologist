import torch
import yaml
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open("embeddings/embed_config.yml", "r") as file:
    embed_config = yaml.safe_load(file)

with open("diffusion/diff_config.yml", "r") as file:
    diff_config = yaml.safe_load(file)

BATCH_SIZE = diff_config['batch_size']
MODEL = embed_config['model']
STEPS = embed_config['steps']

for step in STEPS:
    embeddings = torch.load(f"data/generated/diff_step_{step}/{MODEL}_embeddings.pt")
    embeddings = embeddings.cpu().detach().numpy()

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    labels = np.array([0] * BATCH_SIZE + [1] * BATCH_SIZE)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("2D PCA of CLAP Audio Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"viz/outputs/diff_step_{step}_{MODEL}_outputs.png")
