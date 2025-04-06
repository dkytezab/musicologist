import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embeddings = torch.load("data/embeddings/CLAP/embeddings.pt")
embeddings = embeddings.cpu().detach().numpy()

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Make array of 32 0s and then 32 1s
labels = np.array([0] * 32 + [1] * 32)

# Plot the 2D embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, c=labels, cmap='viridis')
plt.colorbar()
plt.title("2D PCA of CLAP Audio Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("viz/outputs/outputs.png")
