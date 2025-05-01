import torch
import matplotlib.pyplot as plt

from models.sae import Autoencoder

activations_path = 'data/activations'

x = torch.load(f'{activations_path}/diff_step_49/layer_10_prompt_0_batch_0.pt')

plt.imshow(x[0].cpu().numpy(), cmap='viridis', aspect='auto')
plt.savefig('viz/activations/experiment.png')

# steps = [1, 25, 49]
# layers = [2, 10]

# for step in steps:
#     for layer in layers:
#         x = torch.load(f'{activations_path}/diff_step_{step}/layer_{layer}_prompt_0_batch_0.pt')
