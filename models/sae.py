import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 hidden_dim: int,
                 topk: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_features, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_features)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.activation(encoded)
        topk_indices = torch.topk(encoded, k=self.topk, dim=1).indices
        encoded_zeros = torch.zeros_like(encoded)
        encoded_zeros[topk_indices] = encoded[topk_indices]
        encoded = encoded_zeros
        decoded = self.decoder(encoded)
        return encoded, decoded
