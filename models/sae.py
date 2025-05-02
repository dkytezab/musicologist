import torch
import torch.nn as nn

class TopKAutoencoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 topk: int,
                 topk_aux: int):
        super(TopKAutoencoder, self).__init__()
        self.topk = topk
        self.topk_aux = topk_aux
        self.bias = nn.Parameter(torch.zeros(input_size))
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.activation = nn.ReLU()
        self.inactives = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        encoded = torch.flatten(x, start_dim=1)
        encoded = x - self.bias
        encoded = self.encoder(x)
        encoded = self.activation(encoded)
        topk_indices = torch.topk(encoded, k=self.topk, dim=1).indices
        mask = torch.zeros_like(encoded)
        mask.scatter_(1, topk_indices, 1)
        encoded = encoded * mask
        decoded = self.decoder(encoded) + self.bias

        inactive_mask = torch.ones_like(encoded)
        inactive_mask.scatter_(1, topk_indices, 0)
        self.inactives.data = inactive_mask.sum(dim=0)
        most_inactive = torch.topk(self.inactives, k=self.topk_aux, dim=0).indices
        most_inactive_mask = torch.zeros_like(encoded)
        for i in range(most_inactive.shape[0]):
            most_inactive_mask[:, most_inactive[i]] = 1
        encoded_aux = encoded * most_inactive_mask
        decoded_aux = self.decoder(encoded_aux) + self.bias
        return encoded, decoded, encoded_aux, decoded_aux

