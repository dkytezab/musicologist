import torch
import torch.nn as nn

class PromptClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_prompts: int = 128,
                 num_hidden: int = 6,
                 neurons_hidden: int = 512,
                 activation_fn: nn.Module = nn.ReLU):
        super(PromptClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        self.num_hidden = num_hidden
        self.neurons_hidden = neurons_hidden
        self.activation_fn = activation_fn

        self.model = nn.Sequential(
            nn.Linear(embed_dim, neurons_hidden),
            activation_fn(),
            *[nn.Sequential(
                nn.Linear(neurons_hidden, neurons_hidden),
                activation_fn()
            ) for _ in range(num_hidden)],
            nn.Linear(neurons_hidden, num_prompts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
