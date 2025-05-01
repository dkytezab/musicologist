import os
import torch
import torch.nn as nn
from typing import List, Optional

class ActivationCache:
    def __init__(self,
                 model: nn.Module,
                 layer_idxs: List[int],
                 record_ts: Optional[List[int]] = None):
        """
        Initialize the ActivationCache with a model and a list of layer names.
        Args:
            model (nn.Module): The model to cache activations from.
            layers (List[str]): List of layer names to cache.
            record_ts (Optional[List[int]]): List of time steps to record activations for.
        """
        self.hooks = []
        self.layer_idxs = layer_idxs
        self.activations = {idx: { t: [None] * 2 for t in record_ts } for idx in layer_idxs}
        self.model = model
        self.record_ts = record_ts if record_ts is not None else []
        self.cache = {}
        for idx in layer_idxs:
            module_pre = model.model.model.transformer.layers[idx].cross_attend_norm
            module_post = model.model.model.transformer.layers[idx].cross_attn
            self.hooks.append(
                module_pre.register_forward_hook(self._make_hook(idx, pre=0))
            )
            self.hooks.append(
                module_post.register_forward_hook(self._make_hook(idx, pre=1))
            )
        self.current_t = None
    
    def _make_hook(self, idx: int, pre: int):
        def hook(_mod, _inp, out):
            if self.current_t in self.record_ts:
                self.activations[idx][self.current_t][pre] = out.detach().cpu()
        return hook

    def step(self, t: int):
        """
        Update the current time step.
        Args:
            t (int): The current time step.
        """
        self.current_t = t
    
    def clear(self):
        """
        Clear the cached activations.
        """
        for idx in self.layer_idxs:
            for t in self.record_ts:
                if t in self.activations[idx]:
                    del self.activations[idx][t]
        self.activations = {idx: { t: [None] * 2 for t in self.record_ts } for idx in self.layer_idxs}
        self.current_t = None

    def save(self,
             path: str,
             prompt_idx: int = 0,
             batch_idx: int = 0):
        """
        Save the cached activations to a .pt file.
        Args:
            path (str): The path to save the activations.
        """
        for idx in self.layer_idxs:
            for t in self.record_ts:
                out = self.activations[idx][t][1]
                print(f"Layer {idx} | t = {t} | Shape: {out.shape}")
                filename = f"{path}/diff_step_{t}/layer_{idx}_prompt_{prompt_idx}_batch_{batch_idx}.pt"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torch.save(out, filename)
