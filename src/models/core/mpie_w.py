import torch
import torch.nn as nn
from typing import Sequence


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, is_encoder: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1) if is_encoder else None

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MPIEModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = [embed_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor):
        h = self.mlp(tokens)
        return h


class MPIDModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = list(hidden_dims) + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor):
        emb = self.mlp(h)
        return emb
