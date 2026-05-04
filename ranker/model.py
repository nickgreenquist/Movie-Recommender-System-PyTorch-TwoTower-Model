"""
Stage 1 — MLP ranker.

Concatenates user features and candidate item features, runs through a small MLP,
returns scalar logits. NEVER applies sigmoid — BCEWithLogitsLoss handles that.

ZERO src/ imports.
"""
import torch
import torch.nn as nn


class MLPRanker(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, hidden_dims: list = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        in_dim = user_dim + item_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # No activation — outputs raw logits.
        self.mlp = nn.Sequential(*layers)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        user_features: (B, user_dim)
        item_features: (B, item_dim)
        Returns: (B,) raw logits.
        """
        x = torch.cat([user_features, item_features], dim=-1)
        return self.mlp(x).squeeze(-1)
