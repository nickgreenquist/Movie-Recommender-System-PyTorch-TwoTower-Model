"""
Stage 1 — MLP ranker.

Concatenates user features and candidate item features, runs through a small MLP,
returns scalar logits. NEVER applies sigmoid — BCEWithLogitsLoss handles that.

Item feature layout (first genome_dim dims are always genome scores):
  [0:genome_dim]           genome scores (1128)  → genome_bottleneck → bottleneck_dim
  [genome_dim:item_dim]    genre(20) + global_avg(1) + global_count_log1p(1) + year_norm(1) + interact(2)

Without a bottleneck the 1128 genome dims swamp the scalar features in the first
hidden layer. The bottleneck compresses genome to a fixed-size dense rep before
concatenation, so all feature groups arrive at roughly equal scale.

ZERO src/ imports.
"""
import torch
import torch.nn as nn


class MLPRanker(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, hidden_dims: list = None,
                 dropout: float = 0.0, genome_dim: int = 1128, genome_bottleneck_dim: int = 64):
        """
        genome_dim:            number of genome dims at the start of item_features (set 0 to disable).
        genome_bottleneck_dim: output size of the genome compression layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.genome_dim            = genome_dim
        self.genome_bottleneck_dim = genome_bottleneck_dim

        if genome_dim > 0:
            self.genome_bottleneck = nn.Sequential(
                nn.Linear(genome_dim, genome_bottleneck_dim),
                nn.ReLU(),
            )
            mlp_in = user_dim + genome_bottleneck_dim + (item_dim - genome_dim)
        else:
            self.genome_bottleneck = None
            mlp_in = user_dim + item_dim

        layers = []
        in_dim = mlp_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # raw logits — no activation
        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        user_features: (B, user_dim)
        item_features: (B, item_dim)
        Returns: (B,) raw logits.
        """
        if self.genome_bottleneck is not None:
            genome_emb  = self.genome_bottleneck(item_features[:, :self.genome_dim])
            rest        = item_features[:, self.genome_dim:]
            x = torch.cat([user_features, genome_emb, rest], dim=-1)
        else:
            x = torch.cat([user_features, item_features], dim=-1)
        return self.mlp(x).squeeze(-1)
