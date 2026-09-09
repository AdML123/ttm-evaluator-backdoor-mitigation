"""Small trainable prediction heads used after frozen embeddings."""

from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    """A three-affine-layer MLP returning one scalar per leading input item.

    When ``dropout_p > 0``, a dropout layer is inserted after the first ReLU
    (i.e. before the second affine layer), which enables Monte Carlo dropout
    uncertainty estimation at inference time.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        second_hidden_dim: int = 128,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, second_hidden_dim) <= 0:
            raise ValueError("MLP dimensions must be positive")
        if dropout_p < 0 or dropout_p >= 1:
            raise ValueError("dropout_p must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.dropout_p = float(dropout_p)
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if self.dropout_p > 0:
            layers.append(nn.Dropout(self.dropout_p))
        layers += [
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, 1),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(features, dtype=torch.float32)
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected final feature dimension {self.input_dim}, got {values.shape[-1]}"
            )
        return self.network(values).squeeze(-1)
