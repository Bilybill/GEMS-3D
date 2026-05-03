from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)
