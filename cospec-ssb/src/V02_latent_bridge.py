"""Attention-mask-aware latent bridge for V02 split training."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class MaskedLatentBridge(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_model, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, d_model)
        self.gate = nn.Linear(2 * d_model, 1)
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.01)
        nn.init.zeros_(self.decoder.bias)
        nn.init.normal_(self.gate.weight, std=0.001)
        nn.init.zeros_(self.gate.bias)

    def encode(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.encoder(pooled)

    def inject(self, hidden: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        projected = self.decoder(z).unsqueeze(1).expand_as(hidden)
        gate = torch.sigmoid(self.gate(torch.cat([hidden, projected], dim=-1)))
        return hidden + gate * projected

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "d_model": self.d_model,
            "bottleneck_dim": self.bottleneck_dim,
        }, target)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "MaskedLatentBridge":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        bridge = cls(payload["d_model"], payload["bottleneck_dim"])
        bridge.load_state_dict(payload["state_dict"])
        return bridge
