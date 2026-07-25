#!/usr/bin/env python
"""Minimal Coupling bridge modules: linear projection + gated injection."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class LinearProjectionBridge(nn.Module):
    """Mean-pool hidden states across sequence length, then project down to a bottleneck.

    Input:  H_A of shape (batch, seq_len, d_model) or (seq_len, d_model)
    Output: z   of shape (batch, bottleneck_dim) or (1, bottleneck_dim)
    """

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, bottleneck_dim)

    def forward(self, h_a: torch.Tensor) -> torch.Tensor:
        if h_a.dim() == 2:
            # h_a: (seq_len, d_model) -> mean pool -> (d_model) -> (1, d_model)
            pooled = h_a.mean(dim=0, keepdim=True)
        else:
            # h_a: (batch, seq_len, d_model) -> mean pool -> (batch, d_model)
            pooled = h_a.mean(dim=1)
        z = self.proj(pooled)  # (batch, bottleneck_dim) or (1, bottleneck_dim)
        return z


class GatedInjection(nn.Module):
    """Inject a bottleneck vector z into hidden states h_B via a learned gate.

    h_B_new = h_B + sigmoid(gate([h_B, z_proj])) * z_proj

    Where z_proj = up_proj(z) is broadcast across the sequence dimension.
    """

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.gate = nn.Linear(d_model + d_model, 1)

    def forward(self, h_b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h_b: (batch, seq_len, d_model) or (seq_len, d_model)
        # z:   (batch, bottleneck_dim) or (1, bottleneck_dim)
        z_proj = self.up_proj(z)  # (batch, d_model) or (1, d_model)
        
        if h_b.dim() == 2:
            # z_proj: (1, d_model) -> expand -> (seq_len, d_model)
            z_proj_expanded = z_proj.expand_as(h_b)
        else:
            # z_proj: (batch, d_model) -> expand -> (batch, seq_len, d_model)
            z_proj_expanded = z_proj.unsqueeze(1).expand_as(h_b)

        gate_input = torch.cat([h_b, z_proj_expanded], dim=-1)  # (batch, seq_len, 2*d_model) or (seq_len, 2*d_model)
        gate_val = torch.sigmoid(self.gate(gate_input))  # (batch, seq_len, 1) or (seq_len, 1)

        return h_b + gate_val * z_proj_expanded


class MinimalCouplingBridge(nn.Module):
    """End-to-end bridge: extract z from Agent A's hidden states, inject into Agent B.

    Combines LinearProjectionBridge (write encoder) and GatedInjection (read decoder).
    Provides save/load for the bridge state dict independently of PEFT adapters.
    """

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = LinearProjectionBridge(d_model, bottleneck_dim)
        self.decoder = GatedInjection(d_model, bottleneck_dim)
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self._init_weights()

    def _init_weights(self) -> None:
        # Scale down initialization to prevent activation explosion in FP16
        nn.init.normal_(self.encoder.proj.weight, std=0.01)
        nn.init.zeros_(self.encoder.proj.bias)
        nn.init.normal_(self.decoder.up_proj.weight, std=0.01)
        nn.init.zeros_(self.decoder.up_proj.bias)
        nn.init.normal_(self.decoder.gate.weight, std=0.001)
        nn.init.zeros_(self.decoder.gate.bias)

    def encode(self, h_a: torch.Tensor) -> torch.Tensor:
        """Encode Agent A's hidden states into a bottleneck vector z."""
        return self.encoder(h_a)

    def inject(self, h_b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Inject bottleneck vector z into Agent B's hidden states."""
        return self.decoder(h_b, z)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """Full bridge: encode from A, inject into B."""
        z = self.encode(h_a)
        return self.inject(h_b, z)

    def save_bridge(self, path: str | Path) -> None:
        """Save bridge state dict to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "d_model": self.d_model,
                "bottleneck_dim": self.bottleneck_dim,
            },
            str(p),
        )

    @classmethod
    def load_bridge(cls, path: str | Path, device: Optional[str] = None) -> "MinimalCouplingBridge":
        """Load a saved bridge from disk."""
        data = torch.load(str(path), map_location=device or "cpu", weights_only=True)
        bridge = cls(d_model=data["d_model"], bottleneck_dim=data["bottleneck_dim"])
        bridge.load_state_dict(data["state_dict"])
        return bridge
