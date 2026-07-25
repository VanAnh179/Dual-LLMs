"""Pure latent-message interventions used by the S03 causal diagnostic."""
from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor


class NoiseStats(TypedDict):
    mean: Tensor
    std: Tensor


def build_derangement(num_examples: int, seed: int) -> Tensor:
    """Build a reproducible derangement without rejection sampling."""
    if num_examples < 2:
        raise ValueError("A derangement requires at least two examples.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(num_examples, generator=generator)
    shift = int(torch.randint(1, num_examples, (1,), generator=generator).item())
    permutation = torch.empty(num_examples, dtype=torch.long)
    permutation[order] = order.roll(shifts=shift)
    if torch.any(permutation == torch.arange(num_examples)):
        raise RuntimeError("Internal error: generated permutation has a fixed point.")
    return permutation


def apply_matched(z: Tensor) -> Tensor:
    return z.clone()


def apply_shuffled(z: Tensor, permutation: Tensor) -> Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape (N, D), got {tuple(z.shape)}")
    if permutation.ndim != 1 or permutation.numel() != z.shape[0]:
        raise ValueError("Permutation length must equal the number of messages.")
    return z.index_select(0, permutation.to(z.device))


def apply_zero(z: Tensor) -> Tensor:
    return torch.zeros_like(z)


def fit_noise_stats(z_all: Tensor, min_std: float = 1.0e-6) -> NoiseStats:
    if z_all.ndim != 2 or z_all.shape[0] < 1:
        raise ValueError(f"Expected non-empty z with shape (N, D), got {tuple(z_all.shape)}")
    if min_std <= 0:
        raise ValueError("min_std must be positive.")
    mean = z_all.float().mean(dim=0)
    std = z_all.float().std(dim=0, unbiased=False).clamp_min(min_std)
    return {"mean": mean, "std": std}


def apply_noise(
    z: Tensor,
    mean: Tensor,
    std: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    if z.ndim != 2 or mean.shape != z.shape[1:] or std.shape != z.shape[1:]:
        raise ValueError("z must be (N, D), while mean and std must both be (D,).")
    noise = torch.randn(z.shape, generator=generator, device="cpu", dtype=torch.float32)
    sampled = noise * std.detach().float().cpu() + mean.detach().float().cpu()
    return sampled.to(device=z.device, dtype=z.dtype)

