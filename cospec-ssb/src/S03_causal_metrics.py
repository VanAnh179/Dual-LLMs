"""Paired accuracy and bootstrap metrics for S03 interventions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _correctness(rows: Sequence[Mapping[str, object]]) -> tuple[list[str], torch.Tensor]:
    ids = [str(row.get("sample_id", row.get("id"))) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction rows contain duplicate sample IDs.")
    values = torch.tensor([bool(row["correct"]) for row in rows], dtype=torch.float64)
    return ids, values


def paired_bootstrap_delta(
    reference: torch.Tensor,
    control: torch.Tensor,
    *,
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    if reference.ndim != 1 or reference.shape != control.shape or reference.numel() == 0:
        raise ValueError("Paired correctness vectors must be non-empty and have equal shape.")
    if num_resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap configuration.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = reference.numel()
    indices = torch.randint(0, n, (num_resamples, n), generator=generator)
    paired = reference - control
    estimates = paired[indices].mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = torch.quantile(estimates, torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64))
    return {
        "delta": float(paired.mean().item()),
        "ci_low": float(low.item()),
        "ci_high": float(high.item()),
    }


def compute_causal_metrics(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    required = ("matched", "shuffled", "zero", "noise")
    missing = [mode for mode in required if mode not in predictions]
    if missing:
        raise ValueError(f"Missing prediction modes: {missing}")

    vectors: dict[str, torch.Tensor] = {}
    canonical_ids: list[str] | None = None
    counts: dict[str, dict[str, int]] = {}
    accuracies: dict[str, float] = {}
    for mode in required:
        ids, vector = _correctness(predictions[mode])
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise ValueError(f"Sample IDs/order for {mode} do not match matched mode.")
        vectors[mode] = vector
        correct = int(vector.sum().item())
        counts[mode] = {"correct": correct, "total": vector.numel()}
        accuracies[mode] = float(vector.mean().item())

    deltas = {}
    for mode in ("shuffled", "zero", "noise"):
        deltas[mode] = paired_bootstrap_delta(
            vectors["matched"], vectors[mode], num_resamples=num_resamples,
            confidence_level=confidence_level, seed=seed,
        )
    return {
        "n": len(canonical_ids or []),
        "sample_ids": canonical_ids or [],
        "accuracy": accuracies,
        "counts": counts,
        "deltas_vs_matched": deltas,
        "bootstrap": {
            "paired": True,
            "num_resamples": num_resamples,
            "confidence_level": confidence_level,
            "seed": seed,
        },
    }

