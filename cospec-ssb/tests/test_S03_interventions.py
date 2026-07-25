import torch

from src.S03_causal_metrics import compute_causal_metrics
from src.S03_interventions import (
    apply_noise,
    apply_shuffled,
    apply_zero,
    build_derangement,
    fit_noise_stats,
)


def test_derangement_has_no_fixed_points_and_is_reproducible():
    first = build_derangement(100, 42)
    second = build_derangement(100, 42)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(100))
    assert not torch.any(first == torch.arange(100))


def test_zero_shuffle_and_noise_shapes():
    z = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    permutation = build_derangement(4, 42)
    assert apply_shuffled(z, permutation).shape == z.shape
    assert torch.count_nonzero(apply_zero(z)) == 0
    stats = fit_noise_stats(z)
    noise = apply_noise(z, stats["mean"], stats["std"], torch.Generator().manual_seed(42))
    assert noise.shape == z.shape
    assert torch.isfinite(noise).all()


def test_paired_metrics_require_identical_order():
    rows = {
        "matched": [{"sample_id": "a", "correct": True}, {"sample_id": "b", "correct": True}],
        "shuffled": [{"sample_id": "a", "correct": False}, {"sample_id": "b", "correct": True}],
        "zero": [{"sample_id": "a", "correct": False}, {"sample_id": "b", "correct": False}],
        "noise": [{"sample_id": "a", "correct": True}, {"sample_id": "b", "correct": False}],
    }
    metrics = compute_causal_metrics(rows, num_resamples=100, seed=42)
    assert metrics["accuracy"]["matched"] == 1.0
    assert metrics["deltas_vs_matched"]["zero"]["delta"] == 1.0

