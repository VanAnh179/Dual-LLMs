"""Preflight and artifact-integrity guards for V02 experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data_utils import load_config, project_path, read_json, read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_v02_preflight(config_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(config_path)
    dataset_cfg = load_config(cfg["source_dataset_config"])
    validation = read_json(dataset_cfg["output_paths"]["validation"], default={})
    if validation.get("status") != "PASS" or validation.get("smoke"):
        raise SystemExit(
            "V02 preflight blocked: full dataset validation must exist with status PASS."
        )
    manifest = read_json(dataset_cfg["output_paths"]["manifest"], default={})
    if manifest.get("smoke"):
        raise SystemExit("V02 preflight blocked: manifest is marked as smoke.")
    failures = []
    configured_splits = [
        split for split in cfg["data"]
        if split in dataset_cfg["output_paths"]
    ]
    for split in configured_splits:
        path = project_path(cfg["data"][split])
        if not path.exists():
            failures.append(f"missing {split} data: {path}")
            continue
        expected_path = project_path(dataset_cfg["output_paths"][split]).resolve()
        if path.resolve() != expected_path:
            failures.append(f"{split} path differs from validated dataset config")
        expected_hash = manifest.get("splits", {}).get(split, {}).get("sha256")
        if not expected_hash or _sha256(path) != expected_hash:
            failures.append(f"{split} SHA256 differs from validated manifest")
    train = cfg["training"]
    single_effective = (
        int(train["single_batch_size"])
        * int(train["single_gradient_accumulation_steps"])
    )
    split_effective = (
        int(train["split_batch_size"])
        * int(train["split_gradient_accumulation_steps"])
    )
    if single_effective != split_effective:
        failures.append(
            f"effective batch mismatch: single={single_effective}, split={split_effective}"
        )
    if failures:
        raise SystemExit("V02 preflight blocked:\n- " + "\n- ".join(failures))
    return cfg, dataset_cfg


def require_official_external_preflight(
    config_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(config_path)
    if not cfg.get("official_eval_only"):
        raise SystemExit(
            "Official external preflight blocked: config must be evaluation-only."
        )
    validation = read_json(cfg["output_paths"]["validation"], default={})
    manifest = read_json(cfg["output_paths"]["manifest"], default={})
    failures = []
    if validation.get("status") != "PASS" or validation.get("smoke"):
        failures.append("full official validation must exist with status PASS")
    if not manifest.get("official_eval_only") or manifest.get("smoke"):
        failures.append("official manifest is absent, non-eval, or marked smoke")
    test_path = project_path(cfg["output_paths"]["test"])
    if not test_path.exists():
        failures.append(f"missing official test data: {test_path}")
    else:
        expected_hash = (manifest.get("test") or {}).get("sha256")
        if not expected_hash or _sha256(test_path) != expected_hash:
            failures.append("official test SHA256 differs from provenance manifest")
    if failures:
        raise SystemExit(
            "Official external preflight blocked:\n- " + "\n- ".join(failures)
        )
    return cfg, manifest


def require_training_artifacts(
    cfg: dict[str, Any],
    require_full: bool,
    modes: set[str] | None = None,
) -> dict[str, Any]:
    manifest = read_json(cfg["outputs"]["training_manifest"], default={})
    single = manifest.get("single_baselines", {})
    split = manifest.get("split_latent", {})
    failures = []
    requested = modes or {
        "single_full", "single_a", "single_b",
        "split_matched", "split_shuffled", "split_zero",
    }
    single_modes = {
        "full": "single_full",
        "view_a": "single_a",
        "view_b": "single_b",
    }
    required_single = [
        mode for mode, evaluation_mode in single_modes.items()
        if evaluation_mode in requested
    ]
    for mode in required_single:
        item = single.get(mode)
        if not item:
            failures.append(f"missing training manifest for single {mode}")
            continue
        if not project_path(item["adapter_path"]).exists():
            failures.append(f"missing single {mode} adapter")
    needs_split = bool(
        requested & {"split_matched", "split_shuffled", "split_zero"}
    )
    if needs_split:
        for key in ("receiver_adapter_path", "bridge_path"):
            if not split.get(key) or not project_path(split[key]).exists():
                failures.append(f"missing split artifact: {key}")
    if require_full and not failures:
        expected = len(read_jsonl(cfg["data"]["train"]))
        records = [single[mode] for mode in required_single]
        if needs_split:
            records.append(split)
        if records and any(
            int(record["num_train_examples"]) != expected for record in records
        ):
            failures.append(f"not every model used all {expected} training rows")
        row_hashes = {record.get("sample_ids_sha256") for record in records}
        if records and len(row_hashes) != 1:
            failures.append("training row-order hashes differ across conditions")
        optimizer_steps = {int(record["optimizer_steps"]) for record in records}
        if records and len(optimizer_steps) != 1:
            failures.append(
                f"optimizer-step budgets differ across conditions: {sorted(optimizer_steps)}"
            )
        effective_batches = {
            int(record["effective_batch_size"]) for record in records
        }
        if records and len(effective_batches) != 1:
            failures.append(
                f"effective batch sizes differ across conditions: {sorted(effective_batches)}"
            )
    if failures:
        raise SystemExit("V02 artifact preflight blocked:\n- " + "\n- ".join(failures))
    return manifest
