#!/usr/bin/env python
"""Generate the reproducible Kaggle T4x2 V02 training notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "V02_Kaggle_T4x2_full_train.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# V02 Split vs Single: Kaggle T4 x2 Full Run

This notebook verifies the uploaded controlled-training bundle, runs a GPU smoke
gate, trains all matched-budget controls and the latent split model, evaluates the
controlled test, then evaluates the untouched official external holdout.

Kaggle settings required: **Internet on** and **GPU T4 x2**. Upload
`V02_kaggle_training_bundle.zip` as a Kaggle Dataset before running all cells.
Official test data is downloaded at runtime and is never included in training."""
    ),
    code(
        """from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IS_KAGGLE = Path("/kaggle/working").exists()
RUN_FULL = IS_KAGGLE
REPO_URL = "https://github.com/VanAnh179/Dual-LLMs.git"
BUNDLE_NAME = "V02_kaggle_training_bundle.zip"

def run_cmd(args, *, cwd, gpu=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

print({"is_kaggle": IS_KAGGLE, "run_full": RUN_FULL, "python": sys.version})"""
    ),
    code(
        """if IS_KAGGLE:
    checkout = Path("/kaggle/working/Dual-LLMs")
    if not checkout.exists():
        run_cmd(["git", "clone", "--depth", "1", REPO_URL, checkout], cwd="/kaggle/working")
    else:
        run_cmd(["git", "pull", "--ff-only", "origin", "main"], cwd=checkout)
    REPO_ROOT = checkout / "cospec-ssb"
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO_ROOT = next(
        path for path in candidates
        if (path / "scripts/V02_train_split_latent.py").exists()
    )

assert (REPO_ROOT / "configs/v02_split_vs_single.yaml").exists()
print("Repository:", REPO_ROOT)"""
    ),
    code(
        """search_root = Path("/kaggle/input") if IS_KAGGLE else REPO_ROOT / "data"
bundle_candidates = sorted(search_root.rglob(BUNDLE_NAME))
if bundle_candidates:
    BUNDLE_PATH = bundle_candidates[0]
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        names = set(archive.namelist())
        assert "BUNDLE_MANIFEST.json" in names
        bundle_manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        for name, expected in bundle_manifest["files"].items():
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == expected["sha256"], name
        if IS_KAGGLE:
            archive.extractall(REPO_ROOT)
else:
    expanded_manifests = sorted(search_root.rglob("BUNDLE_MANIFEST.json"))
    if not expanded_manifests:
        raise FileNotFoundError(
            f"Neither {BUNDLE_NAME} nor BUNDLE_MANIFEST.json was found below "
            f"{search_root}. Upload the bundle as a Kaggle Dataset."
        )
    BUNDLE_PATH = expanded_manifests[0]
    bundle_root = BUNDLE_PATH.parent
    bundle_manifest = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for name, expected in bundle_manifest["files"].items():
        source = bundle_root / name
        assert source.exists(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected["sha256"]
        if IS_KAGGLE:
            target = REPO_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

assert bundle_manifest["bundle_format"] == "cospec-ssb-v02-kaggle-v1"
assert bundle_manifest["official_test_data_included"] is False
print("Bundle verified:", BUNDLE_PATH)
print("Files:", len(bundle_manifest["files"]))"""
    ),
    code(
        """required = {
    "yaml": "pyyaml",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "peft": "peft",
    "sklearn": "scikit-learn",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing and IS_KAGGLE:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *missing], cwd=REPO_ROOT)
print("Dependency check:", {
    "missing_before": missing,
    "installed_now": missing if IS_KAGGLE else [],
    "local_validation_skips_gpu_packages": not IS_KAGGLE,
})"""
    ),
    code(
        """run_cmd(
    [sys.executable, "scripts/V02_validate_benchmark.py",
     "--config", "configs/v02_multifamily_benchmark.yaml"],
    cwd=REPO_ROOT,
)

validation = json.loads(
    (REPO_ROOT / "outputs/V02_multifamily_benchmark/metrics/dataset_validation.json")
    .read_text(encoding="utf-8")
)
assert validation["status"] == "PASS" and validation["smoke"] is False

if RUN_FULL:
    import torch
    assert torch.cuda.is_available(), "CUDA is required."
    assert torch.cuda.device_count() >= 2, "Select Kaggle GPU T4 x2."
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert all("T4" in name.upper() for name in gpu_names[:2]), gpu_names
    print("GPU gate: PASS", gpu_names)
else:
    print("GPU gate: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_FULL:
    run_cmd(
        [sys.executable, "scripts/V02_build_official_external.py",
         "--config", "configs/v02_official_external.yaml"],
        cwd=REPO_ROOT,
    )
    run_cmd(
        [sys.executable, "scripts/V02_validate_official_external.py",
         "--config", "configs/v02_official_external.yaml"],
        cwd=REPO_ROOT,
    )
    official_validation = json.loads(
        (REPO_ROOT / "outputs/V02_official_external/metrics/dataset_validation.json")
        .read_text(encoding="utf-8")
    )
    assert official_validation["status"] == "PASS"
    print("Official external provenance gate: PASS")
else:
    print("Official download: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_FULL:
    smoke_root = (REPO_ROOT / "outputs/V02_kaggle_smoke").resolve()
    assert smoke_root.is_relative_to(REPO_ROOT.resolve())
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_cfg = "configs/v02_split_vs_single_kaggle_smoke.yaml"
    run_cmd(
        [sys.executable, "scripts/V02_train_single_baselines.py",
         "--config", smoke_cfg, "--max-examples", "16", "--max-steps", "2"],
        cwd=REPO_ROOT, gpu=0,
    )
    run_cmd(
        [sys.executable, "scripts/V02_train_split_latent.py",
         "--config", smoke_cfg, "--max-examples", "16", "--max-steps", "2"],
        cwd=REPO_ROOT, gpu=1,
    )
    smoke_manifest = json.loads(
        (smoke_root / "metrics/training_manifest.json").read_text(encoding="utf-8")
    )
    smoke_records = [
        smoke_manifest["single_baselines"][mode]
        for mode in ("full", "view_a", "view_b")
    ] + [smoke_manifest["split_latent"]]
    assert all(item["optimizer_steps"] == 2 for item in smoke_records)
    assert all(item["all_losses_finite"] for item in smoke_records)
    run_cmd(
        [sys.executable, "scripts/V02_evaluate_split_vs_single.py",
         "--config", smoke_cfg, "--max-examples", "8", "--batch-size", "2"],
        cwd=REPO_ROOT, gpu=0,
    )
    print("GPU smoke train/eval gate: PASS")
else:
    print("GPU smoke train/eval: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_FULL:
    full_root = (REPO_ROOT / "outputs/V02_split_vs_single").resolve()
    assert full_root.is_relative_to(REPO_ROOT.resolve())
    if full_root.exists():
        shutil.rmtree(full_root)
    full_cfg = "configs/v02_split_vs_single.yaml"
    run_cmd(
        [sys.executable, "scripts/V02_train_single_baselines.py",
         "--config", full_cfg],
        cwd=REPO_ROOT, gpu=0,
    )
    run_cmd(
        [sys.executable, "scripts/V02_train_split_latent.py",
         "--config", full_cfg],
        cwd=REPO_ROOT, gpu=1,
    )
    print("Full matched-budget training: COMPLETE")
else:
    print("Full training: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_FULL:
    training_manifest = json.loads(
        (REPO_ROOT / "outputs/V02_split_vs_single/metrics/training_manifest.json")
        .read_text(encoding="utf-8")
    )
    training_records = {
        **training_manifest["single_baselines"],
        "split_latent": training_manifest["split_latent"],
    }
    convergence = {}
    for name, item in training_records.items():
        initial = float(item["initial_optimizer_step_loss"])
        final = float(item["final_optimizer_step_loss"])
        best = float(item["best_optimizer_step_loss"])
        convergence[name] = {
            "steps": item["optimizer_steps"],
            "initial": initial,
            "final": final,
            "best": best,
            "first_window_mean": item["first_window_mean_loss"],
            "last_window_mean": item["last_window_mean_loss"],
            "last_over_first_window": (
                item["last_window_mean_loss"] / item["first_window_mean_loss"]
            ),
        }
        assert item["num_train_examples"] == 8000, (name, item["num_train_examples"])
        assert item["optimizer_steps"] == 500, (name, item["optimizer_steps"])
        assert item["all_losses_finite"], name
        assert best < initial, (name, initial, best)
        assert item["loss_improved"], (
            name, item["first_window_mean_loss"], item["last_window_mean_loss"]
        )
    print(json.dumps(convergence, indent=2))
    print("Finite-loss and convergence gate: PASS")
else:
    print("Convergence gate: deferred to the Kaggle full run")"""
    ),
    code(
        """if RUN_FULL:
    full_cfg = "configs/v02_split_vs_single.yaml"
    run_cmd(
        [sys.executable, "scripts/V02_evaluate_split_vs_single.py",
         "--config", full_cfg],
        cwd=REPO_ROOT, gpu=0,
    )
    run_cmd(
        [sys.executable, "scripts/V02_evaluate_split_vs_single.py",
         "--config", full_cfg,
         "--external-config", "configs/v02_official_external.yaml"],
        cwd=REPO_ROOT, gpu=0,
    )
    controlled = json.loads(
        (REPO_ROOT / "outputs/V02_split_vs_single/metrics/eval_metrics.json")
        .read_text(encoding="utf-8")
    )
    official = json.loads(
        (REPO_ROOT / "outputs/V02_official_external/metrics/eval_metrics.json")
        .read_text(encoding="utf-8")
    )
    print(json.dumps({
        "controlled": {
            "n": controlled["num_examples"],
            "verdict": controlled["primary_verdict"],
            "comparison": controlled["paired_comparisons"]["split_matched_vs_single_full"],
        },
        "official_external": {
            "n": official["num_examples"],
            "verdict": official["primary_verdict"],
            "comparison": official["paired_comparisons"]["split_matched_vs_single_full"],
        },
    }, indent=2))
else:
    print("Evaluation: deferred to the Kaggle full run")"""
    ),
    code(
        """if RUN_FULL:
    result_archive = shutil.make_archive(
        "/kaggle/working/V02_kaggle_results",
        "zip",
        root_dir=REPO_ROOT,
        base_dir="outputs",
    )
    print("Download results from:", result_archive)
else:
    print("LOCAL NOTEBOOK VALIDATION: PASS")
    print("The GPU smoke, full training, convergence, and evaluation cells are enabled automatically on Kaggle.")"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "dockerImageVersionId": None,
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
