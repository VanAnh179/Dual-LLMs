#!/usr/bin/env python
"""Generate the Kaggle T4x2 V03 learning-curve notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_OUTPUT = ROOT / "notebooks" / "V03_Kaggle_T4x2_learning_curve.ipynb"
DATA_OUTPUT = ROOT / "data" / NOTEBOOK_OUTPUT.name


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
        """# V03 Split vs Single: OOD Learning Curve on Kaggle T4 x2

This notebook verifies the controlled V03 bundle, runs a two-GPU smoke gate,
trains nested 2k/4k/8k/16k/32k learning-curve points, evaluates held-out OOD and
IID tests, and finally evaluates the untouched official external benchmark.

Kaggle settings: **Internet on**, **GPU T4 x2**. Upload
`V03_kaggle_learning_curve_bundle.zip` as a Kaggle Dataset, then run all cells."""
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
RUN_GPU = IS_KAGGLE
REPO_URL = "https://github.com/VanAnh179/Dual-LLMs.git"
BUNDLE_NAME = "V03_kaggle_learning_curve_bundle.zip"

def run_cmd(args, *, cwd, gpu=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

print({"is_kaggle": IS_KAGGLE, "run_gpu": RUN_GPU, "python": sys.version})"""
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
        if (path / "scripts/V03_run_learning_curve.py").exists()
    )

assert (REPO_ROOT / "configs/v03_learning_curve.yaml").exists()
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
        visible = sorted(
            str(path.relative_to(search_root))
            for path in search_root.rglob("*")
            if path.is_file()
        )[:50]
        raise FileNotFoundError(
            f"Neither {BUNDLE_NAME} nor an extracted BUNDLE_MANIFEST.json "
            f"was found below {search_root}. Attach the Kaggle Dataset with "
            f"Add Input. Visible files: {visible}"
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

assert bundle_manifest["bundle_format"] == "cospec-ssb-v03-kaggle-v1"
assert bundle_manifest["official_test_data_included"] is False

print("Bundle verified:", BUNDLE_PATH)
print(json.dumps(bundle_manifest["split_roles"], indent=2))"""
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
missing = [
    package for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
if missing and IS_KAGGLE:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *missing], cwd=REPO_ROOT)
print({"missing_before": missing, "installed_on_kaggle": missing if IS_KAGGLE else []})"""
    ),
    code(
        """run_cmd(
    [sys.executable, "scripts/V03_validate_benchmark.py",
     "--config", "configs/v03_ood_benchmark.yaml"],
    cwd=REPO_ROOT,
)
validation = json.loads(
    (REPO_ROOT / "outputs/V03_ood_multifamily_benchmark/metrics/dataset_validation.json")
    .read_text(encoding="utf-8")
)
assert validation["status"] == "PASS" and validation["smoke"] is False
assert validation["splits"]["train"]["reasoning_depths"] == [1, 2]
assert validation["splits"]["test"]["reasoning_depths"] == [3, 4]
print("Dataset, leakage, split-isolation, and OOD-partition gates: PASS")"""
    ),
    code(
        """if RUN_GPU:
    import torch
    assert torch.cuda.is_available(), "CUDA is required."
    assert torch.cuda.device_count() >= 2, "Select Kaggle GPU T4 x2."
    gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
    assert all("T4" in name.upper() for name in gpu_names), gpu_names
    print("GPU gate: PASS", gpu_names)
else:
    print("GPU gate: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_GPU:
    from huggingface_hub import snapshot_download
    model_cache = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
    print("Model snapshot ready before parallel workers:", model_cache)
else:
    print("Model prefetch: skipped during local notebook validation")"""
    ),
    code(
        """if RUN_GPU:
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
        """if RUN_GPU:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    smoke_root = REPO_ROOT / "outputs/V03_learning_curve_smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    run_cmd(
        [sys.executable, "scripts/V03_run_learning_curve.py",
         "--config", "configs/v03_learning_curve.yaml", "--smoke"],
        cwd=REPO_ROOT,
    )
    smoke = json.loads(
        (smoke_root / "metrics/learning_curve.json").read_text(encoding="utf-8")
    )
    assert smoke["nested_train_sizes"] == [16]
    assert all(row["all_losses_finite"] for row in smoke["records"])
    print("Two-GPU train/eval smoke gate: PASS")
else:
    print("Two-GPU train/eval smoke gate: skipped during local validation")"""
    ),
    code(
        """if RUN_GPU:
    run_cmd(
        [sys.executable, "scripts/V03_run_learning_curve.py",
         "--config", "configs/v03_learning_curve.yaml", "--resume"],
        cwd=REPO_ROOT,
    )
    print("Full nested learning curve and IID/OOD controls: COMPLETE")
else:
    print("Full GPU training: deferred to Kaggle")"""
    ),
    code(
        """if RUN_GPU:
    curve = json.loads(
        (REPO_ROOT / "outputs/V03_learning_curve/metrics/learning_curve.json")
        .read_text(encoding="utf-8")
    )
    expected_sizes = [2048, 4096, 8192, 16384, 32768]
    assert curve["nested_train_sizes"] == expected_sizes
    assert len(curve["records"]) == len(expected_sizes)
    assert all(row["all_losses_finite"] for row in curve["records"])
    assert all(
        row["single_steps"] == row["split_steps"] for row in curve["records"]
    )
    convergence_warnings = [
        row["train_examples"] for row in curve["records"]
        if not (row["single_loss_improved"] and row["split_loss_improved"])
    ]
    print(json.dumps(curve["records"], indent=2))
    print({
        "finite_loss_gate": "PASS",
        "matched_step_budget_gate": "PASS",
        "loss_trend_warnings": convergence_warnings,
    })
else:
    print("Convergence audit: deferred to Kaggle; no claim is made before GPU training")"""
    ),
    code(
        """if RUN_GPU:
    final_cfg = "outputs/V03_learning_curve/n32768/configs/eval.yaml"
    run_cmd(
        [sys.executable, "scripts/V02_evaluate_split_vs_single.py",
         "--config", final_cfg,
         "--external-config", "configs/v02_official_external.yaml"],
        cwd=REPO_ROOT,
        gpu=0,
    )
    controlled_ood = json.loads(
        (REPO_ROOT / "outputs/V03_learning_curve/n32768/metrics/eval_metrics.json")
        .read_text(encoding="utf-8")
    )
    controlled_iid = json.loads(
        (REPO_ROOT / "outputs/V03_learning_curve/n32768/metrics/eval_metrics_iid_test.json")
        .read_text(encoding="utf-8")
    )
    official = json.loads(
        (REPO_ROOT / "outputs/V02_official_external/metrics/eval_metrics.json")
        .read_text(encoding="utf-8")
    )
    summary = {
        name: {
            "n": result["num_examples"],
            "verdict": result["primary_verdict"],
            "comparison": result["paired_comparisons"]["split_matched_vs_single_full"],
        }
        for name, result in {
            "controlled_ood": controlled_ood,
            "controlled_iid": controlled_iid,
            "official_external": official,
        }.items()
    }
    print(json.dumps(summary, indent=2))
else:
    print("Official evaluation: deferred to Kaggle")"""
    ),
    code(
        """if RUN_GPU:
    result_archive = shutil.make_archive(
        "/kaggle/working/V03_kaggle_results",
        "zip",
        root_dir=REPO_ROOT,
        base_dir="outputs",
    )
    print("Download results from:", result_archive)
else:
    print("LOCAL NOTEBOOK VALIDATION: PASS")
    print("All cells executed; GPU-only work is guarded and will run automatically on Kaggle.")"""
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
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    payload = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    for output in (NOTEBOOK_OUTPUT, DATA_OUTPUT):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Created {output}")


if __name__ == "__main__":
    main()
