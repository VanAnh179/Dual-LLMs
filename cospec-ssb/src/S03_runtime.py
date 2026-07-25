"""Shared loading, fingerprinting, and preflight helpers for S03 scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data_utils import load_config, project_path


def sha256_path(path: str | Path) -> str:
    target = project_path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    for file_path in files:
        digest.update(str(file_path.relative_to(target.parent)).replace("\\", "/").encode())
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_s03_and_s02(config_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(config_path)
    source_path = cfg.get("source_s02_config")
    if not source_path:
        raise SystemExit("S03 config must define source_s02_config.")
    return cfg, load_config(source_path)


def s02_artifact_paths(s02_cfg: dict[str, Any]) -> dict[str, Path]:
    mc = s02_cfg["minimal_coupling"]
    adapter_root = project_path(s02_cfg["output"]["adapter_dir"])
    return {
        "agent_a_pre_s02": project_path(mc["agent_a_adapter_path"]),
        "agent_b_pre_s02": project_path(mc["agent_b_adapter_path"]),
        "agent_b_s02": adapter_root / "agent_B_minimal_coupling_sft",
        "bridge_s02": adapter_root / "minimal_coupling_bridge.pt",
    }


def require_s02_artifacts(s03_cfg: dict[str, Any], s02_cfg: dict[str, Any]) -> dict[str, Path]:
    paths = s02_artifact_paths(s02_cfg)
    paths["source_metrics"] = project_path(s03_cfg["source_s02_metrics"])
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        detail = "\n  - ".join(missing)
        raise SystemExit(
            "S03 preflight blocked. Required S02 artifacts are missing:\n  - " + detail
            + "\nRefusing to retrain S02 or create placeholder checkpoints."
        )
    return paths


def artifact_fingerprints(
    s03_cfg: dict[str, Any], s02_cfg: dict[str, Any], paths: dict[str, Path]
) -> dict[str, str]:
    source_config = project_path(s03_cfg["source_s02_config"])
    return {
        "source_s02_config_sha256": sha256_path(source_config),
        "agent_a_adapter_sha256": sha256_path(paths["agent_a_pre_s02"]),
        "agent_b_adapter_sha256": sha256_path(paths["agent_b_s02"]),
        "bridge_sha256": sha256_path(paths["bridge_s02"]),
        "model_name": str(s02_cfg["student_model_name"]),
    }


def json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def choose_device_dtype():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for S03 model smoke/full runs.")
    # Match scripts/S02_evaluate_minimal_coupling.py exactly. Evaluation
    # precision is part of the controlled pipeline because it can change a
    # small number of greedy-decoding outcomes.
    return torch.device("cuda"), torch.float16


def choose_training_device_dtype():
    """Match S02 training precision exactly: BF16 when supported, FP32 otherwise."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for S03 zero-control training.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.device("cuda"), dtype


def load_receiver_model(s02_cfg: dict[str, Any], adapter_path: str | Path, device, dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model_name = s02_cfg["student_model_name"]
    mc = s02_cfg["minimal_coupling"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    if mc.get("init_agent_b_from_d11", True):
        model = PeftModel.from_pretrained(
            model, str(project_path(mc["agent_b_adapter_path"])), is_trainable=False
        ).merge_and_unload()
    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
