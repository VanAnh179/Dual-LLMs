"""Small file, config, and dataset helpers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: str | Path) -> Path:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> dict:
    try:
        import yaml
        with open(project_path(path), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ImportError:
        try:
            from ruamel.yaml import YAML
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: pyyaml. Install it with `pip install -r requirements.txt`."
            ) from exc
        with open(project_path(path), "r", encoding="utf-8") as handle:
            return YAML(typ="safe").load(handle) or {}


def get_teacher_model_name(cfg: dict) -> str:
    if "teacher_model_name" in cfg:
        return cfg["teacher_model_name"]
    if "model_name" in cfg:
        print(
            "Warning: config uses legacy `model_name`; using it for both teacher and student. "
            "Prefer `teacher_model_name` and `student_model_name`.",
            file=sys.stderr,
        )
        return cfg["model_name"]
    raise SystemExit("Config must define `teacher_model_name` or legacy `model_name`.")


def get_student_model_name(cfg: dict) -> str:
    if "student_model_name" in cfg:
        return cfg["student_model_name"]
    if "model_name" in cfg:
        print(
            "Warning: config uses legacy `model_name`; using it for both teacher and student. "
            "Prefer `teacher_model_name` and `student_model_name`.",
            file=sys.stderr,
        )
        return cfg["model_name"]
    raise SystemExit("Config must define `student_model_name` or legacy `model_name`.")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(project_path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: str | Path, default: dict | None = None) -> dict:
    p = project_path(path)
    if not p.exists():
        return {} if default is None else default
    with open(p, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_records(
    rows: list[dict],
    max_examples: int | None = None,
    sampling_mode: str = "first_n",
    seed: int = 42,
) -> list[dict]:
    if sampling_mode not in {"first_n", "random"}:
        raise SystemExit(
            f"Unsupported sampling_mode: {sampling_mode!r}. Expected 'first_n' or 'random'."
        )
    selected = list(rows)
    if sampling_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(selected)
    if max_examples is not None:
        selected = selected[: int(max_examples)]
    return selected


def split_train_validation(
    train_rows: list[dict],
    validation_ratio: float = 0.1,
    sampling_mode: str = "first_n",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    if validation_ratio <= 0:
        return train_rows, []
    if validation_ratio >= 1:
        raise SystemExit("validation_ratio must be >= 0 and < 1.")
    ordered = sample_records(train_rows, sampling_mode=sampling_mode, seed=seed)
    dev_size = int(round(len(ordered) * validation_ratio))
    if dev_size <= 0:
        return ordered, []
    dev_ids = {row.get("id") for row in ordered[:dev_size]}
    dev_rows = [row for row in ordered if row.get("id") in dev_ids]
    remaining = [row for row in ordered if row.get("id") not in dev_ids]
    return remaining, dev_rows


def record_sampled_ids(label: str, rows: list[dict], path: str | Path = "outputs/metrics/sampled_ids.json") -> None:
    payload = read_json(path, default={})
    payload[label] = [row.get("id") for row in rows]
    write_json(path, payload)


def reject_test_split_for_training(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if normalized.endswith("data/raw/test.jsonl") or "/test" in normalized or "test.jsonl" in normalized:
        raise SystemExit(
            "Refusing to use a GSM8K test split as training/bootstrap input. "
            "Use data/raw/train.jsonl or train-derived filtered data."
        )


def reject_test_rows_for_training(rows: list[dict]) -> None:
    bad_ids = [row.get("id") for row in rows if str(row.get("id", "")).startswith("test-")]
    if bad_ids:
        raise SystemExit(
            "Refusing to train/bootstrap on records with GSM8K test IDs. "
            f"First offending ID: {bad_ids[0]}"
        )


def reject_train_split_for_final_eval(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "train.jsonl" in normalized or "dev.jsonl" in normalized or "validation" in normalized:
        raise SystemExit(
            "Final evaluation must use the GSM8K test split only. Use data/raw/test.jsonl."
        )


def reject_train_rows_for_final_eval(rows: list[dict]) -> None:
    bad_ids = [
        row.get("id")
        for row in rows
        if str(row.get("id", "")).startswith("train-") or str(row.get("id", "")).startswith("dev-")
    ]
    if bad_ids:
        raise SystemExit(
            "Final evaluation must use GSM8K test records only. "
            f"First non-test ID found: {bad_ids[0]}"
        )


def require_dependencies(*module_names: str) -> None:
    missing = []
    for name in module_names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {joined}. Install them with `pip install -r requirements.txt`."
        )


def require_cuda_if_requested(require_cuda: bool = False) -> None:
    if not require_cuda:
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch. Install requirements first.") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Use a CUDA machine or set require_cuda: false for CPU smoke tests."
        )
