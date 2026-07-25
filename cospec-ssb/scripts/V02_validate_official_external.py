#!/usr/bin/env python
"""Validate official provenance and the adapted V02 external holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V02_benchmark import LABELS
from src.V02_official import ADAPTER_VERSION, OFFICIAL_FAMILIES, canonical_hash
from src.data_utils import load_config, project_path, read_json, write_json


REQUIRED_FIELDS = {
    "sample_id", "generator_version", "family", "difficulty", "block_id",
    "view_a", "view_b", "full_problem", "gold_answer", "answer_index",
    "random_baseline", "metadata",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_strict(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_number}.")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v02_official_external.yaml"
    )
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = read_json(cfg["output_paths"]["manifest"], default={})
    test_path = project_path(cfg["output_paths"]["test"])
    errors: list[str] = []
    if not cfg.get("official_eval_only"):
        errors.append("config must declare official_eval_only: true")
    if not manifest.get("official_eval_only"):
        errors.append("manifest must declare official_eval_only: true")
    if manifest.get("smoke") and not args.allow_smoke:
        errors.append("smoke artifact cannot pass reportable validation")
    try:
        rows = _read_strict(test_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        rows = []
        errors.append(str(exc))

    ids = set()
    source_keys = set()
    family_counts: Counter[str] = Counter()
    family_labels: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        prefix = f"test[{index}]"
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue
        try:
            family = str(row["family"])
            if family not in OFFICIAL_FAMILIES:
                raise ValueError(f"unsupported family {family!r}")
            if row["difficulty"] not in ("easy", "medium", "hard"):
                raise ValueError("invalid difficulty")
            if row["generator_version"] != ADAPTER_VERSION:
                raise ValueError("adapter version mismatch")
            if row["gold_answer"] not in LABELS:
                raise ValueError("invalid gold label")
            if row["answer_index"] != int(row["gold_answer"].rsplit("_", 1)[1]):
                raise ValueError("answer index mismatch")
            metadata = row["metadata"]
            if not metadata.get("official_eval_only"):
                raise ValueError("row is not marked official_eval_only")
            options = metadata["option_values"]
            if set(options) != set(LABELS) or len(set(options.values())) != 4:
                raise ValueError("options must contain four distinct OPTION labels")
            gold_value = options[row["gold_answer"]]
            if hashlib.sha256(gold_value.encode()).hexdigest() != metadata[
                "gold_value_sha256"
            ]:
                raise ValueError("gold value checksum mismatch")
            identity = {
                "adapter_version": ADAPTER_VERSION,
                "family": family,
                "source_name": metadata["source_name"],
                "source_id": metadata["source_id"],
                "source_split": metadata["source_split"],
                "options": options,
                "gold_value": gold_value,
            }
            digest = canonical_hash(identity)
            if digest != metadata["canonical_hash"]:
                raise ValueError("canonical hash mismatch")
            if row["sample_id"] != f"v02o-{family}-{digest[:20]}":
                raise ValueError("sample ID mismatch")
            if row["full_problem"] != (
                f"PRIVATE VIEW A:\n{row['view_a']}\n\n"
                f"PRIVATE VIEW B:\n{row['view_b']}"
            ):
                raise ValueError("full problem is not the exact view composition")
            if "ANSWER:" in (row["view_a"] + row["view_b"]).upper():
                raise ValueError("marked answer leaked into a private view")
            source_key = (metadata["source_name"], metadata["source_id"])
            if row["sample_id"] in ids:
                raise ValueError("duplicate sample ID")
            if source_key in source_keys:
                raise ValueError("duplicate official source item")
            ids.add(row["sample_id"])
            source_keys.add(source_key)
            family_counts[family] += 1
            family_labels[family][row["gold_answer"]] += 1
            source_counts[metadata["source_name"]] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

    expected = int(manifest.get("rows_per_family", cfg["rows_per_family"]))
    expected_counts = {family: expected for family in OFFICIAL_FAMILIES}
    if dict(family_counts) != expected_counts:
        errors.append(
            f"family imbalance: expected={expected_counts}, got={dict(family_counts)}"
        )
    if expected % 4:
        errors.append("rows_per_family must be divisible by four")
    else:
        expected_labels = Counter({label: expected // 4 for label in LABELS})
        for family in OFFICIAL_FAMILIES:
            if family_labels[family] != expected_labels:
                errors.append(
                    f"{family}: label imbalance {dict(family_labels[family])}"
                )
    actual_sha = _sha256(test_path) if test_path.exists() else None
    if actual_sha != (manifest.get("test") or {}).get("sha256"):
        errors.append("test SHA256 differs from provenance manifest")
    source_manifest = manifest.get("sources") or {}
    for source in ("clutrr", "zebralogic", "bbh_arithmetic", "prm800k"):
        item = source_manifest.get(source) or {}
        if not item.get("canonical_repository"):
            errors.append(f"{source}: missing canonical repository")
        if not item.get("license"):
            errors.append(f"{source}: missing license declaration")
        if not item.get("resolved_revision") and not item.get("download_sha256"):
            errors.append(f"{source}: missing resolved revision or download hash")

    result = {
        "experiment_name": cfg["experiment_name"],
        "adapter_version": ADAPTER_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "smoke": bool(manifest.get("smoke")),
        "official_eval_only": True,
        "row_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "family_class_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_labels.items())
        },
        "source_counts": dict(sorted(source_counts.items())),
        "test_sha256": actual_sha,
        "provenance_status": "PASS" if not any(
            "repository" in error or "revision" in error or "license" in error
            for error in errors
        ) else "FAIL",
        "hard_failures": errors,
    }
    write_json(cfg["output_paths"]["validation"], result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
