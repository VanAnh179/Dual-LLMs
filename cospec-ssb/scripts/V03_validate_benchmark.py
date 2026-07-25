#!/usr/bin/env python
"""Validate V03 solutions, counterfactual balance, isolation, and OOD partition."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V03_benchmark import (
    BLOCK_SIZE,
    FAMILIES,
    GENERATOR_VERSION,
    LABELS,
    canonical_hash,
    generate_block,
    solve_spec,
)
from src.data_utils import load_config, project_path, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _validate_split(
    rows: list[dict], split: str, profile: str, expected_per_family: int
) -> dict:
    errors: list[str] = []
    families = Counter(row.get("family") for row in rows)
    labels = Counter(row.get("gold_answer") for row in rows)
    expected_total = expected_per_family * len(FAMILIES)
    if len(rows) != expected_total:
        errors.append(f"expected {expected_total} rows, found {len(rows)}")
    if families != Counter({family: expected_per_family for family in FAMILIES}):
        errors.append(f"family imbalance: {dict(families)}")
    if labels != Counter({label: expected_total // 4 for label in LABELS}):
        errors.append(f"class imbalance: {dict(labels)}")
    if {row.get("distribution_profile") for row in rows} != {profile}:
        errors.append("distribution profile mismatch")
    expected_partition = "held_out" if profile == "ood" else "development"
    if {row.get("template_partition") for row in rows} != {expected_partition}:
        errors.append("template partition mismatch")
    styles = {int(row.get("render_style", -1)) for row in rows}
    allowed_styles = set(range(4, 8)) if profile == "ood" else set(range(4))
    if not styles <= allowed_styles:
        errors.append(f"render styles escape profile partition: {sorted(styles)}")
    depths = {int(row.get("reasoning_depth", -1)) for row in rows}
    allowed_depths = {3, 4} if profile == "ood" else {1, 2}
    if not depths <= allowed_depths:
        errors.append(f"reasoning depths escape profile partition: {sorted(depths)}")

    ids: set[str] = set()
    hashes: set[str] = set()
    blocks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            family = str(row["family"])
            metadata = row["metadata"]
            block_seed = int(metadata["block_seed"])
            identity = {
                "version": GENERATOR_VERSION,
                "family": family,
                "difficulty": row["difficulty"],
                "profile": profile,
                "block_seed": block_seed,
                "variant_a": int(row["variant_a"]),
                "variant_b": int(row["variant_b"]),
                "spec": metadata["task_spec"],
            }
            digest = canonical_hash(identity)
            if digest != metadata["canonical_hash"]:
                raise ValueError("canonical hash mismatch")
            if solve_spec(family, metadata["task_spec"]) != row["gold_answer"]:
                raise ValueError("gold solution mismatch")
            if row["sample_id"] != f"v03-{family}-{digest[:20]}":
                raise ValueError("sample ID mismatch")
            if row["full_problem"] != (
                "PRIVATE VIEW A:\n"
                + row["view_a"]
                + "\n\nPRIVATE VIEW B:\n"
                + row["view_b"]
            ):
                raise ValueError("full problem composition mismatch")
            if "ANSWER:" in (row["view_a"] + row["view_b"]).upper():
                raise ValueError("marked answer leakage")
            ids.add(row["sample_id"])
            hashes.add(digest)
            blocks[(family, block_seed)].append(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{split}[{index}]: {exc}")

    if len(ids) != len(rows):
        errors.append("duplicate sample IDs")
    if len(hashes) != len(rows):
        errors.append("duplicate canonical hashes")
    bad_view_classes = 0
    reproduction_errors = 0
    expected_counter = Counter({label: 1 for label in LABELS})
    for (family, block_seed), block in blocks.items():
        if len(block) != BLOCK_SIZE:
            errors.append(f"{family}/{block_seed}: incomplete counterfactual block")
            continue
        for field in ("view_a", "view_b"):
            groups: dict[str, Counter] = defaultdict(Counter)
            for row in block:
                groups[row[field]][row["gold_answer"]] += 1
            bad_view_classes += sum(counter != expected_counter for counter in groups.values())
        expected = {
            row["sample_id"]: row
            for row in generate_block(
                family, str(block[0]["difficulty"]), block_seed, profile
            )
        }
        for row in block:
            match = expected.get(row["sample_id"])
            if match is None or any(
                match[field] != row[field]
                for field in ("view_a", "view_b", "gold_answer")
            ):
                reproduction_errors += 1
    if bad_view_classes:
        errors.append(f"{bad_view_classes} partial-view equivalence classes are imbalanced")
    if reproduction_errors:
        errors.append(f"{reproduction_errors} rows failed deterministic reproduction")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "row_count": len(rows),
        "profile": profile,
        "family_counts": dict(sorted(families.items())),
        "class_counts": dict(sorted(labels.items())),
        "unique_ids": len(ids),
        "unique_hashes": len(hashes),
        "block_ids": sorted({row["block_id"] for row in rows}),
        "template_partitions": sorted({row["template_partition"] for row in rows}),
        "reasoning_depths": sorted({int(row["reasoning_depth"]) for row in rows}),
        "partial_view_information_balance": {
            "status": "PASS" if not bad_view_classes else "FAIL",
            "bad_classes": bad_view_classes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v03_ood_benchmark.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = json.loads(
        project_path(cfg["output_paths"]["manifest"]).read_text(encoding="utf-8")
    )
    results = {}
    hard_failures = []
    identity_sets = {}
    for split in cfg["rows_per_family"]:
        path = project_path(cfg["output_paths"][split])
        expected = BLOCK_SIZE if args.smoke else int(cfg["rows_per_family"][split])
        try:
            rows = _read_jsonl(path)
            result = _validate_split(
                rows, split, str(cfg["split_profiles"][split]), expected
            )
            actual_hash = _sha256(path)
            if actual_hash != manifest["splits"][split]["sha256"]:
                result["errors"].append("SHA256 differs from manifest")
                result["status"] = "FAIL"
            result["sha256"] = actual_hash
            identity_sets[split] = {
                "ids": {row["sample_id"] for row in rows},
                "blocks": {row["block_id"] for row in rows},
            }
        except (FileNotFoundError, ValueError, KeyError) as exc:
            result = {"status": "FAIL", "errors": [str(exc)]}
            identity_sets[split] = {"ids": set(), "blocks": set()}
        results[split] = result
        hard_failures.extend(result["errors"])

    overlap_errors = []
    splits = list(results)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            for key in ("ids", "blocks"):
                if identity_sets[left][key] & identity_sets[right][key]:
                    overlap_errors.append(f"{key} overlap between {left} and {right}")
    train_partitions = set(results["train"].get("template_partitions", []))
    ood_partitions = set(results["test"].get("template_partitions", []))
    if train_partitions & ood_partitions:
        overlap_errors.append("train and OOD test template partitions overlap")
    for result in results.values():
        result.pop("block_ids", None)
    output = {
        "experiment_name": cfg["experiment_name"],
        "generator_version": GENERATOR_VERSION,
        "status": "PASS" if not hard_failures and not overlap_errors else "FAIL",
        "smoke": bool(args.smoke),
        "splits": results,
        "split_isolation": {
            "status": "PASS" if not overlap_errors else "FAIL",
            "errors": overlap_errors,
        },
        "hard_failures": hard_failures + overlap_errors,
    }
    write_json(cfg["output_paths"]["validation"], output)
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
