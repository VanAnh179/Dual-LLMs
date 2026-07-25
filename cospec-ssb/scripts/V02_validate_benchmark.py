#!/usr/bin/env python
"""Independently validate V02 schema, solutions, balancing, and split isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V02_benchmark import (
    BLOCK_SIZE, DIFFICULTIES, FAMILIES, GENERATOR_VERSION, LABELS,
    canonical_hash, generate_block, solve_spec,
)
from src.data_utils import load_config, project_path, write_json


REQUIRED_FIELDS = {
    "sample_id", "generator_version", "family", "difficulty", "block_id",
    "variant_a", "variant_b", "view_a", "view_b", "full_problem",
    "gold_answer", "answer_index", "random_baseline", "metadata",
}


def _read_strict(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(
    rows: list[dict[str, Any]], split: str, expected_per_family: int
) -> dict[str, Any]:
    errors: list[str] = []
    ids: list[str] = []
    hashes: list[str] = []
    block_ids: list[str] = []
    labels: Counter[str] = Counter()
    families: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    family_labels: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    view_groups: dict[str, dict[str, Counter[str]]] = {
        "view_a": defaultdict(Counter),
        "view_b": defaultdict(Counter),
    }

    for index, row in enumerate(rows):
        prefix = f"{split}[{index}]"
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue
        try:
            family = str(row["family"])
            difficulty = str(row["difficulty"])
            if family not in FAMILIES:
                raise ValueError(f"unsupported family {family!r}")
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"unsupported difficulty {difficulty!r}")
            if row["generator_version"] != GENERATOR_VERSION:
                raise ValueError("generator version mismatch")
            metadata = row["metadata"]
            block_seed = int(metadata["block_seed"])
            spec = metadata["task_spec"]
            independent_gold = solve_spec(family, spec)
            if row["gold_answer"] != independent_gold:
                raise ValueError(
                    f"gold mismatch: stored={row['gold_answer']} solved={independent_gold}"
                )
            if row["answer_index"] != int(independent_gold.rsplit("_", 1)[1]):
                raise ValueError("answer_index mismatch")
            identity = {
                "version": GENERATOR_VERSION,
                "family": family,
                "difficulty": difficulty,
                "block_seed": block_seed,
                "variant_a": int(row["variant_a"]),
                "variant_b": int(row["variant_b"]),
                "spec": spec,
            }
            digest = canonical_hash(identity)
            if digest != metadata["canonical_hash"]:
                raise ValueError("canonical hash mismatch")
            if row["sample_id"] != f"v02-{family}-{digest[:20]}":
                raise ValueError("sample_id mismatch")
            rendered = row["view_a"] + "\n" + row["view_b"]
            if "ANSWER:" in rendered.upper():
                raise ValueError("marked answer leaked into a partial view")
            for forbidden in ("canonical_hash", "answer_index", "gold_answer"):
                if forbidden in rendered:
                    raise ValueError(f"metadata token leaked into a partial view: {forbidden}")
            if row["full_problem"] != (
                "PRIVATE VIEW A:\n" + row["view_a"]
                + "\n\nPRIVATE VIEW B:\n" + row["view_b"]
            ):
                raise ValueError("full_problem composition mismatch")
            ids.append(row["sample_id"])
            hashes.append(digest)
            block_ids.append(row["block_id"])
            labels[row["gold_answer"]] += 1
            families[family] += 1
            difficulties[difficulty] += 1
            family_labels[family][row["gold_answer"]] += 1
            grouped_rows[(family, block_seed)].append(row)
            view_groups["view_a"][row["view_a"]][row["gold_answer"]] += 1
            view_groups["view_b"][row["view_b"]][row["gold_answer"]] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

    expected_total = expected_per_family * len(FAMILIES)
    if len(rows) != expected_total:
        errors.append(f"{split}: expected {expected_total} rows, found {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append(f"{split}: duplicate sample IDs")
    if len(hashes) != len(set(hashes)):
        errors.append(f"{split}: duplicate canonical hashes")
    expected_family_counts = {family: expected_per_family for family in FAMILIES}
    if dict(families) != expected_family_counts:
        errors.append(
            f"{split}: family imbalance: expected={expected_family_counts}, got={dict(families)}"
        )
    expected_label_count = expected_total // len(LABELS)
    if labels != Counter({label: expected_label_count for label in LABELS}):
        errors.append(f"{split}: global class imbalance: {dict(labels)}")
    for family in FAMILIES:
        expected = expected_per_family // len(LABELS)
        if family_labels[family] != Counter({label: expected for label in LABELS}):
            errors.append(f"{split}/{family}: class imbalance {dict(family_labels[family])}")

    reproduction_errors = 0
    for (family, block_seed), block in grouped_rows.items():
        if len(block) != BLOCK_SIZE:
            errors.append(
                f"{split}/{family}/{block_seed}: expected {BLOCK_SIZE} rows, got {len(block)}"
            )
            continue
        difficulty = str(block[0]["difficulty"])
        expected_by_id = {
            row["sample_id"]: row for row in generate_block(family, difficulty, block_seed)
        }
        for row in block:
            expected = expected_by_id.get(row["sample_id"])
            if expected is None or any(
                row[field] != expected[field]
                for field in ("view_a", "view_b", "full_problem", "gold_answer")
            ):
                reproduction_errors += 1
    if reproduction_errors:
        errors.append(f"{split}: deterministic reproduction failed for {reproduction_errors} rows")

    view_balance = {}
    for view_name, groups in view_groups.items():
        expected_counter = Counter({label: 1 for label in LABELS})
        bad = [
            text for text, counter in groups.items()
            if counter != expected_counter
        ]
        if bad:
            errors.append(
                f"{split}: {view_name} information-balance failed for {len(bad)} groups"
            )
        view_balance[view_name] = {
            "unique_equivalence_classes": len(groups),
            "rows_per_class": 4,
            "bad_classes": len(bad),
            "status": "PASS" if not bad else "FAIL",
        }

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "row_count": len(rows),
        "sample_ids": ids,
        "canonical_hashes": hashes,
        "block_ids": sorted(set(block_ids)),
        "family_counts": dict(sorted(families.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "class_counts": dict(sorted(labels.items())),
        "family_class_counts": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(family_labels.items())
        },
        "partial_view_information_balance": view_balance,
        "random_baseline": 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v02_multifamily_benchmark.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest_path = project_path(cfg["output_paths"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {}
    split_sets = {}
    top_errors = []
    for split in ("train", "dev", "test"):
        path = project_path(cfg["output_paths"][split])
        expected = BLOCK_SIZE if args.smoke else int(cfg["rows_per_family"][split])
        try:
            rows = _read_strict(path)
            result = validate_split(rows, split, expected)
            manifest_hash = manifest["splits"][split]["sha256"]
            actual_hash = _file_sha256(path)
            if actual_hash != manifest_hash:
                result["errors"].append("dataset SHA256 does not match manifest")
                result["status"] = "FAIL"
            result["sha256"] = actual_hash
        except (FileNotFoundError, ValueError, KeyError) as exc:
            result = {"status": "FAIL", "errors": [str(exc)]}
        results[split] = result
        split_sets[split] = {
            "ids": set(result.get("sample_ids", [])),
            "hashes": set(result.get("canonical_hashes", [])),
            "blocks": set(result.get("block_ids", [])),
        }
        top_errors.extend(result.get("errors", []))

    overlap_errors = []
    splits = ("train", "dev", "test")
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            for key in ("ids", "hashes", "blocks"):
                if split_sets[left][key] & split_sets[right][key]:
                    overlap_errors.append(f"{key} overlap between {left} and {right}")
    for result in results.values():
        result.pop("sample_ids", None)
        result.pop("canonical_hashes", None)
        result.pop("block_ids", None)
    output = {
        "experiment_name": cfg["experiment_name"],
        "generator_version": cfg["generator_version"],
        "status": "PASS" if not top_errors and not overlap_errors else "FAIL",
        "smoke": bool(args.smoke),
        "splits": results,
        "split_isolation": {
            "status": "PASS" if not overlap_errors else "FAIL",
            "errors": overlap_errors,
        },
        "hard_failures": top_errors + overlap_errors,
    }
    write_json(cfg["output_paths"]["validation"], output)
    print(json.dumps({
        "status": output["status"],
        "generator_version": output["generator_version"],
        "splits": {
            split: {
                "status": result["status"],
                "row_count": result.get("row_count"),
                "family_counts": result.get("family_counts"),
                "class_counts": result.get("class_counts"),
                "partial_view_information_balance": result.get(
                    "partial_view_information_balance"
                ),
                "sha256": result.get("sha256"),
            }
            for split, result in results.items()
        },
        "split_isolation": output["split_isolation"],
        "hard_failures": output["hard_failures"],
    }, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
