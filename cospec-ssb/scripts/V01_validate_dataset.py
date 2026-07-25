#!/usr/bin/env python
"""Re-read and independently validate saved V01 dataset artifacts."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V01_csp_generator import CSPProblem, canonical_hash, solve_csp
from src.V01_split_view_formatter import SplitViewExample, audit_for_direct_leakage, format_split_view
from src.data_utils import load_config, project_path, write_json


REQUIRED_FIELDS = {
    "sample_id", "generator_version", "seed", "n_entities", "template_family",
    "full_problem", "view_a", "view_b", "target", "gold_answer", "answer_index",
    "random_baseline", "solution", "metadata",
}


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _to_example(row: dict[str, Any]) -> SplitViewExample:
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    return SplitViewExample(**{field: row[field] for field in SplitViewExample.__dataclass_fields__})


def validate_rows(rows: list[dict[str, Any]], split: str, expected_count: int | None) -> dict[str, Any]:
    errors: list[str] = []
    sample_ids: list[str] = []
    hashes: list[str] = []
    labels: list[str] = []
    n_values: list[int] = []
    constraints: list[int] = []
    templates: Counter[str] = Counter()
    lengths_a: list[int] = []
    lengths_b: list[int] = []
    seeds: list[int] = []
    view_label_groups: dict[str, dict[str, Counter[str]]] = {
        "view_a": defaultdict(Counter), "view_b": defaultdict(Counter),
    }
    for index, row in enumerate(rows):
        prefix = f"{split}[{index}]"
        try:
            example = _to_example(row)
            problem = CSPProblem.from_dict(example.metadata["problem_spec"])
            solutions = solve_csp(problem)
            digest = canonical_hash(problem)
            if len(solutions) != 1:
                errors.append(f"{prefix}: full CSP has {len(solutions)} solutions")
                continue
            target_slot = solutions[0].entity_to_slot[problem.target_entity]
            if example.answer_index != target_slot or example.gold_answer != f"SLOT_{target_slot}":
                errors.append(f"{prefix}: gold answer is inconsistent with independent solver")
            if digest != example.metadata["canonical_hash"]:
                errors.append(f"{prefix}: canonical hash mismatch")
            reproduced = format_split_view(problem, int(example.metadata["format_seed"]))
            if (reproduced.view_a, reproduced.view_b, reproduced.sample_id) != (
                example.view_a, example.view_b, example.sample_id
            ):
                errors.append(f"{prefix}: deterministic reproduction mismatch")
            findings = audit_for_direct_leakage(example)
            errors.extend(f"{prefix}: {finding}" for finding in findings)
            sample_ids.append(example.sample_id)
            hashes.append(digest)
            labels.append(example.gold_answer)
            n_values.append(example.n_entities)
            constraints.append(int(example.metadata["num_constraints"]))
            templates[example.template_family] += 1
            lengths_a.append(len(example.view_a))
            lengths_b.append(len(example.view_b))
            seeds.append(example.seed)
            view_label_groups["view_a"][example.view_a][example.gold_answer] += 1
            view_label_groups["view_b"][example.view_b][example.gold_answer] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: schema/validation error: {exc}")

    counts = Counter(labels)
    balanced = bool(counts) and max(counts.values()) - min(counts.values()) <= 1
    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"{split}: expected {expected_count} rows, found {len(rows)}")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append(f"{split}: duplicate sample IDs")
    if len(hashes) != len(set(hashes)):
        errors.append(f"{split}: duplicate canonical hashes")
    if len(seeds) != len(set(seeds)):
        errors.append(f"{split}: duplicate generation seeds")
    if not balanced:
        errors.append(f"{split}: answer classes are not balanced: {dict(counts)}")
    group_audit: dict[str, Any] = {}
    for view_name, groups in view_label_groups.items():
        repeated = [counter for counter in groups.values() if sum(counter.values()) >= 4]
        bad = [dict(counter) for counter in repeated if len(counter) != 4 or len(set(counter.values())) != 1]
        if bad:
            errors.append(f"{split}: {view_name} repeated-text label balance failed for {len(bad)} groups")
        group_audit[view_name] = {
            "repeated_groups": len(repeated), "imbalanced_repeated_groups": len(bad)
        }
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "row_count": len(rows), "sample_ids": sample_ids, "canonical_hashes": hashes,
        "seeds": seeds, "repeated_view_label_balance": group_audit,
        "class_counts": dict(sorted(counts.items())),
        "random_baseline": (1.0 / len(counts)) if counts else None,
        "majority_baseline": (max(counts.values()) / len(labels)) if labels else None,
        "n_entities_counts": dict(sorted(Counter(n_values).items())),
        "constraint_stats": {
            "min": min(constraints) if constraints else None,
            "max": max(constraints) if constraints else None,
            "mean": statistics.mean(constraints) if constraints else None,
        },
        "template_counts": dict(sorted(templates.items())),
        "text_length": {
            "view_a_mean": statistics.mean(lengths_a) if lengths_a else None,
            "view_b_mean": statistics.mean(lengths_b) if lengths_b else None,
        },
    }


def validate_dataset(config_path: str, review_only: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    outputs = cfg["output_paths"]
    splits = ("review",) if review_only else ("train", "dev", "test")
    expected = {"review": 20, **{key: int(value) for key, value in cfg["split_sizes"].items()}}
    results: dict[str, Any] = {}
    all_ids: dict[str, set[str]] = {}
    all_hashes: dict[str, set[str]] = {}
    all_seeds: dict[str, set[int]] = {}
    top_errors: list[str] = []
    for split in splits:
        path = project_path(outputs[split])
        if not path.exists():
            results[split] = {"status": "FAIL", "errors": [f"missing file: {path}"]}
            top_errors.append(f"{split}: missing file")
            continue
        try:
            rows = read_jsonl_strict(path)
            result = validate_rows(rows, split, expected[split])
        except ValueError as exc:
            result = {"status": "FAIL", "errors": [str(exc)]}
        results[split] = result
        all_ids[split] = set(result.get("sample_ids", []))
        all_hashes[split] = set(result.get("canonical_hashes", []))
        all_seeds[split] = set(result.get("seeds", []))
        top_errors.extend(result.get("errors", []))

    overlaps: list[str] = []
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            if all_ids.get(left, set()) & all_ids.get(right, set()):
                overlaps.append(f"sample_id overlap: {left}/{right}")
            if all_hashes.get(left, set()) & all_hashes.get(right, set()):
                overlaps.append(f"canonical_hash overlap: {left}/{right}")
            if all_seeds.get(left, set()) & all_seeds.get(right, set()):
                overlaps.append(f"seed overlap: {left}/{right}")
    status = "PASS" if not top_errors and not overlaps else "FAIL"
    return {
        "experiment_name": cfg["experiment_name"], "generator_version": cfg["generator_version"],
        "status": status, "review_only": review_only, "splits": results,
        "split_overlap": {"status": "PASS" if not overlaps else "FAIL", "errors": overlaps},
        "hard_failures": top_errors + overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v01_split_view_dataset.yaml")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = validate_dataset(args.config, args.review_only)
    write_json(cfg["output_paths"]["validation"], result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
