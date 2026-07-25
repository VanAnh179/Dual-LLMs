#!/usr/bin/env python
"""Generate balanced, non-overlapping V01 split-view CSP JSONL files."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V01_csp_generator import canonical_hash, generate_csp_problem
from src.V01_split_view_formatter import audit_for_direct_leakage, format_split_view
from src.data_utils import load_config, project_path


def _atomic_write_jsonl(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _generate_balanced(
    count: int, seed_start: int, n_entities: int, max_attempts: int,
    seen_hashes: set[str], balancing_seed: int,
) -> list[dict]:
    rows: list[dict] = []
    block_size = n_entities * n_entities
    first_seed = seed_start * block_size
    for offset in range(count):
        problem = generate_csp_problem(first_seed + offset, n_entities)
        digest = canonical_hash(problem)
        if digest in seen_hashes:
            raise RuntimeError(f"Duplicate canonical problem generated: {digest}")
        # A whole combinatorial block shares one surface seed. Therefore each
        # fixed A view and each fixed B view occurs once with every answer class.
        format_seed = problem.seed // block_size
        example = format_split_view(problem, seed=format_seed)
        findings = audit_for_direct_leakage(example)
        if findings:
            raise RuntimeError(f"Leakage audit failed for {example.sample_id}: {findings}")
        seen_hashes.add(digest)
        rows.append(example.to_dict())
    if len(rows) != count:
        raise RuntimeError(f"Internal count error: expected {count}, got {len(rows)}")
    class_counts = Counter(row["gold_answer"] for row in rows)
    if max(class_counts.values()) - min(class_counts.values()) > 1:
        raise RuntimeError(f"Class balancing failed: {class_counts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v01_split_view_dataset.yaml")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--num-train", type=int)
    parser.add_argument("--num-dev", type=int)
    parser.add_argument("--num-test", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_seed = args.seed if args.seed is not None else int(cfg["seed"])
    distribution = {int(key): float(value) for key, value in cfg["n_entities_distribution"].items()}
    if distribution != {4: 1.0}:
        raise SystemExit(
            "V01 v01 dataset generation requires n_entities_distribution {4: 1.0} "
            "to keep a fixed four-class answer space. The core generator still supports N=3."
        )
    n_entities = 4
    max_attempts = int(cfg["max_generation_attempts"])
    offsets = cfg["seed_offsets"]
    outputs = cfg["output_paths"]
    seen_hashes: set[str] = set()

    if args.review_only:
        rows = _generate_balanced(
            20, base_seed + int(offsets["review"]), n_entities, max_attempts,
            seen_hashes, balancing_seed=base_seed,
        )
        path = project_path(outputs["review"])
        _atomic_write_jsonl(path, rows, args.overwrite)
        print(f"Generated review set: {path} ({len(rows)} rows)")
        return

    sizes = dict(cfg["split_sizes"])
    if args.num_train is not None:
        sizes["train"] = args.num_train
    if args.num_dev is not None:
        sizes["dev"] = args.num_dev
    if args.num_test is not None:
        sizes["test"] = args.num_test
    generated: dict[str, list[dict]] = {}
    for split_index, split in enumerate(("train", "dev", "test")):
        generated[split] = _generate_balanced(
            int(sizes[split]), base_seed + int(offsets[split]), n_entities,
            max_attempts, seen_hashes, balancing_seed=base_seed + split_index,
        )
    for split in ("train", "dev", "test"):
        path = project_path(outputs[split])
        _atomic_write_jsonl(path, generated[split], args.overwrite)
        counts = Counter(row["gold_answer"] for row in generated[split])
        print(f"Generated {split}: {path} ({len(generated[split])} rows, classes={dict(counts)})")


if __name__ == "__main__":
    main()
