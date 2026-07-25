#!/usr/bin/env python
"""Generate V03 train, IID, and held-out OOD benchmark splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V03_benchmark import BLOCK_SIZE, DIFFICULTIES, FAMILIES, PROFILES, generate_block
from src.data_utils import load_config, project_path, write_json


def _atomic_write(path: Path, rows: list[dict], overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v03_ood_benchmark.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if tuple(cfg["families"]) != FAMILIES:
        raise SystemExit(f"Config families must be exactly {FAMILIES}.")
    if tuple(cfg["difficulties"]) != DIFFICULTIES:
        raise SystemExit(f"Config difficulties must be exactly {DIFFICULTIES}.")
    splits = tuple(cfg["rows_per_family"])
    if set(splits) != {"train", "dev", "iid_test", "test"}:
        raise SystemExit("V03 splits must be train, dev, iid_test, and test.")

    seed = int(cfg["seed"])
    manifest = {
        "experiment_name": cfg["experiment_name"],
        "generator_version": cfg["generator_version"],
        "seed": seed,
        "smoke": bool(args.smoke),
        "splits": {},
    }
    for split_index, split in enumerate(splits):
        profile = str(cfg["split_profiles"][split])
        if profile not in PROFILES:
            raise SystemExit(f"Unsupported profile for {split}: {profile}")
        family_size = BLOCK_SIZE if args.smoke else int(cfg["rows_per_family"][split])
        if family_size % BLOCK_SIZE:
            raise SystemExit(
                f"rows_per_family.{split} must be divisible by {BLOCK_SIZE}; got {family_size}."
            )
        rows = []
        for family_index, family in enumerate(FAMILIES):
            for block_index in range(family_size // BLOCK_SIZE):
                block_seed = (
                    seed
                    + int(cfg["seed_offsets"][split])
                    + family_index * 100000
                    + block_index
                )
                difficulty = DIFFICULTIES[block_index % len(DIFFICULTIES)]
                rows.extend(generate_block(family, difficulty, block_seed, profile))
        random.Random(seed + split_index).shuffle(rows)
        output = project_path(cfg["output_paths"][split])
        sha256 = _atomic_write(output, rows, args.overwrite)
        manifest["splits"][split] = {
            "path": cfg["output_paths"][split],
            "profile": profile,
            "sha256": sha256,
            "rows": len(rows),
            "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
            "difficulty_counts": dict(
                sorted(Counter(row["difficulty"] for row in rows).items())
            ),
            "class_counts": dict(
                sorted(Counter(row["gold_answer"] for row in rows).items())
            ),
            "template_partitions": sorted({row["template_partition"] for row in rows}),
            "reasoning_depths": sorted({int(row["reasoning_depth"]) for row in rows}),
        }
        print(f"Generated {split}/{profile}: {len(rows)} rows -> {output}")
    write_json(cfg["output_paths"]["manifest"], manifest)
    print(f"Saved manifest: {project_path(cfg['output_paths']['manifest'])}")


if __name__ == "__main__":
    main()
