#!/usr/bin/env python
"""Create one reproducible Kaggle upload ZIP for V02 controlled training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import PROJECT_ROOT, load_config, project_path, read_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v02_multifamily_benchmark.yaml"
    )
    parser.add_argument(
        "--output", default="data/V02_kaggle_training_bundle.zip"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_manifest_path = project_path(cfg["output_paths"]["manifest"])
    validation_path = project_path(cfg["output_paths"]["validation"])
    source_manifest = read_json(source_manifest_path, default={})
    validation = read_json(validation_path, default={})
    if validation.get("status") != "PASS" or validation.get("smoke"):
        raise SystemExit(
            "Refusing to package: full dataset validation must be PASS."
        )

    files = []
    for split in ("train", "dev", "test"):
        path = project_path(cfg["output_paths"][split])
        if not path.exists():
            raise SystemExit(f"Missing {split} dataset: {path}")
        expected = (source_manifest.get("splits") or {}).get(split, {}).get("sha256")
        actual = _sha256(path)
        if expected != actual:
            raise SystemExit(
                f"{split} checksum mismatch: manifest={expected}, actual={actual}"
            )
        files.append(path)
    files.extend((source_manifest_path, validation_path))

    bundle_manifest = {
        "bundle_format": "cospec-ssb-v02-kaggle-v1",
        "experiment_name": cfg["experiment_name"],
        "official_test_data_included": False,
        "files": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            name = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
        info = zipfile.ZipInfo(
            "BUNDLE_MANIFEST.json", date_time=(2026, 7, 25, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            (json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n")
            .encode("utf-8"),
            compresslevel=9,
        )
    temporary.replace(output)
    print(f"Created: {output}")
    print(f"SHA256: {_sha256(output)}")
    print(f"Size: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
