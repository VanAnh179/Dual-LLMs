#!/usr/bin/env python
"""Create a deterministic Kaggle upload bundle for the V03 learning curve."""
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
    parser.add_argument("--config", default="configs/v03_ood_benchmark.yaml")
    parser.add_argument(
        "--output", default="data/V03_kaggle_learning_curve_bundle.zip"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = project_path(cfg["output_paths"]["manifest"])
    validation_path = project_path(cfg["output_paths"]["validation"])
    manifest = read_json(manifest_path, default={})
    validation = read_json(validation_path, default={})
    if validation.get("status") != "PASS" or validation.get("smoke"):
        raise SystemExit("Refusing to package: full V03 validation must be PASS.")

    files: list[Path] = []
    for split in cfg["rows_per_family"]:
        path = project_path(cfg["output_paths"][split])
        expected = manifest.get("splits", {}).get(split, {}).get("sha256")
        if not path.exists() or _sha256(path) != expected:
            raise SystemExit(f"Missing or modified validated split: {split}")
        files.append(path)
    files.extend((manifest_path, validation_path))
    bundle_manifest = {
        "bundle_format": "cospec-ssb-v03-kaggle-v1",
        "experiment_name": cfg["experiment_name"],
        "official_test_data_included": False,
        "split_roles": {
            "train": "controlled training",
            "dev": "IID development",
            "iid_test": "IID held-out evaluation",
            "test": "held-out template and structural-depth OOD evaluation",
        },
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
    timestamp = (2026, 7, 26, 0, 0, 0)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            name = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
        info = zipfile.ZipInfo("BUNDLE_MANIFEST.json", date_time=timestamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            (json.dumps(bundle_manifest, indent=2) + "\n").encode(),
            compresslevel=9,
        )
    temporary.replace(output)
    print(f"Created: {output}")
    print(f"SHA256: {_sha256(output)}")
    print(f"Size: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
