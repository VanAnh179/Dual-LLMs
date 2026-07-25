#!/usr/bin/env python
"""Download official holdouts and adapt them into the V02 split-view schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V02_official import (
    ADAPTER_VERSION, adapt_bbh_arithmetic, adapt_clutrr, adapt_prm800k,
    adapt_zebralogic, balanced_interleave, family_counts,
)
from src.data_utils import load_config, project_path, write_json, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_json(url: str) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(url, headers={"User-Agent": "cospec-ssb-v02"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _dataset_revision(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(repo_id, revision=revision).sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v02_official_external.yaml"
    )
    parser.add_argument(
        "--max-per-family", type=int,
        help="Build a non-reportable smoke artifact with fewer rows per family.",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    cfg = load_config(args.config)
    configured_count = int(cfg["rows_per_family"])
    count = int(args.max_per_family or configured_count)
    if count <= 0:
        raise SystemExit("--max-per-family must be positive.")
    smoke = count != configured_count
    seed = int(cfg["seed"])
    sources = cfg["sources"]
    provenance: dict[str, dict[str, Any]] = {}

    clutrr_cfg = sources["clutrr"]
    clutrr_revision = _dataset_revision(
        clutrr_cfg["dataset_id"], str(clutrr_cfg["revision"])
    )
    clutrr = load_dataset(
        clutrr_cfg["dataset_id"],
        clutrr_cfg["dataset_config"],
        split=clutrr_cfg["split"],
        revision=clutrr_revision,
    )
    clutrr_rows = adapt_clutrr(clutrr, count, seed)
    provenance["clutrr"] = {
        **clutrr_cfg,
        "resolved_revision": clutrr_revision,
        "raw_rows": len(clutrr),
        "adapted_rows": len(clutrr_rows),
        "dataset_fingerprint": getattr(clutrr, "_fingerprint", None),
    }
    print(f"CLUTRR: adapted {len(clutrr_rows)}/{len(clutrr)}")

    zebra_cfg = sources["zebralogic"]
    zebra_revision = _dataset_revision(
        zebra_cfg["dataset_id"], str(zebra_cfg["revision"])
    )
    zebra = load_dataset(
        zebra_cfg["dataset_id"],
        zebra_cfg["dataset_config"],
        split=zebra_cfg["split"],
        revision=zebra_revision,
    )
    zebra_rows = adapt_zebralogic(zebra, count, seed)
    provenance["zebralogic"] = {
        **zebra_cfg,
        "resolved_revision": zebra_revision,
        "raw_rows": len(zebra),
        "adapted_rows": len(zebra_rows),
        "dataset_fingerprint": getattr(zebra, "_fingerprint", None),
    }
    print(f"ZebraLogicBench: adapted {len(zebra_rows)}/{len(zebra)}")

    bbh_cfg = sources["bbh_arithmetic"]
    bbh_payload, bbh_sha = _download_json(str(bbh_cfg["url"]))
    bbh_examples = bbh_payload.get("examples")
    if not isinstance(bbh_examples, list):
        raise SystemExit("Unexpected BIG-Bench-Hard JSON: missing examples list.")
    bbh_rows = adapt_bbh_arithmetic(bbh_examples, count, seed)
    provenance["bbh_arithmetic"] = {
        **bbh_cfg,
        "download_sha256": bbh_sha,
        "raw_rows": len(bbh_examples),
        "adapted_rows": len(bbh_rows),
        "canary_present": "canary" in bbh_payload,
    }
    print(f"BIG-Bench-Hard arithmetic: adapted {len(bbh_rows)}/{len(bbh_examples)}")

    prm_cfg = sources["prm800k"]
    prm_revision = _dataset_revision(
        prm_cfg["dataset_id"], str(prm_cfg["revision"])
    )
    prm_path = Path(hf_hub_download(
        repo_id=prm_cfg["dataset_id"],
        filename=prm_cfg["filename"],
        repo_type="dataset",
        revision=prm_revision,
    ))
    prm_sha = _sha256(prm_path)
    expected_prm_sha = str(prm_cfg["expected_sha256"])
    if prm_sha != expected_prm_sha:
        raise SystemExit(
            "PRM800K mirror checksum mismatch: "
            f"expected={expected_prm_sha}, actual={prm_sha}"
        )
    prm_records = []
    with open(prm_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    prm_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Malformed PRM800K JSONL line {line_number}: {exc}"
                    ) from exc
    prm_rows = adapt_prm800k(prm_records, count, seed)
    provenance["prm800k"] = {
        **prm_cfg,
        "resolved_revision": prm_revision,
        "download_sha256": prm_sha,
        "raw_rows": len(prm_records),
        "adapted_rows": len(prm_rows),
        "mirror_note": "Byte-identical mirror of the OpenAI Git LFS artifact.",
    }
    print(f"PRM800K: adapted {len(prm_rows)}/{len(prm_records)}")

    by_family = {
        "relational_csp": clutrr_rows,
        "logic_grid": zebra_rows,
        "arithmetic_constraint": bbh_rows,
        "candidate_verification": prm_rows,
    }
    try:
        rows = balanced_interleave(by_family, count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_path = project_path(cfg["output_paths"]["test"])
    write_jsonl(output_path, rows)
    manifest = {
        "experiment_name": cfg["experiment_name"],
        "adapter_version": ADAPTER_VERSION,
        "official_eval_only": True,
        "smoke": smoke,
        "seed": seed,
        "rows_per_family": count,
        "row_count": len(rows),
        "family_counts": family_counts(rows),
        "sources": provenance,
        "test": {
            "path": cfg["output_paths"]["test"],
            "sha256": _sha256(output_path),
            "row_count": len(rows),
        },
    }
    write_json(cfg["output_paths"]["manifest"], manifest)
    print(f"Saved official external set: {output_path}")
    print(f"Saved provenance manifest: {project_path(cfg['output_paths']['manifest'])}")


if __name__ == "__main__":
    main()
