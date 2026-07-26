#!/usr/bin/env python
"""Run nested V03 learning curves with matched single/split training budgets."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_config, project_path, read_json, read_jsonl, write_json
from src.V02_modeling import select_nested_training_rows


ALL_EVAL_MODES = (
    "base_full",
    "single_full",
    "single_a",
    "single_b",
    "split_matched",
    "split_shuffled",
    "split_zero",
)


def _relative(path: Path) -> str:
    return str(path.relative_to(project_path("."))).replace("\\", "/")


def _write_yaml(path: Path, payload: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyyaml. Install it with `pip install pyyaml`."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _point_config(base: dict, root: Path, size: int, manifest_name: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{base['experiment_name']}_n{size}"
    cfg["outputs"] = {
        "root": _relative(root),
        "single_adapter_root": _relative(root / "adapters/single"),
        "split_adapter": _relative(root / "adapters/split/receiver_lora"),
        "bridge": _relative(root / "adapters/split/latent_bridge.pt"),
        "training_manifest": _relative(root / f"metrics/{manifest_name}"),
        "metrics": _relative(root / "metrics/eval_metrics.json"),
        "generation_dir": _relative(root / "generations"),
    }
    return cfg


def _run(args: list[str], gpu: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[GPU {gpu}] + {' '.join(args)}", flush=True)
    return subprocess.Popen(args, cwd=project_path("."), env=env)


def _wait_pair(left: subprocess.Popen, right: subprocess.Popen, label: str) -> None:
    started = time.monotonic()
    next_heartbeat = started + 30
    processes = (("gpu0", left), ("gpu1", right))
    try:
        while True:
            left_code = left.poll()
            right_code = right.poll()
            if left_code is not None and right_code is not None:
                break
            now = time.monotonic()
            if now >= next_heartbeat:
                states = {
                    name: ("running" if process.poll() is None else process.returncode)
                    for name, process in processes
                }
                print(
                    f"[{label}] elapsed={int(now - started)}s workers={states}",
                    flush=True,
                )
                next_heartbeat = now + 30
            if (
                left_code not in (None, 0)
                or right_code not in (None, 0)
            ):
                for _, process in processes:
                    if process.poll() is None:
                        process.terminate()
                left_code = left.wait()
                right_code = right.wait()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"Interrupting {label}; terminating GPU workers...", flush=True)
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if left_code or right_code:
        raise SystemExit(
            f"{label} failed: gpu0_exit={left_code}, gpu1_exit={right_code}"
        )


def _wait_one(process: subprocess.Popen, label: str) -> None:
    started = time.monotonic()
    next_heartbeat = started + 30
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now >= next_heartbeat:
                print(
                    f"[{label}] elapsed={int(now - started)}s worker=running",
                    flush=True,
                )
                next_heartbeat = now + 30
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"Interrupting {label}; terminating GPU worker...", flush=True)
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if process.returncode:
        raise SystemExit(f"{label} failed: exit={process.returncode}")


def _merge_manifests(paths: list[Path], output: Path) -> dict:
    merged: dict = {"single_baselines": {}}
    for path in paths:
        payload = read_json(path, default={})
        for key in (
            "experiment_name", "model_name", "seed", "train_dataset_sha256"
        ):
            if key in payload:
                merged[key] = payload[key]
        merged["single_baselines"].update(payload.get("single_baselines", {}))
        if payload.get("split_latent"):
            merged["split_latent"] = payload["split_latent"]
    write_json(output, merged)
    return merged


def _evaluate(
    config_path: Path,
    modes: tuple[str, ...],
    test_split: str,
    max_examples: int | None,
) -> None:
    command = [
        sys.executable,
        "scripts/V02_evaluate_split_vs_single.py",
        "--config",
        _relative(config_path),
        "--test-split",
        test_split,
        "--modes",
        *modes,
    ]
    if max_examples is not None:
        command.extend(["--max-examples", str(max_examples)])
    process = _run(command, gpu=0)
    _wait_one(process, f"evaluation {test_split} for {config_path.parent.parent.name}")


def _curve_record(size: int, cfg: dict, metrics: dict) -> dict:
    manifest = read_json(cfg["outputs"]["training_manifest"], default={})
    single = manifest["single_baselines"]["full"]
    split = manifest["split_latent"]
    comparison = metrics["paired_comparisons"][
        "split_matched_vs_single_full"
    ]
    return {
        "train_examples": size,
        "eval_examples": metrics["num_examples"],
        "single_full_accuracy": metrics["summaries"]["single_full"]["overall"]["accuracy"],
        "split_matched_accuracy": metrics["summaries"]["split_matched"]["overall"]["accuracy"],
        "split_minus_single": comparison["delta"],
        "ci_low": comparison["bootstrap_95_ci"][0],
        "ci_high": comparison["bootstrap_95_ci"][1],
        "verdict": metrics["primary_verdict"],
        "single_steps": single["optimizer_steps"],
        "split_steps": split["optimizer_steps"],
        "single_loss_improved": single["loss_improved"],
        "split_loss_improved": split["loss_improved"],
        "all_losses_finite": bool(
            single["all_losses_finite"] and split["all_losses_finite"]
        ),
        "sample_ids_sha256": single["sample_ids_sha256"],
    }


def _training_complete(
    cfg: dict, size: int, expected_sample_hash: str, include_partials: bool
) -> bool:
    manifest = read_json(cfg["outputs"]["training_manifest"], default={})
    single = manifest.get("single_baselines", {})
    required_single = ("full", "view_a", "view_b") if include_partials else ("full",)
    records = [single.get(mode) for mode in required_single]
    records.append(manifest.get("split_latent"))
    if any(not record for record in records):
        return False
    if any(int(record.get("num_train_examples", -1)) != size for record in records):
        return False
    if any(record.get("sample_ids_sha256") != expected_sample_hash for record in records):
        return False
    if any(not record.get("all_losses_finite") for record in records):
        return False
    if len({int(record.get("optimizer_steps", -1)) for record in records}) != 1:
        return False
    artifact_paths = [single[mode]["adapter_path"] for mode in required_single]
    artifact_paths.extend((
        manifest["split_latent"]["receiver_adapter_path"],
        manifest["split_latent"]["bridge_path"],
    ))
    return all(project_path(path).exists() for path in artifact_paths)


def _metrics_complete(
    cfg: dict, test_split: str, modes: tuple[str, ...], expected_examples: int
) -> bool:
    metrics_path = Path(cfg["outputs"]["metrics"])
    if test_split != "test":
        metrics_path = metrics_path.with_name(
            f"{metrics_path.stem}_{test_split}{metrics_path.suffix}"
        )
    metrics = read_json(metrics_path, default={})
    return (
        int(metrics.get("num_examples", -1)) >= expected_examples
        and set(modes).issubset(set(metrics.get("selected_modes", ())))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v03_learning_curve.yaml")
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        help="Override configured nested learning-curve sizes.",
    )
    parser.add_argument("--intermediate-eval-examples", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny sequential point and skip final controls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated completed points and continue incomplete work.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("require_cuda") is not True:
        raise SystemExit("V03 learning curves require require_cuda: true.")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required.") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("V03 learning curves require two visible CUDA GPUs.")

    configured_sizes = [int(value) for value in cfg["learning_curve"]["train_sizes"]]
    sizes = args.train_sizes or configured_sizes
    if args.smoke:
        sizes = [16]
    if sizes != sorted(set(sizes)):
        raise SystemExit("Learning-curve sizes must be unique and increasing.")
    train_rows = read_jsonl(cfg["data"]["train"])
    if sizes[-1] > len(train_rows):
        raise SystemExit(
            f"Largest train size {sizes[-1]} exceeds {len(train_rows)} rows."
        )
    intermediate_eval = (
        args.intermediate_eval_examples
        or int(cfg["learning_curve"]["intermediate_eval_examples"])
    )
    if args.smoke:
        intermediate_eval = 8

    shuffled = select_nested_training_rows(
        train_rows, max_examples=None, seed=int(cfg["seed"])
    )
    previous_ids: set[str] = set()
    curve_root = project_path(cfg["outputs"]["root"])
    if args.smoke:
        curve_root = curve_root.with_name(f"{curve_root.name}_smoke")
    records = []
    final_eval_config: Path | None = None

    for size in sizes:
        point_root = curve_root / f"n{size:05d}"
        selected_ids = [str(row["sample_id"]) for row in shuffled[:size]]
        selected_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
        selected_set = set(selected_ids)
        if not previous_ids.issubset(selected_set):
            raise RuntimeError("Learning-curve subsets are not nested.")
        previous_ids = selected_set
        write_json(point_root / "metrics/selected_sample_ids.json", selected_ids)

        single_cfg = _point_config(
            cfg, point_root, size, "single_training_manifest.json"
        )
        split_cfg = _point_config(
            cfg, point_root, size, "split_training_manifest.json"
        )
        eval_cfg = _point_config(cfg, point_root, size, "training_manifest.json")
        single_cfg_path = point_root / "configs/single.yaml"
        split_cfg_path = point_root / "configs/split.yaml"
        eval_cfg_path = point_root / "configs/eval.yaml"
        _write_yaml(single_cfg_path, single_cfg)
        _write_yaml(split_cfg_path, split_cfg)
        _write_yaml(eval_cfg_path, eval_cfg)

        training_ready = args.resume and _training_complete(
            eval_cfg, size, selected_hash, include_partials=False
        )
        if training_ready:
            print(f"Resume: training n={size} already complete", flush=True)
            merged = read_json(eval_cfg["outputs"]["training_manifest"])
        else:
            single_command = [
                sys.executable,
                "scripts/V02_train_single_baselines.py",
                "--config",
                _relative(single_cfg_path),
                "--modes",
                "full",
                "--max-examples",
                str(size),
            ]
            split_command = [
                sys.executable,
                "scripts/V02_train_split_latent.py",
                "--config",
                _relative(split_cfg_path),
                "--max-examples",
                str(size),
            ]
            single_process = _run(single_command, gpu=0)
            if args.smoke:
                _wait_one(single_process, f"smoke single training n={size}")
                split_process = _run(split_command, gpu=1)
                _wait_one(split_process, f"smoke split training n={size}")
            else:
                split_process = _run(split_command, gpu=1)
                _wait_pair(single_process, split_process, f"training n={size}")
            merged = _merge_manifests(
                [
                    project_path(single_cfg["outputs"]["training_manifest"]),
                    project_path(split_cfg["outputs"]["training_manifest"]),
                ],
                project_path(eval_cfg["outputs"]["training_manifest"]),
            )
        full_record = merged["single_baselines"]["full"]
        split_record = merged["split_latent"]
        if full_record["sample_ids_sha256"] != split_record["sample_ids_sha256"]:
            raise RuntimeError(f"Training row mismatch at n={size}")
        if full_record["optimizer_steps"] != split_record["optimizer_steps"]:
            raise RuntimeError(f"Optimizer-step budget mismatch at n={size}")
        if not (
            full_record["all_losses_finite"] and split_record["all_losses_finite"]
        ):
            raise RuntimeError(f"Non-finite loss at n={size}")

        max_eval = intermediate_eval
        curve_modes = ("single_full", "split_matched")
        if args.resume and _metrics_complete(
            eval_cfg, "test", curve_modes, max_eval
        ):
            print(f"Resume: OOD evaluation n={size} already complete", flush=True)
        else:
            _evaluate(eval_cfg_path, curve_modes, "test", max_eval)
        metrics = read_json(eval_cfg["outputs"]["metrics"])
        records.append(_curve_record(size, eval_cfg, metrics))
        final_eval_config = eval_cfg_path

    if not args.smoke and final_eval_config is not None:
        final_cfg = load_config(_relative(final_eval_config))
        final_root = project_path(final_cfg["outputs"]["root"])
        final_selected_ids = [str(row["sample_id"]) for row in shuffled[:sizes[-1]]]
        final_selected_hash = hashlib.sha256(
            "\n".join(final_selected_ids).encode()
        ).hexdigest()
        partial_a_cfg = _point_config(
            cfg, final_root, sizes[-1], "partial_a_training_manifest.json"
        )
        partial_b_cfg = _point_config(
            cfg, final_root, sizes[-1], "partial_b_training_manifest.json"
        )
        partial_a_path = final_root / "configs/partial_a.yaml"
        partial_b_path = final_root / "configs/partial_b.yaml"
        _write_yaml(partial_a_path, partial_a_cfg)
        _write_yaml(partial_b_path, partial_b_cfg)
        partials_ready = args.resume and _training_complete(
            final_cfg, sizes[-1], final_selected_hash, include_partials=True
        )
        if partials_ready:
            print("Resume: final partial-view controls already complete", flush=True)
        else:
            partial_a = _run(
                [
                    sys.executable,
                    "scripts/V02_train_single_baselines.py",
                    "--config",
                    _relative(partial_a_path),
                    "--modes",
                    "view_a",
                    "--max-examples",
                    str(sizes[-1]),
                ],
                gpu=0,
            )
            partial_b = _run(
                [
                    sys.executable,
                    "scripts/V02_train_single_baselines.py",
                    "--config",
                    _relative(partial_b_path),
                    "--modes",
                    "view_b",
                    "--max-examples",
                    str(sizes[-1]),
                ],
                gpu=1,
            )
            _wait_pair(partial_a, partial_b, "final partial-view controls")
            _merge_manifests(
                [
                    project_path(final_root / "metrics/single_training_manifest.json"),
                    project_path(final_root / "metrics/split_training_manifest.json"),
                    project_path(partial_a_cfg["outputs"]["training_manifest"]),
                    project_path(partial_b_cfg["outputs"]["training_manifest"]),
                ],
                project_path(final_cfg["outputs"]["training_manifest"]),
            )
        final_test_count = len(read_jsonl(final_cfg["data"]["test"]))
        final_iid_count = len(read_jsonl(final_cfg["data"]["iid_test"]))
        if not (
            args.resume
            and _metrics_complete(
                final_cfg, "test", ALL_EVAL_MODES, final_test_count
            )
        ):
            _evaluate(final_eval_config, ALL_EVAL_MODES, "test", None)
        if not (
            args.resume
            and _metrics_complete(
                final_cfg, "iid_test", ALL_EVAL_MODES, final_iid_count
            )
        ):
            _evaluate(final_eval_config, ALL_EVAL_MODES, "iid_test", None)

    summary = {
        "experiment_name": cfg["experiment_name"],
        "nested_train_sizes": sizes,
        "intermediate_eval_examples": intermediate_eval,
        "final_full_controls": not args.smoke,
        "gpu_parallelism": {
            "gpu_0": "single_full",
            "gpu_1": "split_latent",
        },
        "records": records,
        "final_eval_config": (
            _relative(final_eval_config) if final_eval_config else None
        ),
    }
    summary_path = curve_root / "metrics/learning_curve.json"
    write_json(summary_path, summary)
    csv_path = curve_root / "metrics/learning_curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f"Saved learning curve: {summary_path}")
    print(f"Final evaluation config: {summary['final_eval_config']}")


if __name__ == "__main__":
    main()
