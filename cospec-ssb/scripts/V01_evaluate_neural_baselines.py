#!/usr/bin/env python
"""Evaluate frozen single-model A-only, B-only, and full-problem V01 baselines."""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v01_neural_baselines.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--modes", nargs="+", choices=("view_a", "view_b", "full_problem"))
    parser.add_argument(
        "--rescore-only", action="store_true",
        help="Recompute metrics/report from existing predictions without loading a model.",
    )
    args = parser.parse_args()

    from src.data_utils import load_config, project_path, read_jsonl, write_json, write_jsonl
    from src.V01_neural_baselines import (
        VALID_MODES, evaluate_gates, extract_slot, format_prompt,
        summarize_predictions, write_report,
    )

    cfg = load_config(args.config)
    modes = tuple(args.modes or cfg.get("modes", VALID_MODES))
    dataset_path = project_path(cfg["dataset_path"])
    rows = read_jsonl(dataset_path)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise SystemExit("--max-examples must be positive.")
        rows = rows[: args.max_examples]
    batch_size = int(args.batch_size or cfg["generation"]["batch_size"])
    if batch_size <= 0:
        raise SystemExit("Batch size must be positive.")

    generation_dir = project_path(cfg["output"]["generation_dir"])
    if args.rescore_only:
        expected_by_id = {str(row["sample_id"]): str(row["gold_answer"]) for row in rows}
        results = {}
        for mode in modes:
            predictions = read_jsonl(generation_dir / f"{mode}_predictions.jsonl")
            prediction_by_id = {str(row["sample_id"]): row for row in predictions}
            if set(prediction_by_id) != set(expected_by_id):
                raise SystemExit(f"Existing {mode} predictions do not match the selected dataset rows.")
            if any(
                str(prediction_by_id[sample_id]["gold_answer"]) != gold
                for sample_id, gold in expected_by_id.items()
            ):
                raise SystemExit(f"Existing {mode} predictions contain mismatched gold labels.")
            results[mode] = summarize_predictions(predictions)
        metrics = _build_metrics(cfg, dataset_path, modes, results, evaluate_gates)
        write_json(cfg["output"]["metrics_path"], metrics)
        write_report(metrics, cfg)
        print(f"Rescored existing predictions; gate={metrics['gate']['status']}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run inference on the GPU server or use --rescore-only.")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype_name = str(cfg["generation"].get("dtype", "float16"))
    dtype_by_name = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtype_by_name:
        raise SystemExit(f"Unsupported generation dtype: {dtype_name!r}")
    dtype = dtype_by_name[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=dtype, trust_remote_code=True
    ).to(torch.device("cuda"))
    model.eval()

    results = {}
    for mode in modes:
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [format_prompt(tokenizer, mode, row) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text in zip(batch, texts):
                extracted = extract_slot(text)
                predictions.append({
                    "sample_id": row["sample_id"],
                    "mode": mode,
                    "gold_answer": row["gold_answer"],
                    "predicted_text": text.strip(),
                    "extracted_answer": extracted,
                    "correct": extracted == row["gold_answer"],
                })
            print(f"{mode}: {min(start + batch_size, len(rows))}/{len(rows)}", end="\r")
        output_path = generation_dir / f"{mode}_predictions.jsonl"
        write_jsonl(output_path, predictions)
        results[mode] = summarize_predictions(predictions)
        print(f"{mode}: accuracy={results[mode]['accuracy']:.4f} parse={results[mode]['parse_rate']:.4f}")

    metrics = _build_metrics(cfg, dataset_path, modes, results, evaluate_gates)
    write_json(cfg["output"]["metrics_path"], metrics)
    write_report(metrics, cfg)
    print(f"gate={metrics['gate']['status']}")


def _build_metrics(cfg, dataset_path, modes, results, evaluate_gates):
    metrics = {
        "experiment_name": cfg["experiment_name"],
        "model_name": cfg["model_name"],
        "dataset_path": cfg["dataset_path"],
        "dataset_sha256": _sha256(dataset_path),
        "seed": int(cfg.get("seed", 42)),
        "dtype": str(cfg["generation"].get("dtype", "float16")),
        "decoding": "greedy",
        "selected_modes": list(modes),
        "results": results,
    }
    metrics["cooperation_gain_over_best_partial"] = (
        results["full_problem"]["accuracy"]
        - max(results["view_a"]["accuracy"], results["view_b"]["accuracy"])
        if all(mode in results for mode in ("view_a", "view_b", "full_problem")) else None
    )
    metrics["gate"] = evaluate_gates(metrics, cfg)
    return metrics


if __name__ == "__main__":
    main()
