#!/usr/bin/env python
"""Evaluate the zero-trained S03 receiver control on the canonical GSM8K test rows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s03_zero_control_gsm8k.yaml")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from src.answer_extraction import extract_gsm8k_gold_answer
    from src.data_utils import (
        project_path, read_json, read_jsonl, record_sampled_ids,
        reject_train_rows_for_final_eval, reject_train_split_for_final_eval,
        sample_records, write_json, write_jsonl,
    )
    from src.evaluation import score_prediction
    from src.generation import generate_text
    from src.prompts import format_for_generation, messages_for_single
    from src.S01_hook_utils import HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge
    from src.S03_causal_metrics import paired_bootstrap_delta
    from src.S03_reporting import write_s03_report
    from src.S03_runtime import (
        choose_device_dtype, load_receiver_model, load_s03_and_s02, require_s02_artifacts,
    )

    cfg, s02 = load_s03_and_s02(args.config)
    require_s02_artifacts(cfg, s02)
    adapter_path = project_path(cfg["output"]["adapter_dir"])
    if not adapter_path.exists():
        raise SystemExit(f"Zero-control adapter is missing: {adapter_path}")
    device, dtype = choose_device_dtype()
    test_path = s02["data"]["raw_test_path"]
    reject_train_split_for_final_eval(test_path)
    rows = read_jsonl(test_path)
    reject_train_rows_for_final_eval(rows)
    n = args.max_examples or int(cfg.get("max_eval_examples", 100))
    rows = sample_records(rows, n, cfg.get("sampling_mode", "first_n"), int(cfg.get("seed", 42)))
    record_sampled_ids(
        "S03_zero_control_eval", rows,
        path="outputs/S03_zero_control_gsm8k/metrics/sampled_ids.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(s02["student_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_receiver_model(s02, adapter_path, device, dtype)
    model.eval()
    import torch
    mc = s02["minimal_coupling"]
    zero_bridge = MinimalCouplingBridge(
        d_model=int(model.config.hidden_size), bottleneck_dim=int(mc["bottleneck_dim"])
    ).to(device).to(dtype)
    zero_bridge.eval().requires_grad_(False)
    zero_z = torch.zeros((1, int(mc["bottleneck_dim"])), device=device, dtype=dtype)
    layer_b = get_layer_by_index(model, int(mc["layer_index"]))
    predictions = []
    for index, row in enumerate(rows):
        prompt = format_for_generation(tokenizer, messages_for_single(row["problem"]))
        injector = HiddenStateInjector(layer_b, lambda hidden: zero_bridge.inject(hidden, zero_z))
        try:
            text = generate_text(tokenizer, model, prompt, **s02["evaluation"])
        finally:
            injector.remove()
        gold = extract_gsm8k_gold_answer(row["raw_answer"])
        correct, extracted = score_prediction(text, gold)
        predictions.append({
            "sample_id": row["id"], "problem": row["problem"], "gold_answer": gold,
            "predicted_text": text, "extracted_answer": extracted, "correct": bool(correct),
            "intervention": "zero_trained_control", "z_source_sample_id": None,
            "seed": int(cfg.get("seed", 42)),
        })
        print(f"zero-control: {index + 1}/{len(rows)}", end="\r")
    write_jsonl(cfg["output"]["predictions_path"], predictions)
    accuracy = sum(int(row["correct"]) for row in predictions) / len(predictions)
    metrics = {
        "experiment_name": cfg["experiment_name"], "num_examples": len(predictions),
        "zero_control_accuracy": accuracy,
        "training_parity": read_json("outputs/S03_zero_control_gsm8k/metrics/training_parity.json"),
    }

    diagnostic_path = project_path("outputs/S03_causal_diagnostic_gsm8k/generations/matched_predictions.jsonl")
    if diagnostic_path.exists():
        matched_rows = read_jsonl(diagnostic_path)
        matched_by_id = {str(row["sample_id"]): row for row in matched_rows}
        ids = [str(row["sample_id"]) for row in predictions]
        if all(sample_id in matched_by_id for sample_id in ids):
            matched = torch.tensor([bool(matched_by_id[i]["correct"]) for i in ids], dtype=torch.float64)
            control = torch.tensor([bool(row["correct"]) for row in predictions], dtype=torch.float64)
            paired = paired_bootstrap_delta(matched, control, num_resamples=10000, seed=int(cfg.get("seed", 42)))
            metrics["delta_matched_vs_zero_control"] = paired["delta"]
            metrics["paired_bootstrap_95_ci"] = [paired["ci_low"], paired["ci_high"]]
    write_json(cfg["output"]["metrics_path"], metrics)
    write_s03_report()
    print(f"\nzero_control_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
