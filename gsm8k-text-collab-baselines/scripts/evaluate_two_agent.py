#!/usr/bin/env python
"""Evaluate single-agent and two-agent GSM8K runs."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    load_config,
    get_student_model_name,
    read_jsonl,
    record_sampled_ids,
    reject_train_split_for_final_eval,
    reject_train_rows_for_final_eval,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
    write_json,
    write_jsonl,
)
from src.evaluation import score_prediction, solve_alone, solve_with_other
from src.generation import latest_adapter, load_tokenizer_and_model


def clear_model(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_qwen05b.yaml")
    parser.add_argument("--input", default="data/raw/test.jsonl")
    parser.add_argument("--agent-a-adapter", default=None)
    parser.add_argument("--agent-b-adapter", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-metrics", default="outputs/metrics/eval_metrics.json")
    parser.add_argument("--output-predictions", default="outputs/generations/eval_predictions.jsonl")
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))
    reject_train_split_for_final_eval(args.input)

    all_rows = read_jsonl(args.input)
    reject_train_rows_for_final_eval(all_rows)
    rows = sample_records(
        all_rows,
        args.max_examples or cfg.get("max_eval_examples"),
        cfg.get("sampling_mode", "first_n"),
        int(cfg.get("seed", 42)),
    )
    record_sampled_ids("eval_test_ids", rows)
    if not rows:
        raise SystemExit(f"No evaluation rows found in {args.input}. Run prepare_gsm8k.py first.")

    adapter_root = f"{cfg.get('output_dir', 'outputs')}/adapters"
    agent_a_adapter = args.agent_a_adapter or latest_adapter(adapter_root, "agent_A")
    agent_b_adapter = args.agent_b_adapter or latest_adapter(adapter_root, "agent_B")
    student_model_name = get_student_model_name(cfg)

    tokenizer_a, model_a = load_tokenizer_and_model(student_model_name, adapter_path=agent_a_adapter)
    tokenizer_b, model_b = load_tokenizer_and_model(student_model_name, adapter_path=agent_b_adapter)
    gen_config = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "top_p": float(cfg.get("top_p", 0.9)),
    }

    predictions = []
    counts = {"agent_A_alone": 0, "agent_B_alone": 0, "A_then_B": 0, "B_then_A": 0}
    for row in tqdm(rows, desc="evaluate"):
        a_alone = solve_alone(tokenizer_a, model_a, row["problem"], gen_config)
        b_alone = solve_alone(tokenizer_b, model_b, row["problem"], gen_config)
        a_then_b = solve_with_other(tokenizer_b, model_b, row["problem"], a_alone, gen_config)
        b_then_a = solve_with_other(tokenizer_a, model_a, row["problem"], b_alone, gen_config)

        scored = {}
        for key, text in {
            "agent_A_alone": a_alone,
            "agent_B_alone": b_alone,
            "A_then_B": a_then_b,
            "B_then_A": b_then_a,
        }.items():
            ok, pred = score_prediction(text, row["gold_answer"])
            counts[key] += int(ok)
            scored[key] = {"text": text, "pred_answer": pred, "correct": ok}

        predictions.append(
            {
                "id": row.get("id"),
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                **scored,
            }
        )

    n = len(rows)
    metrics = {
        "student_model_name": student_model_name,
        "agent_a_adapter": agent_a_adapter,
        "agent_b_adapter": agent_b_adapter,
        "agent_A_alone_accuracy": counts["agent_A_alone"] / n,
        "agent_B_alone_accuracy": counts["agent_B_alone"] / n,
        "A_then_B_accuracy": counts["A_then_B"] / n,
        "B_then_A_accuracy": counts["B_then_A"] / n,
        "num_examples": n,
    }
    write_json(args.output_metrics, metrics)
    write_jsonl(args.output_predictions, predictions)
    clear_model(model_a, model_b)
    print(metrics)


if __name__ == "__main__":
    main()
