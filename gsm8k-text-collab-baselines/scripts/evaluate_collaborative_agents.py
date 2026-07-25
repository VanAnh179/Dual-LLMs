#!/usr/bin/env python
"""Evaluate D11.2 collaborative agents against base single-model behavior."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    get_student_model_name,
    get_teacher_model_name,
    load_config,
    read_jsonl,
    record_sampled_ids,
    reject_train_rows_for_final_eval,
    reject_train_split_for_final_eval,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
    write_json,
    write_jsonl,
)
from src.evaluation import score_prediction
from src.generation import generate_text, load_tokenizer_and_model
from src.prompts import (
    format_for_generation,
    messages_for_collaborative_standalone,
    messages_for_first_contributor,
    messages_for_second_contributor,
)


EXPERIMENT_NAME = "D11_2_latent_collaborative_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def clear_model(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def solve(tokenizer, model, messages: list[dict[str, str]], gen_config: dict) -> str:
    return generate_text(tokenizer, model, format_for_generation(tokenizer, messages), **gen_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_2_qwen_math7b_teacher.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--agent-a-adapter", default=None)
    parser.add_argument("--agent-b-adapter", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["first_n", "random"], default=None)
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    input_path = args.input or cfg_get(cfg, "data", "raw_test_path", "data/raw/test.jsonl")
    reject_train_split_for_final_eval(input_path)
    all_rows = read_jsonl(input_path)
    reject_train_rows_for_final_eval(all_rows)
    sampling_mode = args.sampling_mode or cfg_get(cfg, "sampling", "sampling_mode", "first_n")
    seed = int(cfg_get(cfg, "sampling", "seed", 42))
    rows = sample_records(
        all_rows,
        args.max_examples or cfg_get(cfg, "sampling", "max_eval_examples", 100),
        sampling_mode,
        seed,
    )
    record_sampled_ids("D11_2_eval_test_ids", rows)
    if not rows:
        raise SystemExit(f"No evaluation rows found in {input_path}.")

    output_root = cfg_get(cfg, "output", "root_dir", f"outputs/{EXPERIMENT_NAME}")
    adapter_dir = cfg_get(cfg, "output", "adapter_dir", f"{output_root}/adapters")
    generation_dir = cfg_get(cfg, "output", "generation_dir", f"{output_root}/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", f"{output_root}/metrics")
    agent_a_adapter = args.agent_a_adapter or f"{adapter_dir}/agent_A_round_1"
    agent_b_adapter = args.agent_b_adapter or f"{adapter_dir}/agent_B_round_1"
    student_model_name = get_student_model_name(cfg)
    gen_config = {
        "max_new_tokens": int(cfg_get(cfg, "bootstrap", "max_new_tokens", 512)),
        "temperature": float(cfg_get(cfg, "bootstrap", "temperature", 0.7)),
        "top_p": float(cfg_get(cfg, "bootstrap", "top_p", 0.9)),
    }

    base_tokenizer, base_model = load_tokenizer_and_model(student_model_name)
    tokenizer_a, model_a = load_tokenizer_and_model(student_model_name, adapter_path=agent_a_adapter)
    tokenizer_b, model_b = load_tokenizer_and_model(student_model_name, adapter_path=agent_b_adapter)

    counts = {
        "base_single": 0,
        "agent_A_alone": 0,
        "agent_B_alone": 0,
        "A_then_B": 0,
        "B_then_A": 0,
    }
    predictions = []
    for row in tqdm(rows, desc="evaluate collaborative agents"):
        base_text = solve(base_tokenizer, base_model, messages_for_collaborative_standalone(row["problem"]), gen_config)
        a_alone_text = solve(tokenizer_a, model_a, messages_for_collaborative_standalone(row["problem"]), gen_config)
        b_alone_text = solve(tokenizer_b, model_b, messages_for_collaborative_standalone(row["problem"]), gen_config)

        a_contribution = solve(tokenizer_a, model_a, messages_for_first_contributor(row["problem"]), gen_config)
        a_then_b_text = solve(
            tokenizer_b,
            model_b,
            messages_for_second_contributor(row["problem"], a_contribution),
            gen_config,
        )
        b_contribution = solve(tokenizer_b, model_b, messages_for_first_contributor(row["problem"]), gen_config)
        b_then_a_text = solve(
            tokenizer_a,
            model_a,
            messages_for_second_contributor(row["problem"], b_contribution),
            gen_config,
        )

        base_ok, base_pred = score_prediction(base_text, row["gold_answer"])
        a_ok, a_pred = score_prediction(a_alone_text, row["gold_answer"])
        b_ok, b_pred = score_prediction(b_alone_text, row["gold_answer"])
        ab_ok, ab_pred = score_prediction(a_then_b_text, row["gold_answer"])
        ba_ok, ba_pred = score_prediction(b_then_a_text, row["gold_answer"])
        counts["base_single"] += int(base_ok)
        counts["agent_A_alone"] += int(a_ok)
        counts["agent_B_alone"] += int(b_ok)
        counts["A_then_B"] += int(ab_ok)
        counts["B_then_A"] += int(ba_ok)
        predictions.append(
            {
                "id": row.get("id"),
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "base_prediction": base_text,
                "base_final_answer": base_pred,
                "is_base_single_correct": base_ok,
                "agent_A_alone_prediction": a_alone_text,
                "agent_A_alone_final_answer": a_pred,
                "is_agent_A_alone_correct": a_ok,
                "agent_B_alone_prediction": b_alone_text,
                "agent_B_alone_final_answer": b_pred,
                "is_agent_B_alone_correct": b_ok,
                "agent_A_contribution": a_contribution,
                "agent_B_contribution": b_contribution,
                "A_then_B_prediction": a_then_b_text,
                "A_then_B_final_answer": ab_pred,
                "B_then_A_prediction": b_then_a_text,
                "B_then_A_final_answer": ba_pred,
                "is_A_then_B_correct": ab_ok,
                "is_B_then_A_correct": ba_ok,
            }
        )

    n = len(rows)
    metrics = {
        "experiment_name": cfg.get("experiment_name", EXPERIMENT_NAME),
        "teacher_model_name": get_teacher_model_name(cfg),
        "student_model_name": student_model_name,
        "agent_a_adapter": agent_a_adapter,
        "agent_b_adapter": agent_b_adapter,
        "base_single_accuracy": counts["base_single"] / n,
        "agent_A_alone_accuracy": counts["agent_A_alone"] / n,
        "agent_B_alone_accuracy": counts["agent_B_alone"] / n,
        "A_then_B_accuracy": counts["A_then_B"] / n,
        "B_then_A_accuracy": counts["B_then_A"] / n,
        "num_examples": n,
        "sampling_mode": sampling_mode,
        "seed": seed,
    }
    write_jsonl(f"{generation_dir}/eval_predictions.jsonl", predictions)
    write_json(f"{metrics_dir}/eval_metrics.json", metrics)
    clear_model(base_model, model_a, model_b)
    print(metrics)


if __name__ == "__main__":
    main()
