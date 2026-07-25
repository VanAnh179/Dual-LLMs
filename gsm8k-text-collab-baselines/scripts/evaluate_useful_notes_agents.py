#!/usr/bin/env python
"""Evaluate D11.3 useful-notes generator plus final-decider agents."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_useful_notes_traces import compact_notes, notes_have_gold_in_last_line
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
from src.prompts import format_for_generation, messages_for_final_decider, messages_for_single, messages_for_useful_notes


EXPERIMENT_NAME = "D11_3_useful_notes_decider_sft"


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
    prompt = format_for_generation(tokenizer, messages)
    return generate_text(tokenizer, model, prompt, **gen_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_3_useful_notes_decider.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--agent-a-notes-adapter", default=None)
    parser.add_argument("--agent-b-decider-adapter", default=None)
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
    record_sampled_ids("D11_3_eval_test_ids", rows)
    if not rows:
        raise SystemExit(f"No evaluation rows found in {input_path}.")

    output_root = cfg_get(cfg, "output", "root_dir", f"outputs/{EXPERIMENT_NAME}")
    adapter_dir = cfg_get(cfg, "output", "adapter_dir", f"{output_root}/adapters")
    generation_dir = cfg_get(cfg, "output", "generation_dir", f"{output_root}/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", f"{output_root}/metrics")
    agent_a_adapter = args.agent_a_notes_adapter or f"{adapter_dir}/agent_A_notes_sft"
    agent_b_adapter = args.agent_b_decider_adapter or f"{adapter_dir}/agent_B_decider_sft"
    student_model_name = get_student_model_name(cfg)

    eval_cfg = cfg.get("evaluation", {})
    common = {
        "temperature": float(eval_cfg.get("temperature", 0.7)),
        "top_p": float(eval_cfg.get("top_p", 0.9)),
    }
    base_gen = {**common, "max_new_tokens": int(eval_cfg.get("base_max_new_tokens", 256))}
    notes_gen = {**common, "max_new_tokens": int(eval_cfg.get("notes_max_new_tokens", 128))}
    decider_gen = {**common, "max_new_tokens": int(eval_cfg.get("decider_max_new_tokens", 256))}

    base_tokenizer, base_model = load_tokenizer_and_model(student_model_name)
    notes_tokenizer, notes_model = load_tokenizer_and_model(student_model_name, adapter_path=agent_a_adapter)
    decider_tokenizer, decider_model = load_tokenizer_and_model(student_model_name, adapter_path=agent_b_adapter)

    counts = {"base_single": 0, "decider_no_notes": 0, "A_then_B": 0}
    predictions = []
    notes_lengths = []
    notes_gold_last_line_count = 0

    for row in tqdm(rows, desc="evaluate D11.3 useful notes"):
        base_text = solve(base_tokenizer, base_model, messages_for_single(row["problem"]), base_gen)
        no_notes_text = solve(
            decider_tokenizer,
            decider_model,
            messages_for_final_decider(row["problem"], "No useful notes provided."),
            decider_gen,
        )
        raw_notes = solve(notes_tokenizer, notes_model, messages_for_useful_notes(row["problem"]), notes_gen)
        notes = compact_notes(raw_notes, cfg)
        a_then_b_text = solve(
            decider_tokenizer,
            decider_model,
            messages_for_final_decider(row["problem"], notes),
            decider_gen,
        )

        base_ok, base_pred = score_prediction(base_text, row["gold_answer"])
        no_notes_ok, no_notes_pred = score_prediction(no_notes_text, row["gold_answer"])
        ab_ok, ab_pred = score_prediction(a_then_b_text, row["gold_answer"])
        counts["base_single"] += int(base_ok)
        counts["decider_no_notes"] += int(no_notes_ok)
        counts["A_then_B"] += int(ab_ok)
        notes_lengths.append(len(notes))
        note_leaks = notes_have_gold_in_last_line(notes, row["gold_answer"])
        notes_gold_last_line_count += int(note_leaks)
        predictions.append(
            {
                "id": row.get("id"),
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "base_prediction": base_text,
                "base_final_answer": base_pred,
                "is_base_single_correct": base_ok,
                "agent_A_notes": notes,
                "agent_A_raw_notes": raw_notes,
                "agent_A_notes_length": len(notes),
                "agent_A_notes_gold_in_last_line": note_leaks,
                "decider_no_notes_prediction": no_notes_text,
                "decider_no_notes_final_answer": no_notes_pred,
                "is_decider_no_notes_correct": no_notes_ok,
                "A_then_B_prediction": a_then_b_text,
                "A_then_B_final_answer": ab_pred,
                "is_A_then_B_correct": ab_ok,
            }
        )

    n = len(rows)
    metrics = {
        "experiment_name": cfg.get("experiment_name", EXPERIMENT_NAME),
        "teacher_model_name": get_teacher_model_name(cfg),
        "student_model_name": student_model_name,
        "agent_a_notes_adapter": agent_a_adapter,
        "agent_b_decider_adapter": agent_b_adapter,
        "base_single_accuracy": counts["base_single"] / n,
        "decider_no_notes_accuracy": counts["decider_no_notes"] / n,
        "A_then_B_accuracy": counts["A_then_B"] / n,
        "num_examples": n,
        "sampling_mode": sampling_mode,
        "seed": seed,
        "average_agent_A_notes_length": sum(notes_lengths) / n if n else 0.0,
        "notes_gold_in_last_line_count": notes_gold_last_line_count,
        "notes_gold_in_last_line_rate": notes_gold_last_line_count / n if n else 0.0,
    }
    write_jsonl(f"{generation_dir}/eval_predictions.jsonl", predictions)
    write_json(f"{metrics_dir}/eval_metrics.json", metrics)
    clear_model(base_model, notes_model, decider_model)
    print(metrics)


if __name__ == "__main__":
    main()
