#!/usr/bin/env python
"""Alternating LoRA SFT for two neutral GSM8K reasoning agents."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    load_config,
    get_student_model_name,
    project_path,
    read_jsonl,
    record_sampled_ids,
    reject_test_split_for_training,
    reject_test_rows_for_training,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
    write_jsonl,
)
from src.generation import generate_text, load_tokenizer_and_model
from src.prompts import format_for_generation, messages_for_single
from src.training import train_lora_sft


def generate_training_rows(tokenizer, model, rows: list[dict], cfg: dict, output_path: str) -> list[dict]:
    from tqdm import tqdm

    gen_config = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "top_p": float(cfg.get("top_p", 0.9)),
    }
    out = []
    for row in tqdm(rows, desc="generate frozen responses"):
        prompt = format_for_generation(tokenizer, messages_for_single(row["problem"]))
        other = generate_text(tokenizer, model, prompt, **gen_config)
        out.append(
            {
                "id": row.get("id"),
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "other_agent_response": other,
                "reasoning_trace": row.get("reasoning_trace", ""),
            }
        )
    write_jsonl(output_path, out)
    return out


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
    parser.add_argument("--train-data", default="data/filtered/bootstrap_train.jsonl")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "datasets", "peft", "trl", "yaml", "tqdm")
    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))
    reject_test_split_for_training(args.train_data)

    all_rows = read_jsonl(args.train_data)
    reject_test_rows_for_training(all_rows)
    rows = sample_records(
        all_rows,
        args.max_train_examples or cfg.get("max_train_examples"),
        cfg.get("sampling_mode", "first_n"),
        int(cfg.get("seed", 42)),
    )
    record_sampled_ids("alternating_train_ids", rows)
    if not rows:
        raise SystemExit(
            f"No training rows found in {args.train_data}. Run bootstrap_traces.py first or provide JSONL with reasoning_trace."
        )

    output_dir = cfg.get("output_dir", "outputs")
    adapter_root = project_path(f"{output_dir}/adapters")
    generation_root = project_path(f"{output_dir}/generations")
    rounds = int(args.num_rounds or cfg.get("num_alternating_rounds", 1))
    student_model_name = get_student_model_name(cfg)
    print(f"Training student agents with: {student_model_name}")
    agent_a_adapter = None
    agent_b_adapter = None

    for round_idx in range(1, rounds + 1):
        tokenizer_a, model_a = load_tokenizer_and_model(student_model_name, adapter_path=agent_a_adapter)
        b_rows = generate_training_rows(
            tokenizer_a,
            model_a,
            rows,
            cfg,
            f"{output_dir}/generations/round_{round_idx}_agent_A_responses.jsonl",
        )
        clear_model(model_a)

        b_out = adapter_root / f"agent_B_round_{round_idx}"
        tokenizer_b, model_b = load_tokenizer_and_model(
            student_model_name,
            adapter_path=agent_b_adapter,
            trainable_lora=True,
            lora_config=cfg.get("lora", {}),
        )
        train_lora_sft(model_b, tokenizer_b, b_rows, b_out, cfg.get("training", {}), two_agent=True)
        agent_b_adapter = str(b_out)
        clear_model(model_b)

        tokenizer_b, model_b = load_tokenizer_and_model(student_model_name, adapter_path=agent_b_adapter)
        a_rows = generate_training_rows(
            tokenizer_b,
            model_b,
            rows,
            cfg,
            f"{output_dir}/generations/round_{round_idx}_agent_B_responses.jsonl",
        )
        clear_model(model_b)

        a_out = adapter_root / f"agent_A_round_{round_idx}"
        tokenizer_a, model_a = load_tokenizer_and_model(
            student_model_name,
            adapter_path=agent_a_adapter,
            trainable_lora=True,
            lora_config=cfg.get("lora", {}),
        )
        train_lora_sft(model_a, tokenizer_a, a_rows, a_out, cfg.get("training", {}), two_agent=True)
        agent_a_adapter = str(a_out)
        clear_model(model_a)

    print(f"Saved Agent A adapter: {agent_a_adapter}")
    print(f"Saved Agent B adapter: {agent_b_adapter}")


if __name__ == "__main__":
    main()
