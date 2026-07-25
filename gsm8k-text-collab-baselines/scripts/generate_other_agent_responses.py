#!/usr/bin/env python
"""Generate neutral other-agent responses for two-agent SFT records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    load_config,
    get_student_model_name,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_qwen05b.yaml")
    parser.add_argument("--input", default="data/filtered/bootstrap_train.jsonl")
    parser.add_argument("--output", default="data/train/two_agent_train.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))
    reject_test_split_for_training(args.input)

    all_rows = read_jsonl(args.input)
    reject_test_rows_for_training(all_rows)
    rows = sample_records(
        all_rows,
        args.max_examples or cfg.get("max_train_examples"),
        cfg.get("sampling_mode", "first_n"),
        int(cfg.get("seed", 42)),
    )
    record_sampled_ids("other_agent_train_ids", rows)
    student_model_name = get_student_model_name(cfg)
    tokenizer, model = load_tokenizer_and_model(student_model_name, adapter_path=args.adapter)
    gen_config = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "top_p": float(cfg.get("top_p", 0.9)),
    }

    out = []
    for row in tqdm(rows, desc="other-agent"):
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
    write_jsonl(args.output, out)
    print(f"Wrote {len(out)} records to {args.output}")


if __name__ == "__main__":
    main()
