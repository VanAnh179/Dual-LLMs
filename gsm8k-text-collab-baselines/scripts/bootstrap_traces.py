#!/usr/bin/env python
"""Generate and filter bootstrapped GSM8K reasoning traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    load_config,
    get_teacher_model_name,
    read_jsonl,
    record_sampled_ids,
    reject_test_split_for_training,
    reject_test_rows_for_training,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
    write_json,
    write_jsonl,
)
from src.filtering import reasoning_from_candidate, trace_is_correct
from src.generation import generate_text, load_tokenizer_and_model
from src.prompts import format_for_generation, messages_for_bootstrap_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_qwen05b.yaml")
    parser.add_argument("--input", default="data/raw/train.jsonl")
    parser.add_argument("--output", default="data/filtered/bootstrap_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
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
    record_sampled_ids("bootstrap_train_ids", rows)
    teacher_model_name = get_teacher_model_name(cfg)
    tokenizer, model = load_tokenizer_and_model(teacher_model_name)
    gen_config = {
        "max_new_tokens": int(args.max_new_tokens or cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "top_p": float(cfg.get("top_p", 0.9)),
    }
    if gen_config["max_new_tokens"] < 256:
        print(
            "Warning: bootstrap max_new_tokens is below 256. Teacher outputs may be truncated "
            "before `Final answer:` and all candidates may be filtered out."
        )
    num_candidates = int(args.num_candidates or cfg.get("num_bootstrap_candidates", 4))

    kept = []
    failed = []
    failed_examples = []
    for row in tqdm(rows, desc="bootstrap"):
        prompt = format_for_generation(tokenizer, messages_for_bootstrap_teacher(row["problem"]))
        found = False
        for candidate_idx in range(num_candidates):
            text = generate_text(tokenizer, model, prompt, **gen_config)
            ok, pred = trace_is_correct(text, row["gold_answer"])
            if ok:
                kept.append(
                    {
                        "id": row.get("id"),
                        "problem": row["problem"],
                        "gold_answer": row["gold_answer"],
                        "reasoning_trace": reasoning_from_candidate(text),
                        "raw_candidate": text,
                    }
                )
                found = True
                break
            reason = "missing_final_answer" if pred is None else "wrong_answer"
            failed.append(
                {"id": row.get("id"), "candidate_idx": candidate_idx, "pred": pred, "reason": reason}
            )
            failed_examples.append(
                {
                    "id": row.get("id"),
                    "problem": row["problem"],
                    "gold_answer": row["gold_answer"],
                    "candidate_idx": candidate_idx,
                    "pred_answer": pred,
                    "failure_reason": reason,
                    "raw_candidate": text,
                }
            )
        if not found:
            failed.append({"id": row.get("id"), "candidate_idx": None, "pred": None})

    write_jsonl(args.output, kept)
    write_jsonl("outputs/generations/bootstrap_failed_candidates.jsonl", failed_examples)
    write_json(
        "outputs/metrics/bootstrap_stats.json",
        {
            "teacher_model_name": teacher_model_name,
            "input_examples": len(rows),
            "kept_examples": len(kept),
            "candidate_attempts": len(failed_examples),
            "failed_examples": len(rows) - len(kept),
        },
    )
    print(f"Kept {len(kept)} of {len(rows)} examples in {args.output}")
    if not kept:
        print(
            "No bootstrapped traces passed filtering. Inspect "
            "outputs/generations/bootstrap_failed_candidates.jsonl to see teacher outputs and extracted answers."
        )


if __name__ == "__main__":
    main()
