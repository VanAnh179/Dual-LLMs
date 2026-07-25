#!/usr/bin/env python
"""Generate D11.2 latent collaborative traces with a teacher model."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answer_extraction import answers_match, normalize_answer
from src.data_utils import (
    get_teacher_model_name,
    load_config,
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
from src.generation import generate_text, load_tokenizer_and_model
from src.prompts import format_for_generation, messages_for_collaborative_bootstrap


EXPERIMENT_NAME = "D11_2_latent_collaborative_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def parse_collaborative_trace(text: str) -> dict | None:
    pattern = re.compile(
        r"Agent 1 contribution:\s*(?P<agent1>.*?)\s*"
        r"Agent 2 contribution:\s*(?P<agent2>.*?)\s*"
        r"Joint solution:\s*(?P<joint>.*?)\s*"
        r"Final answer:\s*(?P<final>.*)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text.strip())
    if not match:
        return None
    parsed = {key: value.strip() for key, value in match.groupdict().items()}
    if not all(parsed.values()):
        return None
    final_text = parsed.pop("final").strip()
    numbers = re.findall(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", final_text)
    parsed["final_answer"] = normalize_answer(numbers[-1]) if numbers else final_text.splitlines()[0].strip()
    parsed["agent1_contribution"] = parsed.pop("agent1")
    parsed["agent2_contribution"] = parsed.pop("agent2")
    parsed["joint_solution"] = parsed.pop("joint")
    return parsed


def avg_len(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(len(row.get(key, "")) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_2_qwen_math7b_teacher.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    input_path = args.input or cfg_get(cfg, "data", "raw_train_path", "data/raw/train.jsonl")
    output_path = args.output or cfg_get(
        cfg,
        "data",
        "filtered_trace_path",
        f"data/filtered/{EXPERIMENT_NAME}/two_agent_traces.jsonl",
    )
    generation_dir = cfg_get(cfg, "output", "generation_dir", f"outputs/{EXPERIMENT_NAME}/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", f"outputs/{EXPERIMENT_NAME}/metrics")

    reject_test_split_for_training(input_path)
    all_rows = read_jsonl(input_path)
    reject_test_rows_for_training(all_rows)
    rows = sample_records(
        all_rows,
        args.max_examples or cfg_get(cfg, "sampling", "max_train_examples", 100),
        cfg_get(cfg, "sampling", "sampling_mode", "first_n"),
        int(cfg_get(cfg, "sampling", "seed", 42)),
    )
    record_sampled_ids("D11_2_bootstrap_train_ids", rows)
    if not rows:
        raise SystemExit(f"No training rows found in {input_path}.")

    teacher_model_name = get_teacher_model_name(cfg)
    tokenizer, model = load_tokenizer_and_model(teacher_model_name)
    gen_config = {
        "max_new_tokens": int(args.max_new_tokens or cfg_get(cfg, "bootstrap", "max_new_tokens", 512)),
        "temperature": float(cfg_get(cfg, "bootstrap", "temperature", 0.7)),
        "top_p": float(cfg_get(cfg, "bootstrap", "top_p", 0.9)),
    }
    num_candidates = int(args.num_candidates or cfg_get(cfg, "bootstrap", "num_candidates", 4))

    kept = []
    failed = []
    parse_success_count = 0
    answer_match_count = 0
    for row in tqdm(rows, desc="bootstrap collaborative traces"):
        prompt = format_for_generation(tokenizer, messages_for_collaborative_bootstrap(row["problem"]))
        found = False
        for candidate_idx in range(num_candidates):
            raw = generate_text(tokenizer, model, prompt, **gen_config)
            parsed = parse_collaborative_trace(raw)
            if parsed is None:
                failed.append(
                    {
                        "id": row.get("id"),
                        "candidate_idx": candidate_idx,
                        "failure_reason": "parse_failed",
                        "raw_candidate": raw,
                    }
                )
                continue
            parse_success_count += 1
            if answers_match(parsed["final_answer"], row["gold_answer"]):
                answer_match_count += 1
                kept.append(
                    {
                        "id": row.get("id"),
                        "problem": row["problem"],
                        "gold_answer": row["gold_answer"],
                        **parsed,
                    }
                )
                found = True
                break
            failed.append(
                {
                    "id": row.get("id"),
                    "candidate_idx": candidate_idx,
                    "failure_reason": "answer_mismatch",
                    "pred_answer": parsed["final_answer"],
                    "gold_answer": row["gold_answer"],
                    "raw_candidate": raw,
                }
            )
        if not found:
            failed.append({"id": row.get("id"), "candidate_idx": None, "failure_reason": "no_kept_candidate"})

    total_candidates = len(rows) * num_candidates
    write_jsonl(output_path, kept)
    write_jsonl(f"{generation_dir}/bootstrap_failed.jsonl", failed)
    write_json(
        f"{metrics_dir}/bootstrap_stats.json",
        {
            "teacher_model_name": teacher_model_name,
            "num_input_examples": len(rows),
            "num_candidates_per_example": num_candidates,
            "num_total_candidates": total_candidates,
            "parse_success_count": parse_success_count,
            "parse_success_rate": parse_success_count / total_candidates if total_candidates else 0.0,
            "answer_match_count": answer_match_count,
            "answer_match_rate": answer_match_count / parse_success_count if parse_success_count else 0.0,
            "kept_example_count": len(kept),
            "average_agent1_contribution_length": avg_len(kept, "agent1_contribution"),
            "average_agent2_contribution_length": avg_len(kept, "agent2_contribution"),
            "average_joint_solution_length": avg_len(kept, "joint_solution"),
        },
    )
    if not kept:
        raise SystemExit(
            "Parsed collaborative traces are empty after filtering. "
            f"Inspect {generation_dir}/bootstrap_failed.jsonl."
        )
    print(f"Kept {len(kept)} collaborative traces in {output_path}")


if __name__ == "__main__":
    main()
