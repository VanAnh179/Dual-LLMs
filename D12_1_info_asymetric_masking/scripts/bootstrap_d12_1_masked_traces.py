#!/usr/bin/env python
"""Bootstrap masked collaboration traces using a teacher model."""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "D12_1_info_asymmetric_masking_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def parse_trace(raw: str) -> dict | None:
    """Parse teacher output into structured fields."""
    a_match = re.search(
        r"(?:^|\n)[#\s*]*(?:Agent A contribution|Agent A's? view|Agent A's? contribution)[\s\:\*\#]*(.*?)(?=(?:\n|$)[#\s*]*(?:Agent B|Joint solution|Final answer))",
        raw, re.IGNORECASE | re.DOTALL
    )
    b_match = re.search(
        r"(?:^|\n)[#\s*]*(?:Agent B contribution|Agent B's? view|Agent B's? contribution)[\s\:\*\#]*(.*?)(?=(?:\n|$)[#\s*]*(?:Joint solution|Final answer))",
        raw, re.IGNORECASE | re.DOTALL
    )
    joint_match = re.search(
        r"(?:^|\n)[#\s*]*(?:Joint solution|Joint's? solution|Joint reasoning)[\s\:\*\#]*(.*?)(?=(?:\n|$)[#\s*]*(?:Final answer|The final answer))",
        raw, re.IGNORECASE | re.DOTALL
    )

    if not (a_match and b_match and joint_match):
        return None

    contrib_a = a_match.group(1).strip()
    contrib_b = b_match.group(1).strip()
    joint_solution = joint_match.group(1).strip()

    from src.answer_extraction import extract_final_answer
    boxed_match = re.search(r"\\boxed\{([^\}]+)\}", raw)
    if boxed_match:
        final_answer = boxed_match.group(1).strip()
    else:
        final_answer = extract_final_answer(raw)

    if not (contrib_a and contrib_b and joint_solution and final_answer):
        return None

    return {
        "contrib_a": contrib_a,
        "contrib_b": contrib_b,
        "joint_solution": joint_solution,
        "final_answer": final_answer,
    }


def clear_model(*objects) -> None:
    import torch
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap masked traces for D12.1")
    parser.add_argument("--config", default="configs/d12_1_info_asymmetric.yaml")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    from src.data_utils import (
        load_config,
        get_teacher_model_name,
        read_jsonl,
        write_jsonl,
        write_json,
        sample_records,
        record_sampled_ids,
        reject_test_split_for_training,
        reject_test_rows_for_training,
        require_cuda_if_requested,
        require_dependencies,
        ensure_parent,
    )
    from src.generation import load_tokenizer_and_model, generate_text
    from src.prompts import format_for_generation
    from src.masking_utils_d12_1 import build_masked_views, validate_views
    from src.prompts_d12_1 import messages_for_masked_bootstrap
    from src.answer_extraction import answers_match

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    train_path = cfg_get(cfg, "data", "raw_train_path", "data/raw/train.jsonl")
    reject_test_split_for_training(train_path)

    all_rows = read_jsonl(train_path)
    reject_test_rows_for_training(all_rows)

    max_ex = args.max_examples or cfg_get(cfg, "sampling", "max_train_examples", 200)
    rows = sample_records(
        all_rows,
        max_ex,
        cfg_get(cfg, "sampling", "sampling_mode", "first_n"),
        int(cfg_get(cfg, "sampling", "seed", 42)),
    )
    record_sampled_ids("d12_1_bootstrap_ids", rows)

    mask_token = cfg_get(cfg, "masking", "mask_token", "[HIDDEN]")
    teacher_model = get_teacher_model_name(cfg)
    gen_config = {
        "max_new_tokens": int(cfg_get(cfg, "bootstrap", "max_new_tokens", 512)),
        "temperature": float(cfg_get(cfg, "bootstrap", "temperature", 0.0)),
        "top_p": float(cfg_get(cfg, "bootstrap", "top_p", 1.0)),
    }

    print(f"[D12.1] Bootstrapping {len(rows)} examples | teacher={teacher_model}")
    load_in_4bit = bool(cfg_get(cfg, "bootstrap", "load_in_4bit", False))
    tokenizer, model = load_tokenizer_and_model(teacher_model, load_in_4bit=load_in_4bit)

    from tqdm import tqdm

    kept, failed = [], []
    skip_mask = 0
    for row in tqdm(rows, desc="bootstrap"):
        problem = row["problem"]
        gold = row["gold_answer"]

        views = build_masked_views(problem, mask_token)
        if views is None:
            skip_mask += 1
            failed.append({**row, "fail_reason": "not_enough_numbers_to_mask"})
            continue
        view_a, view_b = views

        if not validate_views(view_a, view_b, mask_token):
            skip_mask += 1
            failed.append({**row, "fail_reason": "invalid_views"})
            continue

        messages = messages_for_masked_bootstrap(problem, view_a, view_b)
        prompt = format_for_generation(tokenizer, messages)
        raw_output = generate_text(tokenizer, model, prompt, **gen_config)

        parsed = parse_trace(raw_output)
        if parsed is None:
            failed.append({**row, "fail_reason": "parse_failed", "raw_output": raw_output})
            continue

        if not answers_match(parsed["final_answer"], gold):
            failed.append({
                **row, "fail_reason": "answer_mismatch",
                "predicted": parsed["final_answer"], "raw_output": raw_output,
            })
            continue

        kept.append({
            "id": row["id"],
            "problem": problem,
            "gold_answer": gold,
            "view_a": view_a,
            "view_b": view_b,
            "contrib_a": parsed["contrib_a"],
            "contrib_b": parsed["contrib_b"],
            "joint_solution": parsed["joint_solution"],
            "final_answer": parsed["final_answer"],
        })

    clear_model(model, tokenizer)

    # Save outputs
    trace_path = cfg_get(cfg, "data", "filtered_trace_path", "data/filtered/D12_1/masked_traces.jsonl")
    gen_dir = cfg_get(cfg, "output", "generation_dir", "outputs/D12_1_info_asymmetric_masking_sft/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", "outputs/D12_1_info_asymmetric_masking_sft/metrics")

    write_jsonl(trace_path, kept)
    write_jsonl(f"{gen_dir}/bootstrap_failed.jsonl", failed)

    stats = {
        "experiment_name": EXPERIMENT_NAME,
        "teacher_model_name": teacher_model,
        "num_input_examples": len(rows),
        "skip_mask_count": skip_mask,
        "parse_success_count": len(rows) - skip_mask - len([f for f in failed if f.get("fail_reason") == "parse_failed"]),
        "answer_match_count": len(kept),
        "kept_example_count": len(kept),
        "kept_rate": round(len(kept) / max(len(rows), 1), 4),
        "failed_count": len(failed),
    }
    write_json(f"{metrics_dir}/bootstrap_stats.json", stats)

    print(f"\n[D12.1] Bootstrap complete:")
    print(f"  Input:  {len(rows)}")
    print(f"  Skipped (masking): {skip_mask}")
    print(f"  Kept:   {len(kept)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Traces: {trace_path}")

    if not kept:
        raise SystemExit(
            f"No traces kept. Check {gen_dir}/bootstrap_failed.jsonl for details."
        )


if __name__ == "__main__":
    main()
