#!/usr/bin/env python
"""Evaluate D12.1 masked agents: partial views alone vs. combined pipeline."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "D12_1_info_asymmetric_masking_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def clear_model(*objects) -> None:
    import torch
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D12.1 masked agents")
    parser.add_argument("--config", default="configs/d12_1_info_asymmetric.yaml")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sampling-mode", default=None)
    args = parser.parse_args()

    from src.data_utils import (
        load_config,
        get_student_model_name,
        read_jsonl,
        write_jsonl,
        write_json,
        sample_records,
        record_sampled_ids,
        reject_train_split_for_final_eval,
        reject_train_rows_for_final_eval,
        require_cuda_if_requested,
        require_dependencies,
    )
    from src.generation import load_tokenizer_and_model, generate_text
    from src.prompts import format_for_generation
    from src.masking_utils_d12_1 import build_masked_views
    from src.prompts_d12_1 import (
        messages_for_masked_partial,
        messages_for_masked_synthesizer,
    )
    from src.answer_extraction import extract_final_answer, answers_match

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    test_path = cfg_get(cfg, "data", "raw_test_path", "data/raw/test.jsonl")
    reject_train_split_for_final_eval(test_path)

    all_rows = read_jsonl(test_path)
    reject_train_rows_for_final_eval(all_rows)

    max_ex = args.max_examples or cfg_get(cfg, "sampling", "max_eval_examples", 100)
    sampling_mode = args.sampling_mode or cfg_get(cfg, "sampling", "sampling_mode", "first_n")
    seed = int(cfg_get(cfg, "sampling", "seed", 42))
    rows = sample_records(all_rows, max_ex, sampling_mode, seed)
    record_sampled_ids("d12_1_eval_ids", rows)

    student_model = get_student_model_name(cfg)
    adapter_dir = cfg_get(cfg, "output", "adapter_dir",
                          "outputs/D12_1_info_asymmetric_masking_sft/adapters")
    mask_token = cfg_get(cfg, "masking", "mask_token", "[HIDDEN]")

    eval_cfg = cfg.get("evaluation", {})
    gen_config = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 512)),
        "temperature": float(eval_cfg.get("temperature", 0.0)),
        "top_p": float(eval_cfg.get("top_p", 1.0)),
    }

    adapter_a = f"{adapter_dir}/agent_A_partial_view_sft"
    adapter_b = f"{adapter_dir}/agent_B_partial_view_sft"
    adapter_synth = f"{adapter_dir}/final_synthesizer_sft"

    print(f"[D12.1] Evaluating {len(rows)} examples | model={student_model}")
    print(f"[D12.1] Adapter A: {adapter_a}")
    print(f"[D12.1] Adapter B: {adapter_b}")
    print(f"[D12.1] Adapter Synth: {adapter_synth}")

    from tqdm import tqdm

    predictions = []

    # ----------------------------------------------------------------
    # MODE 1: Agent A partial only
    # ----------------------------------------------------------------
    print("\n[D12.1] Mode 1: Agent A partial only")
    tokenizer_a, model_a = load_tokenizer_and_model(student_model, adapter_path=adapter_a)

    a_correct = 0
    a_results = {}
    for row in tqdm(rows, desc="A partial"):
        views = build_masked_views(row["problem"], mask_token)
        if views is None:
            a_results[row["id"]] = {"output": "", "correct": False, "pred": None}
            continue
        view_a, _ = views
        msgs = messages_for_masked_partial(view_a)
        prompt = format_for_generation(tokenizer_a, msgs)
        output = generate_text(tokenizer_a, model_a, prompt, **gen_config)
        pred = extract_final_answer(output)
        correct = pred is not None and answers_match(pred, row["gold_answer"])
        if correct:
            a_correct += 1
        a_results[row["id"]] = {"output": output, "correct": correct, "pred": pred}

    clear_model(model_a, tokenizer_a)
    a_acc = round(a_correct / max(len(rows), 1), 4)
    print(f"  Agent A partial accuracy: {a_acc} ({a_correct}/{len(rows)})")

    # ----------------------------------------------------------------
    # MODE 2: Agent B partial only
    # ----------------------------------------------------------------
    print("\n[D12.1] Mode 2: Agent B partial only")
    tokenizer_b, model_b = load_tokenizer_and_model(student_model, adapter_path=adapter_b)

    b_correct = 0
    b_results = {}
    for row in tqdm(rows, desc="B partial"):
        views = build_masked_views(row["problem"], mask_token)
        if views is None:
            b_results[row["id"]] = {"output": "", "correct": False, "pred": None}
            continue
        _, view_b = views
        msgs = messages_for_masked_partial(view_b)
        prompt = format_for_generation(tokenizer_b, msgs)
        output = generate_text(tokenizer_b, model_b, prompt, **gen_config)
        pred = extract_final_answer(output)
        correct = pred is not None and answers_match(pred, row["gold_answer"])
        if correct:
            b_correct += 1
        b_results[row["id"]] = {"output": output, "correct": correct, "pred": pred}

    clear_model(model_b, tokenizer_b)
    b_acc = round(b_correct / max(len(rows), 1), 4)
    print(f"  Agent B partial accuracy: {b_acc} ({b_correct}/{len(rows)})")

    # ----------------------------------------------------------------
    # MODE 3: A + B -> Synthesizer (full pipeline)
    # ----------------------------------------------------------------
    print("\n[D12.1] Mode 3: A + B -> Synthesizer")

    # Reload A to generate contributions
    tokenizer_a2, model_a2 = load_tokenizer_and_model(student_model, adapter_path=adapter_a)
    contrib_a_map = {}
    for row in tqdm(rows, desc="A contrib"):
        views = build_masked_views(row["problem"], mask_token)
        if views is None:
            contrib_a_map[row["id"]] = ""
            continue
        view_a, _ = views
        msgs = messages_for_masked_partial(view_a)
        prompt = format_for_generation(tokenizer_a2, msgs)
        contrib_a_map[row["id"]] = generate_text(tokenizer_a2, model_a2, prompt, **gen_config)
    clear_model(model_a2, tokenizer_a2)

    # Reload B to generate contributions
    tokenizer_b2, model_b2 = load_tokenizer_and_model(student_model, adapter_path=adapter_b)
    contrib_b_map = {}
    for row in tqdm(rows, desc="B contrib"):
        views = build_masked_views(row["problem"], mask_token)
        if views is None:
            contrib_b_map[row["id"]] = ""
            continue
        _, view_b = views
        msgs = messages_for_masked_partial(view_b)
        prompt = format_for_generation(tokenizer_b2, msgs)
        contrib_b_map[row["id"]] = generate_text(tokenizer_b2, model_b2, prompt, **gen_config)
    clear_model(model_b2, tokenizer_b2)

    # Load synthesizer
    tokenizer_s, model_s = load_tokenizer_and_model(student_model, adapter_path=adapter_synth)
    combined_correct = 0
    collab_essential = 0
    for row in tqdm(rows, desc="Synthesize"):
        rid = row["id"]
        ca = contrib_a_map.get(rid, "")
        cb = contrib_b_map.get(rid, "")
        msgs = messages_for_masked_synthesizer(row["problem"], ca, cb)
        prompt = format_for_generation(tokenizer_s, msgs)
        output = generate_text(tokenizer_s, model_s, prompt, **gen_config)
        pred = extract_final_answer(output)
        correct = pred is not None and answers_match(pred, row["gold_answer"])
        if correct:
            combined_correct += 1

        a_was_correct = a_results.get(rid, {}).get("correct", False)
        b_was_correct = b_results.get(rid, {}).get("correct", False)
        both_wrong_combined_right = (not a_was_correct and not b_was_correct and correct)
        if both_wrong_combined_right:
            collab_essential += 1

        predictions.append({
            "id": rid,
            "problem": row["problem"],
            "gold_answer": row["gold_answer"],
            "a_partial_output": a_results.get(rid, {}).get("output", ""),
            "a_partial_correct": a_was_correct,
            "b_partial_output": b_results.get(rid, {}).get("output", ""),
            "b_partial_correct": b_was_correct,
            "contrib_a": ca,
            "contrib_b": cb,
            "combined_output": output,
            "combined_pred": pred,
            "combined_correct": correct,
            "both_wrong_combined_right": both_wrong_combined_right,
        })

    clear_model(model_s, tokenizer_s)
    combined_acc = round(combined_correct / max(len(rows), 1), 4)
    delta = round(combined_acc - max(a_acc, b_acc), 4)

    print(f"\n{'='*60}")
    print(f"[D12.1] RESULTS on {len(rows)} examples:")
    print(f"  Agent A partial accuracy:    {a_acc}")
    print(f"  Agent B partial accuracy:    {b_acc}")
    print(f"  A+B->Synth accuracy:          {combined_acc}")
    print(f"  Delta (combined - best partial): {delta}")
    print(f"  Collaboration essential:     {collab_essential} cases")
    print(f"{'='*60}")

    # Save outputs
    gen_dir = cfg_get(cfg, "output", "generation_dir",
                      "outputs/D12_1_info_asymmetric_masking_sft/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir",
                          "outputs/D12_1_info_asymmetric_masking_sft/metrics")

    write_jsonl(f"{gen_dir}/eval_predictions.jsonl", predictions)

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "student_model_name": student_model,
        "num_examples": len(rows),
        "sampling_mode": sampling_mode,
        "seed": seed,
        "agent_A_partial_accuracy": a_acc,
        "agent_B_partial_accuracy": b_acc,
        "A_B_then_final_accuracy": combined_acc,
        "delta_vs_best_partial": delta,
        "collaboration_essential_count": collab_essential,
        "collaboration_essential_rate": round(collab_essential / max(len(rows), 1), 4),
    }
    write_json(f"{metrics_dir}/eval_metrics.json", metrics)
    print(f"\nMetrics -> {metrics_dir}/eval_metrics.json")
    print(f"Predictions -> {gen_dir}/eval_predictions.jsonl")


if __name__ == "__main__":
    main()
