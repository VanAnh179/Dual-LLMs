#!/usr/bin/env python
"""Evaluate majority-vote and oracle-vote baselines using D11.0 LoRA agents."""
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
from src.answer_extraction import normalize_answer
from src.evaluation import score_prediction, solve_alone
from src.generation import load_tokenizer_and_model


def clear_model(*objects) -> None:
    """Delete model objects and free GPU memory."""
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D12_0 — Majority Voting Baseline evaluation",
    )
    parser.add_argument(
        "--config",
        default="configs/d12_0_voting_baseline.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override max_eval_examples from config.",
    )
    parser.add_argument(
        "--sampling-mode",
        default=None,
        help="Override sampling_mode from config (first_n | random).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    student_model_name = get_student_model_name(cfg)

    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    adapter_cfg = cfg.get("adapters", {})
    sampling_cfg = cfg.get("sampling", {})
    eval_cfg = cfg.get("evaluation", {})

    test_path = data_cfg.get("raw_test_path", "data/raw/test.jsonl")
    generation_dir = output_cfg.get("generation_dir", "outputs/D12_0_voting/generations")
    metrics_dir = output_cfg.get("metrics_dir", "outputs/D12_0_voting/metrics")

    adapter_a_path = adapter_cfg.get("agent_a")
    adapter_b_path = adapter_cfg.get("agent_b")
    if not adapter_a_path or not adapter_b_path:
        raise SystemExit(
            "Config must define adapters.agent_a and adapters.agent_b "
            "pointing to D11.0 LoRA adapter directories."
        )

    seed = int(sampling_cfg.get("seed", 42))
    sampling_mode = args.sampling_mode or sampling_cfg.get("sampling_mode", "first_n")
    max_examples = args.max_examples or sampling_cfg.get("max_eval_examples")

    gen_config = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 512)),
        "temperature": float(eval_cfg.get("temperature", 0.0)),
        "top_p": float(eval_cfg.get("top_p", 1.0)),
    }

    # ── Load and validate test data ──────────────────────────────────
    reject_train_split_for_final_eval(test_path)
    all_rows = read_jsonl(test_path)
    reject_train_rows_for_final_eval(all_rows)

    rows = sample_records(all_rows, max_examples, sampling_mode, seed)
    record_sampled_ids("D12_0_voting_eval_ids", rows)

    if not rows:
        raise SystemExit(f"No evaluation rows found in {test_path}. Run prepare_gsm8k.py first.")

    print(f"[D12_0] Evaluating {len(rows)} examples  |  model={student_model_name}")
    print(f"[D12_0] Agent A adapter: {adapter_a_path}")
    print(f"[D12_0] Agent B adapter: {adapter_b_path}")

    # ── Phase 1: Agent A solves all problems independently ───────────
    print("\n[D12_0] Loading Agent A …")
    tok_a, model_a = load_tokenizer_and_model(student_model_name, adapter_path=adapter_a_path)

    agent_a_results: list[dict] = []
    for row in tqdm(rows, desc="Agent A solve"):
        raw_text = solve_alone(tok_a, model_a, row["problem"], gen_config)
        correct, pred_answer = score_prediction(raw_text, row["gold_answer"])
        agent_a_results.append({
            "raw_text": raw_text,
            "pred_answer": pred_answer,
            "correct": correct,
        })

    clear_model(tok_a, model_a)
    print("[D12_0] Agent A done — GPU memory cleared.")

    # ── Phase 2: Agent B solves all problems independently ───────────
    print("\n[D12_0] Loading Agent B …")
    tok_b, model_b = load_tokenizer_and_model(student_model_name, adapter_path=adapter_b_path)

    agent_b_results: list[dict] = []
    for row in tqdm(rows, desc="Agent B solve"):
        raw_text = solve_alone(tok_b, model_b, row["problem"], gen_config)
        correct, pred_answer = score_prediction(raw_text, row["gold_answer"])
        agent_b_results.append({
            "raw_text": raw_text,
            "pred_answer": pred_answer,
            "correct": correct,
        })

    clear_model(tok_b, model_b)
    print("[D12_0] Agent B done — GPU memory cleared.")

    # ── Phase 3: Compute Majority Vote and Oracle Vote ───────────────
    predictions: list[dict] = []
    counts = {
        "agent_a_correct": 0,
        "agent_b_correct": 0,
        "vote_correct": 0,
        "oracle_correct": 0,
        "agreements": 0,
        "disagreements": 0,
    }

    for i, row in enumerate(rows):
        a_res = agent_a_results[i]
        b_res = agent_b_results[i]

        ans_a = normalize_answer(a_res["pred_answer"]) if a_res["pred_answer"] is not None else None
        ans_b = normalize_answer(b_res["pred_answer"]) if b_res["pred_answer"] is not None else None
        gold = normalize_answer(row["gold_answer"])

        # Majority vote: agree → take the answer; disagree → None
        if ans_a is not None and ans_b is not None and ans_a == ans_b:
            vote_answer = ans_a
            disagreement = False
        else:
            vote_answer = None
            disagreement = True

        vote_correct = vote_answer is not None and vote_answer == gold
        oracle_correct = bool(a_res["correct"] or b_res["correct"])

        counts["agent_a_correct"] += int(a_res["correct"])
        counts["agent_b_correct"] += int(b_res["correct"])
        counts["vote_correct"] += int(vote_correct)
        counts["oracle_correct"] += int(oracle_correct)
        counts["agreements"] += int(not disagreement)
        counts["disagreements"] += int(disagreement)

        predictions.append({
            "id": row.get("id"),
            "problem": row["problem"],
            "gold_answer": row["gold_answer"],
            "agent_a_answer": a_res["pred_answer"],
            "agent_a_correct": a_res["correct"],
            "agent_b_answer": b_res["pred_answer"],
            "agent_b_correct": b_res["correct"],
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "oracle_correct": oracle_correct,
            "disagreement": disagreement,
        })

    n = len(rows)
    metrics = {
        "experiment_name": cfg.get("experiment_name", "D12_0_voting_baseline"),
        "student_model_name": student_model_name,
        "agent_a_adapter": adapter_a_path,
        "agent_b_adapter": adapter_b_path,
        "num_examples": n,
        "agent_A_alone_accuracy": round(counts["agent_a_correct"] / n, 4),
        "agent_B_alone_accuracy": round(counts["agent_b_correct"] / n, 4),
        "majority_vote_accuracy": round(counts["vote_correct"] / n, 4),
        "oracle_vote_accuracy": round(counts["oracle_correct"] / n, 4),
        "agreement_rate": round(counts["agreements"] / n, 4),
        "disagreement_rate": round(counts["disagreements"] / n, 4),
    }

    # ── Save outputs ─────────────────────────────────────────────────
    pred_path = f"{generation_dir}/voting_predictions.jsonl"
    metrics_path = f"{metrics_dir}/d12_0_metrics.json"

    write_jsonl(pred_path, predictions)
    write_json(metrics_path, metrics)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D12_0 Majority Voting Baseline — Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print(f"\nPredictions → {pred_path}")
    print(f"Metrics     → {metrics_path}")


if __name__ == "__main__":
    main()
