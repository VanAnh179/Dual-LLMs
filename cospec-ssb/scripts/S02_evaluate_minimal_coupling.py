#!/usr/bin/env python
"""Evaluate minimal coupling pipeline: A alone, B alone, and full A->bridge->B coupling."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "S02_minimal_coupling_gsm8k"

from src.data_utils import project_path


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def clear_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_alone(model_name, adapter_path, tokenizer, rows, gen_config, label, device, dtype):
    """Evaluate a single agent with adapter, solving problems independently."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from src.evaluation import solve_alone, score_prediction
    from src.answer_extraction import extract_gsm8k_gold_answer

    print(f"\n--- Evaluating {label} ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    results = []
    correct = 0
    for i, row in enumerate(rows):
        gold = extract_gsm8k_gold_answer(row["raw_answer"])
        prediction_text = solve_alone(tokenizer, model, row["problem"], gen_config)
        is_correct, pred_answer = score_prediction(prediction_text, gold)
        correct += int(is_correct)
        results.append({
            "id": row["id"],
            "problem": row["problem"],
            "gold_answer": gold,
            "mode": label,
            "prediction": prediction_text,
            "predicted_answer": pred_answer,
            "correct": is_correct,
        })
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i+1}/{len(rows)}, running acc={correct/(i+1):.2f}")

    accuracy = correct / len(rows) if rows else 0
    print(f"  {label} accuracy: {accuracy:.4f} ({correct}/{len(rows)})")

    clear_gpu(model)
    return results, accuracy


def evaluate_coupling(model_name, adapter_a_path, adapter_b_trained_path, bridge_path,
                      tokenizer, rows, gen_config, cfg, device, dtype):
    """Evaluate the full minimal coupling pipeline: A->bridge->B."""
    import torch
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge
    from src.evaluation import score_prediction
    from src.answer_extraction import extract_gsm8k_gold_answer
    from src.prompts import format_for_generation, messages_for_single
    from src.generation import generate_text

    mc = cfg["minimal_coupling"]
    layer_index = mc["layer_index"]

    print(f"\n--- Evaluating minimal_coupling ---")

    # Load Agent A (frozen)
    model_a = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model_a = PeftModel.from_pretrained(model_a, adapter_a_path, is_trainable=False)
    model_a.eval()
    model_a.requires_grad_(False)

    # Load Agent B (trained S02 LoRA on merged base)
    # Reconstruct the same way as training: merge D11.0 then load S02 LoRA
    init_from_d11 = mc.get("init_agent_b_from_d11", True)
    adapter_b_d11_path = str(project_path(mc["agent_b_adapter_path"]))

    model_b = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    if init_from_d11:
        model_b = PeftModel.from_pretrained(model_b, adapter_b_d11_path, is_trainable=True)
        model_b = model_b.merge_and_unload()

    # Load the S02-trained LoRA on top
    model_b = PeftModel.from_pretrained(model_b, adapter_b_trained_path, is_trainable=False)
    model_b.eval()

    # Load bridge
    bridge = MinimalCouplingBridge.load_bridge(bridge_path, device=device)
    bridge.to(device).to(dtype)
    bridge.eval()

    # Hooks
    layer_a = get_layer_by_index(model_a, layer_index)
    extractor = HiddenStateExtractor(layer_a)

    results = []
    correct = 0
    for i, row in enumerate(rows):
        gold = extract_gsm8k_gold_answer(row["raw_answer"])
        prompt = format_for_generation(tokenizer, messages_for_single(row["problem"]))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Forward A to get hidden states
        extractor.clear()
        with torch.no_grad():
            model_a(**inputs)
        h_a = extractor.hidden_states.detach()
        z = bridge.encode(h_a)

        # Inject into B
        layer_b = get_layer_by_index(model_b, layer_index)

        def make_injection_fn(z_vec):
            def injection_fn(h_b):
                return bridge.inject(h_b, z_vec)
            return injection_fn

        injector = HiddenStateInjector(layer_b, make_injection_fn(z))

        # Generate from B
        with torch.no_grad():
            prediction_text = generate_text(tokenizer, model_b, prompt, **gen_config)

        injector.remove()

        is_correct, pred_answer = score_prediction(prediction_text, gold)
        correct += int(is_correct)
        results.append({
            "id": row["id"],
            "problem": row["problem"],
            "gold_answer": gold,
            "mode": "minimal_coupling",
            "prediction": prediction_text,
            "predicted_answer": pred_answer,
            "correct": is_correct,
        })
        if (i + 1) % 10 == 0:
            print(f"  coupling: {i+1}/{len(rows)}, running acc={correct/(i+1):.2f}")

    extractor.remove()
    accuracy = correct / len(rows) if rows else 0
    print(f"  minimal_coupling accuracy: {accuracy:.4f} ({correct}/{len(rows)})")

    clear_gpu(model_a, model_b, bridge)
    return results, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S02 minimal coupling pipeline")
    parser.add_argument("--config", type=str, default="configs/s02_minimal_coupling_gsm8k.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Override max eval examples")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from src.data_utils import (
        load_config, project_path, read_jsonl, sample_records,
        reject_train_split_for_final_eval, reject_train_rows_for_final_eval,
        record_sampled_ids, write_jsonl, write_json,
    )

    cfg = load_config(args.config)
    model_name = cfg.get("student_model_name")
    mc = cfg["minimal_coupling"]
    sampling_cfg = cfg.get("sampling", {})
    eval_cfg = cfg.get("evaluation", {})

    # Data guards
    test_path = cfg_get(cfg, "data", "raw_test_path")
    reject_train_split_for_final_eval(test_path)
    rows = read_jsonl(test_path)
    reject_train_rows_for_final_eval(rows)

    max_examples = args.max_examples or sampling_cfg.get("max_eval_examples", 100)
    rows = sample_records(rows, max_examples=max_examples,
                          sampling_mode=sampling_cfg.get("sampling_mode", "first_n"),
                          seed=sampling_cfg.get("seed", 42))
    record_sampled_ids("S02_eval", rows, path="outputs/S02_minimal_coupling_gsm8k/metrics/sampled_ids.json")
    print(f"Evaluating on {len(rows)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 512)),
        "temperature": float(eval_cfg.get("temperature", 0.0)),
        "top_p": float(eval_cfg.get("top_p", 1.0)),
    }

    adapter_a_path = str(project_path(mc["agent_a_adapter_path"]))
    adapter_b_path = str(project_path(mc["agent_b_adapter_path"]))

    # Paths to S02-trained artifacts
    adapter_dir = str(project_path(cfg_get(cfg, "output", "adapter_dir")))
    b_trained_path = str(Path(adapter_dir) / "agent_B_minimal_coupling_sft")
    bridge_path = str(Path(adapter_dir) / "minimal_coupling_bridge.pt")

    # --- Mode 1: Agent A alone ---
    results_a, acc_a = evaluate_alone(
        model_name, adapter_a_path, tokenizer, rows, gen_config,
        "agent_A_alone", device, dtype
    )

    # --- Mode 2: Agent B alone (D11.0 original, NOT S02-trained) ---
    results_b, acc_b = evaluate_alone(
        model_name, adapter_b_path, tokenizer, rows, gen_config,
        "agent_B_alone", device, dtype
    )

    # --- Mode 3: Minimal coupling pipeline ---
    results_c, acc_c = evaluate_coupling(
        model_name, adapter_a_path, b_trained_path, bridge_path,
        tokenizer, rows, gen_config, cfg, device, dtype
    )

    # --- Save predictions ---
    all_predictions = results_a + results_b + results_c
    gen_dir = str(project_path(cfg_get(cfg, "output", "generation_dir")))
    pred_path = str(Path(gen_dir) / "eval_predictions.jsonl")
    write_jsonl(pred_path, all_predictions)
    print(f"\nSaved predictions to {pred_path}")

    # --- Save metrics ---
    max_alone = max(acc_a, acc_b)
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "student_model_name": model_name,
        "num_examples": len(rows),
        "agent_A_alone_accuracy": round(acc_a, 4),
        "agent_B_alone_accuracy": round(acc_b, 4),
        "minimal_coupling_accuracy": round(acc_c, 4),
        "delta_vs_max_alone": round(acc_c - max_alone, 4),
        "delta_vs_d11_0_text_pipeline": round(acc_c - 0.68, 4),
    }
    metrics_dir = str(project_path(cfg_get(cfg, "output", "metrics_dir")))
    metrics_path = str(Path(metrics_dir) / "eval_metrics.json")
    write_json(metrics_path, metrics)
    print(f"Saved metrics to {metrics_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("S02 Evaluation Summary")
    print("=" * 60)
    print(f"  agent_A_alone_accuracy:       {acc_a:.4f}")
    print(f"  agent_B_alone_accuracy:       {acc_b:.4f}")
    print(f"  minimal_coupling_accuracy:    {acc_c:.4f}")
    print(f"  delta_vs_max_alone:           {acc_c - max_alone:+.4f}")
    print(f"  delta_vs_d11_0_text_pipeline: {acc_c - 0.68:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
