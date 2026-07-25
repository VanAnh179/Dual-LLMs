#!/usr/bin/env python
"""Train the S03 receiver control with an exactly zero latent message."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s03_zero_control_gsm8k.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.answer_extraction import extract_gsm8k_gold_answer
    from src.data_utils import (
        ensure_parent, project_path, read_jsonl, record_sampled_ids,
        reject_test_rows_for_training, reject_test_split_for_training, sample_records, write_json,
    )
    from src.prompts import format_messages_with_assistant, messages_for_single
    from src.S01_hook_utils import HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge
    from src.S03_runtime import choose_training_device_dtype, load_s03_and_s02, require_s02_artifacts

    cfg, s02 = load_s03_and_s02(args.config)
    paths = require_s02_artifacts(cfg, s02)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device, dtype = choose_training_device_dtype()

    train_path = s02["data"]["raw_train_path"]
    reject_test_split_for_training(train_path)
    rows = read_jsonl(train_path)
    reject_test_rows_for_training(rows)
    requested_examples = args.max_examples or int(cfg.get("max_train_examples", 180))
    rows = sample_records(rows, requested_examples, cfg.get("sampling_mode", "first_n"), seed)
    record_sampled_ids(
        "S03_zero_control_train", rows,
        path="outputs/S03_zero_control_gsm8k/metrics/sampled_ids.json",
    )

    model_name = s02["student_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    mc = s02["minimal_coupling"]
    if mc.get("init_agent_b_from_d11", True):
        model_b = PeftModel.from_pretrained(
            model_b, str(paths["agent_b_pre_s02"]), is_trainable=False
        ).merge_and_unload()
    lora = s02["lora"]
    model_b = get_peft_model(model_b, LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]), bias="none", task_type="CAUSAL_LM",
        target_modules=lora["target_modules"],
    ))
    model_b.train()

    train_cfg = s02["training"]
    trainable = [parameter for parameter in model_b.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Zero-control receiver has no trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["learning_rate"]))
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    max_length = int(train_cfg["max_seq_length"])
    layer_b = get_layer_by_index(model_b, int(mc["layer_index"]))
    zero_bridge = MinimalCouplingBridge(
        d_model=int(model_b.config.hidden_size), bottleneck_dim=int(mc["bottleneck_dim"])
    ).to(device).to(dtype)
    zero_bridge.eval().requires_grad_(False)
    zero_z = torch.zeros((1, int(mc["bottleneck_dim"])), device=device, dtype=dtype)
    log_path = ensure_parent(cfg["output"]["train_log_path"])
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    with open(log_path, "w", encoding="utf-8") as log_handle:
        if args.max_steps is None:
            training_items = iter(rows)
            total_microsteps = len(rows)
        else:
            total_microsteps = args.max_steps * grad_accum
            training_items = itertools.islice(itertools.cycle(rows), total_microsteps)
        for index, row in enumerate(training_items):
            gold = extract_gsm8k_gold_answer(row["raw_answer"])
            reasoning = row.get("raw_answer", "").split("####")[0].strip()
            target = f"Reasoning:\n{reasoning}\n\nFinal answer:\n{gold}"
            text = format_messages_with_assistant(tokenizer, messages_for_single(row["problem"]), target)
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length, padding=False
            ).to(device)

            injector = HiddenStateInjector(
                layer_b, lambda hidden: zero_bridge.inject(hidden, zero_z)
            )
            try:
                loss = model_b(**encoded, labels=encoded["input_ids"].clone()).loss / grad_accum
            finally:
                injector.remove()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at sample {row['id']}: {loss.item()}")
            loss.backward()

            end_of_accum = (index + 1) % grad_accum == 0
            end_of_data = index + 1 == total_microsteps
            if end_of_accum or end_of_data:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "step": global_step, "examples_seen": index + 1,
                    "loss_scaled": float(loss.detach().cpu()), "zero_message": True,
                }
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                print(f"step={global_step} microexamples={index + 1}/{total_microsteps} loss={loss.item():.6f}")
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

    output_adapter = project_path(cfg["output"]["adapter_dir"])
    output_adapter.mkdir(parents=True, exist_ok=True)
    model_b.save_pretrained(output_adapter)
    tokenizer.save_pretrained(output_adapter)
    parity = {
        "training_parity_status": "exact" if args.max_examples is None and args.max_steps is None else "smoke_override",
        "source_experiment": "S02_minimal_coupling_gsm8k",
        "same_pre_s02_initialization": True,
        "same_data_path": train_path,
        "same_sample_order": True,
        "same_seed": seed,
        "same_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "same_gradient_accumulation_steps": grad_accum,
        "same_optimizer": "AdamW",
        "same_learning_rate": float(train_cfg["learning_rate"]),
        "same_num_train_epochs": int(train_cfg["num_train_epochs"]),
        "same_prompt_and_labels": True,
        "same_precision": str(dtype),
        "differences": [
            "latent message is always zero",
            "bridge parameters are excluded because an exact zero residual has no participating bridge parameters",
        ],
        "num_unique_examples": len(rows),
        "num_example_exposures": total_microsteps,
        "optimization_steps": global_step,
    }
    write_json("outputs/S03_zero_control_gsm8k/metrics/training_parity.json", parity)
    print(f"Saved zero-control adapter: {output_adapter}")


if __name__ == "__main__":
    main()
