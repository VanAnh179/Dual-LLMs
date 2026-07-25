#!/usr/bin/env python
"""Train minimal coupling bridge + Agent B LoRA on GSM8K with frozen Agent A."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "S02_minimal_coupling_gsm8k"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal coupling bridge + Agent B LoRA")
    parser.add_argument("--config", type=str, default="configs/s02_minimal_coupling_gsm8k.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit training examples (default: all)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Limit training steps (default: full epoch)")
    args = parser.parse_args()

    import gc
    import torch
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data_utils import (
        load_config, project_path, read_jsonl, sample_records,
        reject_test_split_for_training, reject_test_rows_for_training,
        record_sampled_ids, ensure_parent,
    )
    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge
    from src.prompts import format_messages_with_assistant, messages_for_single
    from src.answer_extraction import extract_gsm8k_gold_answer

    cfg = load_config(args.config)
    model_name = cfg.get("student_model_name")
    mc = cfg["minimal_coupling"]
    layer_index = mc["layer_index"]
    bottleneck_dim = mc["bottleneck_dim"]
    freeze_a = mc.get("freeze_agent_a", True)
    init_from_d11 = mc.get("init_agent_b_from_d11", True)
    adapter_a_path = str(project_path(mc["agent_a_adapter_path"]))
    adapter_b_path = str(project_path(mc["agent_b_adapter_path"]))

    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    sampling_cfg = cfg.get("sampling", {})

    # Data guards
    train_data_path = cfg_get(cfg, "data", "raw_train_path")
    reject_test_split_for_training(train_data_path)
    rows = read_jsonl(train_data_path)
    reject_test_rows_for_training(rows)

    # Sample
    max_examples = args.max_examples or sampling_cfg.get("max_train_examples")
    rows = sample_records(rows, max_examples=max_examples,
                          sampling_mode=sampling_cfg.get("sampling_mode", "first_n"),
                          seed=sampling_cfg.get("seed", 42))
    record_sampled_ids("S02_train", rows, path="outputs/S02_minimal_coupling_gsm8k/metrics/sampled_ids.json")
    print(f"Training on {len(rows)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16 for stable mixed precision training.")
    else:
        dtype = torch.float32
        print("Using float32 for training.")

    # --- Load Agent A ---
    print("\n[1] Loading Agent A (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model_a = PeftModel.from_pretrained(model_a, adapter_a_path, is_trainable=False)
    model_a.eval()
    if freeze_a:
        model_a.requires_grad_(False)
    print(f"  Agent A loaded, frozen={freeze_a}")

    # --- Load Agent B ---
    print(f"\n[2] Loading Agent B (init_from_d11={init_from_d11})...")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    new_lora = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules",
                                     ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"]),
    )

    if init_from_d11:
        # Merge D11.0 adapter into base weights, then apply fresh LoRA on top.
        # This bakes D11.0 knowledge into the base and trains new LoRA params
        # specifically for bridge integration.
        model_b = PeftModel.from_pretrained(model_b, adapter_b_path, is_trainable=True)
        model_b = model_b.merge_and_unload()
        model_b = get_peft_model(model_b, new_lora)
    else:
        # Fresh LoRA on bare base model — no D11.0 initialization.
        model_b = get_peft_model(model_b, new_lora)

    model_b.train()
    trainable_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"  Agent B trainable params: {trainable_b:,}")

    # --- Bridge ---
    d_model = model_a.config.hidden_size
    bridge = MinimalCouplingBridge(d_model=d_model, bottleneck_dim=bottleneck_dim).to(device).to(dtype)
    bridge.train()
    bridge_params = sum(p.numel() for p in bridge.parameters())
    print(f"\n[3] Bridge params: {bridge_params:,}")

    # --- Optimizer: bridge + B's LoRA only ---
    trainable_params = list(bridge.parameters()) + [
        p for p in model_b.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg.get("learning_rate", 2e-4)))
    max_seq_length = int(train_cfg.get("max_seq_length", 1536))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    logging_steps = int(train_cfg.get("logging_steps", 10))

    # --- Hook setup ---
    layer_a = get_layer_by_index(model_a, layer_index)
    extractor = HiddenStateExtractor(layer_a)

    # --- Training loop ---
    print(f"\n[4] Starting training loop...")
    print(f"    max_steps={args.max_steps}, grad_accum={grad_accum}, lr={optimizer.defaults['lr']}")

    global_step = 0
    accum_loss = 0.0
    max_steps = args.max_steps

    optimizer.zero_grad()

    for i, row in enumerate(rows):
        problem = row["problem"]
        gold_answer = extract_gsm8k_gold_answer(row["raw_answer"])
        reasoning = row.get("raw_answer", "").split("####")[0].strip()

        # Build SFT target text
        messages = messages_for_single(problem)
        full_text = format_messages_with_assistant(tokenizer, messages, f"Reasoning:\n{reasoning}\n\nFinal answer:\n{gold_answer}")

        # Tokenize
        encoding = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=max_seq_length, padding=False).to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # --- Forward Agent A (no grad, frozen) ---
        extractor.clear()
        with torch.no_grad():
            model_a(input_ids=input_ids, attention_mask=attention_mask)
        h_a = extractor.hidden_states.detach()  # detach, keep on GPU

        # --- Encode z through bridge ---
        z = bridge.encode(h_a)

        # --- Hook Agent B for injection ---
        layer_b = get_layer_by_index(model_b, layer_index)

        def make_injection_fn(z_vec):
            def injection_fn(h_b):
                return bridge.inject(h_b, z_vec)
            return injection_fn

        injector = HiddenStateInjector(layer_b, make_injection_fn(z))

        # --- Forward Agent B (with injection, with grad) ---
        labels = input_ids.clone()
        outputs_b = model_b(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs_b.loss / grad_accum

        injector.remove()

        if torch.isnan(loss):
            print(f"  [WARNING] NaN loss at example {i}, skipping")
            optimizer.zero_grad()
            continue

        loss.backward()
        accum_loss += loss.item()

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0 or global_step == 1:
                avg_loss = accum_loss * grad_accum / max(1, grad_accum)
                print(f"  step={global_step}, loss={accum_loss:.4f}, example={i+1}/{len(rows)}")
                accum_loss = 0.0

            if max_steps is not None and global_step >= max_steps:
                print(f"  Reached max_steps={max_steps}, stopping.")
                break

    # Final step for leftover accumulation
    remainder = len(rows) % grad_accum if max_steps is None else 0
    if remainder > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        print(f"  step={global_step} (final partial), loss={accum_loss:.4f}")

    extractor.remove()

    # --- Save ---
    adapter_dir = str(project_path(cfg_get(cfg, "output", "adapter_dir")))
    b_save_path = str(Path(adapter_dir) / "agent_B_minimal_coupling_sft")
    bridge_save_path = str(Path(adapter_dir) / "minimal_coupling_bridge.pt")

    ensure_parent(b_save_path + "/dummy")
    model_b.save_pretrained(b_save_path)
    tokenizer.save_pretrained(b_save_path)
    bridge.save_bridge(bridge_save_path)

    print(f"\n[5] Saved:")
    print(f"  Agent B LoRA: {b_save_path}")
    print(f"  Bridge:       {bridge_save_path}")
    print(f"  Total steps:  {global_step}")
    print("\nDone.")


if __name__ == "__main__":
    main()
