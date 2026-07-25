#!/usr/bin/env python
"""Train LoRA adapters for D12.1 masked agents and synthesizer."""

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
    parser = argparse.ArgumentParser(description="Train LoRA adapters for D12.1")
    parser.add_argument("--config", default="configs/d12_1_info_asymmetric.yaml")
    parser.add_argument("--max-train-examples", type=int, default=None)
    args = parser.parse_args()

    from src.data_utils import (
        load_config,
        get_student_model_name,
        read_jsonl,
        sample_records,
        reject_test_rows_for_training,
        require_cuda_if_requested,
        require_dependencies,
        ensure_parent,
    )
    from src.generation import load_tokenizer_and_model
    from src.prompts import format_messages_with_assistant
    from src.prompts_d12_1 import (
        messages_for_masked_partial,
        messages_for_masked_synthesizer,
    )

    require_dependencies("torch", "transformers", "peft", "yaml", "datasets")
    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    student_model = get_student_model_name(cfg)
    train_dir = cfg_get(cfg, "data", "train_dir", "data/train/D12_1")
    adapter_dir = cfg_get(cfg, "output", "adapter_dir",
                          "outputs/D12_1_info_asymmetric_masking_sft/adapters")
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})

    max_train = args.max_train_examples
    seed = int(cfg_get(cfg, "sampling", "seed", 42))

    # Define adapter training jobs
    jobs = [
        {
            "name": "agent_A_partial_view_sft",
            "data_file": f"{train_dir}/agent_a_partial.jsonl",
            "build_messages": lambda row: messages_for_masked_partial(row["view"]),
            "build_target": lambda row: row["contribution"],
        },
        {
            "name": "agent_B_partial_view_sft",
            "data_file": f"{train_dir}/agent_b_partial.jsonl",
            "build_messages": lambda row: messages_for_masked_partial(row["view"]),
            "build_target": lambda row: row["contribution"],
        },
        {
            "name": "final_synthesizer_sft",
            "data_file": f"{train_dir}/final_synthesis.jsonl",
            "build_messages": lambda row: messages_for_masked_synthesizer(
                row["problem"], row["contrib_a"], row["contrib_b"]
            ),
            "build_target": lambda row: (
                f"Reasoning:\n{row['joint_solution']}\n\nFinal answer:\n{row['final_answer']}"
            ),
        },
    ]

    for job in jobs:
        name = job["name"]
        out_path = f"{adapter_dir}/{name}"
        print(f"\n[D12.1] Training adapter: {name}")

        rows = read_jsonl(job["data_file"])
        if not rows:
            raise SystemExit(
                f"No training data at {job['data_file']}. "
                "Run build_d12_1_masked_sft_data.py first."
            )
        reject_test_rows_for_training(rows)

        if max_train:
            rows = sample_records(rows, max_train, "first_n", seed)

        # Load model with fresh LoRA
        tokenizer, model = load_tokenizer_and_model(
            student_model, trainable_lora=True, lora_config=lora_cfg
        )

        # Build SFT texts
        texts = []
        for row in rows:
            msgs = job["build_messages"](row)
            target = job["build_target"](row)
            texts.append(format_messages_with_assistant(tokenizer, msgs, target))

        if not texts:
            print(f"  WARNING: no SFT texts for {name}, skipping.")
            clear_model(model, tokenizer)
            continue

        # Tokenize and train
        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        max_seq_length = int(training_cfg.get("max_seq_length", 1536))
        ds = Dataset.from_dict({"text": texts})

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"], truncation=True,
                max_length=max_seq_length, padding=False,
            )

        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"], desc=f"Tokenizing {name}")

        train_args = TrainingArguments(
            output_dir=str(ensure_parent(out_path)),
            per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
            num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
            logging_steps=int(training_cfg.get("logging_steps", 10)),
            save_steps=int(training_cfg.get("save_steps", 100)),
            report_to=[],
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            args=train_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        print(f"  Saved: {out_path}")

        clear_model(model, tokenizer, trainer)

    print(f"\n[D12.1] All adapters trained -> {adapter_dir}/")


if __name__ == "__main__":
    main()
