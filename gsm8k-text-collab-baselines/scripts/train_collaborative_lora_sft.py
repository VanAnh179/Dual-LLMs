#!/usr/bin/env python
"""Train D11.2 collaborative LoRA SFT adapters with standard SFT loss."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    get_student_model_name,
    load_config,
    project_path,
    read_jsonl,
    record_sampled_ids,
    reject_test_rows_for_training,
    reject_test_split_for_training,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
)
from src.generation import load_tokenizer_and_model
from src.prompts import (
    format_messages_with_assistant,
    messages_for_first_contributor,
    messages_for_second_contributor,
)


EXPERIMENT_NAME = "D11_2_latent_collaborative_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def clear_model(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_texts(tokenizer, rows: list[dict]) -> list[str]:
    texts = []
    for row in rows:
        mode = row.get("training_mode")
        if mode == "first_contributor":
            messages = messages_for_first_contributor(row["problem"])
        elif mode == "second_contributor":
            messages = messages_for_second_contributor(row["problem"], row.get("other_agent_contribution", ""))
        else:
            raise SystemExit(f"Unsupported training_mode: {mode!r}")
        texts.append(format_messages_with_assistant(tokenizer, messages, row["assistant_target"]))
    return texts


def train_text_sft(model, tokenizer, rows: list[dict], output_dir: str | Path, training_config: dict) -> None:
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    texts = build_texts(tokenizer, rows)
    if not texts:
        raise SystemExit("No collaborative SFT examples were found.")

    max_seq_length = int(training_config.get("max_seq_length", 2048))
    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False)

    dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing collaborative SFT data",
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 100)),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def train_adapter(student_model_name: str, rows: list[dict], output_dir: Path, cfg: dict, adapter_path: str | None = None) -> str:
    tokenizer, model = load_tokenizer_and_model(
        student_model_name,
        adapter_path=adapter_path,
        trainable_lora=True,
        lora_config=cfg.get("lora", {}),
    )
    train_text_sft(model, tokenizer, rows, output_dir, cfg.get("training", {}))
    clear_model(model)
    return str(output_dir)


def assert_standard_loss(cfg: dict) -> None:
    loss = cfg.get("loss", {})
    if loss.get("type", "standard_sft") != "standard_sft":
        raise SystemExit("D11.2 currently supports only loss.type: standard_sft.")
    if float(loss.get("anti_copy_weight", 0.0)) != 0.0 or float(loss.get("diversity_weight", 0.0)) != 0.0:
        raise SystemExit("D11.2 loss regularizers are TODO-only and must remain at weight 0.0.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_2_qwen_math7b_teacher.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--variant", choices=["all", "mixed", "alternating"], default="all")
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "datasets", "peft", "yaml")
    cfg = load_config(args.config)
    assert_standard_loss(cfg)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    train_dir = args.train_dir or cfg_get(cfg, "data", "train_dir", f"data/train/{EXPERIMENT_NAME}")
    reject_test_split_for_training(train_dir)
    mixed_rows = read_jsonl(f"{train_dir}/mixed_collaborative_train.jsonl")
    second_rows = read_jsonl(f"{train_dir}/second_contributor_train.jsonl")
    reject_test_rows_for_training(mixed_rows + second_rows)
    if not mixed_rows or not second_rows:
        raise SystemExit(f"Collaborative SFT data missing under {train_dir}. Run build_collaborative_sft_data.py first.")

    seed = int(cfg_get(cfg, "sampling", "seed", 42))
    sampling_mode = cfg_get(cfg, "sampling", "sampling_mode", "first_n")
    max_train = args.max_train_examples or cfg_get(cfg, "sampling", "max_train_examples", 100)
    mixed_rows = sample_records(mixed_rows, max_train * 2 if max_train else None, sampling_mode, seed)
    second_rows = sample_records(second_rows, max_train, sampling_mode, seed)
    record_sampled_ids("D11_2_mixed_train_ids", mixed_rows)
    record_sampled_ids("D11_2_second_contributor_train_ids", second_rows)

    student_model_name = get_student_model_name(cfg)
    adapter_dir = project_path(cfg_get(cfg, "output", "adapter_dir", f"outputs/{EXPERIMENT_NAME}/adapters"))
    print("D11.2 changes the data objective, not the mathematical loss yet.")
    print("Training with standard token-level SFT loss.")

    if args.variant in {"all", "mixed"}:
        train_adapter(student_model_name, mixed_rows, adapter_dir / "agent_A_mixed_sft", cfg)
        train_adapter(student_model_name, mixed_rows, adapter_dir / "agent_B_mixed_sft", cfg)

    if args.variant in {"all", "alternating"}:
        agent_a_adapter = None
        agent_b_adapter = None
        rounds = int(args.num_rounds or cfg.get("training", {}).get("num_alternating_rounds", 1))
        for round_idx in range(1, rounds + 1):
            agent_b_adapter = train_adapter(
                student_model_name,
                second_rows,
                adapter_dir / f"agent_B_round_{round_idx}",
                cfg,
                adapter_path=agent_b_adapter,
            )
            agent_a_adapter = train_adapter(
                student_model_name,
                second_rows,
                adapter_dir / f"agent_A_round_{round_idx}",
                cfg,
                adapter_path=agent_a_adapter,
            )
        print(f"Saved Agent A adapter: {agent_a_adapter}")
        print(f"Saved Agent B adapter: {agent_b_adapter}")


if __name__ == "__main__":
    main()
