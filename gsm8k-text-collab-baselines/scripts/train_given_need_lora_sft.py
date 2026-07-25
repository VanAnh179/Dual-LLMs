#!/usr/bin/env python
"""Train D11.4 Given/Need generator and final-decider LoRA adapters."""

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
from src.prompts import format_messages_with_assistant, messages_for_given_need, messages_for_given_need_decider


EXPERIMENT_NAME = "D11_4_compact_given_need_decider_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def clear_model(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_standard_loss(cfg: dict) -> None:
    loss = cfg.get("loss", {})
    if loss.get("type", "standard_sft") != "standard_sft":
        raise SystemExit("D11.4 currently supports only loss.type: standard_sft.")


def build_texts(tokenizer, rows: list[dict]) -> list[str]:
    texts = []
    for row in rows:
        mode = row.get("training_mode")
        if mode == "given_need_generator":
            messages = messages_for_given_need(row["problem"])
        elif mode == "given_need_decider":
            messages = messages_for_given_need_decider(row["problem"], row.get("given_need_notes", ""))
        else:
            raise SystemExit(f"Unsupported training_mode: {mode!r}")
        texts.append(format_messages_with_assistant(tokenizer, messages, row["assistant_target"]))
    return texts


def train_text_sft(model, tokenizer, rows: list[dict], output_dir: str | Path, training_config: dict) -> None:
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    texts = build_texts(tokenizer, rows)
    if not texts:
        raise SystemExit("No D11.4 SFT examples were found.")

    max_seq_length = int(training_config.get("max_seq_length", 1536))
    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False)

    dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing D11.4 SFT data",
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


def train_adapter(student_model_name: str, rows: list[dict], output_dir: Path, cfg: dict) -> str:
    tokenizer, model = load_tokenizer_and_model(
        student_model_name,
        trainable_lora=True,
        lora_config=cfg.get("lora", {}),
    )
    train_text_sft(model, tokenizer, rows, output_dir, cfg.get("training", {}))
    clear_model(model)
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_4_compact_given_need_decider.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--role", choices=["all", "notes", "decider"], default="all")
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "datasets", "peft", "yaml")
    cfg = load_config(args.config)
    assert_standard_loss(cfg)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    train_dir = args.train_dir or cfg_get(cfg, "data", "train_dir", f"data/train/{EXPERIMENT_NAME}")
    reject_test_split_for_training(train_dir)
    notes_rows = read_jsonl(f"{train_dir}/given_need_train.jsonl")
    decider_rows = read_jsonl(f"{train_dir}/decider_train.jsonl")
    reject_test_rows_for_training(notes_rows + decider_rows)
    if not notes_rows or not decider_rows:
        raise SystemExit(f"D11.4 SFT data missing under {train_dir}. Run build_given_need_sft_data.py first.")

    seed = int(cfg_get(cfg, "sampling", "seed", 42))
    sampling_mode = cfg_get(cfg, "sampling", "sampling_mode", "first_n")
    max_train = args.max_train_examples or cfg_get(cfg, "sampling", "max_train_examples", 200)
    notes_rows = sample_records(notes_rows, max_train, sampling_mode, seed)
    decider_rows = sample_records(decider_rows, max_train, sampling_mode, seed)
    record_sampled_ids("D11_4_given_need_train_ids", notes_rows)
    record_sampled_ids("D11_4_decider_train_ids", decider_rows)

    student_model_name = get_student_model_name(cfg)
    adapter_dir = project_path(cfg_get(cfg, "output", "adapter_dir", f"outputs/{EXPERIMENT_NAME}/adapters"))
    print("D11.4 uses standard SFT loss: Agent A Given/Need compressor, Agent B final decider.")

    if args.role in {"all", "notes"}:
        path = train_adapter(student_model_name, notes_rows, adapter_dir / "agent_A_given_need_sft", cfg)
        print(f"Saved Agent A Given/Need adapter: {path}")
    if args.role in {"all", "decider"}:
        path = train_adapter(student_model_name, decider_rows, adapter_dir / "agent_B_decider_sft", cfg)
        print(f"Saved Agent B decider adapter: {path}")


if __name__ == "__main__":
    main()
