"""SFT dataset construction and LoRA training helpers."""

from __future__ import annotations

from pathlib import Path

from .prompts import format_for_sft, messages_for_single, messages_for_two_agent


def build_sft_texts(tokenizer, rows: list[dict], two_agent: bool) -> list[str]:
    texts = []
    for row in rows:
        reasoning = row.get("reasoning_trace")
        if not reasoning:
            continue
        if two_agent:
            messages = messages_for_two_agent(row["problem"], row.get("other_agent_response", ""))
        else:
            messages = messages_for_single(row["problem"])
        texts.append(format_for_sft(tokenizer, messages, reasoning, row["gold_answer"]))
    return texts


def train_lora_sft(
    model,
    tokenizer,
    rows: list[dict],
    output_dir: str | Path,
    training_config: dict,
    two_agent: bool = True,
) -> None:
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    texts = build_sft_texts(tokenizer, rows, two_agent=two_agent)
    if not texts:
        raise SystemExit("No SFT examples with reasoning_trace were found.")

    max_seq_length = int(training_config.get("max_seq_length", 2048))
    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing SFT data",
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
