"""Prompt, supervised-tokenization, parsing, and metrics for V02."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from src.V02_benchmark import FAMILIES, LABELS


ANSWER_SYSTEM = """Solve the supplied benchmark task using only the available private view or views.
The valid labels are OPTION_0, OPTION_1, OPTION_2, and OPTION_3.
End with exactly one line: ANSWER: OPTION_X."""

ENCODER_SYSTEM = """Read PRIVATE VIEW A and encode all facts needed by a separate receiver.
Do not assume access to PRIVATE VIEW B. Your hidden states will be used as the message."""

RECEIVER_SYSTEM = """Solve the benchmark task from PRIVATE VIEW B and the injected latent message.
The valid labels are OPTION_0, OPTION_1, OPTION_2, and OPTION_3.
End with exactly one line: ANSWER: OPTION_X."""


def messages_for_mode(mode: str, row: dict[str, Any]) -> list[dict[str, str]]:
    field_by_mode = {
        "full": "full_problem",
        "view_a": "view_a",
        "view_b": "view_b",
        "split_a": "view_a",
        "split_b": "view_b",
    }
    if mode not in field_by_mode:
        raise ValueError(f"Unsupported prompt mode: {mode!r}")
    system = (
        ENCODER_SYSTEM if mode == "split_a"
        else RECEIVER_SYSTEM if mode == "split_b"
        else ANSWER_SYSTEM
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(row[field_by_mode[mode]])},
    ]


def format_prompt(tokenizer, mode: str, row: dict[str, Any]) -> str:
    messages = messages_for_mode(mode, row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    ) + "\n\nASSISTANT:"


def format_supervised(tokenizer, mode: str, row: dict[str, Any]) -> tuple[str, str]:
    messages = messages_for_mode(mode, row)
    target = f"ANSWER: {row['gold_answer']}"
    prompt = format_prompt(tokenizer, mode, row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        full = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full = prompt + target
    return prompt, full


def encode_supervised_batch(
    tokenizer, mode: str, rows: list[dict[str, Any]], max_length: int, device
) -> dict[str, Any]:
    import torch

    all_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    for row in rows:
        prompt, full = format_supervised(tokenizer, mode, row)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
        common = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            common += 1
        if common < min(len(prompt_ids), len(full_ids)) - 2:
            raise RuntimeError("Chat-template prompt is not a prefix of supervised text.")
        labels = [-100] * common + full_ids[common:]
        if not any(label != -100 for label in labels):
            raise RuntimeError("max_seq_length truncates the complete assistant target.")
        all_ids.append(full_ids)
        all_labels.append(labels)
    width = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    padded_labels = []
    for ids, labels in zip(all_ids, all_labels):
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        padded_labels.append(labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }


def encode_prompt_batch(
    tokenizer, mode: str, rows: list[dict[str, Any]], max_length: int, device
):
    prompts = [format_prompt(tokenizer, mode, row) for row in rows]
    return tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length, add_special_tokens=False,
    ).to(device)


def extract_option(text: str) -> str | None:
    explicit = re.findall(
        r"ANSWER\s*:\s*OPTION[_\s-]?([0-3])\b", text, flags=re.IGNORECASE
    )
    if explicit:
        return f"OPTION_{explicit[-1]}"
    mentions = re.findall(r"\bOPTION[_\s-]?([0-3])\b", text, flags=re.IGNORECASE)
    return f"OPTION_{mentions[-1]}" if mentions else None


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty predictions.")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(int(bool(row["correct"])) for row in group)
        parsed = sum(row.get("predicted_answer") in LABELS for row in group)
        return {
            "num_examples": len(group),
            "accuracy": correct / len(group),
            "parse_rate": parsed / len(group),
            "correct": correct,
        }

    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    difficulty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_groups[str(row["family"])].append(row)
        difficulty_groups[str(row["difficulty"])].append(row)
    return {
        "overall": summarize(rows),
        "by_family": {
            family: summarize(family_groups[family]) for family in FAMILIES
        },
        "by_difficulty": {
            difficulty: summarize(group)
            for difficulty, group in sorted(difficulty_groups.items())
        },
        "prediction_counts": dict(sorted(Counter(
            row.get("predicted_answer") or "UNPARSED" for row in rows
        ).items())),
    }


def paired_stratified_bootstrap(
    left: list[dict[str, Any]], right: list[dict[str, Any]],
    num_resamples: int, seed: int,
) -> dict[str, Any]:
    import numpy as np

    right_by_id = {str(row["sample_id"]): row for row in right}
    if {str(row["sample_id"]) for row in left} != set(right_by_id):
        raise ValueError("Paired predictions have different sample IDs.")
    strata: dict[str, list[float]] = defaultdict(list)
    for row in left:
        partner = right_by_id[str(row["sample_id"])]
        strata[str(row["family"])].append(
            float(bool(row["correct"])) - float(bool(partner["correct"]))
        )
    rng = np.random.default_rng(seed)
    estimates = np.zeros(num_resamples, dtype=np.float64)
    total = sum(len(values) for values in strata.values())
    for values in strata.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(num_resamples, len(array)))
        estimates += array[indices].sum(axis=1) / total
    observed = sum(sum(values) for values in strata.values()) / total
    return {
        "delta": float(observed),
        "bootstrap_95_ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "num_resamples": num_resamples,
        "stratified_by": "family",
    }
