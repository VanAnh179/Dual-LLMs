#!/usr/bin/env python
"""Prepare GSM8K JSONL files under data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answer_extraction import extract_gsm8k_gold_answer
from src.data_utils import (
    load_config,
    record_sampled_ids,
    require_dependencies,
    sample_records,
    split_train_validation,
    write_jsonl,
)


def convert_split(dataset, split: str, limit: int | None) -> list[dict]:
    rows = []
    for idx, item in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        raw_answer = item.get("answer", "")
        rows.append(
            {
                "id": f"{split}-{idx}",
                "problem": item.get("question", ""),
                "gold_answer": extract_gsm8k_gold_answer(raw_answer),
                "raw_answer": raw_answer,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_qwen05b.yaml")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["first_n", "random"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=None)
    args = parser.parse_args()

    require_dependencies("datasets")
    from datasets import load_dataset

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    sampling_mode = args.sampling_mode or cfg.get("sampling_mode", "first_n")
    validation_ratio = float(
        args.validation_ratio if args.validation_ratio is not None else cfg.get("validation_ratio", 0.1)
    )

    try:
        data = load_dataset("gsm8k", "main")
    except Exception as exc:
        raise SystemExit(
            "Could not load GSM8K from Hugging Face datasets. Check network access and "
            "install dependencies with `pip install -r requirements.txt`."
        ) from exc

    all_train_rows = convert_split(data["train"], "train", None)
    all_test_rows = convert_split(data["test"], "test", None)
    max_train = args.max_train_examples if args.max_train_examples is not None else cfg.get("max_train_examples")
    max_test = args.max_test_examples if args.max_test_examples is not None else cfg.get("max_eval_examples")
    train_rows = sample_records(all_train_rows, max_train, sampling_mode, seed)
    test_rows = sample_records(all_test_rows, max_test, sampling_mode, seed)
    train_rows, dev_rows = split_train_validation(train_rows, validation_ratio, sampling_mode, seed)
    write_jsonl("data/raw/train.jsonl", train_rows)
    write_jsonl("data/raw/dev.jsonl", dev_rows)
    write_jsonl("data/raw/test.jsonl", test_rows)
    record_sampled_ids("prepare_train_ids", train_rows)
    record_sampled_ids("prepare_dev_ids", dev_rows)
    record_sampled_ids("prepare_test_ids", test_rows)
    print(f"Wrote {len(train_rows)} train rows to data/raw/train.jsonl")
    print(f"Wrote {len(dev_rows)} dev rows to data/raw/dev.jsonl")
    print(f"Wrote {len(test_rows)} test rows to data/raw/test.jsonl")


if __name__ == "__main__":
    main()
