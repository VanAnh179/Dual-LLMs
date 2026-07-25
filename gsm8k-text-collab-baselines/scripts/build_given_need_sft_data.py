#!/usr/bin/env python
"""Build D11.4 SFT datasets for compact Given/Need notes and final decision making."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_config, read_jsonl, write_jsonl


EXPERIMENT_NAME = "D11_4_compact_given_need_decider_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def notes_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "given_need_generator",
        "assistant_target": row["given_need_notes"].strip(),
    }


def decider_record(row: dict, notes: str, notes_quality: str) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "given_need_decider",
        "notes_quality": notes_quality,
        "given_need_notes": notes.strip(),
        "assistant_target": (
            f"Reasoning:\n{row['reasoning'].strip()}\n\n"
            f"Final answer:\n{row['gold_answer']}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_4_compact_given_need_decider.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg_get(
        cfg,
        "data",
        "filtered_trace_path",
        f"data/filtered/{EXPERIMENT_NAME}/given_need_traces.jsonl",
    )
    output_dir = args.output_dir or cfg_get(cfg, "data", "train_dir", f"data/train/{EXPERIMENT_NAME}")
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"No D11.4 traces found in {input_path}. Run bootstrap_given_need_traces.py first.")

    note_rows = [notes_record(row) for row in rows]
    good_rows = []
    empty_rows = []
    decider_cfg = cfg.get("decider_training", {})
    for row in rows:
        if bool(decider_cfg.get("include_good_notes", True)):
            good_rows.append(decider_record(row, row["given_need_notes"], "good"))
        if bool(decider_cfg.get("include_empty_notes", False)):
            empty_rows.append(decider_record(row, "Given: not provided.\nNeed: solve the problem.", "empty"))

    decider_rows = good_rows + empty_rows
    write_jsonl(f"{output_dir}/given_need_train.jsonl", note_rows)
    write_jsonl(f"{output_dir}/decider_good_notes_train.jsonl", good_rows)
    write_jsonl(f"{output_dir}/decider_empty_notes_train.jsonl", empty_rows)
    write_jsonl(f"{output_dir}/decider_train.jsonl", decider_rows)
    print(f"Wrote {len(note_rows)} Given/Need rows and {len(decider_rows)} decider rows.")


if __name__ == "__main__":
    main()
