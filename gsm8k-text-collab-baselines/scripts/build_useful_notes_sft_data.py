#!/usr/bin/env python
"""Build D11.3 SFT datasets for useful notes and final decision making."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_config, read_jsonl, write_jsonl


EXPERIMENT_NAME = "D11_3_useful_notes_decider_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def notes_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "note_generator",
        "assistant_target": f"Notes:\n{row['useful_notes'].strip()}",
    }


def decider_record(row: dict, notes: str, notes_quality: str) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "final_decider",
        "notes_quality": notes_quality,
        "useful_notes": notes.strip(),
        "assistant_target": (
            f"Reasoning:\n{row['reasoning'].strip()}\n\n"
            f"Final answer:\n{row['gold_answer']}"
        ),
    }


def first_note_lines(notes: str, max_lines: int = 2) -> str:
    lines = [line.strip() for line in notes.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_3_useful_notes_decider.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg_get(
        cfg,
        "data",
        "filtered_trace_path",
        f"data/filtered/{EXPERIMENT_NAME}/useful_note_traces.jsonl",
    )
    output_dir = args.output_dir or cfg_get(cfg, "data", "train_dir", f"data/train/{EXPERIMENT_NAME}")
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"No D11.3 traces found in {input_path}. Run bootstrap_useful_notes_traces.py first.")

    note_rows = [notes_record(row) for row in rows]
    good_rows = []
    partial_rows = []
    empty_rows = []
    decider_cfg = cfg.get("decider_training", {})
    for row in rows:
        if bool(decider_cfg.get("include_good_notes", True)):
            good_rows.append(decider_record(row, row["useful_notes"], "good"))
        if bool(decider_cfg.get("include_partial_notes", True)):
            partial = first_note_lines(row["useful_notes"], max_lines=2)
            if partial and partial != row["useful_notes"].strip():
                partial_rows.append(decider_record(row, partial, "partial"))
        if bool(decider_cfg.get("include_empty_notes", False)):
            empty_rows.append(decider_record(row, "No useful notes provided.", "empty"))

    decider_rows = good_rows + partial_rows + empty_rows
    write_jsonl(f"{output_dir}/notes_train.jsonl", note_rows)
    write_jsonl(f"{output_dir}/decider_good_notes_train.jsonl", good_rows)
    write_jsonl(f"{output_dir}/decider_partial_notes_train.jsonl", partial_rows)
    write_jsonl(f"{output_dir}/decider_train.jsonl", decider_rows)
    print(f"Wrote {len(note_rows)} notes rows and {len(decider_rows)} decider rows.")


if __name__ == "__main__":
    main()
