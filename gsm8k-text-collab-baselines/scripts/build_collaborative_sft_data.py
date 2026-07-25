#!/usr/bin/env python
"""Build D11.2 SFT JSONL datasets from collaborative traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_config, read_jsonl, write_jsonl


EXPERIMENT_NAME = "D11_2_latent_collaborative_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def first_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "first_contributor",
        "assistant_target": f"Contribution:\n{row['agent1_contribution'].strip()}",
    }


def second_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "training_mode": "second_contributor",
        "other_agent_contribution": row["agent1_contribution"],
        "assistant_target": (
            f"Contribution:\n{row['agent2_contribution'].strip()}\n\n"
            f"Joint solution:\n{row['joint_solution'].strip()}\n\n"
            f"Final answer:\n{row['gold_answer']}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_2_qwen_math7b_teacher.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg_get(
        cfg,
        "data",
        "filtered_trace_path",
        f"data/filtered/{EXPERIMENT_NAME}/two_agent_traces.jsonl",
    )
    output_dir = args.output_dir or cfg_get(cfg, "data", "train_dir", f"data/train/{EXPERIMENT_NAME}")
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"No collaborative traces found in {input_path}. Run bootstrap_collaborative_traces.py first.")

    first_rows = [first_record(row) for row in rows]
    second_rows = [second_record(row) for row in rows]
    mixed_rows = [item for pair in zip(first_rows, second_rows) for item in pair]

    write_jsonl(f"{output_dir}/first_contributor_train.jsonl", first_rows)
    write_jsonl(f"{output_dir}/second_contributor_train.jsonl", second_rows)
    write_jsonl(f"{output_dir}/mixed_collaborative_train.jsonl", mixed_rows)
    print(f"Wrote {len(first_rows)} first contributor rows, {len(second_rows)} second contributor rows.")


if __name__ == "__main__":
    main()
