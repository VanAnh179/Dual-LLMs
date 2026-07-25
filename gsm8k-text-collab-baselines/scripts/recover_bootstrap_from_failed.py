#!/usr/bin/env python
"""Recover correct bootstrapped traces from saved failed candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import write_jsonl
from src.filtering import reasoning_from_candidate, trace_is_correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/generations/bootstrap_failed_candidates.jsonl")
    parser.add_argument("--output", default="data/filtered/bootstrap_train.jsonl")
    args = parser.parse_args()

    kept_by_id = {}
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ok, pred = trace_is_correct(row.get("raw_candidate", ""), row["gold_answer"])
            if ok and row.get("id") not in kept_by_id:
                kept_by_id[row.get("id")] = {
                    "id": row.get("id"),
                    "problem": row["problem"],
                    "gold_answer": row["gold_answer"],
                    "reasoning_trace": reasoning_from_candidate(row["raw_candidate"]),
                    "raw_candidate": row["raw_candidate"],
                    "recovered_pred_answer": pred,
                }

    kept = list(kept_by_id.values())
    write_jsonl(args.output, kept)
    print(f"Recovered {len(kept)} examples to {args.output}")


if __name__ == "__main__":
    main()
