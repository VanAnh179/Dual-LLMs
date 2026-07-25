#!/usr/bin/env python
"""Prepare training data for baseline adapter retraining by extracting reasoning_trace from raw_answer."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import read_jsonl, write_jsonl, ensure_parent

rows = read_jsonl("data/raw/train.jsonl")
out = []
for r in rows:
    raw = r.get("raw_answer", "")
    parts = raw.split("####")
    reasoning = parts[0].strip() if parts else raw
    out.append({
        "id": r["id"],
        "problem": r["problem"],
        "gold_answer": r["gold_answer"],
        "raw_answer": r["raw_answer"],
        "reasoning_trace": reasoning,
    })

write_jsonl("data/filtered/baseline_retrain.jsonl", out)
print(f"Wrote {len(out)} rows to data/filtered/baseline_retrain.jsonl")
sample = out[0]["reasoning_trace"][:100]
print(f"Sample reasoning_trace: {sample}")
