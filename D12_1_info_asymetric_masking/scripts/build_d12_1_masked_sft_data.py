#!/usr/bin/env python
"""Build SFT training data from D12.1 masked traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "D12_1_info_asymmetric_masking_sft"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT data for D12.1")
    parser.add_argument("--config", default="configs/d12_1_info_asymmetric.yaml")
    args = parser.parse_args()

    from src.data_utils import (
        load_config,
        read_jsonl,
        write_jsonl,
        reject_test_rows_for_training,
        require_dependencies,
    )

    require_dependencies("yaml")
    cfg = load_config(args.config)

    trace_path = cfg_get(cfg, "data", "filtered_trace_path", "data/filtered/D12_1/masked_traces.jsonl")
    train_dir = cfg_get(cfg, "data", "train_dir", "data/train/D12_1")

    traces = read_jsonl(trace_path)
    if not traces:
        raise SystemExit(
            f"No traces found at {trace_path}. Run bootstrap_d12_1_masked_traces.py first."
        )
    reject_test_rows_for_training(traces)

    agent_a_rows, agent_b_rows, synthesis_rows = [], [], []
    for t in traces:
        base = {
            "id": t["id"],
            "problem": t["problem"],
            "gold_answer": t["gold_answer"],
        }

        # Agent A partial view → contribution
        agent_a_rows.append({
            **base,
            "view": t["view_a"],
            "contribution": t["contrib_a"],
            "training_mode": "partial_a",
        })

        # Agent B partial view → contribution
        agent_b_rows.append({
            **base,
            "view": t["view_b"],
            "contribution": t["contrib_b"],
            "training_mode": "partial_b",
        })

        # Synthesizer: full problem + both contributions → joint solution + answer
        synthesis_rows.append({
            **base,
            "contrib_a": t["contrib_a"],
            "contrib_b": t["contrib_b"],
            "joint_solution": t["joint_solution"],
            "final_answer": t["final_answer"],
            "training_mode": "synthesis",
        })

    write_jsonl(f"{train_dir}/agent_a_partial.jsonl", agent_a_rows)
    write_jsonl(f"{train_dir}/agent_b_partial.jsonl", agent_b_rows)
    write_jsonl(f"{train_dir}/final_synthesis.jsonl", synthesis_rows)

    print(f"[D12.1] SFT data built:")
    print(f"  Agent A partial:    {len(agent_a_rows)} -> {train_dir}/agent_a_partial.jsonl")
    print(f"  Agent B partial:    {len(agent_b_rows)} -> {train_dir}/agent_b_partial.jsonl")
    print(f"  Final synthesis:    {len(synthesis_rows)} -> {train_dir}/final_synthesis.jsonl")


if __name__ == "__main__":
    main()
