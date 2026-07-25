#!/usr/bin/env python
"""Post-hoc D11.2 contribution analysis."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import read_jsonl, write_json


EXPERIMENT_NAME = "D11_2_latent_collaborative_sft"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def numbers(text: str) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", text))


def operator_counts(text: str) -> dict[str, int]:
    return {op: text.count(op) for op in ["+", "-", "*", "/", "="]}


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = tokens(a)
    b_tokens = tokens(b)
    if not a_tokens and not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=f"outputs/{EXPERIMENT_NAME}/generations/eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default=f"outputs/{EXPERIMENT_NAME}/metrics/contribution_analysis.json",
    )
    parser.add_argument(
        "--output-md",
        default=f"outputs/{EXPERIMENT_NAME}/analysis/interesting_examples.md",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    if not rows:
        raise SystemExit(f"No predictions found in {args.predictions}. Run evaluate_collaborative_agents.py first.")

    overlaps = []
    a_lengths = []
    b_lengths = []
    new_number_counts = []
    operator_totals = {"agent_A": Counter(), "agent_B": Counter()}
    a_then_b_correct_a_alone_wrong = []
    short_a_b_correct = []
    near_duplicate = []

    for row in rows:
        a = row.get("agent_A_contribution", "")
        b = row.get("agent_B_contribution", "")
        overlap = lexical_overlap(a, b)
        a_nums = numbers(a)
        b_nums = numbers(b)
        new_nums = b_nums - a_nums
        overlaps.append(overlap)
        a_lengths.append(len(a))
        b_lengths.append(len(b))
        new_number_counts.append(len(new_nums))
        operator_totals["agent_A"].update(operator_counts(a))
        operator_totals["agent_B"].update(operator_counts(b))

        if "is_agent_A_alone_correct" in row:
            a_alone_wrong = not row["is_agent_A_alone_correct"]
        else:
            a_alone_wrong = row.get("agent_A_alone_final_answer") != row.get("gold_answer")
        if row.get("is_A_then_B_correct") and a_alone_wrong:
            a_then_b_correct_a_alone_wrong.append(row)
        if len(a.split()) <= 25 and row.get("is_A_then_B_correct"):
            short_a_b_correct.append(row)
        if overlap >= 0.85:
            near_duplicate.append(row)

    report = {
        "num_examples": len(rows),
        "average_agent_A_contribution_length": avg(a_lengths),
        "average_agent_B_contribution_length": avg(b_lengths),
        "average_lexical_overlap": avg(overlaps),
        "average_new_numbers_introduced_by_B": avg(new_number_counts),
        "operator_counts": {
            "agent_A": dict(operator_totals["agent_A"]),
            "agent_B": dict(operator_totals["agent_B"]),
        },
        "A_then_B_correct_agent_A_alone_wrong_count": len(a_then_b_correct_a_alone_wrong),
        "short_A_contribution_B_correct_count": len(short_a_b_correct),
        "near_duplicate_contribution_count": len(near_duplicate),
    }
    write_json(args.output_json, report)

    lines = [
        "# D11.2 Interesting Examples",
        "",
        "## A_then_B correct while Agent A alone differs from gold",
        "",
    ]
    for row in a_then_b_correct_a_alone_wrong[:10]:
        lines.extend(
            [
                f"### {row.get('id')}",
                "",
                f"Gold answer: `{row.get('gold_answer')}`",
                "",
                f"Agent A alone final: `{row.get('agent_A_alone_final_answer')}`",
                "",
                "Agent A contribution:",
                "",
                row.get("agent_A_contribution", "").strip(),
                "",
                "A_then_B prediction:",
                "",
                row.get("A_then_B_prediction", "").strip(),
                "",
            ]
        )
    lines.extend(["## Near-duplicate contributions", ""])
    for row in near_duplicate[:10]:
        lines.extend(
            [
                f"### {row.get('id')}",
                "",
                "Agent A contribution:",
                "",
                row.get("agent_A_contribution", "").strip(),
                "",
                "Agent B contribution:",
                "",
                row.get("agent_B_contribution", "").strip(),
                "",
            ]
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
