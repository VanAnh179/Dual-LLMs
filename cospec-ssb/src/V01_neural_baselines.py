"""Prompting, parsing, metrics, and reporting for V01 neural baselines."""
from __future__ import annotations

import re
from typing import Any

from src.data_utils import project_path


VALID_MODES = ("view_a", "view_b", "full_problem")
LABELS = tuple(f"SLOT_{index}" for index in range(4))

SYSTEM_PROMPT = """You solve position lookup tasks. There are exactly 4 slots, zero-indexed:
- The 1st link in the sequence is at SLOT_0
- The 2nd link in the sequence is at SLOT_1
- The 3rd link in the sequence is at SLOT_2
- The 4th link in the sequence is at SLOT_3
Think step-by-step. Use only the information given. If information is missing, guess.
End with exactly: ANSWER: SLOT_X (where X is 0, 1, 2, or 3)."""


def build_messages(mode: str, row: dict[str, Any]) -> list[dict[str, str]]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if mode == "full_problem":
        user_shot = (
            "Link assignments:\n"
            "- ENTITY_ALPHA is associated with LINK_P.\n"
            "- ENTITY_BETA is associated with LINK_Q.\n"
            "- ENTITY_GAMMA is associated with LINK_R.\n"
            "- ENTITY_DELTA is associated with LINK_S.\n"
            "Query: Which position contains ENTITY_GAMMA?\n\n"
            "Additional registry:\n"
            "The left-to-right link sequence is:\n"
            "LINK_S -> LINK_P -> LINK_R -> LINK_Q\n"
            "Query: Which position contains ENTITY_GAMMA?"
        )
        assistant_shot = (
            "Reasoning:\n"
            "1. Target: ENTITY_GAMMA.\n"
            "2. ENTITY_GAMMA is associated with LINK_R.\n"
            "3. The sequence is: LINK_S (SLOT_0) -> LINK_P (SLOT_1) -> LINK_R (SLOT_2) -> LINK_Q (SLOT_3).\n"
            "4. Looking up LINK_R in the annotated sequence above: LINK_R (SLOT_2).\n"
            "ANSWER: SLOT_2"
        )
    elif mode == "view_a":
        user_shot = (
            "Link assignments:\n"
            "- ENTITY_ALPHA is associated with LINK_P.\n"
            "- ENTITY_BETA is associated with LINK_Q.\n"
            "- ENTITY_GAMMA is associated with LINK_R.\n"
            "- ENTITY_DELTA is associated with LINK_S.\n"
            "Query: Which position contains ENTITY_GAMMA?"
        )
        assistant_shot = (
            "Reasoning:\n"
            "1. Target: ENTITY_GAMMA.\n"
            "2. ENTITY_GAMMA is associated with LINK_R.\n"
            "3. No link sequence is provided, so I cannot determine the slot.\n"
            "4. Guessing randomly.\n"
            "ANSWER: SLOT_0"
        )
    else:  # view_b
        user_shot = (
            "The left-to-right link sequence is:\n"
            "LINK_S -> LINK_P -> LINK_R -> LINK_Q\n"
            "Query: Which position contains ENTITY_GAMMA?"
        )
        assistant_shot = (
            "Reasoning:\n"
            "1. Target: ENTITY_GAMMA.\n"
            "2. The sequence is: LINK_S (SLOT_0) -> LINK_P (SLOT_1) -> LINK_R (SLOT_2) -> LINK_Q (SLOT_3).\n"
            "3. I do not know which link ENTITY_GAMMA is associated with.\n"
            "4. Guessing randomly.\n"
            "ANSWER: SLOT_1"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_shot},
        {"role": "assistant", "content": assistant_shot},
        {"role": "user", "content": str(row[mode])},
    ]


def format_prompt(tokenizer, mode: str, row: dict[str, Any]) -> str:
    messages = build_messages(mode, row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    ) + "\n\nASSISTANT:"


def extract_slot(text: str) -> str | None:
    explicit = re.findall(r"ANSWER\s*:\s*SLOT[_\s-]?([0-3])\b", text, flags=re.IGNORECASE)
    if explicit:
        return f"SLOT_{explicit[-1]}"
    # Fallback to ANSWER: X (numeric slot)
    explicit_num = re.findall(r"ANSWER\s*:\s*([0-3])\b", text, flags=re.IGNORECASE)
    if explicit_num:
        return f"SLOT_{explicit_num[-1]}"
    mentions = re.findall(r"\bSLOT[_\s-]?([0-3])\b", text, flags=re.IGNORECASE)
    return f"SLOT_{mentions[-1]}" if mentions else None


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty prediction set.")
    correct = [int(bool(row["correct"])) for row in rows]
    per_label_f1 = []
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    for row in rows:
        gold = str(row["gold_answer"])
        predicted = row.get("extracted_answer")
        if predicted in label_to_index:
            confusion[label_to_index[gold]][label_to_index[str(predicted)]] += 1
    for label in LABELS:
        true_positive = sum(
            1 for row in rows
            if row["gold_answer"] == label and row.get("extracted_answer") == label
        )
        false_positive = sum(
            1 for row in rows
            if row["gold_answer"] != label and row.get("extracted_answer") == label
        )
        false_negative = sum(
            1 for row in rows
            if row["gold_answer"] == label and row.get("extracted_answer") != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_label_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    parsed_rows = [row for row in rows if row.get("extracted_answer") in LABELS]
    parsed_accuracy = (
        sum(int(bool(row["correct"])) for row in parsed_rows) / len(parsed_rows)
        if parsed_rows else None
    )
    return {
        "num_examples": len(rows),
        "accuracy": sum(correct) / len(correct),
        "macro_f1": sum(per_label_f1) / len(per_label_f1),
        "parse_rate": len(parsed_rows) / len(rows),
        "accuracy_on_parsed": parsed_accuracy,
        "confusion_matrix": confusion,
        "labels": list(LABELS),
    }


def evaluate_gates(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    expected = int(cfg["expected_test_examples"])
    results = metrics["results"]
    complete = all(
        mode in results and int(results[mode]["num_examples"]) == expected
        for mode in VALID_MODES
    )
    if not complete:
        return {
            "status": "SMOKE_ONLY",
            "reason": f"All three modes must contain exactly {expected} examples.",
        }
    partial_limit = float(cfg["gates"]["partial_view_max_accuracy"])
    full_minimum = float(cfg["gates"]["full_problem_min_accuracy"])
    partial_pass = all(
        results[mode]["accuracy"] <= partial_limit
        and results[mode].get("accuracy_on_parsed") is not None
        and results[mode]["accuracy_on_parsed"] <= partial_limit
        for mode in ("view_a", "view_b")
    )
    full_pass = results["full_problem"]["accuracy"] >= full_minimum
    return {
        "status": "PASS" if partial_pass and full_pass else "FAIL",
        "partial_view_gate": "PASS" if partial_pass else "FAIL",
        "full_problem_sanity": "PASS" if full_pass else "FAIL",
        "partial_view_max_accuracy": partial_limit,
        "full_problem_min_accuracy": full_minimum,
        "partial_view_gate_metric": "accuracy_and_accuracy_on_parsed",
    }


def write_report(metrics: dict[str, Any], cfg: dict[str, Any]) -> None:
    result_rows = []
    for mode in VALID_MODES:
        result = metrics.get("results", {}).get(mode, {})
        result_rows.append(
            f"| {mode} | {result.get('num_examples', 'PENDING')} | "
            f"{result.get('accuracy', 'PENDING')} | {result.get('macro_f1', 'PENDING')} | "
            f"{result.get('parse_rate', 'PENDING')} | "
            f"{result.get('accuracy_on_parsed', 'PENDING')} |"
        )
    gate = metrics.get("gate", {"status": "PENDING"})
    results = metrics.get("results", {})
    interpretation = "Results are pending."
    if all(mode in results for mode in VALID_MODES):
        best_partial = max(results["view_a"]["accuracy"], results["view_b"]["accuracy"])
        cooperation_gain = results["full_problem"]["accuracy"] - best_partial
        interpretation = (
            f"The full problem exceeds the best raw partial-view accuracy by "
            f"{cooperation_gain:.2f}. View B's raw accuracy is "
            f"{results['view_b']['accuracy']:.2f} at parse rate "
            f"{results['view_b']['parse_rate']:.2f}; among parsed responses its accuracy is "
            f"{results['view_b'].get('accuracy_on_parsed', 0.0):.2f}. The conditional value is "
            "the appropriate diagnostic for distinguishing chance performance from abstention."
        )
    content = f"""# V01 Neural Baseline Results

## Protocol

The same frozen instruction model is evaluated with deterministic greedy decoding on three input
conditions: View A alone, View B alone, and the matched full problem. No V01 example is used for
training or prompt selection. Partial-view runs test single-view solvability; the full-problem run
checks that the task itself is solvable by the selected model.

Model: `{metrics.get('model_name', cfg['model_name'])}`

| Condition | N | Accuracy | Macro-F1 | Parse rate | Accuracy on parsed |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(result_rows)}

Random baseline: 0.25

## Interpretation

{interpretation}

## Gate

```yaml
status: {gate.get('status', 'PENDING')}
partial_view_gate: {gate.get('partial_view_gate', 'PENDING')}
full_problem_sanity: {gate.get('full_problem_sanity', 'PENDING')}
partial_view_max_accuracy: {cfg['gates']['partial_view_max_accuracy']}
full_problem_min_accuracy: {cfg['gates']['full_problem_min_accuracy']}
partial_view_gate_metric: accuracy_and_accuracy_on_parsed
```

`SMOKE_ONLY` means fewer than all {cfg['expected_test_examples']} test examples or fewer than all
three conditions were evaluated; it must not be used as a scientific conclusion.
"""
    project_path(cfg["output"]["report_path"]).write_text(content, encoding="utf-8")
