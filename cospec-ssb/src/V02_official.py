"""Adapters from official reasoning benchmarks to the V02 split-view schema."""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from typing import Any, Iterable

from src.V02_benchmark import LABELS


ADAPTER_VERSION = "v02o.4"
OFFICIAL_FAMILIES = (
    "relational_csp",
    "logic_grid",
    "arithmetic_constraint",
    "candidate_verification",
)

CLUTRR_RELATIONS = (
    "aunt", "son-in-law", "grandfather", "brother", "sister", "father",
    "mother", "grandmother", "uncle", "daughter-in-law", "grandson",
    "granddaughter", "father-in-law", "mother-in-law", "nephew", "son",
    "daughter", "niece", "husband", "wife", "sister-in-law",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _option_map(
    gold: str, candidates: Iterable[str], desired_index: int, seed: int, namespace: str
) -> dict[str, str]:
    distractors = sorted({str(value) for value in candidates if str(value) != gold})
    if len(distractors) < 3:
        raise ValueError("At least three distinct distractors are required.")
    rng = _stable_rng(seed, namespace)
    selected = rng.sample(distractors, 3)
    values = list(selected)
    values.insert(desired_index, gold)
    return {label: value for label, value in zip(LABELS, values)}


def _render_options(options: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {value}" for label, value in options.items())


def _neutralize_protocol_markers(text: str) -> str:
    """Prevent source text from impersonating the benchmark answer delimiter."""
    return re.sub(
        r"\bANSWER\s*:", "[source answer]", str(text), flags=re.IGNORECASE
    )


def _row(
    *,
    family: str,
    source_name: str,
    source_id: str,
    source_split: str,
    difficulty: str,
    view_a: str,
    view_b: str,
    options: dict[str, str],
    gold_value: str,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    view_a = _neutralize_protocol_markers(view_a)
    view_b = _neutralize_protocol_markers(view_b)
    gold_answer = next(label for label, value in options.items() if value == gold_value)
    identity = {
        "adapter_version": ADAPTER_VERSION,
        "family": family,
        "source_name": source_name,
        "source_id": source_id,
        "source_split": source_split,
        "options": options,
        "gold_value": gold_value,
    }
    digest = canonical_hash(identity)
    return {
        "sample_id": f"v02o-{family}-{digest[:20]}",
        "generator_version": ADAPTER_VERSION,
        "family": family,
        "difficulty": difficulty,
        "block_id": f"v02o-{family}-{digest[:12]}",
        "variant_a": 0,
        "variant_b": 0,
        "view_a": view_a,
        "view_b": view_b,
        "full_problem": f"PRIVATE VIEW A:\n{view_a}\n\nPRIVATE VIEW B:\n{view_b}",
        "gold_answer": gold_answer,
        "answer_index": int(gold_answer.rsplit("_", 1)[1]),
        "random_baseline": 0.25,
        "metadata": {
            "canonical_hash": digest,
            "official_eval_only": True,
            "source_name": source_name,
            "source_id": source_id,
            "source_split": source_split,
            "source_metadata": source_metadata,
            "option_values": options,
            "gold_value_sha256": hashlib.sha256(gold_value.encode()).hexdigest(),
        },
    }


def _sentence_facts(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=\[?[A-Z])", normalized)
        if part.strip()
    ]


def adapt_clutrr(
    records: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        facts = _sentence_facts(str(record.get("clean_story") or record.get("story") or ""))
        if len(facts) < 2:
            continue
        source_id = str(record.get("id") or canonical_hash(record)[:20])
        query = record.get("query")
        if isinstance(query, str):
            try:
                parsed_query = ast.literal_eval(query)
            except (SyntaxError, ValueError):
                parsed_query = None
            if isinstance(parsed_query, (list, tuple)) and len(parsed_query) == 2:
                query = parsed_query
        if isinstance(query, (list, tuple)) and len(query) == 2:
            query_text = f"What is {query[1]}'s family relation to {query[0]}?"
        else:
            query_text = f"Infer the queried family relation for {query}."
        target = str(record["target_text"])
        task_name = str(record.get("task_name", ""))
        match = re.search(r"\.(\d+)$", task_name)
        chain_length = int(match.group(1)) if match else len(facts)
        difficulty = "easy" if chain_length <= 3 else "medium" if chain_length <= 5 else "hard"
        candidates.append((source_id, record, facts, query_text, target, difficulty))

    candidates.sort(key=lambda item: canonical_hash([seed, item[0]]))
    rows = []
    for index, (source_id, record, facts, query_text, target, difficulty) in enumerate(
        candidates[:count]
    ):
        options = _option_map(
            target, CLUTRR_RELATIONS, index % 4, seed, f"clutrr:{source_id}"
        )
        left = facts[::2]
        right = facts[1::2]
        common = f"{query_text}\nCandidate relations:\n{_render_options(options)}"
        rows.append(_row(
            family="relational_csp",
            source_name="CLUTRR",
            source_id=source_id,
            source_split="test",
            difficulty=difficulty,
            view_a="Family facts (subset A):\n- " + "\n- ".join(left) + "\n\n" + common,
            view_b="Family facts (subset B):\n- " + "\n- ".join(right) + "\n\n" + common,
            options=options,
            gold_value=target,
            source_metadata={"task_name": record.get("task_name"), "query": query},
        ))
    return rows


def _split_zebra_puzzle(puzzle: str) -> tuple[str, list[str]]:
    text = puzzle.replace("\r\n", "\n").strip()
    marker = re.search(r"\n##\s*Clues?\s*:?\s*\n", text, flags=re.IGNORECASE)
    if marker:
        introduction = text[:marker.start()].strip()
        clue_text = text[marker.end():]
    else:
        introduction = text
        clue_text = ""
    clues = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in clue_text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", line)
    ]
    return introduction, clues


def adapt_zebralogic(
    records: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        solution = record.get("solution") or {}
        header = list(solution.get("header") or [])
        table = list(solution.get("rows") or [])
        introduction, clues = _split_zebra_puzzle(str(record.get("puzzle") or ""))
        if len(header) < 3 or len(table) < 4 or len(clues) < 2:
            continue
        source_id = str(record.get("id") or canonical_hash(record)[:20])
        candidates.append((source_id, record, introduction, clues, header, table))
    candidates.sort(key=lambda item: canonical_hash([seed, item[0]]))

    rows = []
    for index, (source_id, record, introduction, clues, header, table) in enumerate(
        candidates[:count]
    ):
        rng = _stable_rng(seed, f"zebra:{source_id}")
        column_index = 1 + rng.randrange(len(header) - 1)
        row_index = rng.randrange(len(table))
        gold = str(table[row_index][column_index])
        column_values = [str(row[column_index]) for row in table]
        options = _option_map(
            gold, column_values, index % 4, seed, f"zebra-options:{source_id}"
        )
        query = (
            f"For house {table[row_index][0]}, which {header[column_index]} is correct?\n"
            f"Candidates:\n{_render_options(options)}"
        )
        size = len(table) * (len(header) - 1)
        difficulty = "easy" if size <= 16 else "medium" if size <= 25 else "hard"
        rows.append(_row(
            family="logic_grid",
            source_name="ZebraLogicBench",
            source_id=source_id,
            source_split="test",
            difficulty=difficulty,
            view_a=f"{introduction}\n\nClues (subset A):\n- "
            + "\n- ".join(clues[::2]) + "\n\n" + query,
            view_b=f"{introduction}\n\nClues (subset B):\n- "
            + "\n- ".join(clues[1::2]) + "\n\n" + query,
            options=options,
            gold_value=gold,
            source_metadata={
                "size": record.get("size"),
                "target_house": str(table[row_index][0]),
                "target_column": str(header[column_index]),
            },
        ))
    return rows


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _numeric_distractors(target: str) -> list[str]:
    try:
        value = int(target.replace(",", "").strip())
    except ValueError:
        value = int(float(target.replace(",", "").strip()))
    deltas = (1, -1, 2, -2, 5, -5, 10, -10)
    return [str(value + delta) for delta in deltas if value + delta != value]


def adapt_bbh_arithmetic(
    records: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = []
    for index, record in enumerate(records):
        prompt = str(record.get("input") or "")
        target = str(record.get("target") or "").strip()
        expression = prompt
        if "=" in expression:
            expression = expression.rsplit("=", 1)[0]
        matches = list(_NUMBER.finditer(expression))
        if len(matches) < 2:
            continue
        source_id = f"multistep_arithmetic_two:{index}"
        candidates.append((source_id, expression.strip(), target, matches))
    candidates.sort(key=lambda item: canonical_hash([seed, item[0]]))

    rows = []
    for index, (source_id, expression, target, _) in enumerate(candidates[:count]):
        values: list[str] = []

        def replace(match: re.Match[str]) -> str:
            values.append(match.group(0))
            return f"N{len(values) - 1}"

        template = _NUMBER.sub(replace, expression)
        known_a = [f"- N{i} = {value}" for i, value in enumerate(values) if i % 2 == 0]
        known_b = [f"- N{i} = {value}" for i, value in enumerate(values) if i % 2 == 1]
        options = _option_map(
            target, _numeric_distractors(target), index % 4, seed, f"bbh:{source_id}"
        )
        query = f"Return the value of the expression.\nCandidates:\n{_render_options(options)}"
        difficulty = "easy" if len(values) <= 5 else "medium" if len(values) <= 9 else "hard"
        rows.append(_row(
            family="arithmetic_constraint",
            source_name="BIG-Bench-Hard/multistep_arithmetic_two",
            source_id=source_id,
            source_split="test",
            difficulty=difficulty,
            view_a=f"Expression template:\n{template}\n\nKnown quantities (A):\n"
            + "\n".join(known_a) + "\n\n" + query,
            view_b=f"Expression template:\n{template}\n\nKnown quantities (B):\n"
            + "\n".join(known_b) + "\n\n" + query,
            options=options,
            gold_value=target,
            source_metadata={"num_numeric_leaves": len(values)},
        ))
    return rows


def _chosen_prefix(steps: list[dict[str, Any]], stop: int) -> list[str]:
    prefix = []
    for step in steps[:stop]:
        chosen = step.get("chosen_completion")
        completions = step.get("completions") or []
        if chosen is not None and 0 <= int(chosen) < len(completions):
            prefix.append(str(completions[int(chosen)].get("text") or ""))
        elif step.get("human_completion"):
            prefix.append(str(step["human_completion"]))
    return [text for text in prefix if text]


def adapt_prm800k(
    records: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = []
    for record_index, record in enumerate(records):
        if record.get("is_quality_control_question") or record.get(
            "is_initial_screening_question"
        ):
            continue
        question = record.get("question") or {}
        problem = str(question.get("problem") or "").strip()
        reference = str(question.get("ground_truth_answer") or "").strip()
        steps = list((record.get("label") or {}).get("steps") or [])
        if not problem or not reference:
            continue
        for step_index, step in enumerate(steps):
            completions = [
                completion for completion in (step.get("completions") or [])
                if not completion.get("flagged") and completion.get("rating") in (-1, 0, 1)
            ]
            positives = [item for item in completions if item.get("rating") == 1]
            nonpositives = [item for item in completions if item.get("rating") != 1]
            if len(positives) != 1:
                continue
            positive_text = str(positives[0].get("text") or "").strip()
            unique_nonpositives: dict[str, dict[str, Any]] = {}
            for completion in nonpositives:
                text = str(completion.get("text") or "").strip()
                if text and text != positive_text:
                    unique_nonpositives.setdefault(text, completion)
            if not positive_text or len(unique_nonpositives) < 3:
                continue
            source_id = f"phase2_test:{record_index}:step:{step_index}"
            candidates.append((
                source_id, problem, reference, steps, step_index,
                positives[0], list(unique_nonpositives.values()),
            ))
    candidates.sort(key=lambda item: canonical_hash([seed, item[0]]))

    rows = []
    for index, item in enumerate(candidates[:count]):
        source_id, problem, reference, steps, step_index, positive, negatives = item
        rng = _stable_rng(seed, f"prm:{source_id}")
        chosen_negatives = rng.sample(negatives, 3)
        gold = str(positive["text"]).strip()
        options = _option_map(
            gold,
            [str(candidate["text"]).strip() for candidate in chosen_negatives],
            index % 4,
            seed,
            f"prm-options:{source_id}",
        )
        prefix = _chosen_prefix(steps, step_index)[-2:]
        prefix_text = "\n".join(f"- {text}" for text in prefix) or "- No prior step."
        view_a = (
            "Planner trajectory immediately before the next step:\n"
            f"{prefix_text}\n\nCandidate next steps:\n{_render_options(options)}\n\n"
            "Which candidate is a correct and useful next reasoning step?"
        )
        view_b = (
            f"Problem to verify:\n{problem}\n\n"
            f"Reference final result: {reference}\n\n"
            "Use this verifier context to judge the planner's candidate next steps."
        )
        difficulty = "easy" if len(problem) < 180 else "medium" if len(problem) < 360 else "hard"
        rows.append(_row(
            family="candidate_verification",
            source_name="PRM800K/phase2_test",
            source_id=source_id,
            source_split="phase2_test",
            difficulty=difficulty,
            view_a=view_a,
            view_b=view_b,
            options=options,
            gold_value=gold,
            source_metadata={
                "record_index": int(source_id.split(":")[1]),
                "step_index": step_index,
                "human_rating": 1,
            },
        ))
    return rows


def balanced_interleave(
    by_family: dict[str, list[dict[str, Any]]], rows_per_family: int
) -> list[dict[str, Any]]:
    missing = {
        family: len(by_family.get(family, []))
        for family in OFFICIAL_FAMILIES
        if len(by_family.get(family, [])) < rows_per_family
    }
    if missing:
        raise ValueError(
            f"Insufficient official examples for requested balance: {missing}; "
            f"required={rows_per_family}"
        )
    rows = []
    for index in range(rows_per_family):
        for family in OFFICIAL_FAMILIES:
            rows.append(by_family[family][index])
    return rows


def family_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["family"])] += 1
    return dict(sorted(counts.items()))
