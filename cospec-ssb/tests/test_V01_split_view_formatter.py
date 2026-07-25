import json

import pytest

from scripts.V01_validate_dataset import read_jsonl_strict
from src.V01_csp_generator import canonical_hash, generate_csp_problem
from src.V01_split_view_formatter import audit_for_direct_leakage, format_split_view


def test_formatter_is_deterministic_and_does_not_leak():
    problem = generate_csp_problem(42, 4)
    first = format_split_view(problem, 99)
    second = format_split_view(problem, 99)
    assert first == second
    assert first.gold_answer not in first.view_a
    assert first.gold_answer not in first.view_b
    assert audit_for_direct_leakage(first) == []


def test_balanced_classes_and_nonoverlap_can_be_constructed():
    buckets = {index: [] for index in range(4)}
    seen: set[str] = set()
    for seed in range(1000):
        problem = generate_csp_problem(seed, 4)
        example = format_split_view(problem, seed)
        if len(buckets[example.answer_index]) < 5:
            buckets[example.answer_index].append(example)
            digest = canonical_hash(problem)
            assert digest not in seen
            seen.add(digest)
        if all(len(values) == 5 for values in buckets.values()):
            break
    assert {key: len(values) for key, values in buckets.items()} == {0: 5, 1: 5, 2: 5, 3: 5}


def test_malformed_jsonl_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"ok": True}) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        read_jsonl_strict(path)

