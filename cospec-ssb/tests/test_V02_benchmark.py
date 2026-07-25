from collections import Counter, defaultdict

import pytest

from src.V02_benchmark import (
    DIFFICULTIES, FAMILIES, LABELS, generate_block, solve_spec,
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_each_block_has_unique_full_solutions_and_balanced_labels(family, difficulty):
    rows = generate_block(family, difficulty, 12345)
    assert len(rows) == 16
    assert len({row["sample_id"] for row in rows}) == 16
    assert Counter(row["gold_answer"] for row in rows) == Counter(
        {label: 4 for label in LABELS}
    )
    for row in rows:
        assert solve_spec(family, row["metadata"]["task_spec"]) == row["gold_answer"]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_partial_view_equivalence_class_contains_all_labels(family):
    rows = generate_block(family, "hard", 98765)
    for field in ("view_a", "view_b"):
        groups = defaultdict(Counter)
        for row in rows:
            groups[row[field]][row["gold_answer"]] += 1
        assert len(groups) == 4
        assert all(counter == Counter({label: 1 for label in LABELS}) for counter in groups.values())


def test_full_problem_is_exact_composition_without_marked_answer_leakage():
    row = generate_block("logic_grid", "medium", 77)[0]
    assert row["full_problem"] == (
        "PRIVATE VIEW A:\n" + row["view_a"]
        + "\n\nPRIVATE VIEW B:\n" + row["view_b"]
    )
    assert "ANSWER:" not in row["view_a"].upper()
    assert "ANSWER:" not in row["view_b"].upper()
