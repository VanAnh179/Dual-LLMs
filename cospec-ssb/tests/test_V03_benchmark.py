from collections import Counter, defaultdict

import pytest

from src.V03_benchmark import (
    DIFFICULTIES, FAMILIES, LABELS, PROFILES, generate_block, solve_spec,
)
from src.V02_modeling import select_nested_training_rows


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_v03_blocks_are_solved_balanced_and_partial_views_are_uninformative(
    profile, family, difficulty
):
    rows = generate_block(family, difficulty, 12345, profile)
    assert len(rows) == 16
    assert Counter(row["gold_answer"] for row in rows) == Counter(
        {label: 4 for label in LABELS}
    )
    for row in rows:
        assert solve_spec(family, row["metadata"]["task_spec"]) == row["gold_answer"]
    for field in ("view_a", "view_b"):
        groups = defaultdict(Counter)
        for row in rows:
            groups[row[field]][row["gold_answer"]] += 1
        assert len(groups) == 4
        assert all(
            counter == Counter({label: 1 for label in LABELS})
            for counter in groups.values()
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_v03_ood_templates_and_depth_are_held_out(family):
    train = generate_block(family, "hard", 77, "train")
    ood = generate_block(family, "hard", 88, "ood")
    assert {row["template_partition"] for row in train} == {"development"}
    assert {row["template_partition"] for row in ood} == {"held_out"}
    assert min(row["reasoning_depth"] for row in ood) > max(
        row["reasoning_depth"] for row in train
    )


def test_v03_learning_curve_prefix_is_family_balanced_and_nested():
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for block_index in range(3):
            rows.extend(
                generate_block(
                    family, "medium", 1000 + family_index * 10 + block_index
                )
            )
    first = select_nested_training_rows(rows, 64, 42)
    second = select_nested_training_rows(rows, 128, 42)
    assert Counter(row["family"] for row in first) == Counter(
        {family: 16 for family in FAMILIES}
    )
    assert Counter(row["family"] for row in second) == Counter(
        {family: 32 for family in FAMILIES}
    )
    assert {row["sample_id"] for row in first} <= {
        row["sample_id"] for row in second
    }
