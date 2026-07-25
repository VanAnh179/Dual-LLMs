from collections import Counter

from src.V02_official import (
    adapt_bbh_arithmetic, adapt_clutrr, adapt_prm800k, adapt_zebralogic,
)


def _assert_valid_row(row, family):
    assert row["family"] == family
    assert row["metadata"]["official_eval_only"] is True
    assert row["gold_answer"] in {"OPTION_0", "OPTION_1", "OPTION_2", "OPTION_3"}
    assert len(set(row["metadata"]["option_values"].values())) == 4
    assert row["full_problem"] == (
        "PRIVATE VIEW A:\n" + row["view_a"]
        + "\n\nPRIVATE VIEW B:\n" + row["view_b"]
    )
    assert "ANSWER:" not in row["view_a"].upper()
    assert "ANSWER:" not in row["view_b"].upper()


def test_clutrr_adapter_splits_facts_and_balances_option_positions():
    records = [
        {
            "id": f"clutrr-{index}",
            "clean_story": (
                "[Ava] is [Bea]'s mother. [Bea] is [Cy]'s sister. "
                "[Cy] is [Dan]'s father."
            ),
            "query": ["Ava", "Dan"],
            "target_text": "grandson",
            "task_name": "task_1.3",
        }
        for index in range(4)
    ]
    rows = adapt_clutrr(records, 4, 11)
    assert len(rows) == 4
    assert Counter(row["gold_answer"] for row in rows) == Counter({
        "OPTION_0": 1, "OPTION_1": 1, "OPTION_2": 1, "OPTION_3": 1,
    })
    for row in rows:
        _assert_valid_row(row, "relational_csp")


def test_clutrr_adapter_parses_csv_string_query():
    record = {
        "id": "csv-clutrr",
        "clean_story": "[Ava] is [Bea]'s mother. [Bea] is [Cy]'s sister.",
        "query": "('Ava', 'Cy')",
        "target_text": "granddaughter",
        "task_name": "task_1.2",
    }
    row = adapt_clutrr([record], 1, 17)[0]
    assert "What is Cy's family relation to Ava?" in row["full_problem"]


def test_zebralogic_adapter_uses_solution_cell_without_rendering_solution_table():
    record = {
        "id": "lgp-test-fixture",
        "size": "4*2",
        "puzzle": (
            "There are four houses.\n## Clues\n"
            "1. Ava is in house 1.\n"
            "2. Bea is immediately right of Ava.\n"
            "3. Cy keeps the dog.\n"
            "4. Dan is in house 4."
        ),
        "solution": {
            "header": ["House", "Name", "Pet"],
            "rows": [
                ["1", "Ava", "cat"],
                ["2", "Bea", "dog"],
                ["3", "Cy", "fish"],
                ["4", "Dan", "bird"],
            ],
        },
    }
    rows = adapt_zebralogic([record], 1, 7)
    assert len(rows) == 1
    _assert_valid_row(rows[0], "logic_grid")
    assert '"rows"' not in rows[0]["full_problem"]


def test_bbh_adapter_masks_complementary_numeric_leaves():
    rows = adapt_bbh_arithmetic(
        [{"input": "Calculate ((2 + 3) * (4 - 1)) =", "target": "15"}],
        1,
        9,
    )
    assert len(rows) == 1
    row = rows[0]
    _assert_valid_row(row, "arithmetic_constraint")
    assert "N0 = 2" in row["view_a"]
    assert "N1 = 3" in row["view_b"]


def test_prm_adapter_uses_human_step_ratings_as_gold():
    completions = [
        {"text": "Correct next step.", "rating": 1, "flagged": None},
        {"text": "Wrong step A.", "rating": -1, "flagged": None},
        {"text": "No progress.", "rating": 0, "flagged": None},
        {"text": "Wrong step B.", "rating": -1, "flagged": None},
    ]
    record = {
        "question": {
            "problem": "Compute 2 + 2.",
            "ground_truth_answer": "4",
        },
        "label": {
            "steps": [{
                "completions": completions,
                "chosen_completion": 0,
                "human_completion": None,
            }]
        },
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
    }
    rows = adapt_prm800k([record], 1, 13)
    assert len(rows) == 1
    row = rows[0]
    _assert_valid_row(row, "candidate_verification")
    assert row["metadata"]["option_values"][row["gold_answer"]] == "Correct next step."


def test_prm_adapter_skips_steps_without_three_distinct_negative_texts():
    record = {
        "question": {
            "problem": "Compute 3 + 3.",
            "ground_truth_answer": "6",
        },
        "label": {
            "steps": [{
                "completions": [
                    {"text": "Correct.", "rating": 1, "flagged": None},
                    {"text": "Repeated.", "rating": -1, "flagged": None},
                    {"text": "Repeated.", "rating": 0, "flagged": None},
                    {"text": "Another.", "rating": -1, "flagged": None},
                ],
                "chosen_completion": 0,
                "human_completion": None,
            }]
        },
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
    }
    assert adapt_prm800k([record], 1, 13) == []


def test_prm_adapter_neutralizes_source_answer_marker_in_rendered_views():
    record = {
        "question": {
            "problem": "Compute 5 + 5.",
            "ground_truth_answer": "10",
        },
        "label": {
            "steps": [{
                "completions": [
                    {"text": "ANSWER: 10.", "rating": 1, "flagged": None},
                    {"text": "Use 9.", "rating": -1, "flagged": None},
                    {"text": "Use 11.", "rating": -1, "flagged": None},
                    {"text": "Stop early.", "rating": 0, "flagged": None},
                ],
                "chosen_completion": 0,
                "human_completion": None,
            }]
        },
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
    }
    row = adapt_prm800k([record], 1, 21)[0]
    _assert_valid_row(row, "candidate_verification")
    assert "[source answer] 10." in row["view_a"]
    assert row["metadata"]["option_values"][row["gold_answer"]] == "ANSWER: 10."
