from src.V02_benchmark import FAMILIES, generate_block
from src.V02_modeling import extract_option, messages_for_mode, summarize_predictions
from scripts.V02_evaluate_split_vs_single import _shuffled_sources


def _row():
    return generate_block("arithmetic_constraint", "medium", 321)[0]


def test_prompt_modes_never_cross_partial_views():
    row = _row()
    assert messages_for_mode("view_a", row)[-1]["content"] == row["view_a"]
    assert messages_for_mode("view_b", row)[-1]["content"] == row["view_b"]
    assert messages_for_mode("split_a", row)[-1]["content"] == row["view_a"]
    assert messages_for_mode("split_b", row)[-1]["content"] == row["view_b"]
    assert messages_for_mode("full", row)[-1]["content"] == row["full_problem"]


def test_answer_parser_prefers_marked_final_answer():
    assert extract_option("OPTION_0 may fit.\nANSWER: OPTION_3") == "OPTION_3"
    assert extract_option("option-2") == "OPTION_2"
    assert extract_option("the third one") is None


def test_summary_reports_family_and_difficulty_slices():
    predictions = []
    for family in FAMILIES:
        row = generate_block(family, "easy", 44)[0]
        predictions.append({
            "family": family,
            "difficulty": "easy",
            "predicted_answer": row["gold_answer"],
            "correct": True,
        })
    summary = summarize_predictions(predictions)
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["overall"]["parse_rate"] == 1.0
    assert set(summary["by_family"]) == set(FAMILIES)


def test_shuffled_control_uses_a_different_block_in_the_same_stratum():
    rows = (
        generate_block("logic_grid", "hard", 100)
        + generate_block("logic_grid", "hard", 101)
    )
    source = _shuffled_sources(rows, 9)
    assert all(index != origin for index, origin in enumerate(source))
    assert all(rows[index]["family"] == rows[origin]["family"] for index, origin in enumerate(source))
    assert all(rows[index]["difficulty"] == rows[origin]["difficulty"] for index, origin in enumerate(source))
    assert all(rows[index]["block_id"] != rows[origin]["block_id"] for index, origin in enumerate(source))
