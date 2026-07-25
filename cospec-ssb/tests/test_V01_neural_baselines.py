from src.V01_neural_baselines import (
    build_messages, evaluate_gates, extract_slot, summarize_predictions,
)


def _row():
    return {
        "view_a": "A_ONLY_SENTINEL",
        "view_b": "B_ONLY_SENTINEL",
        "full_problem": "FULL_SENTINEL",
        "gold_answer": "SLOT_2",
    }


def test_each_mode_receives_only_its_requested_field():
    row = _row()
    for mode, sentinel in (
        ("view_a", "A_ONLY_SENTINEL"),
        ("view_b", "B_ONLY_SENTINEL"),
        ("full_problem", "FULL_SENTINEL"),
    ):
        user_content = build_messages(mode, row)[-1]["content"]
        assert user_content == sentinel


def test_extract_slot_prefers_explicit_final_answer():
    assert extract_slot("Maybe SLOT_0.\nANSWER: SLOT_3") == "SLOT_3"
    assert extract_slot("slot-2") == "SLOT_2"
    assert extract_slot("second position") is None


def test_metrics_count_unparseable_outputs_as_incorrect():
    predictions = [
        {"gold_answer": "SLOT_0", "extracted_answer": "SLOT_0", "correct": True},
        {"gold_answer": "SLOT_1", "extracted_answer": None, "correct": False},
    ]
    result = summarize_predictions(predictions)
    assert result["accuracy"] == 0.5
    assert result["parse_rate"] == 0.5
    assert result["accuracy_on_parsed"] == 1.0


def test_gate_is_smoke_only_until_all_full_test_modes_exist():
    cfg = {
        "expected_test_examples": 100,
        "gates": {"partial_view_max_accuracy": 0.28, "full_problem_min_accuracy": 0.8},
    }
    metrics = {"results": {"view_a": {"num_examples": 4, "accuracy": 0.25}}}
    assert evaluate_gates(metrics, cfg)["status"] == "SMOKE_ONLY"


def test_full_gate_requires_chance_partial_views_and_solvable_full_problem():
    cfg = {
        "expected_test_examples": 100,
        "gates": {"partial_view_max_accuracy": 0.28, "full_problem_min_accuracy": 0.8},
    }
    metrics = {"results": {
        "view_a": {"num_examples": 100, "accuracy": 0.25, "accuracy_on_parsed": 0.25},
        "view_b": {"num_examples": 100, "accuracy": 0.25, "accuracy_on_parsed": 0.25},
        "full_problem": {"num_examples": 100, "accuracy": 0.95, "accuracy_on_parsed": 0.95},
    }}
    assert evaluate_gates(metrics, cfg)["status"] == "PASS"


def test_partial_gate_does_not_pass_only_because_outputs_are_unparseable():
    cfg = {
        "expected_test_examples": 100,
        "gates": {"partial_view_max_accuracy": 0.28, "full_problem_min_accuracy": 0.8},
    }
    metrics = {"results": {
        "view_a": {"num_examples": 100, "accuracy": 0.25, "accuracy_on_parsed": 0.25},
        "view_b": {"num_examples": 100, "accuracy": 0.10, "accuracy_on_parsed": 0.50},
        "full_problem": {"num_examples": 100, "accuracy": 0.95, "accuracy_on_parsed": 0.95},
    }}
    assert evaluate_gates(metrics, cfg)["partial_view_gate"] == "FAIL"
