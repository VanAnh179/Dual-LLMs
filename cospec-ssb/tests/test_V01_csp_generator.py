from dataclasses import replace

from src.V01_csp_generator import (
    canonicalize_problem, generate_csp_problem, solve_csp, validate_unique_solution,
)


def test_same_seed_is_identical_and_different_seeds_are_diverse():
    first = generate_csp_problem(42, 4)
    assert canonicalize_problem(first) == canonicalize_problem(generate_csp_problem(42, 4))
    hashes = {canonicalize_problem(generate_csp_problem(seed, 4)) for seed in range(20)}
    assert len(hashes) == 20


def test_every_sample_has_one_full_solution_and_all_partial_answers():
    for seed in range(30):
        problem = generate_csp_problem(seed, 3 + seed % 2)
        result = validate_unique_solution(problem)
        assert result.valid and result.solution_count == 1
        for partial in ("a", "b"):
            answers = {item.entity_to_slot[problem.target_entity] for item in solve_csp(problem, partial)}
            assert answers == set(range(problem.n_entities))


def test_removing_slot_sequence_constraint_creates_multiple_solutions():
    problem = generate_csp_problem(7, 4)
    mutated = replace(problem, slot_link_sequence=None)
    assert len(solve_csp(mutated)) > 1

