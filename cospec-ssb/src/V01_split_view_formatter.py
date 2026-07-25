"""Leakage-aware text formatting for V01 CSP examples."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.V01_csp_generator import (
    GENERATOR_VERSION, CSPProblem, canonical_hash, solve_csp, validate_unique_solution,
)


TEMPLATES = (
    ("Registry facts:\n{facts}", "Reading positions from left to right, the link IDs are:\n{sequence}"),
    ("Entity-link records:\n{facts}", "The ordered bays, from first through last, contain:\n{sequence}"),
    ("Association ledger:\n{facts}", "In positional order, beginning with the first slot:\n{sequence}"),
    ("Link assignments:\n{facts}", "The left-to-right link sequence is:\n{sequence}"),
)


@dataclass(frozen=True)
class SplitViewExample:
    sample_id: str
    generator_version: str
    seed: int
    n_entities: int
    template_family: str
    full_problem: str
    view_a: str
    view_b: str
    target: str
    gold_answer: str
    answer_index: int
    random_baseline: float
    solution: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _render(problem: CSPProblem, seed: int) -> tuple[str, str, str]:
    rng = random.Random(seed ^ 0x51A7)
    template_index = rng.randrange(len(TEMPLATES))
    facts = list(problem.entity_link_constraints)
    rng.shuffle(facts)
    fact_lines = [f"- {entity} is associated with {link}." for entity, link in facts]
    sequence = list(problem.slot_link_sequence or ())
    sequence_text = " -> ".join(sequence)
    target = f"Which position contains {problem.target_entity}?"
    view_a = f"{TEMPLATES[template_index][0].format(facts=chr(10).join(fact_lines))}\n\nQuery: {target}"
    view_b = f"{TEMPLATES[template_index][1].format(sequence=sequence_text)}\n\nQuery: {target}"
    return view_a, view_b, f"template_{template_index}"


def format_split_view(problem: CSPProblem, seed: int) -> SplitViewExample:
    validation = validate_unique_solution(problem)
    if not validation.valid or validation.target_slot is None:
        raise ValueError(f"Cannot format invalid CSP: {validation.errors}")
    view_a, view_b, template = _render(problem, seed)
    digest = canonical_hash(problem)
    solution = solve_csp(problem)[0]
    answer_index = validation.target_slot
    target = f"Which position contains {problem.target_entity}?"
    full_problem = f"{view_a}\n\nAdditional registry:\n{view_b}"
    return SplitViewExample(
        sample_id=f"{GENERATOR_VERSION}-{digest[:20]}", generator_version=GENERATOR_VERSION,
        seed=problem.seed, n_entities=problem.n_entities, template_family=template,
        full_problem=full_problem, view_a=view_a, view_b=view_b, target=target,
        gold_answer=f"SLOT_{answer_index}", answer_index=answer_index,
        random_baseline=1.0 / problem.n_entities,
        solution={
            "entity_to_link": solution.entity_to_link,
            "slot_to_link": list(solution.slot_to_link),
            "entity_to_slot": solution.entity_to_slot,
        },
        metadata={
            "num_constraints": len(problem.entity_link_constraints) + 1,
            "unique_solution_count": validation.solution_count,
            "canonical_hash": digest,
            "format_seed": seed,
            "problem_spec": problem.to_dict(),
        },
    )


def audit_for_direct_leakage(example: SplitViewExample) -> list[str]:
    findings: list[str] = []
    forbidden = (
        example.gold_answer, str(example.solution), "answer_index", "canonical_hash",
        "generator_version", "template_family",
    )
    for view_name, text in (("view_a", example.view_a), ("view_b", example.view_b)):
        for token in forbidden:
            if token and token in text:
                findings.append(f"{view_name}: forbidden value/token {token!r} is rendered")
    problem = CSPProblem.from_dict(example.metadata["problem_spec"])
    for view_name, partial in (("view_a", "a"), ("view_b", "b")):
        possible = {solution.entity_to_slot[problem.target_entity] for solution in solve_csp(problem, partial)}
        if possible != set(range(problem.n_entities)):
            findings.append(f"{view_name}: target slots are not uniformly underdetermined: {sorted(possible)}")
    return findings

