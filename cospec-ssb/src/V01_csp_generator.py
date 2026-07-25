"""Deterministic split-view CSP generator and independent enumerating solver."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from typing import Any, Literal


GENERATOR_VERSION = "v01.2"


@dataclass(frozen=True)
class CSPProblem:
    seed: int
    n_entities: int
    entities: tuple[str, ...]
    links: tuple[str, ...]
    target_entity: str
    entity_link_constraints: tuple[tuple[str, str], ...]
    slot_link_sequence: tuple[str, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_entities": self.n_entities,
            "entities": list(self.entities),
            "links": list(self.links),
            "target_entity": self.target_entity,
            "entity_link_constraints": [list(item) for item in self.entity_link_constraints],
            "slot_link_sequence": list(self.slot_link_sequence) if self.slot_link_sequence else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CSPProblem":
        sequence = payload.get("slot_link_sequence")
        return cls(
            seed=int(payload["seed"]), n_entities=int(payload["n_entities"]),
            entities=tuple(payload["entities"]), links=tuple(payload["links"]),
            target_entity=str(payload["target_entity"]),
            entity_link_constraints=tuple(tuple(item) for item in payload["entity_link_constraints"]),
            slot_link_sequence=tuple(sequence) if sequence is not None else None,
        )


@dataclass(frozen=True)
class Solution:
    entity_to_link: dict[str, str]
    slot_to_link: tuple[str, ...]
    entity_to_slot: dict[str, int]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    solution_count: int
    target_slot: int | None
    errors: tuple[str, ...]


def _opaque_tokens(rng: random.Random, prefix: str, count: int) -> tuple[str, ...]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    seen: set[str] = set()
    values: list[str] = []
    while len(values) < count:
        token = "".join(rng.choice(alphabet) for _ in range(5))
        value = f"{prefix}_{token}"
        if value not in seen:
            seen.add(value)
            values.append(value)
    rng.shuffle(values)
    return tuple(values)


def generate_csp_problem(seed: int, n_entities: int) -> CSPProblem:
    if n_entities not in (3, 4):
        raise ValueError("V01 supports exactly 3 or 4 entities.")
    block_size = n_entities * n_entities
    block_index, cell_index = divmod(seed, block_size)
    entity_variant, slot_variant = divmod(cell_index, n_entities)
    rng = random.Random((block_index << 8) ^ n_entities ^ 0xC05EEC)
    entities = _opaque_tokens(rng, "ENTITY", n_entities)
    links = _opaque_tokens(rng, "LINK", n_entities)
    target = rng.choice(entities)
    target_link = links[entity_variant]
    other_entities = [entity for entity in entities if entity != target]
    other_links = [link for link in links if link != target_link]
    constraints = tuple([(target, target_link), *zip(other_entities, other_links)])
    slot_links = tuple(links[(slot + slot_variant) % n_entities] for slot in range(n_entities))
    problem = CSPProblem(
        seed=seed, n_entities=n_entities, entities=entities, links=links,
        target_entity=target, entity_link_constraints=constraints,
        slot_link_sequence=slot_links,
    )
    validation = validate_unique_solution(problem)
    if not validation.valid:
        raise RuntimeError(f"Generated invalid CSP at seed {seed}: {validation.errors}")
    return problem


def solve_csp(
    problem: CSPProblem, partial_view: Literal["a", "b"] | None = None
) -> list[Solution]:
    if partial_view not in (None, "a", "b"):
        raise ValueError("partial_view must be None, 'a', or 'b'.")
    entity_constraints = dict(problem.entity_link_constraints) if partial_view != "b" else {}
    fixed_sequence = problem.slot_link_sequence if partial_view != "a" else None
    solutions: list[Solution] = []
    for entity_assignment in itertools.permutations(problem.links):
        entity_to_link = dict(zip(problem.entities, entity_assignment))
        if any(entity_to_link.get(entity) != link for entity, link in entity_constraints.items()):
            continue
        slot_candidates = [fixed_sequence] if fixed_sequence is not None else itertools.permutations(problem.links)
        for slot_to_link in slot_candidates:
            if slot_to_link is None or sorted(slot_to_link) != sorted(problem.links):
                continue
            link_to_slot = {link: slot for slot, link in enumerate(slot_to_link)}
            entity_to_slot = {entity: link_to_slot[link] for entity, link in entity_to_link.items()}
            solutions.append(Solution(entity_to_link, tuple(slot_to_link), entity_to_slot))
    return solutions


def canonicalize_problem(problem: CSPProblem) -> str:
    payload = {
        "version": GENERATOR_VERSION,
        "n_entities": problem.n_entities,
        "entities": sorted(problem.entities),
        "links": sorted(problem.links),
        "target_entity": problem.target_entity,
        "entity_link_constraints": sorted([list(item) for item in problem.entity_link_constraints]),
        "slot_link_sequence": list(problem.slot_link_sequence) if problem.slot_link_sequence else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def canonical_hash(problem: CSPProblem) -> str:
    return hashlib.sha256(canonicalize_problem(problem).encode()).hexdigest()


def validate_unique_solution(problem: CSPProblem) -> ValidationResult:
    errors: list[str] = []
    if len(set(problem.entities)) != problem.n_entities or len(set(problem.links)) != problem.n_entities:
        errors.append("entities and links must be unique")
    if problem.target_entity not in problem.entities:
        errors.append("target entity is not in the entity domain")
    solutions = solve_csp(problem) if not errors else []
    if len(solutions) != 1:
        errors.append(f"expected one solution, found {len(solutions)}")
    target_slot = solutions[0].entity_to_slot[problem.target_entity] if len(solutions) == 1 else None
    return ValidationResult(not errors, len(solutions), target_slot, tuple(errors))
