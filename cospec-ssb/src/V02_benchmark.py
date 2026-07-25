"""Deterministic multi-family forced-cooperation benchmark generation."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any, Callable


GENERATOR_VERSION = "v02.0"
FAMILIES = (
    "relational_csp",
    "logic_grid",
    "arithmetic_constraint",
    "candidate_verification",
)
DIFFICULTIES = ("easy", "medium", "hard")
LABELS = tuple(f"OPTION_{index}" for index in range(4))
BLOCK_SIZE = 16


def _token(seed: int, namespace: str, index: int) -> str:
    raw = f"{seed}:{namespace}:{index}".encode()
    suffix = hashlib.blake2s(raw, digest_size=4).hexdigest().upper()
    return f"{namespace}_{suffix}"


def _noise(seed: int, side: str, difficulty: str) -> str:
    count = {"easy": 0, "medium": 2, "hard": 4}[difficulty]
    if count == 0:
        return ""
    lines = [
        f"- {_token(seed, f'NOTE_{side}', index)} is an unrelated audit marker."
        for index in range(count)
    ]
    return "\n\nIrrelevant audit notes:\n" + "\n".join(lines)


def _shuffled_lines(lines: list[str], seed: int) -> str:
    ordered = list(lines)
    random.Random(seed).shuffle(ordered)
    return "\n".join(ordered)


def _relational(seed: int, difficulty: str, variant_a: int, variant_b: int):
    entities = [_token(seed, "ENTITY", index) for index in range(4)]
    links = [_token(seed, "LINK", index) for index in range(4)]
    entity_to_link = {
        entities[index]: links[(index + variant_a) % 4] for index in range(4)
    }
    slot_to_link = [links[(slot - variant_b) % 4] for slot in range(4)]
    target = entities[0]
    facts = [
        f"- {entity} is associated with {entity_to_link[entity]}."
        for entity in entities
    ]
    view_a = (
        "Entity-link registry:\n"
        f"{_shuffled_lines(facts, seed + 17 * variant_a)}"
        f"{_noise(seed, 'A', difficulty)}\n\n"
        f"Target entity: {target}. Which OPTION is its left-to-right slot?"
    )
    view_b = (
        "The four slots contain these links from left to right:\n"
        + " -> ".join(slot_to_link)
        + f"{_noise(seed, 'B', difficulty)}\n\n"
        f"Target entity: {target}. Which OPTION is its left-to-right slot?\n"
        "OPTION_0=first, OPTION_1=second, OPTION_2=third, OPTION_3=fourth."
    )
    spec = {
        "target_entity": target,
        "entity_to_link": entity_to_link,
        "slot_to_link": slot_to_link,
    }
    return view_a, view_b, spec


def _logic_grid(seed: int, difficulty: str, variant_a: int, variant_b: int):
    people = [_token(seed, "PERSON", index) for index in range(4)]
    pets = [_token(seed, "PET", index) for index in range(4)]
    people_by_house = [people[(position - variant_a) % 4] for position in range(4)]
    pets_by_house = [pets[(position + variant_b) % 4] for position in range(4)]
    target = people[0]
    person_clues = [f"- {people_by_house[0]} lives in the leftmost house."]
    person_clues.extend(
        f"- {people_by_house[position + 1]} lives immediately right of "
        f"{people_by_house[position]}."
        for position in range(3)
    )
    pet_clues = [f"- The leftmost house keeps {pets_by_house[0]}."]
    pet_clues.extend(
        f"- The house keeping {pets_by_house[position + 1]} is immediately right of "
        f"the house keeping {pets_by_house[position]}."
        for position in range(3)
    )
    options = "\n".join(f"- OPTION_{index}: {pet}" for index, pet in enumerate(pets))
    view_a = (
        "Four people live in four adjacent houses. Person clues:\n"
        f"{_shuffled_lines(person_clues, seed + 29 * variant_a)}"
        f"{_noise(seed, 'A', difficulty)}\n\n"
        f"Which pet belongs to {target}? Return its OPTION label."
    )
    view_b = (
        "Four pets belong to four adjacent houses. Pet clues:\n"
        f"{_shuffled_lines(pet_clues, seed + 31 * variant_b)}\n\n"
        f"Candidate pets:\n{options}"
        f"{_noise(seed, 'B', difficulty)}\n\n"
        f"Which pet belongs to {target}? Return its OPTION label."
    )
    spec = {
        "target_person": target,
        "people_by_house": people_by_house,
        "pets_by_house": pets_by_house,
        "option_to_pet": {f"OPTION_{index}": pet for index, pet in enumerate(pets)},
    }
    return view_a, view_b, spec


def _arithmetic(seed: int, difficulty: str, variant_a: int, variant_b: int):
    entities = [_token(seed, "ACCOUNT", index) for index in range(4)]
    quantities = [_token(seed, "QTY", index) for index in range(4)]
    entity_to_quantity = {
        entities[index]: quantities[(index + variant_a) % 4] for index in range(4)
    }
    target = entities[0]
    base = 3 + (seed % 7)
    inputs = [base + 3 * index for index in range(4)]
    multiplier = 2 + (seed % 3)
    offset = 5 + (seed % 11)
    quantity_values = {
        quantities[index]: inputs[(index + variant_b) % 4] for index in range(4)
    }
    scores = [multiplier * value + offset for value in inputs]
    equations = []
    equation_offset = 2 + (seed % 5)
    for quantity in quantities:
        value = quantity_values[quantity]
        equations.append(
            f"- {quantity} + {equation_offset} = {value + equation_offset}."
        )
    mappings = [
        f"- {entity} uses quantity symbol {entity_to_quantity[entity]}."
        for entity in entities
    ]
    options = "\n".join(
        f"- OPTION_{index}: score {score}" for index, score in enumerate(scores)
    )
    view_a = (
        "Account-to-quantity constraints:\n"
        f"{_shuffled_lines(mappings, seed + 37 * variant_a)}"
        f"{_noise(seed, 'A', difficulty)}\n\n"
        f"Find the score for {target} and return its OPTION label."
    )
    view_b = (
        "Quantity equations:\n"
        f"{_shuffled_lines(equations, seed + 41 * variant_b)}\n\n"
        f"For any solved quantity q, score = {multiplier} * q + {offset}.\n"
        f"Candidate scores:\n{options}"
        f"{_noise(seed, 'B', difficulty)}\n\n"
        f"Find the score for {target} and return its OPTION label."
    )
    spec = {
        "target_entity": target,
        "entity_to_quantity": entity_to_quantity,
        "quantity_values": quantity_values,
        "multiplier": multiplier,
        "offset": offset,
        "option_to_score": {
            f"OPTION_{index}": score for index, score in enumerate(scores)
        },
    }
    return view_a, view_b, spec


def _verification(seed: int, difficulty: str, variant_a: int, variant_b: int):
    candidates = [_token(seed, "CANDIDATE", index) for index in range(4)]
    checks = [_token(seed, "CHECK", index) for index in range(4)]
    common = _token(seed, "COMMON_CHECK", 0)
    requirements = {
        candidates[index]: [common, checks[(index - variant_a) % 4]]
        for index in range(4)
    }
    statuses = {
        common: "PASS",
        **{
            check: ("PASS" if index == variant_b else "FAIL")
            for index, check in enumerate(checks)
        },
    }
    candidate_lines = [
        f"- OPTION_{index} ({candidate}) requires {requirements[candidate][0]} "
        f"and {requirements[candidate][1]}."
        for index, candidate in enumerate(candidates)
    ]
    status_lines = [f"- {check}: {status}." for check, status in statuses.items()]
    view_a = (
        "Planner candidate requirements:\n"
        f"{_shuffled_lines(candidate_lines, seed + 43 * variant_a)}"
        f"{_noise(seed, 'A', difficulty)}\n\n"
        "A candidate is valid only when every required check passes. Which OPTION is valid?"
    )
    view_b = (
        "Verifier evidence ledger:\n"
        f"{_shuffled_lines(status_lines, seed + 47 * variant_b)}"
        f"{_noise(seed, 'B', difficulty)}\n\n"
        "A candidate is valid only when every required check passes. Which OPTION is valid?"
    )
    spec = {
        "candidates": candidates,
        "requirements": requirements,
        "statuses": statuses,
    }
    return view_a, view_b, spec


GENERATORS: dict[str, Callable] = {
    "relational_csp": _relational,
    "logic_grid": _logic_grid,
    "arithmetic_constraint": _arithmetic,
    "candidate_verification": _verification,
}


def solve_spec(family: str, spec: dict[str, Any]) -> str:
    if family == "relational_csp":
        link = spec["entity_to_link"][spec["target_entity"]]
        return f"OPTION_{spec['slot_to_link'].index(link)}"
    if family == "logic_grid":
        position = spec["people_by_house"].index(spec["target_person"])
        pet = spec["pets_by_house"][position]
        inverse = {pet_name: option for option, pet_name in spec["option_to_pet"].items()}
        return inverse[pet]
    if family == "arithmetic_constraint":
        quantity = spec["entity_to_quantity"][spec["target_entity"]]
        value = spec["quantity_values"][quantity]
        score = spec["multiplier"] * value + spec["offset"]
        inverse = {value: option for option, value in spec["option_to_score"].items()}
        return inverse[score]
    if family == "candidate_verification":
        valid = [
            candidate for candidate, requirements in spec["requirements"].items()
            if all(spec["statuses"][check] == "PASS" for check in requirements)
        ]
        if len(valid) != 1:
            raise ValueError(f"Expected one valid candidate, found {len(valid)}")
        return f"OPTION_{spec['candidates'].index(valid[0])}"
    raise ValueError(f"Unsupported family: {family}")


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_block(family: str, difficulty: str, block_seed: int) -> list[dict[str, Any]]:
    if family not in GENERATORS:
        raise ValueError(f"Unsupported family: {family}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    rows = []
    for variant_a in range(4):
        for variant_b in range(4):
            view_a, view_b, spec = GENERATORS[family](
                block_seed, difficulty, variant_a, variant_b
            )
            gold = solve_spec(family, spec)
            expected = f"OPTION_{(variant_a + variant_b) % 4}"
            if gold != expected:
                raise RuntimeError(
                    f"Latin-square construction failed: expected {expected}, got {gold}"
                )
            identity = {
                "version": GENERATOR_VERSION,
                "family": family,
                "difficulty": difficulty,
                "block_seed": block_seed,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "spec": spec,
            }
            digest = canonical_hash(identity)
            rows.append({
                "sample_id": f"v02-{family}-{digest[:20]}",
                "generator_version": GENERATOR_VERSION,
                "family": family,
                "difficulty": difficulty,
                "block_id": f"{family}-{block_seed}",
                "variant_a": variant_a,
                "variant_b": variant_b,
                "view_a": view_a,
                "view_b": view_b,
                "full_problem": (
                    "PRIVATE VIEW A:\n" + view_a
                    + "\n\nPRIVATE VIEW B:\n" + view_b
                ),
                "gold_answer": gold,
                "answer_index": int(gold.rsplit("_", 1)[1]),
                "random_baseline": 0.25,
                "metadata": {
                    "block_seed": block_seed,
                    "canonical_hash": digest,
                    "task_spec": spec,
                },
            })
    counts = Counter(row["gold_answer"] for row in rows)
    if counts != Counter({label: 4 for label in LABELS}):
        raise RuntimeError(f"Block label balancing failed: {counts}")
    return rows
