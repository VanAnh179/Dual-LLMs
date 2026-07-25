"""Deterministic V03 benchmark with explicit IID and held-out OOD profiles."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any, Callable


GENERATOR_VERSION = "v03.0"
FAMILIES = (
    "relational_csp",
    "logic_grid",
    "arithmetic_constraint",
    "candidate_verification",
)
DIFFICULTIES = ("easy", "medium", "hard")
PROFILES = ("train", "iid", "ood")
LABELS = tuple(f"OPTION_{index}" for index in range(4))
BLOCK_SIZE = 16


def _token(seed: int, namespace: str, index: int) -> str:
    raw = f"{seed}:{namespace}:{index}".encode()
    suffix = hashlib.blake2s(raw, digest_size=4).hexdigest().upper()
    return f"{namespace}_{suffix}"


def _profile_settings(seed: int, profile: str, difficulty: str) -> dict[str, int | str]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    base_depth = {"easy": 1, "medium": 2, "hard": 2}[difficulty]
    if profile == "ood":
        return {
            "style": 4 + seed % 4,
            "depth": base_depth + 2,
            "noise": {"easy": 2, "medium": 4, "hard": 6}[difficulty],
            "template_partition": "held_out",
        }
    return {
        "style": seed % 4,
        "depth": base_depth,
        "noise": {"easy": 0, "medium": 1, "hard": 3}[difficulty],
        "template_partition": "development",
    }


def _noise(seed: int, side: str, count: int, held_out: bool) -> str:
    if count == 0:
        return ""
    if held_out:
        lines = [
            f"* Archive annotation {_token(seed, f'ARCHIVE_{side}', index)} has no bearing on the query."
            for index in range(count)
        ]
        return "\n\nNon-operative archive annotations:\n" + "\n".join(lines)
    lines = [
        f"- {_token(seed, f'NOTE_{side}', index)} is an unrelated audit marker."
        for index in range(count)
    ]
    return "\n\nIrrelevant audit notes:\n" + "\n".join(lines)


def _shuffled(lines: list[str], seed: int) -> str:
    result = list(lines)
    random.Random(seed).shuffle(result)
    return "\n".join(result)


def _chain_lines(
    seed: int,
    namespace: str,
    starts: list[str],
    ends: list[str],
    depth: int,
    held_out: bool,
) -> list[str]:
    lines: list[str] = []
    for item_index, (start, end) in enumerate(zip(starts, ends)):
        nodes = [start]
        nodes.extend(
            _token(seed, f"{namespace}_RELAY_{level}", item_index)
            for level in range(max(0, depth - 1))
        )
        nodes.append(end)
        for left, right in zip(nodes, nodes[1:]):
            if held_out:
                lines.append(f"* Ledger pointer: {left} => {right}.")
            else:
                lines.append(f"- {left} maps to {right}.")
    return lines


def _relational(seed: int, difficulty: str, profile: str, variant_a: int, variant_b: int):
    settings = _profile_settings(seed, profile, difficulty)
    held_out = profile == "ood"
    depth = int(settings["depth"])
    entities = [_token(seed, "ENTITY", index) for index in range(4)]
    links = [_token(seed, "LINK", index) for index in range(4)]
    mapped_links = [links[(index + variant_a) % 4] for index in range(4)]
    slot_to_link = [links[(slot - variant_b) % 4] for slot in range(4)]
    target = entities[0]
    facts = _chain_lines(
        seed, "RELATION", entities, mapped_links, depth, held_out
    )
    if held_out:
        view_a = (
            "Trace each registry pointer transitively; intermediate relay names are not answers.\n"
            f"{_shuffled(facts, seed + 17 * variant_a)}"
            f"{_noise(seed, 'A', int(settings['noise']), True)}\n\n"
            f"Locate the final registry token reached from {target}."
        )
        view_b = (
            "A separate shelf ledger lists terminal registry tokens in positional order:\n"
            + " | ".join(slot_to_link)
            + f"{_noise(seed, 'B', int(settings['noise']), True)}\n\n"
            f"Which OPTION is the shelf position for {target}?\n"
            "Use OPTION_0 for the first position through OPTION_3 for the fourth."
        )
    else:
        view_a = (
            "Entity registry. Follow mappings until a LINK token is reached:\n"
            f"{_shuffled(facts, seed + 17 * variant_a)}"
            f"{_noise(seed, 'A', int(settings['noise']), False)}\n\n"
            f"Target entity: {target}. Which OPTION is its slot?"
        )
        view_b = (
            "The four slots contain these terminal LINK tokens from left to right:\n"
            + " -> ".join(slot_to_link)
            + f"{_noise(seed, 'B', int(settings['noise']), False)}\n\n"
            f"Target entity: {target}. Which OPTION is its slot?\n"
            "OPTION_0=first, OPTION_1=second, OPTION_2=third, OPTION_3=fourth."
        )
    spec = {
        "target_entity": target,
        "entity_to_link": dict(zip(entities, mapped_links)),
        "slot_to_link": slot_to_link,
    }
    return view_a, view_b, spec, settings


def _logic_grid(seed: int, difficulty: str, profile: str, variant_a: int, variant_b: int):
    settings = _profile_settings(seed, profile, difficulty)
    held_out = profile == "ood"
    depth = int(settings["depth"])
    people = [_token(seed, "PERSON", index) for index in range(4)]
    pets = [_token(seed, "PET", index) for index in range(4)]
    people_by_house = [people[(position - variant_a) % 4] for position in range(4)]
    pets_by_house = [pets[(position + variant_b) % 4] for position in range(4)]
    people_aliases = [_token(seed, "RESIDENT_ALIAS", index) for index in range(4)]
    pet_aliases = [_token(seed, "PET_ALIAS", index) for index in range(4)]
    person_alias = dict(zip(people, people_aliases))
    pet_alias = dict(zip(pets, pet_aliases))
    person_resolution = _chain_lines(
        seed, "PERSON_LOGIC", people, people_aliases, depth, held_out
    )
    pet_resolution = _chain_lines(
        seed, "PET_LOGIC", pets, pet_aliases, depth, held_out
    )
    displayed_people = [person_alias[person] for person in people_by_house]
    displayed_pets = [pet_alias[pet] for pet in pets_by_house]
    target = people[0]
    if held_out:
        person_clues = [
            f"* {displayed_people[0]} occupies the western endpoint.",
            f"* Exactly one home separates {displayed_people[0]} and {displayed_people[2]}.",
            f"* {displayed_people[1]} is directly west of {displayed_people[2]}.",
            f"* {displayed_people[3]} occupies the eastern endpoint.",
            f"* {displayed_people[2]} is directly west of {displayed_people[3]}.",
        ]
        pet_clues = [
            f"* {displayed_pets[3]} is kept at the eastern endpoint.",
            f"* Exactly one home separates {displayed_pets[1]} and {displayed_pets[3]}.",
            f"* {displayed_pets[0]} is directly west of {displayed_pets[1]}.",
            f"* {displayed_pets[1]} is directly west of {displayed_pets[2]}.",
        ]
        intro_a = "Infer the unique west-to-east resident order from non-adjacent and endpoint clues:"
        intro_b = "Independently infer the west-to-east pet order:"
    else:
        person_clues = [f"- {displayed_people[0]} lives in the leftmost house."]
        person_clues.extend(
            f"- {displayed_people[position + 1]} lives immediately right of {displayed_people[position]}."
            for position in range(3)
        )
        pet_clues = [f"- The leftmost house keeps {displayed_pets[0]}."]
        pet_clues.extend(
            f"- The house keeping {displayed_pets[position + 1]} is immediately right of the house keeping {displayed_pets[position]}."
            for position in range(3)
        )
        intro_a = "Four people live in four adjacent houses. Person clues:"
        intro_b = "Four pets belong to four adjacent houses. Pet clues:"
    options = "\n".join(f"- OPTION_{index}: {pet}" for index, pet in enumerate(pets))
    view_a = (
        "Resolve resident names through the alias ledger:\n"
        f"{_shuffled(person_resolution, seed + 23 * variant_a)}\n\n"
        f"{intro_a}\n{_shuffled(person_clues, seed + 29 * variant_a)}"
        f"{_noise(seed, 'A', int(settings['noise']), held_out)}\n\n"
        f"Which pet belongs to {target}? Return its OPTION label."
    )
    view_b = (
        "Resolve pet names through the alias ledger:\n"
        f"{_shuffled(pet_resolution, seed + 27 * variant_b)}\n\n"
        f"{intro_b}\n{_shuffled(pet_clues, seed + 31 * variant_b)}\n\n"
        f"Candidate pets:\n{options}"
        f"{_noise(seed, 'B', int(settings['noise']), held_out)}\n\n"
        f"Which pet belongs to {target}? Return its OPTION label."
    )
    spec = {
        "target_person": target,
        "people_by_house": people_by_house,
        "pets_by_house": pets_by_house,
        "option_to_pet": {f"OPTION_{index}": pet for index, pet in enumerate(pets)},
    }
    return view_a, view_b, spec, settings


def _arithmetic(seed: int, difficulty: str, profile: str, variant_a: int, variant_b: int):
    settings = _profile_settings(seed, profile, difficulty)
    held_out = profile == "ood"
    depth = int(settings["depth"])
    entities = [_token(seed, "ACCOUNT", index) for index in range(4)]
    quantities = [_token(seed, "QTY", index) for index in range(4)]
    mapped_quantities = [quantities[(index + variant_a) % 4] for index in range(4)]
    target = entities[0]
    base = 3 + seed % 7
    inputs = [base + 3 * index for index in range(4)]
    multiplier = 2 + seed % 3
    offset = 5 + seed % 11
    quantity_values = {
        quantities[index]: inputs[(index + variant_b) % 4] for index in range(4)
    }
    mappings = _chain_lines(
        seed, "ARITH", entities, mapped_quantities, depth, held_out
    )
    equations = []
    adjustment = 2 + seed % 5
    for quantity in quantities:
        value = quantity_values[quantity]
        if held_out:
            equations.append(
                f"* (({quantity} + {adjustment}) * 2) - {adjustment} = "
                f"{(value + adjustment) * 2 - adjustment}."
            )
        else:
            equations.append(f"- {quantity} + {adjustment} = {value + adjustment}.")
    scores = [multiplier * value + offset for value in inputs]
    options = "\n".join(
        f"- OPTION_{index}: score {score}" for index, score in enumerate(scores)
    )
    view_a = (
        "Resolve the account-to-quantity mapping transitively:\n"
        f"{_shuffled(mappings, seed + 37 * variant_a)}"
        f"{_noise(seed, 'A', int(settings['noise']), held_out)}\n\n"
        f"Find the score for {target} and return its OPTION label."
    )
    view_b = (
        "Solve the quantity equations:\n"
        f"{_shuffled(equations, seed + 41 * variant_b)}\n\n"
        f"For a solved quantity q, score = {multiplier} * q + {offset}.\n"
        f"Candidate scores:\n{options}"
        f"{_noise(seed, 'B', int(settings['noise']), held_out)}\n\n"
        f"Find the score for {target} and return its OPTION label."
    )
    spec = {
        "target_entity": target,
        "entity_to_quantity": dict(zip(entities, mapped_quantities)),
        "quantity_values": quantity_values,
        "multiplier": multiplier,
        "offset": offset,
        "option_to_score": {
            f"OPTION_{index}": score for index, score in enumerate(scores)
        },
    }
    return view_a, view_b, spec, settings


def _verification(seed: int, difficulty: str, profile: str, variant_a: int, variant_b: int):
    settings = _profile_settings(seed, profile, difficulty)
    held_out = profile == "ood"
    candidates = [_token(seed, "CANDIDATE", index) for index in range(4)]
    checks = [_token(seed, "CHECK", index) for index in range(4)]
    common = _token(seed, "COMMON_CHECK", 0)
    extra = [
        _token(seed, "EXTRA_CHECK", index)
        for index in range(max(0, int(settings["depth"]) - 1))
    ]
    requirements = {
        candidates[index]: [common, *extra, checks[(index - variant_a) % 4]]
        for index in range(4)
    }
    statuses = {
        common: "PASS",
        **{check: "PASS" for check in extra},
        **{
            check: ("PASS" if index == variant_b else "FAIL")
            for index, check in enumerate(checks)
        },
    }
    if held_out:
        candidate_lines = [
            f"* OPTION_{index} ({candidate}) is admissible iff all of "
            f"{', '.join(requirements[candidate])} are cleared."
            for index, candidate in enumerate(candidates)
        ]
        status_lines = [
            f"* Evidence marks {check} as {'cleared' if status == 'PASS' else 'rejected'}."
            for check, status in statuses.items()
        ]
    else:
        candidate_lines = [
            f"- OPTION_{index} ({candidate}) requires "
            f"{', '.join(requirements[candidate])}."
            for index, candidate in enumerate(candidates)
        ]
        status_lines = [f"- {check}: {status}." for check, status in statuses.items()]
    view_a = (
        "Planner candidate requirements:\n"
        f"{_shuffled(candidate_lines, seed + 43 * variant_a)}"
        f"{_noise(seed, 'A', int(settings['noise']), held_out)}\n\n"
        "A candidate is valid only when every required check passes. Which OPTION is valid?"
    )
    view_b = (
        "Verifier evidence ledger:\n"
        f"{_shuffled(status_lines, seed + 47 * variant_b)}"
        f"{_noise(seed, 'B', int(settings['noise']), held_out)}\n\n"
        "A candidate is valid only when every required check passes. Which OPTION is valid?"
    )
    spec = {
        "candidates": candidates,
        "requirements": requirements,
        "statuses": statuses,
    }
    return view_a, view_b, spec, settings


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
        inverse = {value: key for key, value in spec["option_to_pet"].items()}
        return inverse[pet]
    if family == "arithmetic_constraint":
        quantity = spec["entity_to_quantity"][spec["target_entity"]]
        value = spec["quantity_values"][quantity]
        score = spec["multiplier"] * value + spec["offset"]
        inverse = {value: key for key, value in spec["option_to_score"].items()}
        return inverse[score]
    if family == "candidate_verification":
        valid = [
            candidate
            for candidate, requirements in spec["requirements"].items()
            if all(spec["statuses"][check] == "PASS" for check in requirements)
        ]
        if len(valid) != 1:
            raise ValueError(f"Expected one valid candidate, found {len(valid)}")
        return f"OPTION_{spec['candidates'].index(valid[0])}"
    raise ValueError(f"Unsupported family: {family}")


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_block(
    family: str,
    difficulty: str,
    block_seed: int,
    profile: str = "train",
) -> list[dict[str, Any]]:
    if family not in GENERATORS:
        raise ValueError(f"Unsupported family: {family}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    rows = []
    for variant_a in range(4):
        for variant_b in range(4):
            view_a, view_b, spec, settings = GENERATORS[family](
                block_seed, difficulty, profile, variant_a, variant_b
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
                "profile": profile,
                "block_seed": block_seed,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "spec": spec,
            }
            digest = canonical_hash(identity)
            rows.append({
                "sample_id": f"v03-{family}-{digest[:20]}",
                "generator_version": GENERATOR_VERSION,
                "distribution_profile": profile,
                "template_partition": settings["template_partition"],
                "render_style": settings["style"],
                "reasoning_depth": settings["depth"],
                "family": family,
                "difficulty": difficulty,
                "block_id": f"{profile}-{family}-{block_seed}",
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
