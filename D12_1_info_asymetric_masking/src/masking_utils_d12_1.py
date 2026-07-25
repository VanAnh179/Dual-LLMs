"""Number masking utilities for Info-Asymmetric experiments."""

from __future__ import annotations

import re


# Matches integers, decimals, dollar amounts, comma-separated numbers.
# Avoids partial word matches via lookaround.
NUMBER_RE = re.compile(r"(?<![a-zA-Z])\$?\d[\d,]*(?:\.\d+)?(?![a-zA-Z])")


def find_numbers(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, matched_string) for every number in *text*."""
    return [(m.start(), m.end(), m.group()) for m in NUMBER_RE.finditer(text)]


def split_question(problem: str) -> tuple[str, str]:
    """Split a GSM8K problem into (context, question).

    The *question* is the last sentence that contains '?'.
    If no '?' exists the last sentence is used as the question.
    Numbers in the question part must NEVER be masked.
    """
    sentences = re.split(r"(?<=[.?!])\s+", problem.strip())
    if not sentences:
        return problem, ""

    # Walk backwards to find the last sentence with '?'
    q_idx = len(sentences) - 1
    for i in range(len(sentences) - 1, -1, -1):
        if "?" in sentences[i]:
            q_idx = i
            break

    context = " ".join(sentences[:q_idx]).strip()
    question = " ".join(sentences[q_idx:]).strip()
    return context, question


def build_masked_views(
    problem: str, mask_token: str = "[HIDDEN]"
) -> tuple[str, str] | None:
    """Build two complementary masked views of a GSM8K problem.

    View A keeps even-indexed numbers (0, 2, 4, …) and masks odd.
    View B keeps odd-indexed numbers (1, 3, 5, …) and masks even.
    Numbers in the final question sentence are NEVER masked.

    Returns ``(view_a, view_b)`` or ``None`` when there are fewer than
    2 maskable numbers (both views would be identical).
    """
    context, question = split_question(problem)
    if not context:
        return None

    numbers = find_numbers(context)
    if len(numbers) < 2:
        return None

    # Build views by replacing appropriate numbers
    def _mask(text: str, nums: list[tuple[int, int, str]], keep_indices: set[int]) -> str:
        parts: list[str] = []
        prev = 0
        for idx, (start, end, _val) in enumerate(nums):
            parts.append(text[prev:start])
            if idx in keep_indices:
                parts.append(text[start:end])
            else:
                parts.append(mask_token)
            prev = end
        parts.append(text[prev:])
        return "".join(parts)

    even_indices = {i for i in range(len(numbers)) if i % 2 == 0}
    odd_indices = {i for i in range(len(numbers)) if i % 2 == 1}

    context_a = _mask(context, numbers, keep_indices=even_indices)
    context_b = _mask(context, numbers, keep_indices=odd_indices)

    # Append question unchanged
    sep = " " if context else ""
    view_a = (context_a + sep + question).strip()
    view_b = (context_b + sep + question).strip()
    return view_a, view_b


def validate_views(
    view_a: str, view_b: str, mask_token: str = "[HIDDEN]"
) -> bool:
    """Check both views have at least 1 real number and 1 mask token."""
    has_num_a = bool(NUMBER_RE.search(view_a))
    has_num_b = bool(NUMBER_RE.search(view_b))
    has_mask_a = mask_token in view_a
    has_mask_b = mask_token in view_b
    return has_num_a and has_num_b and has_mask_a and has_mask_b
