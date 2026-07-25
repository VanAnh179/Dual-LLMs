"""Filtering helpers for bootstrapped reasoning traces."""

from __future__ import annotations

from .answer_extraction import answers_match, extract_explicit_final_answer, extract_final_answer


def trace_is_correct(
    candidate_text: str,
    gold_answer: str,
    require_explicit_final_answer: bool = False,
) -> tuple[bool, str | None]:
    extractor = extract_explicit_final_answer if require_explicit_final_answer else extract_final_answer
    pred = extractor(candidate_text)
    return (pred is not None and answers_match(pred, gold_answer), pred)


def reasoning_from_candidate(candidate_text: str) -> str:
    text = candidate_text.strip()
    lower = text.lower()
    final_idx = lower.rfind("final answer:")
    if final_idx >= 0:
        text = text[:final_idx].strip()
    if text.lower().startswith("reasoning:"):
        text = text[len("reasoning:") :].strip()
    return text
