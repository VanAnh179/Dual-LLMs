"""Evaluation helpers for one-agent and two-agent GSM8K runs."""

from __future__ import annotations

from .answer_extraction import answers_match, extract_final_answer
from .generation import generate_text
from .prompts import format_for_generation, messages_for_single, messages_for_two_agent


def solve_alone(tokenizer, model, problem: str, gen_config: dict) -> str:
    prompt = format_for_generation(tokenizer, messages_for_single(problem))
    return generate_text(tokenizer, model, prompt, **gen_config)


def solve_with_other(tokenizer, model, problem: str, other_response: str, gen_config: dict) -> str:
    prompt = format_for_generation(tokenizer, messages_for_two_agent(problem, other_response))
    return generate_text(tokenizer, model, prompt, **gen_config)


def score_prediction(text: str, gold_answer: str) -> tuple[bool, str | None]:
    pred = extract_final_answer(text)
    return (pred is not None and answers_match(pred, gold_answer), pred)
