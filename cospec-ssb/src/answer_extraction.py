"""Answer extraction and matching helpers for GSM8K-style outputs."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


FINAL_PATTERNS = [
    re.compile(r"final answer\s*:", re.IGNORECASE),
    re.compile(r"answer\s*:", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
]

NUMBER_RE = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_gold_answer(raw_answer: str) -> str:
    if raw_answer is None:
        raise ValueError("raw_answer is missing")
    if "####" in raw_answer:
        return normalize_answer(raw_answer.rsplit("####", 1)[1])
    final = extract_final_answer(raw_answer)
    if final is None:
        raise ValueError(f"Could not extract GSM8K gold answer from: {raw_answer[:120]!r}")
    return normalize_answer(final)


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    for pattern in FINAL_PATTERNS[:2]:
        matches = list(pattern.finditer(text))
        if matches:
            tail = text[matches[-1].end() :]
            number = extract_last_number(tail)
            if number is not None:
                return number
            first_content = first_nonempty_line(tail)
            if first_content:
                return normalize_answer(first_content)
    hash_matches = FINAL_PATTERNS[2].findall(text)
    if hash_matches:
        return normalize_answer(hash_matches[-1])
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def extract_explicit_final_answer(text: str) -> str | None:
    if not text:
        return None
    for pattern in FINAL_PATTERNS[:2]:
        matches = list(pattern.finditer(text))
        if matches:
            tail = text[matches[-1].end() :]
            number = extract_last_number(tail)
            if number is not None:
                return number
            first_content = first_nonempty_line(tail)
            if first_content:
                return normalize_answer(first_content)
    hash_matches = FINAL_PATTERNS[2].findall(text)
    if hash_matches:
        return normalize_answer(hash_matches[-1])
    return None


def extract_last_number(text: str) -> str | None:
    numbers = NUMBER_RE.findall(text or "")
    if not numbers:
        return None
    return normalize_answer(numbers[-1])


def first_nonempty_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and stripped not in {"**", "\\[", "\\]", "$$"}:
            return stripped
    return None


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    cleaned = str(answer).strip()
    cleaned = cleaned.split("\n", 1)[0]
    cleaned = cleaned.replace("$", "").replace(",", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.rstrip(".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        value = Decimal(cleaned)
        if value == value.to_integral_value():
            return str(value.to_integral_value())
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError):
        return cleaned.lower()


def answers_match(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)
