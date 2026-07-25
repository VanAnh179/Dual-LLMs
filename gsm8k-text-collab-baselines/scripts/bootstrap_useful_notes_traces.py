#!/usr/bin/env python
"""Generate D11.3 useful-notes/final-decider traces."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answer_extraction import answers_match, extract_final_answer, normalize_answer
from src.data_utils import (
    get_teacher_model_name,
    load_config,
    read_jsonl,
    record_sampled_ids,
    reject_test_rows_for_training,
    reject_test_split_for_training,
    require_cuda_if_requested,
    require_dependencies,
    sample_records,
    write_json,
    write_jsonl,
)
from src.generation import generate_text, load_tokenizer_and_model
from src.prompts import format_for_generation, messages_for_d11_3_bootstrap


EXPERIMENT_NAME = "D11_3_useful_notes_decider_sft"
TRACE_RE = re.compile(
    r"Notes:\s*(?P<notes>.*?)\s*Reasoning:\s*(?P<reasoning>.*?)\s*Final answer:\s*(?P<final>.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
AGENT_NOTES_RE = re.compile(
    r"Agent\s*1\s*:\s*(?:write\s+the\s+notes|notes)?\s*(?P<notes>.*?)\s*"
    r"Reasoning:\s*(?P<reasoning>.*?)\s*Final answer:\s*(?P<final>.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
NOTES_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*\s*)?Notes\s*(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
REASONING_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*\s*)?Reasoning\s*(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
FINAL_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*\s*)?Final answer\s*(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
STRUCTURED_REASONING_RE = re.compile(
    r"(?:^|\n)\s*\d+[.)]\s*(?:\*\*\s*)?Reasoning\s*(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
FORBIDDEN_NOTE_PHRASES = (
    "final answer",
    "answer is",
    "the answer",
    "boxed",
    "therefore",
    "so the answer",
)


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def parse_trace(text: str) -> dict | None:
    stripped = text.strip()
    match = TRACE_RE.search(stripped) or AGENT_NOTES_RE.search(stripped)
    if not match:
        return parse_late_notes_trace(stripped) or parse_structured_notes_trace(stripped)
    return build_trace(
        match.group("notes"),
        match.group("reasoning"),
        f"Final answer:\n{match.group('final').strip()}",
    )


def parse_late_notes_trace(text: str) -> dict | None:
    notes_matches = list(NOTES_LABEL_RE.finditer(text))
    if not notes_matches:
        return None

    notes_match = notes_matches[-1]
    prefix = text[: notes_match.start()]
    final = extract_final_answer(prefix)
    if final is None:
        return None

    tail = text[notes_match.end() :]
    reasoning_match = REASONING_LABEL_RE.search(tail)
    if reasoning_match:
        notes = tail[: reasoning_match.start()]
        reasoning = tail[reasoning_match.end() :]
    else:
        notes = tail
        final_label_matches = list(FINAL_LABEL_RE.finditer(prefix))
        reasoning = prefix[: final_label_matches[-1].start()] if final_label_matches else prefix

    return build_trace(notes, reasoning, final)


def parse_structured_notes_trace(text: str) -> dict | None:
    if not re.search(r"\b(?:quantities|relationships|next computation)\b", text, re.IGNORECASE):
        return None
    reasoning_match = STRUCTURED_REASONING_RE.search(text)
    if not reasoning_match:
        return None

    notes = text[: reasoning_match.start()]
    tail = text[reasoning_match.end() :]
    final_label_match = FINAL_LABEL_RE.search(tail)
    if final_label_match:
        reasoning = tail[: final_label_match.start()]
        final_source = f"Final answer:\n{tail[final_label_match.end():].strip()}"
    else:
        reasoning = tail
        final_source = tail

    return build_trace(notes, reasoning, final_source)


def build_trace(notes_text: str, reasoning_text: str, final_source: str) -> dict | None:
    notes = clean_block(notes_text)
    reasoning = clean_block(reasoning_text)
    final = extract_final_answer(final_source)
    if not notes or not reasoning or final is None:
        return None
    return {"useful_notes": notes, "reasoning": reasoning, "final_answer": final}


def clean_block(text: str) -> str:
    cleaned = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.strip("*").strip()
        lowered = stripped.lower().rstrip(":")
        if lowered in {"notes", "reasoning", "next computation", "relationships", "quantities"}:
            continue
        if stripped in {"[", "]", "\\[", "\\]"}:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def compact_notes(notes: str, cfg: dict) -> str:
    max_bullets = int(cfg_get(cfg, "notes", "max_bullets", 4))
    max_words = int(cfg_get(cfg, "notes", "max_words_per_bullet", 20))
    compacted = []
    lines = note_lines(notes)
    structured_lines = [
        line for line in lines if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
    ]
    for line in structured_lines or lines:
        lowered = line.lower()
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = line.strip().strip("*").strip()
        lowered = line.lower().strip().rstrip(":")
        if lowered in {"notes", "reasoning", "next computation", "relationships", "quantities"}:
            continue
        if any(phrase in lowered for phrase in FORBIDDEN_NOTE_PHRASES):
            continue
        if looks_like_final_computation(line):
            continue
        words = re.findall(r"\S+", line)
        if not words:
            continue
        if len(words) > max_words:
            line = " ".join(words[:max_words])
        compacted.append(f"- {line.strip()}")
        if len(compacted) >= max_bullets:
            break
    return "\n".join(compacted)


def looks_like_final_computation(line: str) -> bool:
    lowered = line.lower()
    if "total" not in lowered and "profit" not in lowered:
        return False
    if line.count("=") < 2:
        return False
    return bool(re.search(r"=\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s+\w+)?\.?$", line.strip()))


def note_lines(notes: str) -> list[str]:
    return [line.strip() for line in notes.splitlines() if line.strip()]


def note_bullets(notes: str) -> list[str]:
    lines = note_lines(notes)
    bullets = [line for line in lines if line.startswith(("-", "*")) and line.strip("*").strip()]
    return bullets or lines


def notes_have_gold_in_last_line(notes: str, gold_answer: str) -> bool:
    lines = note_lines(notes)
    if not lines:
        return False
    numbers = re.findall(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", lines[-1])
    return bool(numbers and normalize_answer(numbers[-1]) == normalize_answer(gold_answer))


def notes_are_valid(notes: str, gold_answer: str, cfg: dict) -> tuple[bool, str]:
    lowered = notes.lower()
    for phrase in FORBIDDEN_NOTE_PHRASES:
        if phrase in lowered:
            return False, f"forbidden_note_phrase:{phrase}"

    bullets = note_bullets(notes)
    if not bullets:
        return False, "empty_notes"
    if len(bullets) > int(cfg_get(cfg, "notes", "max_bullets", 4)):
        return False, "too_many_note_bullets"
    if bool(cfg_get(cfg, "notes", "reject_gold_in_last_note_line", False)):
        if notes_have_gold_in_last_line(notes, gold_answer):
            return False, "gold_answer_in_last_note_line"
    return True, "ok"


def avg_len(rows: list[dict], key: str) -> float:
    return sum(len(row.get(key, "")) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_3_useful_notes_decider.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    require_dependencies("torch", "transformers", "peft", "yaml", "tqdm")
    from tqdm import tqdm

    cfg = load_config(args.config)
    require_cuda_if_requested(bool(cfg.get("require_cuda", False)))

    input_path = args.input or cfg_get(cfg, "data", "raw_train_path", "data/raw/train.jsonl")
    output_path = args.output or cfg_get(
        cfg,
        "data",
        "filtered_trace_path",
        f"data/filtered/{EXPERIMENT_NAME}/useful_note_traces.jsonl",
    )
    generation_dir = cfg_get(cfg, "output", "generation_dir", f"outputs/{EXPERIMENT_NAME}/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", f"outputs/{EXPERIMENT_NAME}/metrics")

    reject_test_split_for_training(input_path)
    all_rows = read_jsonl(input_path)
    reject_test_rows_for_training(all_rows)
    rows = sample_records(
        all_rows,
        args.max_examples or cfg_get(cfg, "sampling", "max_train_examples", 100),
        cfg_get(cfg, "sampling", "sampling_mode", "first_n"),
        int(cfg_get(cfg, "sampling", "seed", 42)),
    )
    record_sampled_ids("D11_3_bootstrap_train_ids", rows)
    if not rows:
        raise SystemExit(f"No training rows found in {input_path}.")

    teacher_model_name = get_teacher_model_name(cfg)
    tokenizer, model = load_tokenizer_and_model(teacher_model_name)
    gen_config = {
        "max_new_tokens": int(args.max_new_tokens or cfg_get(cfg, "bootstrap", "max_new_tokens", 384)),
        "temperature": float(cfg_get(cfg, "bootstrap", "temperature", 0.7)),
        "top_p": float(cfg_get(cfg, "bootstrap", "top_p", 0.9)),
    }
    num_candidates = int(args.num_candidates or cfg_get(cfg, "bootstrap", "num_candidates", 1))

    kept = []
    failed = []
    parse_success_count = 0
    answer_match_count = 0
    valid_notes_count = 0
    gold_in_last_note_count = 0

    for row in tqdm(rows, desc="bootstrap useful-note traces"):
        prompt = format_for_generation(tokenizer, messages_for_d11_3_bootstrap(row["problem"]))
        found = False
        for candidate_idx in range(num_candidates):
            raw = generate_text(tokenizer, model, prompt, **gen_config)
            parsed = parse_trace(raw)
            if parsed is None:
                failed.append(
                    {
                        "id": row.get("id"),
                        "candidate_idx": candidate_idx,
                        "failure_reason": "parse_failed",
                        "raw_candidate": raw,
                    }
                )
                continue
            parse_success_count += 1
            parsed["useful_notes"] = compact_notes(parsed["useful_notes"], cfg)
            if not parsed["useful_notes"]:
                failed.append(
                    {
                        "id": row.get("id"),
                        "candidate_idx": candidate_idx,
                        "failure_reason": "empty_compacted_notes",
                        "pred_answer": parsed["final_answer"],
                        "gold_answer": row["gold_answer"],
                        "raw_candidate": raw,
                    }
                )
                continue
            if notes_have_gold_in_last_line(parsed["useful_notes"], row["gold_answer"]):
                gold_in_last_note_count += 1
            notes_ok, notes_reason = notes_are_valid(parsed["useful_notes"], row["gold_answer"], cfg)
            if not notes_ok:
                failed.append(
                    {
                        "id": row.get("id"),
                        "candidate_idx": candidate_idx,
                        "failure_reason": notes_reason,
                        "pred_answer": parsed["final_answer"],
                        "gold_answer": row["gold_answer"],
                        "raw_candidate": raw,
                    }
                )
                continue
            valid_notes_count += 1
            if not answers_match(parsed["final_answer"], row["gold_answer"]):
                failed.append(
                    {
                        "id": row.get("id"),
                        "candidate_idx": candidate_idx,
                        "failure_reason": "answer_mismatch",
                        "pred_answer": parsed["final_answer"],
                        "gold_answer": row["gold_answer"],
                        "raw_candidate": raw,
                    }
                )
                continue
            answer_match_count += 1
            kept.append(
                {
                    "id": row.get("id"),
                    "problem": row["problem"],
                    "gold_answer": row["gold_answer"],
                    **parsed,
                    "raw_candidate": raw,
                }
            )
            found = True
            break
        if not found:
            failed.append({"id": row.get("id"), "candidate_idx": None, "failure_reason": "no_kept_candidate"})

    total_candidates = len(rows) * num_candidates
    write_jsonl(output_path, kept)
    write_jsonl(f"{generation_dir}/bootstrap_failed.jsonl", failed)
    write_json(
        f"{metrics_dir}/bootstrap_stats.json",
        {
            "teacher_model_name": teacher_model_name,
            "num_input_examples": len(rows),
            "num_candidates_per_example": num_candidates,
            "num_total_candidates": total_candidates,
            "parse_success_count": parse_success_count,
            "parse_success_rate": parse_success_count / total_candidates if total_candidates else 0.0,
            "valid_notes_count": valid_notes_count,
            "valid_notes_rate": valid_notes_count / parse_success_count if parse_success_count else 0.0,
            "answer_match_count": answer_match_count,
            "answer_match_rate": answer_match_count / valid_notes_count if valid_notes_count else 0.0,
            "kept_example_count": len(kept),
            "gold_in_last_note_line_count": gold_in_last_note_count,
            "average_notes_length": avg_len(kept, "useful_notes"),
            "average_reasoning_length": avg_len(kept, "reasoning"),
        },
    )
    if not kept:
        raise SystemExit(
            "No D11.3 useful-note traces passed filtering. "
            f"Inspect {generation_dir}/bootstrap_failed.jsonl."
        )
    print(f"Kept {len(kept)} useful-note traces in {output_path}")


if __name__ == "__main__":
    main()
