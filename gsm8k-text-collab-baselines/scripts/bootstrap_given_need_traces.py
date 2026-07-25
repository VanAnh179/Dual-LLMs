#!/usr/bin/env python
"""Generate D11.4 compact Given/Need notes plus solution traces."""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answer_extraction import answers_match, extract_final_answer, normalize_answer
from src.data_utils import (
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
from src.prompts import (
    format_for_generation,
    messages_for_d11_4_given_need_teacher,
    messages_for_d11_4_solution_teacher,
)


EXPERIMENT_NAME = "D11_4_compact_given_need_decider_sft"
GIVEN_NEED_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?\s*Given\s*(?:\*\*)?\s*:\s*(?P<given>.*?)"
    r"(?:^|\n)\s*(?:\*\*)?\s*Need\s*(?:\*\*)?\s*:\s*(?P<need>.*?)(?=\n\s*(?:\*\*)?\s*(?:Plan|Reasoning|Final answer|Answer|Solution)\s*(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SOLUTION_RE = re.compile(
    r"Reasoning:\s*(?P<reasoning>.*?)\s*Final answer:\s*(?P<final>.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
FORBIDDEN_NOTE_PHRASES = (
    "final answer",
    "answer is",
    "the answer",
    "boxed",
    "therefore",
    "so the answer",
    "plan:",
    "reasoning:",
    "solution:",
    "####",
)


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def teacher_notes_model_name(cfg: dict) -> str:
    return cfg.get("teacher_notes_model_name") or cfg.get("teacher_model_name")


def teacher_solution_model_name(cfg: dict) -> str:
    return cfg.get("teacher_solution_model_name") or cfg.get("teacher_model_name")


def clear_model(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_given_need(text: str, cfg: dict) -> dict | None:
    match = GIVEN_NEED_RE.search(text.strip())
    if not match:
        return None
    given = compact_field(match.group("given"), int(cfg_get(cfg, "notes", "max_given_words", 36)))
    need = compact_field(match.group("need"), int(cfg_get(cfg, "notes", "max_need_words", 14)))
    if not given or not need:
        return None
    notes = format_given_need(given, need)
    return {"given": given, "need": need, "given_need_notes": notes}


def compact_field(text: str, max_words: int) -> str:
    pieces = []
    for line in (text or "").splitlines():
        stripped = clean_note_line(line)
        if not stripped:
            continue
        lowered = stripped.lower().strip().rstrip(":")
        if lowered in {"given", "need", "notes"}:
            continue
        if re.match(r"^(?:plan|reasoning|final answer|answer|solution)\s*:", lowered):
            break
        pieces.append(stripped)
    joined = "; ".join(pieces).strip(" ;")
    words = re.findall(r"\S+", joined)
    if len(words) > max_words:
        joined = " ".join(words[:max_words])
    return joined.strip()


def clean_note_line(line: str) -> str:
    stripped = (line or "").strip()
    stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", stripped)
    stripped = stripped.strip().strip("*").strip()
    stripped = re.sub(r"^(?:Given|Need)\s*:\s*", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def format_given_need(given: str, need: str) -> str:
    return f"Given: {given.strip()}\nNeed: {need.strip()}"


def notes_have_gold(notes: str, gold_answer: str) -> bool:
    gold = normalize_answer(gold_answer)
    if not gold:
        return False
    for number in re.findall(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", notes):
        if normalize_answer(number) == gold:
            return True
    return False


def notes_are_valid(notes: str, gold_answer: str, cfg: dict) -> tuple[bool, str]:
    lowered = notes.lower()
    if "=" in notes:
        return False, "note_contains_equals"
    for phrase in FORBIDDEN_NOTE_PHRASES:
        if phrase in lowered:
            return False, f"forbidden_note_phrase:{phrase}"
    if len(notes) > int(cfg_get(cfg, "notes", "max_chars", 320)):
        return False, "notes_too_long_after_compact"
    if bool(cfg_get(cfg, "notes", "reject_gold_in_notes", False)) and notes_have_gold(notes, gold_answer):
        return False, "gold_answer_in_notes"
    return True, "ok"


def parse_solution(text: str) -> dict | None:
    stripped = text.strip()
    match = SOLUTION_RE.search(stripped)
    if match:
        reasoning = clean_block(match.group("reasoning"))
        final = extract_final_answer(f"Final answer:\n{match.group('final').strip()}")
    else:
        final = extract_final_answer(stripped)
        reasoning = clean_block(remove_final_answer_tail(stripped))
    if not reasoning or final is None:
        return None
    return {"reasoning": reasoning, "final_answer": final}


def remove_final_answer_tail(text: str) -> str:
    matches = list(re.finditer(r"final answer\s*:", text, re.IGNORECASE))
    if matches:
        return text[: matches[-1].start()]
    return text


def clean_block(text: str) -> str:
    cleaned = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.strip("*").strip()
        if stripped in {"[", "]", "\\[", "\\]"}:
            continue
        lowered = stripped.lower().rstrip(":")
        if lowered in {"reasoning", "final answer"}:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def avg_len(rows: list[dict], key: str) -> float:
    return sum(len(row.get(key, "")) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d11_4_compact_given_need_decider.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--notes-max-new-tokens", type=int, default=None)
    parser.add_argument("--solution-max-new-tokens", type=int, default=None)
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
        f"data/filtered/{EXPERIMENT_NAME}/given_need_traces.jsonl",
    )
    generation_dir = cfg_get(cfg, "output", "generation_dir", f"outputs/{EXPERIMENT_NAME}/generations")
    metrics_dir = cfg_get(cfg, "output", "metrics_dir", f"outputs/{EXPERIMENT_NAME}/metrics")

    reject_test_split_for_training(input_path)
    all_rows = read_jsonl(input_path)
    reject_test_rows_for_training(all_rows)
    rows = sample_records(
        all_rows,
        args.max_examples or cfg_get(cfg, "sampling", "max_train_examples", 200),
        cfg_get(cfg, "sampling", "sampling_mode", "first_n"),
        int(cfg_get(cfg, "sampling", "seed", 42)),
    )
    record_sampled_ids("D11_4_bootstrap_train_ids", rows)
    if not rows:
        raise SystemExit(f"No training rows found in {input_path}.")

    notes_model_name = teacher_notes_model_name(cfg)
    solution_model_name = teacher_solution_model_name(cfg)
    if not notes_model_name or not solution_model_name:
        raise SystemExit("Config must define teacher_notes_model_name and teacher_solution_model_name.")

    notes_gen_config = {
        "max_new_tokens": int(
            args.notes_max_new_tokens or cfg_get(cfg, "bootstrap", "notes_max_new_tokens", 96)
        ),
        "temperature": float(cfg_get(cfg, "bootstrap", "notes_temperature", 0.0)),
        "top_p": float(cfg_get(cfg, "bootstrap", "notes_top_p", 1.0)),
    }
    solution_gen_config = {
        "max_new_tokens": int(
            args.solution_max_new_tokens or cfg_get(cfg, "bootstrap", "solution_max_new_tokens", 512)
        ),
        "temperature": float(cfg_get(cfg, "bootstrap", "solution_temperature", 0.0)),
        "top_p": float(cfg_get(cfg, "bootstrap", "solution_top_p", 1.0)),
    }

    failed = []
    note_rows = []
    note_parse_success_count = 0
    valid_notes_count = 0
    gold_in_notes_count = 0

    notes_tokenizer, notes_model = load_tokenizer_and_model(notes_model_name)
    for row in tqdm(rows, desc="bootstrap D11.4 given/need notes"):
        prompt = format_for_generation(notes_tokenizer, messages_for_d11_4_given_need_teacher(row["problem"]))
        raw_notes = generate_text(notes_tokenizer, notes_model, prompt, **notes_gen_config)
        parsed = parse_given_need(raw_notes, cfg)
        if parsed is None:
            failed.append(
                {
                    "id": row.get("id"),
                    "stage": "notes",
                    "failure_reason": "notes_parse_failed",
                    "raw_notes": raw_notes,
                }
            )
            continue
        note_parse_success_count += 1
        notes = parsed["given_need_notes"]
        if notes_have_gold(notes, row["gold_answer"]):
            gold_in_notes_count += 1
        notes_ok, notes_reason = notes_are_valid(notes, row["gold_answer"], cfg)
        if not notes_ok:
            failed.append(
                {
                    "id": row.get("id"),
                    "stage": "notes",
                    "failure_reason": notes_reason,
                    "gold_answer": row["gold_answer"],
                    "raw_notes": raw_notes,
                    **parsed,
                }
            )
            continue
        valid_notes_count += 1
        note_rows.append(
            {
                "id": row.get("id"),
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "raw_notes": raw_notes,
                **parsed,
            }
        )
    clear_model(notes_model)

    kept = []
    solution_parse_success_count = 0
    answer_match_count = 0
    solution_tokenizer, solution_model = load_tokenizer_and_model(solution_model_name)
    for row in tqdm(note_rows, desc="bootstrap D11.4 solution traces"):
        prompt = format_for_generation(
            solution_tokenizer,
            messages_for_d11_4_solution_teacher(row["problem"], row["given_need_notes"]),
        )
        raw_solution = generate_text(solution_tokenizer, solution_model, prompt, **solution_gen_config)
        parsed = parse_solution(raw_solution)
        if parsed is None:
            failed.append(
                {
                    "id": row.get("id"),
                    "stage": "solution",
                    "failure_reason": "solution_parse_failed",
                    "gold_answer": row["gold_answer"],
                    "raw_solution": raw_solution,
                }
            )
            continue
        solution_parse_success_count += 1
        if not answers_match(parsed["final_answer"], row["gold_answer"]):
            failed.append(
                {
                    "id": row.get("id"),
                    "stage": "solution",
                    "failure_reason": "solution_answer_mismatch",
                    "pred_answer": parsed["final_answer"],
                    "gold_answer": row["gold_answer"],
                    "raw_solution": raw_solution,
                }
            )
            continue
        answer_match_count += 1
        kept.append(
            {
                **row,
                **parsed,
                "raw_solution": raw_solution,
            }
        )
    clear_model(solution_model)

    write_jsonl(output_path, kept)
    write_jsonl(f"{generation_dir}/bootstrap_failed.jsonl", failed)
    write_json(
        f"{metrics_dir}/bootstrap_stats.json",
        {
            "teacher_notes_model_name": notes_model_name,
            "teacher_solution_model_name": solution_model_name,
            "num_input_examples": len(rows),
            "note_parse_success_count": note_parse_success_count,
            "note_parse_success_rate": note_parse_success_count / len(rows) if rows else 0.0,
            "valid_notes_count": valid_notes_count,
            "valid_notes_rate": valid_notes_count / note_parse_success_count if note_parse_success_count else 0.0,
            "gold_in_notes_count": gold_in_notes_count,
            "gold_in_notes_rate": gold_in_notes_count / note_parse_success_count if note_parse_success_count else 0.0,
            "solution_input_count": len(note_rows),
            "solution_parse_success_count": solution_parse_success_count,
            "solution_parse_success_rate": solution_parse_success_count / len(note_rows) if note_rows else 0.0,
            "answer_match_count": answer_match_count,
            "answer_match_rate": answer_match_count / solution_parse_success_count if solution_parse_success_count else 0.0,
            "kept_example_count": len(kept),
            "average_notes_length": avg_len(kept, "given_need_notes"),
            "average_reasoning_length": avg_len(kept, "reasoning"),
        },
    )
    if not kept:
        raise SystemExit(
            "No D11.4 Given/Need traces passed filtering. "
            f"Inspect {generation_dir}/bootstrap_failed.jsonl."
        )
    print(f"Kept {len(kept)} D11.4 Given/Need traces in {output_path}")


if __name__ == "__main__":
    main()
