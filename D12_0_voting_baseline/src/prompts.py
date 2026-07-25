"""Prompt templates for neutral two-agent GSM8K reasoning."""

SINGLE_AGENT_SYSTEM = (
    "You are a careful reasoning agent. Solve the problem step by step and "
    "provide the final answer."
)

BOOTSTRAP_TEACHER_SYSTEM = (
    "You are a careful math reasoning teacher. Solve the problem step by step.\n"
    "Keep the reasoning concise enough to finish the full answer.\n"
    "Return exactly this format:\n"
    "Reasoning:\n"
    "<your reasoning>\n\n"
    "Final answer:\n"
    "<the final numeric answer only>"
)

TWO_AGENT_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "Another agent may provide a useful, partially useful, or incorrect response.\n"
    "Your goal is to produce the best reasoning trace and final answer.\n"
    "You may use, revise, ignore, or contradict the other agent's response.\n"
    "Do not assume a fixed role such as solver or verifier."
)

COLLABORATIVE_BOOTSTRAP_SYSTEM = (
    "You are generating training data for a two-agent reasoning system.\n"
    "Do not assign fixed roles such as solver, verifier, critic, calculator, planner, or executor.\n"
    "The two agents should provide useful and complementary contributions.\n"
    "They may decompose the problem, identify quantities, form equations, compute intermediate values, "
    "check consistency, or synthesize the final solution.\n"
    "The goal is to produce a correct final answer."
)

FIRST_CONTRIBUTOR_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "Your job is to contribute useful information toward solving the problem.\n"
    "The other agent may later add a different but useful contribution.\n"
    "Do not assume a fixed role."
)

SECOND_CONTRIBUTOR_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "Another agent has provided a contribution.\n"
    "Your job is to add useful complementary information and produce a final joint solution.\n"
    "You may use, extend, refine, or synthesize the other contribution.\n"
    "Do not assume a fixed role such as solver or verifier."
)

COLLABORATIVE_STANDALONE_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "Produce a useful contribution, a complete joint solution, and a concise final answer.\n"
    "Do not assume a fixed role."
)

D11_3_BOOTSTRAP_SYSTEM = (
    "You are generating compact training data for a small two-agent GSM8K system.\n"
    "Agent 1 writes short useful notes for another agent, without giving the final answer.\n"
    "Agent 2 is the final decision maker and must solve the problem correctly from the problem and notes.\n"
    "Keep all text concise."
)

USEFUL_NOTES_SYSTEM = (
    "You write short useful notes for another agent solving a GSM8K problem.\n"
    "Do not provide the final answer.\n"
    "Use at most 4 bullets.\n"
    "Each bullet should be short and actionable."
)

FINAL_DECIDER_SYSTEM = (
    "You are the final decision maker for a GSM8K problem.\n"
    "Use the problem as the source of truth.\n"
    "The notes from another agent may be useful, incomplete, or partially wrong.\n"
    "Produce concise reasoning and the final numeric answer."
)

D11_4_GIVEN_NEED_TEACHER_SYSTEM = (
    "You rewrite GSM8K problems into very short semantic notes for a small solver model.\n"
    "Do not solve the problem.\n"
    "Do not include calculations, equations, plans, or the final answer.\n"
    "Preserve important quantities, units, and semantic relationships exactly."
)

D11_4_SOLUTION_TEACHER_SYSTEM = (
    "You are a careful math reasoning teacher.\n"
    "Solve the GSM8K problem using the problem as the source of truth.\n"
    "The Given/Need notes may help clarify the problem, but they are not a substitute for the problem."
)

GIVEN_NEED_SYSTEM = (
    "You write very short semantic notes for another agent solving a GSM8K problem.\n"
    "Do not solve the problem.\n"
    "Do not include calculations, equations, plans, or the final answer.\n"
    "Output only Given and Need."
)

GIVEN_NEED_DECIDER_SYSTEM = (
    "You are the final decision maker for a GSM8K problem.\n"
    "Use the problem as the source of truth.\n"
    "The Given/Need notes may help clarify the problem but may be incomplete.\n"
    "Produce concise reasoning and the final numeric answer."
)


def single_agent_user(problem: str) -> str:
    return f"Problem:\n{problem}"


def bootstrap_teacher_user(problem: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        "Give concise reasoning and always finish with `Final answer:` followed by only the number."
    )


def two_agent_user(problem: str, other_agent_response: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        f"Another agent's response:\n{other_agent_response}"
    )


def collaborative_bootstrap_user(problem: str) -> str:
    return (
        "Solve the following GSM8K problem as a collaboration between two agents.\n\n"
        f"Problem:\n{problem}\n\n"
        "Return exactly this format:\n\n"
        "Agent 1 contribution:\n"
        "...\n\n"
        "Agent 2 contribution:\n"
        "...\n\n"
        "Joint solution:\n"
        "...\n\n"
        "Final answer:\n"
        "...\n\n"
        "Requirements:\n"
        "- Agent 1 and Agent 2 should not merely copy each other.\n"
        "- Agent 2 should add useful information beyond Agent 1.\n"
        "- The joint solution should be complete and clear.\n"
        "- The final answer should be concise."
    )


def first_contributor_user(problem: str) -> str:
    return f"Problem:\n{problem}"


def second_contributor_user(problem: str, other_contribution: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        f"Other agent's contribution:\n{other_contribution}"
    )


def collaborative_standalone_user(problem: str) -> str:
    return f"Problem:\n{problem}"


def d11_3_bootstrap_user(problem: str) -> str:
    return (
        "Solve the following GSM8K problem as a compact two-agent training trace.\n\n"
        f"Problem:\n{problem}\n\n"
        "Return exactly this format:\n\n"
        "Notes:\n"
        "- quantities: ...\n"
        "- relationships: ...\n"
        "- next computation: ...\n\n"
        "Reasoning:\n"
        "...\n\n"
        "Final answer:\n"
        "...\n\n"
        "Requirements:\n"
        "- Start directly with `Notes:` and do not add text before it.\n"
        "- Do not add any text after the numeric final answer.\n"
        "- Notes must not include the final answer.\n"
        "- Notes must be at most 4 bullets.\n"
        "- Each note bullet should be under 20 words.\n"
        "- Reasoning should be at most 4 short lines.\n"
        "- Final answer should be the numeric answer only."
    )


def useful_notes_user(problem: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        "Write useful notes only.\n"
        "Format:\n"
        "Notes:\n"
        "- quantities: ...\n"
        "- relationships: ...\n"
        "- next computation: ..."
    )


def final_decider_user(problem: str, notes: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        f"Useful notes from another agent:\n{notes.strip()}\n\n"
        "Solve the problem and end with `Final answer:`."
    )


def d11_4_given_need_teacher_user(problem: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        "Write compact semantic notes in exactly this format:\n\n"
        "Given: <one short line with the relevant facts and relationships>\n"
        "Need: <one short line saying what the question asks for>\n\n"
        "Rules:\n"
        "- Do not solve.\n"
        "- Do not write a plan.\n"
        "- Do not use equations or equals signs.\n"
        "- Do not include the final numeric answer.\n"
        "- Keep Given under 36 words and Need under 14 words."
    )


def d11_4_solution_teacher_user(problem: str, notes: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        f"Given/Need notes:\n{notes.strip()}\n\n"
        "Solve the problem. Return exactly:\n\n"
        "Reasoning:\n"
        "<concise reasoning>\n\n"
        "Final answer:\n"
        "<numeric answer only>"
    )


def given_need_user(problem: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        "Write compact notes only.\n"
        "Format:\n"
        "Given: ...\n"
        "Need: ..."
    )


def given_need_decider_user(problem: str, notes: str) -> str:
    return (
        f"Problem:\n{problem}\n\n"
        f"Given/Need notes from another agent:\n{notes.strip()}\n\n"
        "Solve the problem and end with `Final answer:`."
    )


def assistant_target(reasoning_trace: str, gold_answer: str) -> str:
    return f"Reasoning:\n{reasoning_trace.strip()}\n\nFinal answer:\n{gold_answer}"


def messages_for_single(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM},
        {"role": "user", "content": single_agent_user(problem)},
    ]


def messages_for_bootstrap_teacher(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BOOTSTRAP_TEACHER_SYSTEM},
        {"role": "user", "content": bootstrap_teacher_user(problem)},
    ]


def messages_for_two_agent(problem: str, other_agent_response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TWO_AGENT_SYSTEM},
        {"role": "user", "content": two_agent_user(problem, other_agent_response)},
    ]


def messages_for_collaborative_bootstrap(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COLLABORATIVE_BOOTSTRAP_SYSTEM},
        {"role": "user", "content": collaborative_bootstrap_user(problem)},
    ]


def messages_for_first_contributor(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FIRST_CONTRIBUTOR_SYSTEM},
        {"role": "user", "content": first_contributor_user(problem)},
    ]


def messages_for_second_contributor(problem: str, other_contribution: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SECOND_CONTRIBUTOR_SYSTEM},
        {"role": "user", "content": second_contributor_user(problem, other_contribution)},
    ]


def messages_for_collaborative_standalone(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COLLABORATIVE_STANDALONE_SYSTEM},
        {"role": "user", "content": collaborative_standalone_user(problem)},
    ]


def messages_for_d11_3_bootstrap(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": D11_3_BOOTSTRAP_SYSTEM},
        {"role": "user", "content": d11_3_bootstrap_user(problem)},
    ]


def messages_for_useful_notes(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": USEFUL_NOTES_SYSTEM},
        {"role": "user", "content": useful_notes_user(problem)},
    ]


def messages_for_final_decider(problem: str, notes: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FINAL_DECIDER_SYSTEM},
        {"role": "user", "content": final_decider_user(problem, notes)},
    ]


def messages_for_d11_4_given_need_teacher(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": D11_4_GIVEN_NEED_TEACHER_SYSTEM},
        {"role": "user", "content": d11_4_given_need_teacher_user(problem)},
    ]


def messages_for_d11_4_solution_teacher(problem: str, notes: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": D11_4_SOLUTION_TEACHER_SYSTEM},
        {"role": "user", "content": d11_4_solution_teacher_user(problem, notes)},
    ]


def messages_for_given_need(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GIVEN_NEED_SYSTEM},
        {"role": "user", "content": given_need_user(problem)},
    ]


def messages_for_given_need_decider(problem: str, notes: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GIVEN_NEED_DECIDER_SYSTEM},
        {"role": "user", "content": given_need_decider_user(problem, notes)},
    ]


def format_for_generation(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(f"{m['role'].title()}:\n{m['content']}" for m in messages) + "\n\nAssistant:\n"


def format_for_sft(
    tokenizer,
    messages: list[dict[str, str]],
    reasoning_trace: str,
    gold_answer: str,
) -> str:
    target = assistant_target(reasoning_trace, gold_answer)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
    prompt = "\n\n".join(f"{m['role'].title()}:\n{m['content']}" for m in messages)
    return f"{prompt}\n\nAssistant:\n{target}"


def format_messages_with_assistant(
    tokenizer,
    messages: list[dict[str, str]],
    assistant_content: str,
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": assistant_content.strip()}],
            tokenize=False,
            add_generation_prompt=False,
        )
    prompt = "\n\n".join(f"{m['role'].title()}:\n{m['content']}" for m in messages)
    return f"{prompt}\n\nAssistant:\n{assistant_content.strip()}"
