"""Prompt templates for D12.1 Info-Asymmetric Masking experiment."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MASKED_PARTIAL_VIEW_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "You see only PART of the information — some numbers are hidden as [HIDDEN].\n"
    "Another agent sees the hidden numbers but not yours.\n"
    "Contribute useful reasoning based on what you CAN see.\n"
    "Do not guess hidden numbers. State what you know and what you need."
)

MASKED_SYNTHESIZER_SYSTEM = (
    "You are the final synthesizer in a two-agent reasoning system.\n"
    "Two agents each saw different parts of the problem.\n"
    "Each has provided a contribution based on their partial view.\n"
    "Combine their information to solve the complete problem.\n"
    "Produce concise reasoning and the final numeric answer."
)

MASKED_BOOTSTRAP_SYSTEM = (
    "You are generating training data for a two-agent math reasoning system.\n"
    "Two agents each see a DIFFERENT partial view of the problem "
    "(some numbers replaced by [HIDDEN]).\n"
    "Generate realistic contributions from each agent's limited view, "
    "then a joint solution that combines both.\n"
    "The final answer must be correct."
)

# ---------------------------------------------------------------------------
# User prompt builders
# ---------------------------------------------------------------------------


def masked_partial_user(view: str) -> str:
    return (
        f"Problem (partial view — some numbers are [HIDDEN]):\n{view}\n\n"
        "Reason about what you CAN see. State clearly what information you "
        "have and what is hidden. Do not guess hidden values.\n"
        "End with `Final answer:` followed by your best numeric answer "
        "(or 'INSUFFICIENT' if you truly cannot solve)."
    )


def masked_synthesizer_user(
    problem: str, contrib_a: str, contrib_b: str
) -> str:
    return (
        f"Full problem:\n{problem}\n\n"
        f"Agent A contribution (saw some numbers, others hidden):\n{contrib_a}\n\n"
        f"Agent B contribution (saw the other numbers):\n{contrib_b}\n\n"
        "Combine both contributions to solve the problem.\n"
        "End with `Final answer:` followed by the numeric answer only."
    )


def masked_bootstrap_user(
    problem: str, view_a: str, view_b: str
) -> str:
    return (
        f"Full problem:\n{problem}\n\n"
        f"Agent A view:\n{view_a}\n\n"
        f"Agent B view:\n{view_b}\n\n"
        "Generate a training trace in exactly this format:\n\n"
        "Agent A contribution:\n"
        "(reasoning from A's partial view)\n\n"
        "Agent B contribution:\n"
        "(reasoning from B's partial view)\n\n"
        "Joint solution:\n"
        "(combining both)\n\n"
        "Final answer:\n"
        "(numeric answer only)\n\n"
        "Requirements:\n"
        "- Each contribution should reason from that agent's visible numbers only.\n"
        "- The joint solution should combine both contributions.\n"
        "- The final answer must be the correct numeric answer."
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def messages_for_masked_partial(view: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MASKED_PARTIAL_VIEW_SYSTEM},
        {"role": "user", "content": masked_partial_user(view)},
    ]


def messages_for_masked_synthesizer(
    problem: str, contrib_a: str, contrib_b: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MASKED_SYNTHESIZER_SYSTEM},
        {"role": "user", "content": masked_synthesizer_user(problem, contrib_a, contrib_b)},
    ]


def messages_for_masked_bootstrap(
    problem: str, view_a: str, view_b: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MASKED_BOOTSTRAP_SYSTEM},
        {"role": "user", "content": masked_bootstrap_user(problem, view_a, view_b)},
    ]
