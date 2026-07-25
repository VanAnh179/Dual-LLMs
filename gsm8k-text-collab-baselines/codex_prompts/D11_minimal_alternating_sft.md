# D11 Minimal Alternating SFT

Implement a minimal training-based dual-LLM collaboration pipeline for GSM8K.

Core constraints:

- Use two neutral reasoning agents, Agent A and Agent B.
- Do not assign fixed solver/verifier roles.
- Train with alternating optimization: freeze one agent while fine-tuning the other.
- Use LoRA adapters, not full fine-tuning.
- Use supervised targets containing reasoning traces and final answers.
- If reasoning traces are missing, bootstrap candidate traces and keep only candidates whose final answer matches the gold answer.

Target assistant format:

```text
Reasoning:
...

Final answer:
...
```

Two-agent prompt wording must remain neutral: the responding agent may use, revise, ignore, or contradict the other agent's response.
