"""Model loading and text generation helpers."""

from __future__ import annotations

from pathlib import Path

from .data_utils import project_path


def load_tokenizer_and_model(
    model_name: str,
    adapter_path: str | None = None,
    trainable_lora: bool = False,
    lora_config: dict | None = None,
):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path:
        adapter = project_path(adapter_path)
        if not adapter.exists():
            raise SystemExit(f"Adapter path does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable_lora)
    elif trainable_lora:
        params = lora_config or {}
        config = LoraConfig(
            r=int(params.get("r", 16)),
            lora_alpha=int(params.get("alpha", 32)),
            lora_dropout=float(params.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=params.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, config)

    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def latest_adapter(root: str | Path, agent_name: str) -> str | None:
    base = project_path(root)
    candidates = sorted(base.glob(f"{agent_name}_round_*"))
    return str(candidates[-1]) if candidates else None
