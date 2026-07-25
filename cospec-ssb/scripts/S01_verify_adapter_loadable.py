#!/usr/bin/env python
"""Verify that imported D11.0 LoRA adapters load correctly and produce valid outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "S01_verify_adapter_loadable"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def verify_adapter(model_name: str, adapter_path: str, adapter_label: str, sample_text: str) -> bool:
    """Load base model + adapter, run 1 forward pass, check for NaN."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n--- Verifying {adapter_label}: {adapter_path} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )

    try:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    except Exception as exc:
        print(f"[FAIL] Could not load adapter {adapter_label}: {exc}")
        print(f"  Adapter path: {adapter_path}")
        print(f"  To retrain, run:")
        print(f"    python scripts/train_alternating_lora.py (1 Round)")
        print(f"  Output new adapters to: outputs/S01_baseline_retrained/adapters/")
        return False

    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    if torch.isnan(logits).any():
        print(f"[FAIL] {adapter_label} produces NaN logits!")
        return False

    if logits.numel() == 0:
        print(f"[FAIL] {adapter_label} produces empty logits!")
        return False

    print(f"[OK] {adapter_label} loaded successfully. Logits shape: {logits.shape}, "
          f"mean={logits.float().mean().item():.4f}")

    # Clean up
    del model, tokenizer, inputs, outputs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify D11.0 adapter loadability")
    parser.add_argument("--config", type=str, default="configs/s02_minimal_coupling_gsm8k.yaml")
    args = parser.parse_args()

    from src.data_utils import load_config, project_path, read_jsonl

    cfg = load_config(args.config)
    model_name = cfg.get("student_model_name", "Qwen/Qwen2.5-1.5B-Instruct")

    adapter_a_path = str(project_path(cfg_get(cfg, "minimal_coupling", "agent_a_adapter_path")))
    adapter_b_path = str(project_path(cfg_get(cfg, "minimal_coupling", "agent_b_adapter_path")))

    # Load one sample from train for forward test
    train_rows = read_jsonl(cfg_get(cfg, "data", "raw_train_path"), limit=1)
    if not train_rows:
        raise SystemExit("No rows found in train.jsonl")
    sample_text = f"Problem:\n{train_rows[0]['problem']}"

    # Verify both adapters
    ok_a = verify_adapter(model_name, adapter_a_path, "Agent_A", sample_text)
    ok_b = verify_adapter(model_name, adapter_b_path, "Agent_B", sample_text)

    print("\n" + "=" * 60)
    if ok_a and ok_b:
        print("ADAPTER_STATUS: OK")
        print("Both adapters loaded and forward pass succeeded.")
    else:
        print("ADAPTER_STATUS: FAIL")
        if not ok_a:
            print("  - Agent A adapter failed to load or produced invalid output.")
        if not ok_b:
            print("  - Agent B adapter failed to load or produced invalid output.")
        print("\nTo retrain adapters, run:")
        print("  python scripts/train_alternating_lora.py")
        print("  Save output to: outputs/S01_baseline_retrained/adapters/")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
