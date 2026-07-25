#!/usr/bin/env python
"""Smoke test: forward one GSM8K example through the minimal coupling pipeline, verify gradients."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "S02_test_forward_single_example"


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, cfg.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward smoke test for minimal coupling pipeline")
    parser.add_argument("--config", type=str, default="configs/s02_minimal_coupling_gsm8k.yaml")
    args = parser.parse_args()

    import torch
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data_utils import load_config, project_path, read_jsonl
    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge

    cfg = load_config(args.config)
    model_name = cfg.get("student_model_name")
    mc = cfg["minimal_coupling"]
    layer_index = mc["layer_index"]
    bottleneck_dim = mc["bottleneck_dim"]
    adapter_a_path = str(project_path(mc["agent_a_adapter_path"]))
    adapter_b_path = str(project_path(mc["agent_b_adapter_path"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load 1 train example
    rows = read_jsonl(cfg_get(cfg, "data", "raw_train_path"), limit=1)
    sample = rows[0]
    problem_text = f"Problem:\n{sample['problem']}"

    print("=" * 60)
    print("S02 Forward Smoke Test")
    print("=" * 60)

    # --- Load Agent A (frozen) ---
    print("\n[1] Loading Agent A (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model_a = PeftModel.from_pretrained(model_a, adapter_a_path, is_trainable=False)
    model_a.eval()
    model_a.requires_grad_(False)
    print(f"  Agent A loaded. Device: {device}")

    # --- Load Agent B (trainable LoRA) ---
    # NOTE: this MUST match the exact same procedure used in S02_train_minimal_coupling.py
    # to ensure the smoke test validates the real training pipeline.
    init_from_d11 = mc.get("init_agent_b_from_d11", True)
    print(f"\n[2] Loading Agent B (init_from_d11={init_from_d11})...")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    lora_cfg = cfg.get("lora", {})
    new_lora = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    if init_from_d11:
        # Merge D11.0 adapter into base weights, then apply fresh LoRA on top
        model_b = PeftModel.from_pretrained(model_b, adapter_b_path, is_trainable=True)
        model_b = model_b.merge_and_unload()
        model_b = get_peft_model(model_b, new_lora)
    else:
        # Fresh LoRA on bare base model (no D11.0 knowledge)
        model_b = get_peft_model(model_b, new_lora)

    model_b.train()
    print(f"  Agent B loaded. Trainable params: "
          f"{sum(p.numel() for p in model_b.parameters() if p.requires_grad):,}")

    # --- Bridge ---
    d_model = model_a.config.hidden_size
    print(f"\n[3] Creating MinimalCouplingBridge (d_model={d_model}, bottleneck={bottleneck_dim})...")
    bridge = MinimalCouplingBridge(d_model=d_model, bottleneck_dim=bottleneck_dim).to(device).to(dtype)
    bridge.train()
    bridge_param_count = sum(p.numel() for p in bridge.parameters())
    print(f"  Bridge params: {bridge_param_count:,}")

    # --- Hook Agent A ---
    layer_a = get_layer_by_index(model_a, layer_index)
    extractor = HiddenStateExtractor(layer_a)
    print(f"\n[4] Hooked extractor on Agent A layer {layer_index} ({type(layer_a).__name__})")

    # --- Tokenize ---
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    print(f"\n[5] Input tokens: {inputs['input_ids'].shape}")

    # --- Forward Agent A (no grad) ---
    print("\n[6] Forward Agent A (torch.no_grad)...")
    with torch.no_grad():
        model_a(**inputs)
    h_a = extractor.hidden_states
    print(f"  H_A shape: {h_a.shape}")
    extractor.remove()

    # --- Encode z ---
    # h_a doesn't need grad (A is frozen), but bridge needs grad
    h_a_detached = h_a.detach()  # detach from A's graph, but keep on GPU for bridge
    z = bridge.encode(h_a_detached)
    print(f"  z shape:   {z.shape}")

    # --- Hook Agent B with injection ---
    # verbose_first_call=True to print whether hidden_states arrives in args or kwargs
    layer_b = get_layer_by_index(model_b, layer_index)

    def injection_fn(h_b: torch.Tensor) -> torch.Tensor:
        return bridge.inject(h_b, z)

    injector = HiddenStateInjector(layer_b, injection_fn, verbose_first_call=True)
    print(f"\n[7] Hooked injector on Agent B layer {layer_index} (verbose_first_call=True)")

    # --- Forward Agent B (with grad) ---
    print("\n[8] Forward Agent B (with injection)...")
    labels = inputs["input_ids"].clone()
    outputs_b = model_b(**inputs, labels=labels)
    loss = outputs_b.loss
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss is NaN: {torch.isnan(loss).item()}")

    # --- Backward ---
    print("\n[9] Backward pass...")
    loss.backward()

    # Check gradients on bridge parameters
    print("\n[10] Gradient check on bridge parameters:")
    all_grads_ok = True
    for name, param in bridge.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad norm = {grad_norm:.6f}")
        else:
            print(f"  {name}: grad = None  [PROBLEM]")
            all_grads_ok = False

    # Check some LoRA params in B
    print("\n[11] Gradient check on Agent B LoRA (sample):")
    checked = 0
    for name, param in model_b.named_parameters():
        if param.requires_grad and param.grad is not None and "lora" in name.lower():
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
            checked += 1
            if checked >= 3:
                break

    injector.remove()

    # --- Summary ---
    print("\n" + "=" * 60)
    if not torch.isnan(loss) and all_grads_ok:
        print("SMOKE TEST: PASS")
        print(f"  Loss:        {loss.item():.4f}")
        print(f"  H_A shape:   {h_a.shape}")
        print(f"  z shape:     {z.shape}")
        print(f"  Bridge grad: OK (all params have non-None gradients)")
    else:
        print("SMOKE TEST: FAIL")
        if torch.isnan(loss):
            print("  Loss is NaN!")
        if not all_grads_ok:
            print("  Some bridge parameters have None gradients!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
