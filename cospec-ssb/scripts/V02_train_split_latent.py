#!/usr/bin/env python
"""Train the V02 latent split model: frozen A, trainable bridge and receiver LoRA."""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v02_split_vs_single.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.V02_latent_bridge import MaskedLatentBridge
    from src.V02_modeling import encode_prompt_batch, encode_supervised_batch
    from src.V02_runtime import require_v02_preflight
    from src.data_utils import project_path, read_json, read_jsonl, write_json

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for V02 split training.")
    cfg, _ = require_v02_preflight(args.config)
    train_cfg = cfg["training"]
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    train_path = project_path(cfg["data"]["train"])
    rows = read_jsonl(train_path)
    random.Random(seed).shuffle(rows)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_a = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=dtype, trust_remote_code=True
    ).to(device)
    model_a.eval().requires_grad_(False)
    model_a.config.use_cache = False
    model_b = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=dtype, trust_remote_code=True
    ).to(device)
    model_b.config.use_cache = False
    lora = cfg["lora"]
    model_b = get_peft_model(model_b, LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    ))
    model_b.train()
    bridge_cfg = cfg["bridge"]
    bridge = MaskedLatentBridge(
        int(model_a.config.hidden_size), int(bridge_cfg["bottleneck_dim"])
    ).to(device=device, dtype=dtype)
    bridge.train()
    receiver_params = [parameter for parameter in model_b.parameters() if parameter.requires_grad]
    trainable = list(bridge.parameters()) + receiver_params
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    layer_index = int(bridge_cfg["layer_index"])
    extractor = HiddenStateExtractor(get_layer_by_index(model_a, layer_index))
    receiver_layer = get_layer_by_index(model_b, layer_index)
    batch_size = int(train_cfg["split_batch_size"])
    grad_accum = int(train_cfg["split_gradient_accumulation_steps"])
    max_length = int(train_cfg["max_seq_length"])
    epochs = int(train_cfg["epochs"])
    logging_steps = int(train_cfg["logging_steps"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    stop = False
    for epoch in range(epochs):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            a_inputs = encode_prompt_batch(
                tokenizer, "split_a", batch, max_length, device
            )
            extractor.clear()
            with torch.no_grad():
                model_a(**a_inputs, use_cache=False)
            if extractor.hidden_states is None:
                raise RuntimeError("Agent A hidden-state hook did not fire.")
            z = bridge.encode(
                extractor.hidden_states.detach(), a_inputs["attention_mask"]
            )
            b_inputs = encode_supervised_batch(
                tokenizer, "split_b", batch, max_length, device
            )
            injector = HiddenStateInjector(
                receiver_layer, lambda hidden, message=z: bridge.inject(hidden, message)
            )
            try:
                loss = model_b(**b_inputs, use_cache=False).loss
            finally:
                injector.remove()
            (loss / grad_accum).backward()
            running_loss += float(loss.detach())
            micro_step += 1
            should_step = micro_step % grad_accum == 0 or start + batch_size >= len(rows)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % logging_steps == 0:
                    print(
                        f"split: step={global_step} epoch={epoch + 1}/{epochs} "
                        f"loss={running_loss / min(grad_accum, micro_step):.6f}"
                    )
                running_loss = 0.0
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break
    extractor.remove()
    adapter_path = project_path(cfg["outputs"]["split_adapter"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    model_b.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    bridge_path = project_path(cfg["outputs"]["bridge"])
    bridge.save(bridge_path)
    manifest = read_json(cfg["outputs"]["training_manifest"], default={})
    manifest.update({
        "experiment_name": cfg["experiment_name"],
        "model_name": cfg["model_name"],
        "seed": seed,
        "train_dataset_sha256": _sha256(train_path),
    })
    manifest["split_latent"] = {
        "receiver_adapter_path": cfg["outputs"]["split_adapter"],
        "bridge_path": cfg["outputs"]["bridge"],
        "num_train_examples": len(rows),
        "optimizer_steps": global_step,
        "epochs_configured": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "receiver_trainable_parameters": sum(p.numel() for p in receiver_params),
        "bridge_trainable_parameters": sum(p.numel() for p in bridge.parameters()),
        "total_trainable_parameters": sum(p.numel() for p in trainable),
        "dtype": str(dtype).replace("torch.", ""),
        "layer_index": layer_index,
        "bottleneck_dim": int(bridge_cfg["bottleneck_dim"]),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(str(row["sample_id"]) for row in rows).encode()
        ).hexdigest(),
    }
    write_json(cfg["outputs"]["training_manifest"], manifest)
    print(f"Saved receiver adapter: {adapter_path}")
    print(f"Saved latent bridge: {bridge_path}")


if __name__ == "__main__":
    main()
