#!/usr/bin/env python
"""LoRA-SFT full-input and partial-view single-model V02 baselines."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
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
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "view_a", "view_b"),
        default=("full", "view_a", "view_b"),
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.V02_modeling import encode_supervised_batch, select_nested_training_rows
    from src.V02_runtime import require_v02_preflight
    from src.data_utils import project_path, read_json, read_jsonl, write_json

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for V02 training.")
    cfg, _ = require_v02_preflight(args.config)
    train_cfg = cfg["training"]
    train_path = project_path(cfg["data"]["train"])
    original_rows = read_jsonl(train_path)
    seed = int(cfg["seed"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    manifest_path = cfg["outputs"]["training_manifest"]
    manifest = read_json(manifest_path, default={})
    manifest.update({
        "experiment_name": cfg["experiment_name"],
        "model_name": cfg["model_name"],
        "seed": seed,
        "train_dataset_sha256": _sha256(train_path),
        "single_baselines": manifest.get("single_baselines", {}),
    })

    for mode in args.modes:
        rows = select_nested_training_rows(
            original_rows, args.max_examples, seed
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], dtype=dtype, trust_remote_code=True
        ).to(device)
        model.config.use_cache = False
        lora = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
        ))
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        batch_size = int(train_cfg["single_batch_size"])
        grad_accum = int(train_cfg["single_gradient_accumulation_steps"])
        max_length = int(train_cfg["max_seq_length"])
        epochs = int(train_cfg["epochs"])
        logging_steps = int(train_cfg["logging_steps"])
        max_grad_norm = float(train_cfg["max_grad_norm"])
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        micro_step = 0
        running_loss = 0.0
        window_micro_steps = 0
        optimizer_step_losses: list[float] = []
        stop = False
        for epoch in range(epochs):
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                encoded = encode_supervised_batch(
                    tokenizer, mode, batch, max_length, device
                )
                with torch.autocast(
                    device_type="cuda", dtype=dtype,
                    enabled=dtype == torch.float16,
                ):
                    loss = model(**encoded, use_cache=False).loss
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"{mode}: non-finite loss at micro step {micro_step + 1}: "
                        f"{loss_value}"
                    )
                scaler.scale(loss / grad_accum).backward()
                running_loss += loss_value
                window_micro_steps += 1
                micro_step += 1
                should_step = micro_step % grad_accum == 0 or start + batch_size >= len(rows)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    step_loss = running_loss / window_micro_steps
                    optimizer_step_losses.append(step_loss)
                    if global_step == 1 or global_step % logging_steps == 0:
                        print(
                            f"{mode}: step={global_step} epoch={epoch + 1}/{epochs} "
                            f"loss={step_loss:.6f}"
                        )
                    running_loss = 0.0
                    window_micro_steps = 0
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                        break
            if stop:
                break
        output_dir = (
            project_path(cfg["outputs"]["single_adapter_root"]) / mode
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        convergence_window = min(20, len(optimizer_step_losses))
        first_window_mean = (
            sum(optimizer_step_losses[:convergence_window]) / convergence_window
        )
        last_window_mean = (
            sum(optimizer_step_losses[-convergence_window:]) / convergence_window
        )
        manifest["single_baselines"][mode] = {
            "adapter_path": str(output_dir.relative_to(project_path("."))).replace("\\", "/"),
            "num_train_examples": len(rows),
            "optimizer_steps": global_step,
            "epochs_configured": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "dtype": str(dtype).replace("torch.", ""),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in rows).encode()
            ).hexdigest(),
            "initial_optimizer_step_loss": optimizer_step_losses[0],
            "final_optimizer_step_loss": optimizer_step_losses[-1],
            "best_optimizer_step_loss": min(optimizer_step_losses),
            "convergence_window_steps": convergence_window,
            "first_window_mean_loss": first_window_mean,
            "last_window_mean_loss": last_window_mean,
            "loss_improved": last_window_mean < first_window_mean,
            "all_losses_finite": all(
                math.isfinite(value) for value in optimizer_step_losses
            ),
            "optimizer_step_loss_curve": optimizer_step_losses,
        }
        write_json(manifest_path, manifest)
        print(f"Saved {mode} adapter: {output_dir}")
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
