#!/usr/bin/env python
"""Evaluate V02 single-model controls and matched/shuffled/zero latent split modes."""
from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction(row, mode, text, extract_option):
    answer = extract_option(text)
    return {
        "sample_id": row["sample_id"],
        "family": row["family"],
        "difficulty": row["difficulty"],
        "block_id": row["block_id"],
        "mode": mode,
        "gold_answer": row["gold_answer"],
        "predicted_answer": answer,
        "predicted_text": text.strip(),
        "correct": answer == row["gold_answer"],
    }


def _shuffled_sources(rows: list[dict], seed: int) -> list[int]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["family"], row["difficulty"])].append(index)
    source = list(range(len(rows)))
    rng = random.Random(seed)
    for indices in groups.values():
        by_block = defaultdict(list)
        for index in indices:
            by_block[rows[index]["block_id"]].append(index)
        blocks = list(by_block)
        rng.shuffle(blocks)
        if len(blocks) < 2:
            continue
        for block_index, target_block in enumerate(blocks):
            origin_block = blocks[(block_index + 1) % len(blocks)]
            targets = list(by_block[target_block])
            origins = list(by_block[origin_block])
            rng.shuffle(targets)
            rng.shuffle(origins)
            if len(targets) != len(origins):
                raise RuntimeError("Shuffled-control blocks have unequal sizes.")
            for target, origin in zip(targets, origins):
                source[target] = origin
    if len(rows) > 1 and any(index == origin for index, origin in enumerate(source)):
        fallback = list(range(1, len(rows))) + [0]
        source = [
            fallback[index] if origin == index else origin
            for index, origin in enumerate(source)
        ]
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v02_split_vs_single.yaml")
    parser.add_argument(
        "--external-config",
        help="Evaluate trained V02 artifacts on a validated official holdout.",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--test-split",
        default="test",
        help="Configured data split to evaluate (for example test or iid_test).",
    )
    all_modes = (
        "base_full", "single_full", "single_a", "single_b",
        "split_matched", "split_shuffled", "split_zero",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=all_modes,
        default=all_modes,
        help="Evaluate only these conditions. Useful for learning curves.",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.V02_latent_bridge import MaskedLatentBridge
    from src.V02_modeling import (
        encode_prompt_batch, extract_option, paired_stratified_bootstrap,
        select_nested_training_rows, summarize_predictions,
    )
    from src.V02_runtime import (
        require_official_external_preflight, require_training_artifacts,
        require_v02_preflight,
    )
    from src.data_utils import (
        project_path, read_json, read_jsonl, write_json, write_jsonl,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for V02 model evaluation.")
    cfg, _ = require_v02_preflight(args.config)
    selected_modes = set(args.modes)
    require_training_artifacts(
        cfg,
        require_full=args.max_examples is None,
        modes=selected_modes,
    )
    eval_cfg = cfg["evaluation"]
    external_cfg = None
    external_manifest = None
    if args.external_config:
        external_cfg, external_manifest = require_official_external_preflight(
            args.external_config
        )
        test_path = project_path(external_cfg["output_paths"]["test"])
        metrics_path = external_cfg["output_paths"]["metrics"]
        generation_path = external_cfg["output_paths"]["generation_dir"]
    else:
        if args.test_split not in cfg["data"]:
            raise SystemExit(
                f"Unknown configured test split {args.test_split!r}; "
                f"available={sorted(cfg['data'])}"
            )
        test_path = project_path(cfg["data"][args.test_split])
        if args.test_split == "test":
            metrics_path = cfg["outputs"]["metrics"]
            generation_path = cfg["outputs"]["generation_dir"]
        else:
            metrics_base = Path(cfg["outputs"]["metrics"])
            metrics_path = str(
                metrics_base.with_name(
                    f"{metrics_base.stem}_{args.test_split}{metrics_base.suffix}"
                )
            ).replace("\\", "/")
            generation_path = str(
                Path(cfg["outputs"]["generation_dir"]) / args.test_split
            ).replace("\\", "/")
    rows = read_jsonl(test_path)
    if args.max_examples is not None:
        if external_cfg is None and all("block_id" in row for row in rows):
            rows = select_nested_training_rows(
                rows, args.max_examples, int(cfg["seed"]) + 1009
            )
        else:
            rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit("No test examples selected.")
    device = torch.device("cuda")
    dtype_name = str(eval_cfg.get("dtype", "float16"))
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise SystemExit(f"Unsupported evaluation dtype: {dtype_name}")
    dtype = dtype_map[dtype_name]
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    batch_size = int(args.batch_size or eval_cfg["batch_size"])
    max_length = int(
        (external_cfg or {}).get(
            "max_seq_length", cfg["training"]["max_seq_length"]
        )
    )
    max_new_tokens = int(eval_cfg["max_new_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    generation_dir = project_path(generation_path)
    generation_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, list[dict]] = {}
    model_source = os.environ.get("COSPEC_MODEL_PATH", cfg["model_name"])

    def load_base_model(label: str):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"Loading {label} from {model_source}: "
            f"free_gpu={free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB",
            flush=True,
        )
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                dtype=dtype,
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True,
            ).to(device)
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(f"Loaded {label} on GPU", flush=True)
        return model

    def evaluate_single(mode_name: str, prompt_mode: str, adapter_path: Path | None):
        model = load_base_model(mode_name)
        if adapter_path is not None:
            if not adapter_path.exists():
                raise SystemExit(f"Missing adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = encode_prompt_batch(
                tokenizer, prompt_mode, batch, max_length, device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            predictions.extend(
                _prediction(row, mode_name, text, extract_option)
                for row, text in zip(batch, texts)
            )
            print(f"{mode_name}: {min(start + batch_size, len(rows))}/{len(rows)}", end="\r")
        print(
            f"{mode_name}: accuracy="
            f"{sum(int(row['correct']) for row in predictions) / len(predictions):.4f}"
        )
        write_jsonl(generation_dir / f"{mode_name}.jsonl", predictions)
        all_predictions[mode_name] = predictions
        del model
        gc.collect()
        torch.cuda.empty_cache()

    single_root = project_path(cfg["outputs"]["single_adapter_root"])
    single_plan = (
        ("base_full", "full", None),
        ("single_full", "full", single_root / "full"),
        ("single_a", "view_a", single_root / "view_a"),
        ("single_b", "view_b", single_root / "view_b"),
    )
    for mode_name, prompt_mode, adapter_path in single_plan:
        if mode_name in selected_modes:
            evaluate_single(mode_name, prompt_mode, adapter_path)

    split_modes = selected_modes & {
        "split_matched", "split_shuffled", "split_zero"
    }
    if split_modes:
        bridge_path = project_path(cfg["outputs"]["bridge"])
        receiver_path = project_path(cfg["outputs"]["split_adapter"])
        if not bridge_path.exists() or not receiver_path.exists():
            raise SystemExit("Split adapter/bridge artifacts are missing.")
        bridge = MaskedLatentBridge.load(bridge_path, map_location=device).to(
            device=device, dtype=dtype
        )
        bridge.eval()
        model_a = load_base_model("split encoder")
        model_a.eval().requires_grad_(False)
        extractor = HiddenStateExtractor(
            get_layer_by_index(model_a, int(cfg["bridge"]["layer_index"]))
        )
        latent_chunks = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = encode_prompt_batch(
                tokenizer, "split_a", batch, max_length, device
            )
            extractor.clear()
            with torch.inference_mode():
                model_a(**encoded, use_cache=False)
            latent_chunks.append(
                bridge.encode(
                    extractor.hidden_states, encoded["attention_mask"]
                ).detach().cpu()
            )
            print(
                f"encode_a: {min(start + batch_size, len(rows))}/{len(rows)}",
                end="\r",
            )
        extractor.remove()
        extractor.clear()
        all_z = torch.cat(latent_chunks, dim=0)
        print(f"Encoded {len(all_z)} latent messages.")
        del model_a, extractor, latent_chunks, encoded
        gc.collect()
        torch.cuda.empty_cache()

        model_b = load_base_model("split receiver")
        model_b = PeftModel.from_pretrained(model_b, receiver_path, is_trainable=False)
        model_b.eval()
        receiver_layer = get_layer_by_index(
            model_b, int(cfg["bridge"]["layer_index"])
        )
        shuffled_source = _shuffled_sources(rows, int(cfg["seed"]) + 91)
        for intervention in (
            "split_matched", "split_shuffled", "split_zero"
        ):
            if intervention not in split_modes:
                continue
            predictions = []
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                indices = list(range(start, start + len(batch)))
                if intervention == "split_matched":
                    z = all_z[indices].to(device=device, dtype=dtype)
                    source_ids = [rows[index]["sample_id"] for index in indices]
                elif intervention == "split_shuffled":
                    origins = [shuffled_source[index] for index in indices]
                    z = all_z[origins].to(device=device, dtype=dtype)
                    source_ids = [rows[index]["sample_id"] for index in origins]
                else:
                    z = torch.zeros(
                        (len(batch), int(cfg["bridge"]["bottleneck_dim"])),
                        device=device,
                        dtype=dtype,
                    )
                    source_ids = [None] * len(batch)
                encoded = encode_prompt_batch(
                    tokenizer, "split_b", batch, max_length, device
                )
                injector = HiddenStateInjector(
                    receiver_layer,
                    lambda hidden, message=z: bridge.inject(hidden, message),
                )
                try:
                    with torch.inference_mode():
                        generated = model_b.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                finally:
                    injector.remove()
                texts = tokenizer.batch_decode(
                    generated[:, encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                for row, text, source_id in zip(batch, texts, source_ids):
                    prediction = _prediction(
                        row, intervention, text, extract_option
                    )
                    prediction["message_source_sample_id"] = source_id
                    predictions.append(prediction)
                print(
                    f"{intervention}: "
                    f"{min(start + batch_size, len(rows))}/{len(rows)}",
                    end="\r",
                )
            print(
                f"{intervention}: accuracy="
                f"{sum(int(row['correct']) for row in predictions) / len(predictions):.4f}"
            )
            write_jsonl(generation_dir / f"{intervention}.jsonl", predictions)
            all_predictions[intervention] = predictions
        del model_b, bridge
        gc.collect()
        torch.cuda.empty_cache()

    summaries = {
        mode: summarize_predictions(predictions)
        for mode, predictions in all_predictions.items()
    }
    resamples = int(eval_cfg["bootstrap_resamples"])
    seed = int(cfg["seed"])
    comparison_specs = (
        ("split_matched_vs_single_full", "split_matched", "single_full", seed),
        ("split_matched_vs_base_full", "split_matched", "base_full", seed + 1),
        ("split_matched_vs_shuffled", "split_matched", "split_shuffled", seed + 2),
        ("split_matched_vs_zero", "split_matched", "split_zero", seed + 3),
    )
    comparisons = {
        name: paired_stratified_bootstrap(
            all_predictions[left], all_predictions[right], resamples, pair_seed
        )
        for name, left, right, pair_seed in comparison_specs
        if left in all_predictions and right in all_predictions
    }
    primary = comparisons.get("split_matched_vs_single_full")
    if primary is None:
        primary_verdict = "NOT_EVALUATED"
        low = high = None
    else:
        low, high = primary["bootstrap_95_ci"]
        if low > 0:
            primary_verdict = "SPLIT_BETTER"
        elif high < 0:
            primary_verdict = "SINGLE_FULL_BETTER"
        else:
            primary_verdict = "INCONCLUSIVE"
    metrics = {
        "experiment_name": (
            external_cfg["experiment_name"] if external_cfg else cfg["experiment_name"]
        ),
        "training_experiment_name": cfg["experiment_name"],
        "evaluation_track": (
            "official_external"
            if external_cfg
            else f"controlled_synthetic_{args.test_split}"
        ),
        "selected_modes": list(args.modes),
        "official_eval_only": bool(external_cfg),
        "model_name": cfg["model_name"],
        "num_examples": len(rows),
        "test_dataset_sha256": _sha256(test_path),
        "dtype": dtype_name,
        "decoding": "greedy",
        "training_manifest": read_json(cfg["outputs"]["training_manifest"], default={}),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "primary_endpoint": "split_matched_accuracy_minus_single_full_accuracy",
        "primary_verdict": primary_verdict,
        "report_status": "AWAITING_RESEARCH_REPORT",
    }
    if external_manifest is not None:
        metrics["official_provenance_manifest"] = external_manifest
    write_json(metrics_path, metrics)
    print(f"Saved metrics: {project_path(metrics_path)}")
    if primary is None:
        print(f"primary_verdict={primary_verdict}")
    else:
        print(
            f"primary_verdict={primary_verdict} "
            f"delta={primary['delta']:+.4f} CI={[low, high]}"
        )


if __name__ == "__main__":
    main()
