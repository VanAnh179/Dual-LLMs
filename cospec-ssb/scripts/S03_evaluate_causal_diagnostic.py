#!/usr/bin/env python
"""Evaluate matched/shuffled/zero/noise interventions on the trained S02 pipeline."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s03_causal_diagnostic_gsm8k.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--modes", nargs="+", choices=("matched", "shuffled", "zero", "noise"))
    parser.add_argument("--rebuild-z-cache", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.answer_extraction import extract_gsm8k_gold_answer
    from src.data_utils import (
        project_path, read_jsonl, record_sampled_ids, reject_train_rows_for_final_eval,
        reject_train_split_for_final_eval, sample_records, write_json, write_jsonl,
    )
    from src.evaluation import score_prediction
    from src.generation import generate_text
    from src.prompts import format_for_generation, messages_for_single
    from src.S01_hook_utils import HiddenStateExtractor, HiddenStateInjector, get_layer_by_index
    from src.S02_minimal_coupling import MinimalCouplingBridge
    from src.S03_causal_metrics import compute_causal_metrics
    from src.S03_interventions import (
        apply_matched, apply_noise, apply_shuffled, apply_zero, build_derangement, fit_noise_stats,
    )
    from src.S03_reporting import write_s03_report
    from src.S03_runtime import (
        artifact_fingerprints, choose_device_dtype, json_fingerprint, load_receiver_model,
        load_s03_and_s02, require_s02_artifacts,
    )

    cfg, s02 = load_s03_and_s02(args.config)
    paths = require_s02_artifacts(cfg, s02)
    device, dtype = choose_device_dtype()
    rows = read_jsonl(s02["data"]["raw_test_path"])
    reject_train_split_for_final_eval(s02["data"]["raw_test_path"])
    reject_train_rows_for_final_eval(rows)
    n = args.max_examples or int(cfg.get("max_eval_examples", 100))
    rows = sample_records(rows, n, cfg.get("sampling_mode", "first_n"), int(cfg.get("seed", 42)))
    if len(rows) < 2:
        raise SystemExit("S03 shuffle diagnostic requires at least two examples.")
    sample_ids = [str(row["id"]) for row in rows]
    record_sampled_ids("S03_causal_diagnostic", rows, path=cfg["output"]["metrics_path"] + ".sampled_ids.json")
    fingerprints = artifact_fingerprints(cfg, s02, paths)
    fingerprints["selection_sha256"] = json_fingerprint(sample_ids)
    fingerprints["evaluation_dtype"] = str(dtype)

    tokenizer = AutoTokenizer.from_pretrained(s02["student_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache_path = project_path(cfg["output"]["cache_path"])
    cache = None
    if cache_path.exists() and not args.rebuild_z_cache:
        candidate = torch.load(cache_path, map_location="cpu", weights_only=True)
        if candidate.get("fingerprints") == fingerprints and candidate.get("sample_ids") == sample_ids:
            cache = candidate
            print(f"Reusing valid z cache: {cache_path}")

    mc = s02["minimal_coupling"]
    if cache is None:
        model_a = AutoModelForCausalLM.from_pretrained(
            s02["student_model_name"], torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model_a = PeftModel.from_pretrained(model_a, str(paths["agent_a_pre_s02"]), is_trainable=False)
        model_a.eval().requires_grad_(False)
        bridge = MinimalCouplingBridge.load_bridge(paths["bridge_s02"], device=str(device)).to(device).to(dtype)
        bridge.eval()
        extractor = HiddenStateExtractor(get_layer_by_index(model_a, int(mc["layer_index"])))
        messages = []
        with torch.no_grad():
            for index, row in enumerate(rows):
                prompt = format_for_generation(tokenizer, messages_for_single(row["problem"]))
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                extractor.clear()
                model_a(**inputs)
                if extractor.hidden_states is None:
                    raise RuntimeError("Sender hidden-state hook did not capture a tensor.")
                z = bridge.encode(extractor.hidden_states.detach())
                messages.append(z.squeeze(0).float().cpu())
                print(f"cache z: {index + 1}/{len(rows)}", end="\r")
        extractor.remove()
        z_all = torch.stack(messages)
        expected_dim = int(mc["bottleneck_dim"])
        if tuple(z_all.shape) != (len(rows), expected_dim) or not torch.isfinite(z_all).all():
            raise RuntimeError(f"Invalid z cache shape/values: {tuple(z_all.shape)}")
        cache = {
            "z": z_all, "sample_ids": sample_ids, "shape": list(z_all.shape),
            "source_dtype": str(dtype), "stored_dtype": str(z_all.dtype),
            "fingerprints": fingerprints,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        del model_a, bridge
        gc.collect()
        torch.cuda.empty_cache()

    z_all = cache["z"].float()
    permutation = build_derangement(len(rows), int(cfg["shuffle"]["seed"]))
    if torch.any(permutation == torch.arange(len(rows))):
        raise RuntimeError("Shuffle permutation contains a fixed point.")
    stats = fit_noise_stats(z_all, float(cfg["noise"].get("min_std", 1e-6)))
    noise_generator = torch.Generator(device="cpu").manual_seed(int(cfg["noise"]["seed"]))
    mode_z = {
        "matched": apply_matched(z_all),
        "shuffled": apply_shuffled(z_all, permutation),
        "zero": apply_zero(z_all),
        "noise": apply_noise(z_all, stats["mean"], stats["std"], noise_generator),
    }
    if torch.count_nonzero(mode_z["zero"]) != 0:
        raise RuntimeError("Zero intervention is not exactly zero.")
    requested = args.modes or list(cfg["interventions"])
    if set(requested) != set(cfg["interventions"]):
        print("Partial mode run requested; aggregate metrics will be written only when all modes exist.")

    model_b = load_receiver_model(s02, paths["agent_b_s02"], device, dtype)
    model_b.eval()
    bridge = MinimalCouplingBridge.load_bridge(paths["bridge_s02"], device=str(device)).to(device).to(dtype)
    bridge.eval()
    layer_b = get_layer_by_index(model_b, int(mc["layer_index"]))
    gen_cfg = s02["evaluation"]
    generation_dir = project_path(cfg["output"]["generation_dir"])
    generation_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    for mode in requested:
        results = []
        for index, row in enumerate(rows):
            z = mode_z[mode][index:index + 1].to(device=device, dtype=dtype)
            injector = HiddenStateInjector(layer_b, lambda h, z_vec=z: bridge.inject(h, z_vec))
            try:
                prompt = format_for_generation(tokenizer, messages_for_single(row["problem"]))
                prediction = generate_text(tokenizer, model_b, prompt, **gen_cfg)
            finally:
                injector.remove()
            gold = extract_gsm8k_gold_answer(row["raw_answer"])
            correct, extracted = score_prediction(prediction, gold)
            source_index = int(permutation[index]) if mode == "shuffled" else index
            source_id = sample_ids[source_index] if mode in ("matched", "shuffled") else None
            results.append({
                "sample_id": row["id"], "problem": row["problem"], "gold_answer": gold,
                "predicted_text": prediction, "extracted_answer": extracted, "correct": bool(correct),
                "intervention": mode, "z_source_sample_id": source_id,
                "seed": int(cfg.get("seed", 42)), "checkpoint_identifiers": fingerprints,
            })
            print(f"{mode}: {index + 1}/{len(rows)}", end="\r")
        write_jsonl(generation_dir / f"{mode}_predictions.jsonl", results)
        all_predictions[mode] = results

    prediction_paths = {mode: generation_dir / f"{mode}_predictions.jsonl" for mode in cfg["interventions"]}
    if all(path.exists() for path in prediction_paths.values()):
        loaded = {mode: read_jsonl(path) for mode, path in prediction_paths.items()}
        valid_outputs = all(
            [str(row.get("sample_id")) for row in loaded[mode]] == sample_ids
            and all(row.get("checkpoint_identifiers") == fingerprints for row in loaded[mode])
            for mode in cfg["interventions"]
        )
        if not valid_outputs:
            raise RuntimeError(
                "Existing intervention outputs do not match current sample IDs/fingerprints. "
                "Rerun all four modes before aggregating metrics."
            )
        bootstrap = cfg["bootstrap"]
        metrics = compute_causal_metrics(
            loaded, num_resamples=int(bootstrap["num_resamples"]),
            confidence_level=float(bootstrap["confidence_level"]), seed=int(cfg.get("seed", 42)),
        )
        metrics.update({
            "experiment_name": cfg["experiment_name"], "fingerprints": fingerprints,
            "shuffle_permutation": permutation.tolist(),
            "shuffle_source_sample_ids": [sample_ids[int(i)] for i in permutation],
            "noise_statistics": "per_dimension_empirical_mean_std",
        })
        write_json(cfg["output"]["metrics_path"], metrics)
        write_s03_report()
        print(f"\nSaved metrics: {project_path(cfg['output']['metrics_path'])}")


if __name__ == "__main__":
    main()
