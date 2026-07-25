"""Shared deterministic TF-IDF leakage probes and V01 gate reporting."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from src.data_utils import load_config, project_path, read_json, read_jsonl, write_json


def _pipeline(c_value: float, seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression

    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000)),
    ])
    classifier = LogisticRegression(
        C=c_value, max_iter=2000, solver="lbfgs", random_state=seed,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _bootstrap_accuracy(
    gold: list[str], predicted: list[str], seed: int, num_resamples: int = 5000
) -> list[float]:
    correct = np.asarray([a == b for a, b in zip(gold, predicted)], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(correct), size=(num_resamples, len(correct)))
    estimates = correct[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _fit_eval(
    train_rows: list[dict], dev_rows: list[dict], test_rows: list[dict],
    field: str, seed: int, shuffled_labels: bool = False,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    x_train = [str(row[field]) for row in train_rows]
    y_train = [str(row["gold_answer"]) for row in train_rows]
    x_dev = [str(row[field]) for row in dev_rows]
    y_dev = [str(row["gold_answer"]) for row in dev_rows]
    x_test = [str(row[field]) for row in test_rows]
    y_test = [str(row["gold_answer"]) for row in test_rows]
    if shuffled_labels:
        y_train = list(y_train)
        np.random.default_rng(seed).shuffle(y_train)

    candidates = []
    for c_value in (0.3, 1.0, 3.0):
        model = _pipeline(c_value, seed)
        model.fit(x_train, y_train)
        dev_prediction = model.predict(x_dev)
        candidates.append((float(accuracy_score(y_dev, dev_prediction)), c_value, model))
    _, best_c, model = max(candidates, key=lambda item: (item[0], -item[1]))
    prediction = model.predict(x_test).tolist()
    labels = sorted(set(y_train) | set(y_dev) | set(y_test))
    accuracy = float(accuracy_score(y_test, prediction))
    return {
        "field": field, "accuracy": accuracy,
        "macro_f1": float(f1_score(y_test, prediction, average="macro", labels=labels)),
        "bootstrap_95_ci": _bootstrap_accuracy(y_test, prediction, seed),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "labels": labels, "class_counts": dict(sorted(Counter(y_test).items())),
        "selected_c": best_c, "dev_candidate_accuracies": {
            str(c_value): score for score, c_value, _ in candidates
        },
        "label_shuffled_training": shuffled_labels,
    }


def run_probe(config_path: str, view: str, seed: int = 42) -> dict[str, Any]:
    if view not in ("a", "b"):
        raise ValueError("view must be 'a' or 'b'")
    cfg = load_config(config_path)
    paths = cfg["output_paths"]
    train_rows = read_jsonl(paths["train"])
    dev_rows = read_jsonl(paths["dev"])
    test_rows = read_jsonl(paths["test"])
    field = f"view_{view}"
    primary = _fit_eval(train_rows, dev_rows, test_rows, field, seed)
    shuffled = _fit_eval(train_rows, dev_rows, test_rows, field, seed + 1, shuffled_labels=True)
    full = _fit_eval(train_rows, dev_rows, test_rows, "full_problem", seed)
    classes = sorted({str(row["gold_answer"]) for row in train_rows})
    random_baseline = 1.0 / len(classes)
    majority = max(Counter(row["gold_answer"] for row in test_rows).values()) / len(test_rows)
    threshold = random_baseline + 0.03
    warnings: list[str] = []
    if shuffled["accuracy"] >= random_baseline + 0.10:
        warnings.append("Label-shuffled sanity accuracy is unexpectedly high.")
    if full["accuracy"] < threshold:
        warnings.append(
            "Full-problem linear probe is also near random; the probe may be too weak for relational matching."
        )
    result = {
        "view": view, "seed": seed, "primary": primary,
        "label_shuffled_sanity": shuffled, "full_problem_diagnostic": full,
        "random_baseline": random_baseline, "majority_baseline": majority,
        "threshold_random_plus_0_03": threshold,
        "gate": "PASS" if primary["accuracy"] < threshold else "FAIL",
        "warnings": warnings,
    }
    combined = read_json(paths["probes"], default={})
    combined[f"view_{view}"] = result
    validation = read_json(paths["validation"], default={})
    if "view_a" in combined and "view_b" in combined:
        combined["gate"] = (
            "PASS" if validation.get("status") == "PASS"
            and combined["view_a"]["gate"] == "PASS"
            and combined["view_b"]["gate"] == "PASS" else "FAIL"
        )
        combined["dataset_validation_status"] = validation.get("status", "MISSING")
        combined["generator_version"] = cfg["generator_version"]
        generator_bytes = project_path("src/V01_csp_generator.py").read_bytes()
        combined["generator_sha256"] = hashlib.sha256(generator_bytes).hexdigest()
    write_json(paths["probes"], combined)
    write_v01_report(config_path)
    return result


def write_v01_report(config_path: str) -> None:
    cfg = load_config(config_path)
    validation = read_json(cfg["output_paths"]["validation"], default={})
    probes = read_json(cfg["output_paths"]["probes"], default={})
    splits = validation.get("splits", {})
    rows = []
    for split in ("train", "dev", "test"):
        item = splits.get(split, {})
        rows.append(
            f"| {split} | {item.get('row_count', 'PENDING')} | {item.get('class_counts', 'PENDING')} |"
        )
    probe_rows = []
    for key in ("view_a", "view_b"):
        item = probes.get(key, {})
        primary = item.get("primary", {})
        probe_rows.append(
            f"| {key} | {primary.get('accuracy', 'PENDING')} | {primary.get('macro_f1', 'PENDING')} | "
            f"{primary.get('bootstrap_95_ci', 'PENDING')} | {item.get('threshold_random_plus_0_03', 'PENDING')} |"
        )
    gate = probes.get("gate", "INTERIM_REVIEW_ONLY")
    sanity_rows = []
    all_warnings: list[str] = []
    for key in ("view_a", "view_b"):
        item = probes.get(key, {})
        sanity_rows.append(
            f"| {key} | {item.get('label_shuffled_sanity', {}).get('accuracy', 'PENDING')} | "
            f"{item.get('full_problem_diagnostic', {}).get('accuracy', 'PENDING')} |"
        )
        all_warnings.extend(item.get("warnings", []))
    warning_text = "\n".join(f"- {warning}" for warning in sorted(set(all_warnings))) or "- PENDING"
    content = f"""# V01 Split-View Dataset Results

## Objective and schema

V01 is a forced-cooperation synthetic CSP. View A maps opaque entities to shared link IDs; View B orders those link IDs across four slots. The target entity's slot is uniquely recoverable only after joining the views. The answer classes are `SLOT_0` through `SLOT_3`, with a 0.25 random baseline.

## Validation

The validator re-read every JSONL row, independently enumerated CSP solutions, checked gold consistency, reproduced formatting from seeds, audited both partial views, and checked duplicate/overlap and class balance.

| Split | Count | Class counts |
| --- | ---: | --- |
{chr(10).join(rows)}

Dataset validation: **{validation.get('status', 'PENDING')}**

## Leakage probes

| Probe | Accuracy | Macro-F1 | 95% bootstrap CI | Gate threshold |
| --- | ---: | ---: | --- | ---: |
{chr(10).join(probe_rows)}

Each probe uses word+character TF-IDF and multinomial logistic regression fit on train, minimally tuned on dev, and evaluated once on test. Label-shuffled training and a full-problem probe are diagnostics. Passing a linear probe only excludes a family of surface shortcuts; it does not prove that an LLM cannot solve one view, so A-only and B-only neural baselines remain necessary.

### Sanity checks and warnings

| Probe | Label-shuffled accuracy | Full-problem accuracy |
| --- | ---: | ---: |
{chr(10).join(sanity_rows)}

{warning_text}

The v01 and v01.1 development artifacts failed the strict A-view threshold because rejection sampling allowed surface/class correlations. They were superseded as whole datasets. Version v01.2 uses combinatorial blocks that balance every repeated A and B view across labels; no test-row filtering was used.

## Gate

```yaml
generator_version: {cfg['generator_version']}
generator_sha256: {probes.get('generator_sha256', 'PENDING')}
dataset_validation: {validation.get('status', 'PENDING')}
gate: {gate}
```
"""
    project_path("notes/V01_split_view_dataset_results.md").write_text(content, encoding="utf-8")
