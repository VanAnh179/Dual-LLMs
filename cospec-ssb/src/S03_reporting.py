"""Render the S03 report from available machine-readable artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_utils import project_path, read_json


def _value(payload: dict[str, Any], *keys: str, default: Any = "PENDING_FULL_RUN") -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_s03_report() -> None:
    diagnostic = read_json(
        "outputs/S03_causal_diagnostic_gsm8k/metrics/causal_diagnostic.json", default={}
    )
    zero_control = read_json(
        "outputs/S03_zero_control_gsm8k/metrics/eval_metrics.json", default={}
    )
    full_ready = diagnostic.get("n") == 100 and zero_control.get("num_examples") == 100
    parity_status = _value(zero_control, "training_parity", "training_parity_status", default="missing")
    acc = diagnostic.get("accuracy", {})
    deltas = diagnostic.get("deltas_vs_matched", {})
    matched = acc.get("matched")
    zero_trained = zero_control.get("zero_control_accuracy")
    d_shuffle = _value(deltas, "shuffled", "delta")
    d_zero = _value(deltas, "zero", "delta")
    d_control = zero_control.get("delta_matched_vs_zero_control", "PENDING_FULL_RUN")

    verdict = "PENDING_FULL_RUN"
    gate = "HOLD"
    if full_ready and parity_status != "exact":
        verdict = "INCONCLUSIVE_CONTROL_MISMATCH"
    elif full_ready and isinstance(matched, float) and isinstance(zero_trained, float):
        ci_shuffle = _value(deltas, "shuffled", "ci_low", default=-1.0)
        ci_zero = _value(deltas, "zero", "ci_low", default=-1.0)
        dependency = (d_shuffle > 0.10 and ci_shuffle > 0) or (d_zero > 0.10 and ci_zero > 0)
        useful = dependency and d_control >= 0.03 and matched >= 0.58
        if useful:
            verdict, gate = "PASS_USEFUL_COMMUNICATION", "PASS"
        elif dependency:
            verdict = "DEPENDENT_BUT_HARMFUL"
        else:
            verdict = "NO_CAUSAL_DEPENDENCY"

    completion_note = (
        "Both 100-example evaluations are complete and the zero-control training parity "
        f"status is `{parity_status}`."
        if full_ready
        else "The report is interim until both 100-example evaluations exist."
    )

    text = f"""# S03 Causal Diagnostic Verdict

## Artifact audit

S03 uses the model, adapters, bridge, prompt, generation settings, test ordering, and data guards inherited from `configs/s02_minimal_coupling_gsm8k.yaml`. Runtime fingerprints and exact sample IDs are recorded in the diagnostic JSON and z cache. Local development was blocked from GPU execution when S02 artifacts were absent; no S02 artifact was recreated or overwritten.

## Method

The diagnostic caches sender messages for the canonical test order, then evaluates the same trained S02 receiver under matched, whole-dataset deranged shuffle, zero, and per-dimension empirical Gaussian noise interventions. Deltas use a paired bootstrap over correctness vectors. The zero-trained control starts from the same pre-S02 receiver initialization and uses the same S02 data/order/budget while receiving an exactly zero residual message throughout training and evaluation.

## Results

| Quantity | Value |
| --- | ---: |
| matched | {_fmt(acc.get('matched', 'PENDING_FULL_RUN'))} |
| shuffled | {_fmt(acc.get('shuffled', 'PENDING_FULL_RUN'))} |
| zero inference | {_fmt(acc.get('zero', 'PENDING_FULL_RUN'))} |
| noise | {_fmt(acc.get('noise', 'PENDING_FULL_RUN'))} |
| zero-trained control | {_fmt(zero_trained if zero_trained is not None else 'PENDING_FULL_RUN')} |
| delta matched-shuffled | {_fmt(d_shuffle)} |
| delta matched-zero | {_fmt(d_zero)} |
| delta matched-zero-trained | {_fmt(d_control)} |

Reference accuracies are B-alone=0.61, D11 text A-to-B=0.68, and S02 matched=0.37. Dependence and usefulness are assessed separately: a large intervention effect can still be harmful when matched performance remains below B-alone or fails to beat the trained control.

## Limitations

{completion_note} Bootstrap intervals quantify paired sample uncertainty, not training-seed uncertainty. Only one training seed was evaluated, so training-seed variance remains unknown.

```yaml
verdict: {verdict}
s04_gate: {gate}
acc_matched: {_fmt(acc.get('matched', 'PENDING_FULL_RUN'))}
acc_shuffled: {_fmt(acc.get('shuffled', 'PENDING_FULL_RUN'))}
acc_zero_inference: {_fmt(acc.get('zero', 'PENDING_FULL_RUN'))}
acc_noise: {_fmt(acc.get('noise', 'PENDING_FULL_RUN'))}
acc_zero_trained_control: {_fmt(zero_trained if zero_trained is not None else 'PENDING_FULL_RUN')}
delta_shuffle: {_fmt(d_shuffle)}
delta_zero: {_fmt(d_zero)}
delta_matched_vs_zero_control: {_fmt(d_control)}
```
"""
    path = project_path("notes/S03_causal_diagnostic_verdict.md")
    path.write_text(text, encoding="utf-8")
