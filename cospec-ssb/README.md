# GSM8K Dual-Agent Fine-Tuning

This project is a separate minimal pipeline for training-based two-agent collaboration on GSM8K. It intentionally does not depend on or modify `../gsm8k-llm-cascade/`, which is a separate cascade-style workspace.

The goal is to fine-tune two general reasoning agents without fixed roles such as solver and verifier. Agent A and Agent B can each act as the first or second responder depending on the alternating training phase.

Bootstrapping can use a larger teacher model than the student agents. By default, candidate reasoning traces are generated with `Qwen/Qwen2.5-Math-7B-Instruct`, filtered by gold final-answer match, and then used as supervised targets for student LoRA training with `Qwen/Qwen2.5-1.5B-Instruct`. The teacher is not used as Agent A or Agent B during training or final evaluation unless the config explicitly sets the student model to the same model.

## Pipeline

1. Prepare GSM8K-style JSONL data:

```bash
python scripts/prepare_gsm8k.py
```

2. Bootstrap reasoning traces when the raw data does not already provide them:

```bash
python scripts/bootstrap_traces.py --config configs/d11_qwen05b.yaml
```

3. Train LoRA adapters with alternating optimization:

```bash
python scripts/train_alternating_lora.py --config configs/d11_qwen05b.yaml
```

4. Evaluate Agent A alone and A then B:

```bash
python scripts/evaluate_two_agent.py --config configs/d11_qwen05b.yaml
```

## Data Format

`data/raw/train.jsonl` and `data/raw/dev.jsonl` are derived only from the GSM8K train split. `data/raw/test.jsonl` is reserved for final evaluation only. Training, bootstrapping, and other-agent response generation refuse obvious test-split inputs; final evaluation refuses obvious train/dev inputs.

Sampling is config driven:

```yaml
teacher_model_name: Qwen/Qwen2.5-Math-7B-Instruct
student_model_name: Qwen/Qwen2.5-1.5B-Instruct
seed: 42
sampling_mode: first_n  # first_n or random
max_train_examples: 100
max_eval_examples: 100
validation_ratio: 0.1
```

Use `first_n` for deterministic smoke tests. Use `random` to shuffle with the fixed seed before selecting examples for real experiments. Each script records the exact selected IDs in `outputs/metrics/sampled_ids.json`.

Input JSONL records should contain at least:

```json
{"problem": "...", "gold_answer": "..."}
```

Records may also include a supervised reasoning target:

```json
{"problem": "...", "gold_answer": "...", "reasoning_trace": "..."}
```

GSM8K Hugging Face records are converted into:

```json
{"id": "train-0", "problem": "...", "gold_answer": "...", "raw_answer": "..."}
```

## Smoke Test With 10 Examples

```bash
python scripts/prepare_gsm8k.py --max-train-examples 10 --max-test-examples 10
python scripts/bootstrap_traces.py --config configs/d11_qwen05b.yaml --max-examples 10 --num-candidates 1 --max-new-tokens 128
python scripts/train_alternating_lora.py --config configs/d11_qwen05b.yaml --max-train-examples 10 --num-rounds 1
python scripts/evaluate_two_agent.py --config configs/d11_qwen05b.yaml --max-examples 10
```

If dependencies are missing, scripts exit with a clear message pointing to `pip install -r requirements.txt`. CUDA is optional by default for smoke tests; set `require_cuda: true` in the config when you want scripts to fail early on non-CUDA machines.
# CoSpec-SSB

## V02 benchmark tracks

V02 deliberately separates two kinds of evidence:

- `V02_multifamily_split_benchmark` is a controlled synthetic benchmark. Its
  counterfactual blocks guarantee balanced labels and identical partial-view
  equivalence classes, so it tests whether the communication channel is causally
  useful under controlled leakage and difficulty.
- `V02_official_external` is an evaluation-only external-validity track. It adapts
  held-out examples from CLUTRR, ZebraLogicBench, BIG-Bench Hard
  `multistep_arithmetic_two`, and PRM800K. Official test examples are never added
  to V02 training.

The official adapter records the upstream repository, resolved dataset revision,
download hash, license declaration, original source ID, and original split in a
local provenance manifest. Generated data, model outputs, and reports are ignored
by Git.

The official sources are:

- CLUTRR: <https://github.com/facebookresearch/clutrr>
- ZebraLogicBench official leaderboard:
  <https://huggingface.co/spaces/allenai/ZebraLogic>; its public
  answer-bearing evaluation mirror is
  <https://huggingface.co/datasets/WildEval/ZebraLogic>
- BIG-Bench Hard: <https://github.com/suzgunmirac/BIG-Bench-Hard>
- PRM800K: <https://github.com/openai/prm800k>
