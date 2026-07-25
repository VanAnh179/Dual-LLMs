# Codex Plan — D11.2 Latent Collaborative SFT

Use this prompt/plan for Codex. It is designed for the project:

```text
DualLLMs/gsm8k-dual-agent-finetune/
```

---

## Initial instruction for Codex

```text
You are working in:

DualLLMs/gsm8k-dual-agent-finetune/

Current status:
- We already have a minimal alternating LoRA-SFT pipeline.
- Student model: Qwen/Qwen2.5-1.5B-Instruct.
- Previous LoRA-SFT result on 100 GSM8K test examples:
  - Agent A alone: 0.58
  - Agent B alone: 0.65
  - A_then_B: 0.68
  - B_then_A: 0.63
- This suggests early collaboration behavior.
- Now we want to revise the training objective toward latent collaboration, not solver/verifier correction.

Important conceptual requirement:
Do NOT frame the system as solver/verifier.
Do NOT use labels such as rescue, keep_correct, accept, reject, verdict, correct/incorrect assessment of the other agent.
We want the agents to learn flexible collaboration patterns:
- one agent may decompose the problem,
- another may compute,
- one may sketch,
- another may complete,
- one may identify quantities,
- another may synthesize,
without hard-coding these roles.

Default teacher model:
Qwen/Qwen2.5-Math-7B-Instruct

Default student model:
Qwen/Qwen2.5-1.5B-Instruct

The teacher is only used to generate high-quality collaborative traces.
The teacher must NOT be used during final evaluation.

Implement the following plan step by step.
```

---

## PHASE 0 — Preserve existing baseline

```text
Goal:
Before changing anything, preserve the current D11.0 baseline.

Tasks:
1. Create a notes file:
   notes/D11_0_baseline_results.md

2. Record the current result:
   student_model_name: Qwen/Qwen2.5-1.5B-Instruct
   agent_a_adapter: outputs/adapters/agent_A_round_1
   agent_b_adapter: outputs/adapters/agent_B_round_1
   num_examples: 100
   agent_A_alone_accuracy: 0.58
   agent_B_alone_accuracy: 0.65
   A_then_B_accuracy: 0.68
   B_then_A_accuracy: 0.63

3. Do not overwrite old adapters.
4. Create a new experiment name:
   D11_2_latent_collaborative_sft
5. All new outputs should go under:
   outputs/D11_2_latent_collaborative_sft/
```

---

## PHASE 1 — Change data objective to latent collaborative traces

```text
Goal:
Replace ordinary reasoning traces with teacher-generated two-agent complementary traces.

Do NOT generate:
- verifier labels
- accept/reject labels
- rescue labels
- correctness labels for the other agent

New data format:
Each JSONL example should contain:

{
  "id": "...",
  "problem": "...",
  "gold_answer": "...",
  "agent1_contribution": "...",
  "agent2_contribution": "...",
  "joint_solution": "...",
  "final_answer": "..."
}

Teacher prompt:
Use this exact conceptual style:

System:
You are generating training data for a two-agent reasoning system.
Do not assign fixed roles such as solver, verifier, critic, calculator, planner, or executor.
The two agents should provide useful and complementary contributions.
They may decompose the problem, identify quantities, form equations, compute intermediate values, check consistency, or synthesize the final solution.
The goal is to produce a correct final answer.

User:
Solve the following GSM8K problem as a collaboration between two agents.

Problem:
{problem}

Return exactly this format:

Agent 1 contribution:
...

Agent 2 contribution:
...

Joint solution:
...

Final answer:
...

Requirements:
- Agent 1 and Agent 2 should not merely copy each other.
- Agent 2 should add useful information beyond Agent 1.
- The joint solution should be complete and clear.
- The final answer should be concise.

Implementation tasks:
1. Modify or create:
   scripts/bootstrap_collaborative_traces.py

2. It should load:
   teacher_model_name from config.

3. Default:
   teacher_model_name: Qwen/Qwen2.5-Math-7B-Instruct

4. It should generate K candidate collaborative traces per problem.

5. It should parse:
   Agent 1 contribution
   Agent 2 contribution
   Joint solution
   Final answer

6. It should filter only by final answer:
   keep example if extracted Final answer matches gold_answer.

7. Save kept examples to:
   data/filtered/D11_2_latent_collaborative_sft/two_agent_traces.jsonl

8. Save failed/unparsed examples to:
   outputs/D11_2_latent_collaborative_sft/generations/bootstrap_failed.jsonl

9. Save stats:
   outputs/D11_2_latent_collaborative_sft/metrics/bootstrap_stats.json

Stats should include:
- teacher_model_name
- num_input_examples
- num_candidates_per_example
- num_total_candidates
- parse_success_count
- parse_success_rate
- answer_match_count
- answer_match_rate
- kept_example_count
- average_agent1_contribution_length
- average_agent2_contribution_length
- average_joint_solution_length
```

---

## PHASE 2 — Build training datasets from collaborative traces

```text
Goal:
Construct SFT datasets for two types of behavior:
1. first contributor behavior
2. second contributor + final synthesis behavior

Do not use solver/verifier wording.

Create or modify:
scripts/build_collaborative_sft_data.py

Input:
data/filtered/D11_2_latent_collaborative_sft/two_agent_traces.jsonl

Output files:
data/train/D11_2_latent_collaborative_sft/first_contributor_train.jsonl
data/train/D11_2_latent_collaborative_sft/second_contributor_train.jsonl
data/train/D11_2_latent_collaborative_sft/mixed_collaborative_train.jsonl

First contributor training example:

System:
You are one agent in a two-agent reasoning system.
Your job is to contribute useful information toward solving the problem.
The other agent may later add a different but useful contribution.
Do not assume a fixed role.

User:
Problem:
{problem}

Assistant:
Contribution:
{agent1_contribution}

Second contributor / final synthesis training example:

System:
You are one agent in a two-agent reasoning system.
Another agent has provided a contribution.
Your job is to add useful complementary information and produce a final joint solution.
You may use, extend, refine, or synthesize the other contribution.
Do not assume a fixed role such as solver or verifier.

User:
Problem:
{problem}

Other agent's contribution:
{agent1_contribution}

Assistant:
Contribution:
{agent2_contribution}

Joint solution:
{joint_solution}

Final answer:
{gold_answer}

Mixed dataset:
- Include both first contributor and second contributor examples.
- Add a field:
  "training_mode": "first_contributor" or "second_contributor"

Important:
- Do not include assessment of whether the other agent is correct.
- Do not include words like verdict, accept, reject, rescue.
```

---

## PHASE 3 — Train D11.2 with standard SFT first

```text
Goal:
Before changing the loss, train the new data objective with standard LoRA-SFT.
This isolates whether the new collaborative data objective helps.

Create or modify:
scripts/train_collaborative_lora_sft.py

Training setup:
- student_model_name: Qwen/Qwen2.5-1.5B-Instruct
- LoRA adapters for Agent A and Agent B
- Use standard token-level cross-entropy SFT loss.
- Do NOT implement custom loss in this phase.

Training variants:
Variant 1:
- Train Agent A and Agent B on mixed_collaborative_train.jsonl independently.
- Save:
  outputs/D11_2_latent_collaborative_sft/adapters/agent_A_mixed_sft/
  outputs/D11_2_latent_collaborative_sft/adapters/agent_B_mixed_sft/

Variant 2:
- Alternating collaborative SFT:
  Round 1:
    Freeze Agent A.
    Train Agent B as second contributor/final synthesizer.
  Round 2:
    Freeze Agent B.
    Train Agent A as second contributor/final synthesizer.
- Save:
  outputs/D11_2_latent_collaborative_sft/adapters/agent_A_round_1/
  outputs/D11_2_latent_collaborative_sft/adapters/agent_B_round_1/

Keep it simple:
- num_alternating_rounds configurable
- default 1 round
- max_train_examples configurable
- default 100 for smoke test
- do not overwrite D11.0 adapters
```

---

## PHASE 4 — Evaluate D11.2 against D11.0 and base

```text
Goal:
Evaluate whether the new data objective improves over:
1. base single model
2. D11.0 standard LoRA-SFT
3. D11.2 latent collaborative SFT

Create or modify:
scripts/evaluate_collaborative_agents.py

Evaluation modes:
1. base_single
2. agent_A_alone
3. agent_B_alone
4. A_then_B
5. B_then_A

For A_then_B:
- Agent A receives problem and produces Contribution.
- Agent B receives problem + Agent A contribution.
- Agent B produces:
  Contribution
  Joint solution
  Final answer

For B_then_A:
- Symmetric.

Metrics:
{
  "experiment_name": "D11_2_latent_collaborative_sft",
  "teacher_model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
  "student_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "agent_a_adapter": "...",
  "agent_b_adapter": "...",
  "base_single_accuracy": ...,
  "agent_A_alone_accuracy": ...,
  "agent_B_alone_accuracy": ...,
  "A_then_B_accuracy": ...,
  "B_then_A_accuracy": ...,
  "num_examples": ...,
  "sampling_mode": ...,
  "seed": ...
}

Also save per-example predictions:
outputs/D11_2_latent_collaborative_sft/generations/eval_predictions.jsonl

Each prediction should include:
{
  "id": "...",
  "problem": "...",
  "gold_answer": "...",
  "base_prediction": "...",
  "agent_A_contribution": "...",
  "agent_B_contribution": "...",
  "A_then_B_final_answer": "...",
  "B_then_A_final_answer": "...",
  "is_A_then_B_correct": true/false,
  "is_B_then_A_correct": true/false
}

Important:
Use the same 100 test examples as before for comparability:
- input: data/raw/test.jsonl
- sampling_mode: first_n
- max_eval_examples: 100
- seed: 42
```

---

## PHASE 5 — Add post-hoc collaboration analysis

```text
Goal:
Analyze whether agents are actually contributing differently.
This is analysis only, not training supervision.

Create:
scripts/analyze_contributions.py

Input:
outputs/D11_2_latent_collaborative_sft/generations/eval_predictions.jsonl

Report:
1. Average length of Agent A contribution.
2. Average length of Agent B contribution.
3. Lexical overlap between A and B contributions.
4. Number overlap:
   - numbers appearing in A contribution
   - numbers appearing in B contribution
   - new numbers introduced by B
5. Equation/operator overlap:
   - +, -, *, /, = occurrences
6. Cases where:
   - A_then_B correct
   - Agent A alone wrong
7. Cases where:
   - A contribution is short but B produces correct full solution.
8. Cases where:
   - A and B contributions are almost identical.

Save:
outputs/D11_2_latent_collaborative_sft/metrics/contribution_analysis.json
outputs/D11_2_latent_collaborative_sft/analysis/interesting_examples.md

Do not use this analysis as labels during training.
```

---

## PHASE 6 — Prepare custom loss, but do not enable by default

```text
Goal:
Add infrastructure for future loss changes without changing the main experiment yet.

Do not implement rescue-weighted loss.
Do not use other_agent_correct.
Do not use accept/reject labels.

Add TODO infrastructure for two optional future losses:

1. Anti-copy regularization:
   Penalize second contribution if it copies too much from first contribution.
   This should be a TODO only unless explicitly enabled later.

2. Contribution diversity regularization:
   Encourage the two contributions to contain complementary information.
   This should be a TODO only unless explicitly enabled later.

Config fields:

loss:
  type: standard_sft
  anti_copy_weight: 0.0
  diversity_weight: 0.0

For now:
- type must remain standard_sft
- anti_copy_weight = 0.0
- diversity_weight = 0.0

Add clear comments in code:
D11.2 changes the data objective, not the mathematical loss yet.
The first controlled experiment should compare old data objective vs new collaborative data objective under the same SFT loss.
```

---

## PHASE 7 — Smoke test commands

```text
After implementation, print exact commands for a smoke test.

Commands should include:

1. Bootstrap collaborative traces with 10 examples:

python scripts/bootstrap_collaborative_traces.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-examples 10 \
  --num-candidates 1 \
  --max-new-tokens 512

2. Build collaborative SFT data:

python scripts/build_collaborative_sft_data.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml

3. Train with 10 examples:

python scripts/train_collaborative_lora_sft.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-train-examples 10

4. Evaluate with 10 examples:

python scripts/evaluate_collaborative_agents.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-examples 10 \
  --sampling-mode first_n

5. Analyze contributions:

python scripts/analyze_contributions.py \
  --predictions outputs/D11_2_latent_collaborative_sft/generations/eval_predictions.jsonl

Make sure all scripts fail clearly if:
- teacher model cannot be loaded
- adapters are missing
- CUDA memory is insufficient
- parsed collaborative traces are empty
```

---

## PHASE 8 — Config file

```text
Create:

configs/d11_2_qwen_math7b_teacher.yaml

Content:

experiment_name: D11_2_latent_collaborative_sft

teacher_model_name: Qwen/Qwen2.5-Math-7B-Instruct
student_model_name: Qwen/Qwen2.5-1.5B-Instruct

data:
  raw_train_path: data/raw/train.jsonl
  raw_test_path: data/raw/test.jsonl
  filtered_trace_path: data/filtered/D11_2_latent_collaborative_sft/two_agent_traces.jsonl
  train_dir: data/train/D11_2_latent_collaborative_sft

output:
  root_dir: outputs/D11_2_latent_collaborative_sft
  adapter_dir: outputs/D11_2_latent_collaborative_sft/adapters
  generation_dir: outputs/D11_2_latent_collaborative_sft/generations
  metrics_dir: outputs/D11_2_latent_collaborative_sft/metrics
  analysis_dir: outputs/D11_2_latent_collaborative_sft/analysis

sampling:
  seed: 42
  sampling_mode: first_n
  max_train_examples: 100
  max_eval_examples: 100

bootstrap:
  num_candidates: 4
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9

training:
  num_alternating_rounds: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_train_epochs: 1
  logging_steps: 10
  save_steps: 100

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

loss:
  type: standard_sft
  anti_copy_weight: 0.0
  diversity_weight: 0.0
```

---

## PHASE 9 — Final report back to user

```text
After implementation and smoke test, return a concise report:

1. Files created/modified.
2. Folder tree for D11.2 outputs.
3. Bootstrap stats:
   - parse success
   - answer match rate
   - number of kept traces
4. Training status:
   - adapters saved where
5. Evaluation metrics:
   - base_single_accuracy
   - agent_A_alone_accuracy
   - agent_B_alone_accuracy
   - A_then_B_accuracy
   - B_then_A_accuracy
6. Contribution analysis summary.
7. Any errors or limitations.

Do not claim improvement unless D11.2 beats:
- base_single_accuracy
- D11.0 A_then_B accuracy = 0.68 on the same 100 examples.
```

---

## Short design summary

```text
D11.0: Existing LoRA-SFT baseline, best A_then_B = 0.68 on 100 GSM8K test examples.
D11.2: Change data objective to latent collaborative SFT.
Teacher: Qwen/Qwen2.5-Math-7B-Instruct.
Student: Qwen/Qwen2.5-1.5B-Instruct.
Training loss for first controlled experiment: standard SFT cross-entropy.
Custom loss: prepare TODO infrastructure only, do not enable yet.
Main comparison: old data objective vs new collaborative data objective under the same SFT loss.
```
