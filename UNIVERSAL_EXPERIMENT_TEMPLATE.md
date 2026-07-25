# 🧬 UNIVERSAL EXPERIMENT PROMPT — Dual LLMs

> **Cách dùng**: Copy toàn bộ prompt → Tìm & thay thế các `{{PLACEHOLDER}}` → Paste vào Codex/AI coder.
> **Tham khảo**: `dual_llms_comprehensive_guide.md` cho chi tiết 8 phương án.

---

## DANH SÁCH PLACEHOLDERS CẦN THAY

| Placeholder | Ý nghĩa | Ví dụ |
|---|---|---|
| `{{EXP_ID}}` | Mã thí nghiệm ngắn | `D12_0`, `D12_1`, `D12_2` |
| `{{EXP_SNAKE}}` | Tên snake_case (dùng cho file/folder) | `info_asymmetric_masking`, `debate_collaboration`, `compression_game` |
| `{{EXP_FULL_NAME}}` | Tên đầy đủ snake_case + `_sft` | `D12_0_info_asymmetric_masking_sft` |
| `{{EXP_TITLE}}` | Tên tiêu đề đọc được | `Info-Asymmetric Masking`, `Debate Framework` |
| `{{HYPOTHESIS}}` | 2-3 câu mô tả giả thuyết | `Nếu ép information asymmetry...` |
| `{{METHOD_DESCRIPTION}}` | 1 đoạn mô tả cách hoạt động | `Agent A chỉ thấy 50% số liệu...` |
| `{{TEACHER_MODEL}}` | Model teacher | `Qwen/Qwen2.5-Math-7B-Instruct` |
| `{{STUDENT_MODEL}}` | Model student | `Qwen/Qwen2.5-1.5B-Instruct` |
| `{{SCRIPT_NAMES}}` | Danh sách tên scripts (xem bảng Phase) | |
| `{{HELPER_MODULE}}` | File helper mới nếu cần | `src/masking_utils.py` hoặc `(không cần)` |
| `{{SYSTEM_PROMPTS}}` | Các system prompt constants cho src/prompts.py | |
| `{{USER_PROMPT_FUNCTIONS}}` | Các hàm user prompt cho src/prompts.py | |
| `{{BOOTSTRAP_TRACE_FORMAT}}` | Format trace mà teacher phải sinh | |
| `{{BOOTSTRAP_FILTER_RULES}}` | Điều kiện lọc trace | |
| `{{BOOTSTRAP_STATS_FIELDS}}` | Các field thống kê bootstrap | |
| `{{SFT_DATA_FILES}}` | Danh sách file SFT training data | |
| `{{SFT_DATA_DESCRIPTION}}` | Mô tả từng file SFT | |
| `{{ADAPTER_NAMES}}` | Tên các LoRA adapter output | |
| `{{EVAL_MODES}}` | Danh sách evaluation modes | |
| `{{EVAL_PREDICTION_SCHEMA}}` | Schema JSON cho prediction JSONL | |
| `{{EVAL_METRICS_SCHEMA}}` | Schema JSON cho eval metrics | |
| `{{KEY_METRIC}}` | Metric quan trọng nhất cần report | |
| `{{ANALYSIS_POINTS}}` | Danh sách điểm phân tích | |
| `{{EXTRA_CONFIG_SECTIONS}}` | Các section config YAML đặc thù | |
| `{{NOTES_COMPARISON_BLOCK}}` | Block so sánh với D11.0/D11.4 | |

---

## BẮT ĐẦU PROMPT — COPY TỪ ĐÂY

````text
Bạn là senior ML engineer + AI researcher. Hãy làm việc trong repo:

D:\Program\Dual LLMs\gsm8k-dual-agent-finetune

================================================================
PHẦN 1 — BỐI CẢNH DỰ ÁN (KHÔNG ĐỔI CHO MỌI THÍ NGHIỆM)
================================================================

Dự án nghiên cứu câu hỏi: "Hai mô hình ngôn ngữ nhỏ (1.5B params) có thể HỌC ĐƯỢC
cách cộng tác suy luận — tức là cùng nhau giải được bài mà từng model riêng lẻ
không giải được — hay không?"

Metric cốt lõi:
  Δ = Acc(A→B) − max(Acc(A_alone), Acc(B_alone))
  Δ > 0 → cộng tác thật. Δ ≈ 0 → ensembling. Δ < 0 → cộng tác phản tác dụng.
  Metric phụ (mạnh nhất): số bài mà A sai + B sai nhưng A→B đúng.

Kết quả hiện tại (baseline):
  D11.0 (Alternating LoRA SFT): A→B = 0.68, Δ = +0.03 (100 GSM8K test, first_n, seed=42)
  D11.4 (Given/Need + Decider): A→B = 0.67, Δ = 0.00

Codebase hiện tại:
  - src/: answer_extraction.py, data_utils.py, evaluation.py, filtering.py,
          generation.py, prompts.py, training.py
  - scripts/: 19 pipeline scripts cho D11.0, D11.2, D11.3, D11.4
  - configs/: 4 YAML config files
  - data/raw/: train.jsonl, dev.jsonl, test.jsonl (từ GSM8K)
  - outputs/: adapters, generations, metrics cho từng experiment

================================================================
PHẦN 2 — NGUYÊN TẮC BẮT BUỘC (KHÔNG ĐỔI CHO MỌI THÍ NGHIỆM)
================================================================

A. Triết lý thiết kế:
  - KHÔNG dùng solver/verifier framing.
  - KHÔNG dùng nhãn: rescue, accept, reject, verdict, keep_correct, other_agent_correct.
  - Agent A và Agent B là hai reasoning agents TRUNG TÍNH.
  - Teacher chỉ dùng bootstrap/generate traces, KHÔNG dùng trong final evaluation.
  - Student mặc định: {{STUDENT_MODEL}}.
  - Teacher mặc định: {{TEACHER_MODEL}}.

B. Data integrity:
  - Evaluation PHẢI dùng data/raw/test.jsonl (hoặc test split tương ứng).
  - Training/bootstrap TUYỆT ĐỐI KHÔNG dùng GSM8K test rows.
  - Gọi reject_test_split_for_training() và reject_test_rows_for_training() trong training scripts.
  - Gọi reject_train_split_for_final_eval() và reject_train_rows_for_final_eval() trong eval scripts.
  - Gọi record_sampled_ids() để ghi lại IDs đã sample.

C. Code style bắt buộc:
  - Mỗi Python script bắt đầu bằng:
      #!/usr/bin/env python
      """One-line clear docstring."""
      from __future__ import annotations
  - Dòng tiếp theo:
      import argparse
      import sys
      from pathlib import Path
      sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  - Dùng EXPERIMENT_NAME = "{{EXP_FULL_NAME}}" ở đầu file.
  - Dùng cfg_get(cfg, section, key, default=None) cho nested config:
      def cfg_get(cfg: dict, section: str, key: str, default=None):
          return cfg.get(section, {}).get(key, cfg.get(key, default))
  - Dùng argparse cho MỌI script. Mọi đường dẫn lấy từ config, không hardcode.
  - Tái dùng tối đa: src/data_utils.py, src/generation.py, src/evaluation.py,
    src/answer_extraction.py, src/prompts.py, src/training.py.
  - Comment ngắn, chỉ giải thích block khó. Không comment hiển nhiên.
  - Function nhỏ, tên rõ: parse_..., build_..., format_..., score_..., avg_len_...
  - JSONL records schema nhất quán.
  - Metrics lưu JSON indent=2 qua write_json().
  - Generations/predictions lưu JSONL qua write_jsonl().
  - Clear GPU memory: clear_model(*objects) — gc.collect + torch.cuda.empty_cache.
  - Không overwrite bất kỳ D11 adapters/results nào.
  - Mọi script phải fail RÕ RÀNG khi:
    * Thiếu dependency → raise SystemExit với message chỉ pip install -r requirements.txt
    * Thiếu data file → raise SystemExit với message chỉ script nào cần chạy trước
    * Thiếu adapter → raise SystemExit với message chỉ đường dẫn adapter mong đợi
    * Parse traces rỗng → raise SystemExit với message chỉ file failed để debug
    * CUDA unavailable khi require_cuda=true → raise SystemExit rõ ràng

D. Cấu trúc tổ chức (KHÔNG nhồi mọi thứ vào 1 file):
  - Mỗi experiment CÓ: config riêng, scripts riêng, data paths riêng, output paths riêng,
    notes riêng, và RUN.md section riêng.
  - Prompt constants và helper functions đặt trong src/prompts.py (hoặc module mới trong src/).
  - Logic tái dùng đặt trong src/ modules, không copy-paste giữa scripts.

================================================================
PHẦN 3 — THÍ NGHIỆM CỤ THỂ: {{EXP_ID}} — {{EXP_TITLE}}
================================================================

Tên thí nghiệm: {{EXP_FULL_NAME}}

Giả thuyết:
{{HYPOTHESIS}}

Mô tả phương pháp:
{{METHOD_DESCRIPTION}}

================================================================
PHẦN 4 — CẤU TRÚC FILE
================================================================

4.1. Config:
  configs/{{EXP_ID | lowercase}}_{{EXP_SNAKE}}.yaml

4.2. Scripts (mỗi script 1 chức năng duy nhất):
  {{SCRIPT_NAMES}}
  (Mẫu tên: scripts/bootstrap_{{EXP_SNAKE}}_traces.py
             scripts/build_{{EXP_SNAKE}}_sft_data.py
             scripts/train_{{EXP_SNAKE}}_lora_sft.py
             scripts/evaluate_{{EXP_SNAKE}}_agents.py
             scripts/analyze_{{EXP_SNAKE}}.py)

4.3. Helper module mới (nếu cần):
  {{HELPER_MODULE}}
  Nếu không cần module mới, tái dùng src/ hiện có.

4.4. Prompts (cập nhật src/prompts.py):
  Thêm constants và functions. KHÔNG hard-code prompt dài trong scripts.

4.5. Data outputs:
  data/filtered/{{EXP_FULL_NAME}}/  — traces đã lọc
  data/train/{{EXP_FULL_NAME}}/     — SFT training data
  Các file cụ thể:
  {{SFT_DATA_FILES}}

4.6. Experiment outputs:
  outputs/{{EXP_FULL_NAME}}/adapters/     — LoRA adapters
  outputs/{{EXP_FULL_NAME}}/generations/  — bootstrap_failed.jsonl, eval_predictions.jsonl
  outputs/{{EXP_FULL_NAME}}/metrics/      — bootstrap_stats.json, eval_metrics.json
  outputs/{{EXP_FULL_NAME}}/analysis/     — interesting_examples.md (nếu có analyze script)
  Adapter names cụ thể:
  {{ADAPTER_NAMES}}

4.7. Notes:
  notes/{{EXP_FULL_NAME | replace("_sft","") | hoặc giữ nguyên}}_results.md

4.8. RUN.md:
  Thêm section ## {{EXP_ID}} {{EXP_TITLE}} với exact smoke-test và wider-run commands.

================================================================
PHẦN 5 — CONFIG YAML
================================================================

experiment_name: {{EXP_FULL_NAME}}

teacher_model_name: {{TEACHER_MODEL}}
student_model_name: {{STUDENT_MODEL}}

require_cuda: true

data:
  raw_train_path: data/raw/train.jsonl
  raw_test_path: data/raw/test.jsonl
  filtered_trace_path: data/filtered/{{EXP_FULL_NAME}}/traces.jsonl
  train_dir: data/train/{{EXP_FULL_NAME}}

output:
  root_dir: outputs/{{EXP_FULL_NAME}}
  adapter_dir: outputs/{{EXP_FULL_NAME}}/adapters
  generation_dir: outputs/{{EXP_FULL_NAME}}/generations
  metrics_dir: outputs/{{EXP_FULL_NAME}}/metrics
  analysis_dir: outputs/{{EXP_FULL_NAME}}/analysis

sampling:
  seed: 42
  sampling_mode: first_n
  max_train_examples: 200
  max_eval_examples: 100

# --- Section đặc thù cho phương pháp này ---
{{EXTRA_CONFIG_SECTIONS}}
# --- Hết section đặc thù ---

bootstrap:
  num_candidates: 1
  max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0

evaluation:
  base_max_new_tokens: 512
  method_max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0

training:
  max_seq_length: 1536
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

================================================================
PHẦN 6 — PROMPTS (thêm vào src/prompts.py)
================================================================

System prompts (constants ALL_CAPS):
{{SYSTEM_PROMPTS}}

User prompt functions (lowercase, return str):
{{USER_PROMPT_FUNCTIONS}}

Message builder functions (return list[dict[str, str]]):
  Tạo messages_for_{{EXP_SNAKE}}_*() tương ứng, mỗi hàm return:
    [{"role": "system", "content": CONSTANT}, {"role": "user", "content": function(args)}]

================================================================
PHẦN 7 — IMPLEMENTATION PHASES
================================================================

PHASE 0 — Inspect codebase hiện có:
  Đọc: README.md, RUN.md, configs/d11_4_compact_given_need_decider.yaml,
        scripts/bootstrap_given_need_traces.py, src/prompts.py, src/data_utils.py.
  Tóm tắt ngắn trong reasoning nội bộ: naming conventions, path patterns,
  metrics format, prompt style, note style.

--- BOOTSTRAP ---

PHASE 1 — scripts/bootstrap_{{EXP_SNAKE}}_traces.py:
  Input: data/raw/train.jsonl (từ config: data.raw_train_path).
  Với mỗi problem:
    1. [Bước tiền xử lý đặc thù nếu có — ví dụ: mask numbers, split clues, v.v.]
    2. Teacher nhận prompt → sinh trace theo format:

{{BOOTSTRAP_TRACE_FORMAT}}

    3. Parse trace → extract các fields.
    4. Filter rules:
{{BOOTSTRAP_FILTER_RULES}}
       + Luôn áp dụng: parse đủ fields + final answer match gold_answer.
       + Không dùng forbidden labels (solver/verifier/rescue/verdict/accept/reject).

  Save:
    - Kept traces → data/filtered/{{EXP_FULL_NAME}}/traces.jsonl
    - Failed traces → outputs/{{EXP_FULL_NAME}}/generations/bootstrap_failed.jsonl
    - Stats → outputs/{{EXP_FULL_NAME}}/metrics/bootstrap_stats.json

  Bootstrap stats fields:
    {
      "experiment_name": "{{EXP_FULL_NAME}}",
      "teacher_model_name": "...",
      "num_input_examples": ...,
      "parse_success_count": ...,
      "parse_success_rate": ...,
      "answer_match_count": ...,
      "answer_match_rate": ...,
      "kept_example_count": ...,
{{BOOTSTRAP_STATS_FIELDS}}
    }

  Nếu kept = 0 → raise SystemExit chỉ file bootstrap_failed.jsonl để debug.

--- BUILD SFT DATA ---

PHASE 2 — scripts/build_{{EXP_SNAKE}}_sft_data.py:
  Input: data/filtered/{{EXP_FULL_NAME}}/traces.jsonl
  Output files trong data/train/{{EXP_FULL_NAME}}/:
{{SFT_DATA_DESCRIPTION}}

  Mỗi training row phải có field "training_mode" cho mixed data.
  Dùng format_messages_with_assistant() từ src/prompts.py cho SFT text.
  Không dùng wording fixed-role. Giữ neutral.

--- TRAIN ---

PHASE 3 — scripts/train_{{EXP_SNAKE}}_lora_sft.py:
  Train LoRA adapters:
{{ADAPTER_NAMES}}

  Dùng standard SFT loss (cross-entropy trên output tokens).
  Flow:
    1. Load student model.
    2. Cho mỗi adapter cần train:
       a. Load/init LoRA config.
       b. Load SFT data tương ứng.
       c. Build SFT texts bằng format_messages_with_assistant() từ src/prompts.py.
       d. Tokenize, tạo Dataset.
       e. Trainer.train().
       f. model.save_pretrained(adapter_path).
       g. clear_model().
    3. Print đường dẫn adapters đã save.

--- EVALUATE ---

PHASE 4 — scripts/evaluate_{{EXP_SNAKE}}_agents.py:
  Input: data/raw/test.jsonl (từ config: data.raw_test_path).
  Evaluation modes:
{{EVAL_MODES}}

  QUAN TRỌNG:
    - max_eval_examples = 100, sampling_mode = first_n, seed = 42 (giống D11 để so sánh).
    - Không claim improvement nếu không so cùng điều kiện.

  Prediction JSONL — mỗi row:
{{EVAL_PREDICTION_SCHEMA}}

  Metrics JSON:
{{EVAL_METRICS_SCHEMA}}

  Key metric đặc biệt cho thí nghiệm này:
{{KEY_METRIC}}

--- ANALYZE ---

PHASE 5 — scripts/analyze_{{EXP_SNAKE}}.py:
  Input: outputs/{{EXP_FULL_NAME}}/generations/eval_predictions.jsonl
  Phân tích:
{{ANALYSIS_POINTS}}

  Save:
    - outputs/{{EXP_FULL_NAME}}/metrics/analysis.json
    - outputs/{{EXP_FULL_NAME}}/analysis/interesting_examples.md
      (5-10 ví dụ thú vị nhất, mỗi ví dụ gồm: problem, agent outputs, gold, nhận xét ngắn)

--- NOTES ---

PHASE 6 — notes/{{EXP_ID}}_{{EXP_SNAKE}}_results.md:
  Format bắt buộc:

# {{EXP_ID}} {{EXP_TITLE}} Results

Short design summary:
(1-2 câu mô tả ý tưởng thí nghiệm)

Config:
```yaml
teacher_model_name: {{TEACHER_MODEL}}
student_model_name: {{STUDENT_MODEL}}
max_train_examples: ...
max_eval_examples: 100
sampling_mode: first_n
seed: 42
```

Bootstrap/train:
```yaml
num_input_examples: ...
parse_success_count: ...
answer_match_count: ...
kept_example_count: ...
(thêm các field đặc thù)
```

Eval on first 100 GSM8K test examples:
```yaml
(liệt kê accuracy cho từng eval mode)
```

Comparison with baselines:
```yaml
D11_0_A_then_B_accuracy: 0.68
D11_0_delta: 0.03
D11_4_A_then_B_accuracy: 0.67
D11_4_delta: 0.00
{{NOTES_COMPARISON_BLOCK}}
```

Initial read:
Viết 1-2 đoạn NGẮN, KHÁCH QUAN, không phóng đại. Nêu rõ:
- Metric chính (Δ) so với D11.0 baseline.
- Có bằng chứng cộng tác thật hay không (bao nhiêu cases both_wrong → combined_correct).
- Hạn chế chính của thí nghiệm.
- 1 câu "next step" gợi ý thí nghiệm tiếp theo nếu kết quả tốt/xấu.

--- RUN.MD ---

PHASE 7 — Cập nhật RUN.md:
  Thêm section:

## {{EXP_ID}} {{EXP_TITLE}}

Smoke test (10 examples):

```bash
python scripts/bootstrap_{{EXP_SNAKE}}_traces.py \
  --config configs/{{EXP_ID | lowercase}}_{{EXP_SNAKE}}.yaml \
  --max-examples 10

python scripts/build_{{EXP_SNAKE}}_sft_data.py \
  --config configs/{{EXP_ID | lowercase}}_{{EXP_SNAKE}}.yaml

python scripts/train_{{EXP_SNAKE}}_lora_sft.py \
  --config configs/{{EXP_ID | lowercase}}_{{EXP_SNAKE}}.yaml \
  --max-train-examples 10

python scripts/evaluate_{{EXP_SNAKE}}_agents.py \
  --config configs/{{EXP_ID | lowercase}}_{{EXP_SNAKE}}.yaml \
  --max-examples 10 \
  --sampling-mode first_n

python scripts/analyze_{{EXP_SNAKE}}.py \
  --predictions outputs/{{EXP_FULL_NAME}}/generations/eval_predictions.jsonl
```

Wider run (200 train / 100 eval):

```bash
(Giống trên nhưng bỏ --max-examples hoặc đặt 200/100)
```

================================================================
PHẦN 8 — TIÊU CHUẨN HOÀN THÀNH
================================================================

Checklist:
[ ] Config YAML file tạo đúng đường dẫn.
[ ] Mỗi script chạy được hoặc fail với thông báo rõ ràng.
[ ] Không nhồi logic vào 1 file duy nhất — mỗi script 1 chức năng.
[ ] Prompts trong src/prompts.py, không hard-code trong scripts.
[ ] Không overwrite bất kỳ D11 adapters/results nào.
[ ] Notes viết khách quan, có YAML code blocks cho số liệu.
[ ] RUN.md có exact commands cho smoke test và wider run.
[ ] Data guards đầy đủ (reject test for train, reject train for eval).
[ ] Metrics JSON có experiment_name, student_model_name, num_examples.
[ ] Prediction JSONL có id, problem, gold_answer, correctness flags.

Final response sau khi implement:
1. Files created/modified — liệt kê đầy đủ.
2. Folder tree chính cho {{EXP_FULL_NAME}}.
3. Smoke test status — đã chạy hay chưa, lỗi gì nếu có.
4. Metrics chính nếu có.
5. Limitations / TODO còn lại.
````

---

## VÍ DỤ ĐIỀN: D12.0 Info-Asymmetric Masking

Dưới đây là ví dụ cách điền placeholder cho phương pháp ①:

| Placeholder | Giá trị |
|---|---|
| `{{EXP_ID}}` | `D12_0` |
| `{{EXP_SNAKE}}` | `info_asymmetric_masking` |
| `{{EXP_FULL_NAME}}` | `D12_0_info_asymmetric_masking_sft` |
| `{{EXP_TITLE}}` | `Info-Asymmetric Masking` |
| `{{TEACHER_MODEL}}` | `Qwen/Qwen2.5-Math-7B-Instruct` |
| `{{STUDENT_MODEL}}` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `{{HYPOTHESIS}}` | `Nếu ép information asymmetry (mỗi agent chỉ thấy 50% số liệu), Δ phải tăng đáng kể vì single agent không đủ thông tin giải một mình.` |
| `{{METHOD_DESCRIPTION}}` | `Che 50% các con số trong bài toán GSM8K bằng [HIDDEN]. Agent A thấy số ở index chẵn, Agent B thấy số ở index lẻ. Final agent nhận problem đầy đủ + 2 contributions → tổng hợp và giải.` |
| `{{HELPER_MODULE}}` | `src/masking_utils.py — chứa find_numbers(), build_masked_views(), validate_views()` |
| `{{EXTRA_CONFIG_SECTIONS}}` | (xem ví dụ bên dưới) |

```yaml
# {{EXTRA_CONFIG_SECTIONS}} cho D12.0:
masking:
  mask_token: "[HIDDEN]"
  split_mode: deterministic_alternating
  min_numbers_per_view: 1
  mask_question_numbers: false
```

| Placeholder | Giá trị |
|---|---|
| `{{SCRIPT_NAMES}}` | `scripts/bootstrap_info_asymmetric_masking_traces.py`, `scripts/build_info_asymmetric_masking_sft_data.py`, `scripts/train_info_asymmetric_masking_lora_sft.py`, `scripts/evaluate_info_asymmetric_masking_agents.py`, `scripts/analyze_info_asymmetric_masking.py` |

```
{{SYSTEM_PROMPTS}}:

MASKED_PARTIAL_VIEW_SYSTEM = (
    "You are one agent in a two-agent reasoning system.\n"
    "You see only PART of the information — some numbers are hidden as [HIDDEN].\n"
    "Another agent sees the hidden numbers but not yours.\n"
    "Contribute useful reasoning based on what you CAN see.\n"
    "Do not guess hidden numbers. State what you know and what you need."
)

MASKED_FINAL_SYNTHESIZER_SYSTEM = (
    "You are the final synthesizer in a two-agent reasoning system.\n"
    "Two agents each saw different parts of the problem.\n"
    "Each has provided a contribution based on their partial view.\n"
    "Combine their information to solve the complete problem.\n"
    "Produce concise reasoning and the final numeric answer."
)
```

```
{{BOOTSTRAP_TRACE_FORMAT}}:

Agent A view:
{agent_a_view}

Agent B view:
{agent_b_view}

Agent A contribution:
(reasoning from A's partial view)

Agent B contribution:
(reasoning from B's partial view)

Joint solution:
(combining both)

Final answer:
(numeric)
```

```
{{BOOTSTRAP_FILTER_RULES}}:
- parse đủ 6 fields (a_view, b_view, a_contribution, b_contribution, joint_solution, final_answer)
- final answer match gold_answer
- both views non-empty, both contributions non-empty
- mỗi view có ít nhất 1 số thật (không toàn [HIDDEN])
```

```
{{SFT_DATA_FILES}}:
data/train/D12_0_info_asymmetric_masking_sft/agent_a_partial_train.jsonl
data/train/D12_0_info_asymmetric_masking_sft/agent_b_partial_train.jsonl
data/train/D12_0_info_asymmetric_masking_sft/final_synthesis_train.jsonl
data/train/D12_0_info_asymmetric_masking_sft/mixed_train.jsonl
```

```
{{ADAPTER_NAMES}}:
outputs/D12_0_info_asymmetric_masking_sft/adapters/agent_A_partial_view_sft/
outputs/D12_0_info_asymmetric_masking_sft/adapters/agent_B_partial_view_sft/
outputs/D12_0_info_asymmetric_masking_sft/adapters/final_synthesizer_sft/
```

```
{{EVAL_MODES}}:
1. base_single_full_problem: base model (no LoRA) giải full problem.
2. agent_A_partial_only: Agent A (LoRA) giải partial view A alone.
3. agent_B_partial_only: Agent B (LoRA) giải partial view B alone.
4. A_and_B_then_final: A sinh contribution (view A) + B sinh contribution (view B)
                        → Final agent nhận full problem + 2 contributions → answer.
```

```
{{KEY_METRIC}}:
delta_vs_best_partial = A_and_B_then_final_accuracy − max(agent_A_partial_only, agent_B_partial_only)
collaboration_essential_count = số cases cả 2 partial sai nhưng combined đúng
```

```
{{NOTES_COMPARISON_BLOCK}}:
D12_0_A_and_B_then_final_accuracy: ...
D12_0_delta_vs_best_partial: ...
D12_0_collaboration_essential_count: ...
D12_0_collaboration_essential_rate: ...
```

---

## BẢNG TRA CỨU NHANH: 8 PHƯƠNG ÁN → PLACEHOLDERS

Dùng bảng này để tra nhanh giá trị placeholder cho từng phương án trong `dual_llms_comprehensive_guide.md`.

### ① Info-Asymmetric Masking

| Field | Value |
|---|---|
| EXP_ID | `D12_0` |
| EXP_SNAKE | `info_asymmetric_masking` |
| EXP_FULL_NAME | `D12_0_info_asymmetric_masking_sft` |
| EXP_TITLE | `Info-Asymmetric Masking` |
| HELPER_MODULE | `src/masking_utils.py` |
| Adapters | `agent_A_partial_view_sft`, `agent_B_partial_view_sft`, `final_synthesizer_sft` |
| Key eval modes | `base_single_full_problem`, `agent_A_partial_only`, `agent_B_partial_only`, `A_and_B_then_final` |
| Key metric | `delta_vs_best_partial`, `collaboration_essential_count` |

### ② Majority Voting Baseline

| Field | Value |
|---|---|
| EXP_ID | `D12_0B` |
| EXP_SNAKE | `voting_baseline` |
| EXP_FULL_NAME | `D12_0B_voting_baseline` |
| EXP_TITLE | `Majority Voting Baseline` |
| HELPER_MODULE | `(không cần)` |
| Scripts | Chỉ 1: `scripts/evaluate_voting_baseline.py` (không cần bootstrap/build/train) |
| Adapters | Tái dùng D11.0: `outputs/adapters/agent_A_round_1`, `agent_B_round_1` |
| Key metric | `majority_vote_accuracy vs A_then_B_accuracy`, `atb_correct_but_vote_wrong` |
| Đặc biệt | Không cần PHASE 1-3 (bootstrap/build/train). Chỉ cần eval script. |

### ③ Debate Framework

| Field | Value |
|---|---|
| EXP_ID | `D12_1` |
| EXP_SNAKE | `debate_collaboration` |
| EXP_FULL_NAME | `D12_1_debate_collaboration_sft` |
| EXP_TITLE | `Debate Framework` |
| HELPER_MODULE | `(không cần)` |
| Adapters | `agent_A_debate_sft`, `agent_B_debate_sft` |
| Key eval modes | `agent_A_initial`, `agent_B_initial`, `debate_1_round`, `debate_symmetric` |
| Key metric | `debate_net_benefit = fix_rate − break_rate`, `agreement_rate` |
| Extra config | `debate: { max_rounds: 1, require_disagreement: true }` |

### ④ Specialized Decomposition

| Field | Value |
|---|---|
| EXP_ID | `D12_1B` |
| EXP_SNAKE | `decomposition` |
| EXP_FULL_NAME | `D12_1B_decomposition_sft` |
| EXP_TITLE | `Specialized Decomposition` |
| HELPER_MODULE | `src/decomposition_utils.py` — chứa `decompose_problem()`, `merge_sub_answers()` |
| Adapters | `agent_A_sub_solver_sft`, `agent_B_sub_solver_sft`, `merge_agent_sft` |
| Key eval modes | `base_single`, `decomposed_A_only`, `decomposed_B_only`, `decomposed_merged` |
| Key metric | `merged_accuracy vs base_single`, `sub_problem_accuracy_A/B` |
| Extra config | `decomposition: { method: rule_based, max_sub_problems: 4 }` |

### ⑤ Soft Message V1

| Field | Value |
|---|---|
| EXP_ID | `D12_3` |
| EXP_SNAKE | `soft_message_v1` |
| EXP_FULL_NAME | `D12_3_soft_message_v1_sft` |
| EXP_TITLE | `Soft Message V1` |
| HELPER_MODULE | `src/soft_message.py` — chứa `BottleneckMLP`, `SoftMessageBridge`, `GaussianNoise` |
| Scripts thêm | `scripts/probe_soft_message_leakage.py` (6 scripts thay vì 5) |
| Adapters | `agent_A_lora`, `agent_B_lora`, `bottleneck_mlp` (3 components) |
| Key eval modes | `base_single`, `agent_B_alone`, `soft_message_pipeline`, `text_message_comparison` |
| Key metric | `soft_pipeline_accuracy vs agent_B_alone`, `probe_leakage_accuracy` |
| Extra config | `soft_message: { num_soft_tokens: 1, bottleneck_dim: 32, noise_std: 0.1 }` |
| Đặc biệt | Train end-to-end (gradient xuyên A→bridge→B). Cần gradient checkpointing. |

### ⑥ Adversarial Compression Game

| Field | Value |
|---|---|
| EXP_ID | `D12_2` |
| EXP_SNAKE | `compression_game` |
| EXP_FULL_NAME | `D12_2_compression_game_sft` |
| EXP_TITLE | `Adversarial Compression Game` |
| HELPER_MODULE | `(không cần — logic nằm trong prompts)` |
| Adapters | `agent_A_compressor_sft`, `agent_B_solver_sft` |
| Key eval modes | `base_single`, `solver_full_no_code`, `solver_blind_code_only`, `solver_full_with_code` |
| Key metric | `solver_full_with_code − solver_full_no_code`, `blind_solver_accuracy` |
| Extra config | `compression: { max_code_tokens: 8, format: "key:value", forbid_gold_answer: true }` |

### ⑦ Spectral Reasoning Fusion

| Field | Value |
|---|---|
| EXP_ID | `D12_4` |
| EXP_SNAKE | `spectral_fusion` |
| EXP_FULL_NAME | `D12_4_spectral_fusion_sft` |
| EXP_TITLE | `Spectral Reasoning Fusion` |
| HELPER_MODULE | `src/spectral_fusion.py` — chứa `FreqWeightedPooling`, `CrossAttentionFusion`, `SpectralDiversityLoss` |
| Adapters | `agent_A_lowfreq_lora`, `agent_B_highfreq_lora`, `fusion_module` |
| Key eval modes | `base_single`, `agent_A_alone`, `agent_B_alone`, `spectral_fused` |
| Key metric | `fused_accuracy vs base_single`, `spectral_diversity_score` |
| Extra config | `spectral: { num_layer_groups: 3, a_weights: [0.7,0.2,0.1], b_weights: [0.1,0.2,0.7] }` |
| Đặc biệt | Train end-to-end. Fusion module ~1M params. Spectral Diversity Loss bắt buộc. |

### ⑧ Cognitive Oscillator

| Field | Value |
|---|---|
| EXP_ID | `D12_5` |
| EXP_SNAKE | `cognitive_oscillator` |
| EXP_FULL_NAME | `D12_5_cognitive_oscillator_sft` |
| EXP_TITLE | `Cognitive Oscillator` |
| HELPER_MODULE | `src/oscillator.py` — chứa `GatedResidual`, `OscillationLoop`, `DeepSupervisionLoss`, `AdaptiveHalt` |
| Scripts thêm | Không cần bootstrap — dùng D11.0 data. Chỉ train + eval + analyze. |
| Adapters | `agent_A_oscillator_lora`, `agent_B_oscillator_lora`, `gating_module` |
| Key eval modes | `base_single`, `oscillation_T2`, `oscillation_T3`, `oscillation_adaptive` |
| Key metric | `oscillation_T2_accuracy vs base_single`, `avg_oscillations_before_halt` |
| Extra config | `oscillator: { max_T: 3, gate_init: 0.5, convergence_threshold: 0.01, deep_supervision: true }` |
| Đặc biệt | Forward qua model 2T lần (T oscillations × 2 agents). Cần gradient checkpointing nghiêm ngặt. Batch size=1 bắt buộc. |
