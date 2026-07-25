Dưới đây là bản kế hoạch viết lại theo đúng 3 giai đoạn và 8 phương pháp bạn đưa ra, trong đó **D12 được giữ đúng là Soft Message**, còn các phương pháp khác được đặt thành các D riêng hoặc nhánh con nếu là bản mở rộng trực tiếp.

---

# Kế Hoạch Nghiên Cứu D12+ Cho Dual LLMs

## Câu hỏi nghiên cứu trung tâm

Dự án cần trả lời một câu hỏi duy nhất:

> Hai LLM nhỏ có thực sự học được cách cộng tác suy luận, hay kết quả A→B hiện tại chỉ là một dạng ensemble/voting đắt đỏ?

Metric chính:

```text
Delta = Acc(combined) - max(Acc(agent_A_alone), Acc(agent_B_alone))
```

Diễn giải:

```text
Delta > 0.10 trên masked data  -> có tín hiệu cộng tác thật
Delta ~= 0                     -> cộng tác giả / ensemble effect
A→B ~= voting                  -> pipeline hiện tại có thể chỉ là expensive majority voting
```

Baseline cần giữ cố định:

```yaml
D11_0_A_then_B_accuracy: 0.68
D11_4_A_then_B_accuracy: 0.67
student_model: Qwen/Qwen2.5-1.5B-Instruct
eval_set: first_100_gsm8k_test
sampling_mode: first_n
seed: 42
```

---

# Tổng Quan Các Giai Đoạn

```text
Giai đoạn 0:
  Chẩn đoán nền tảng.
  Chạy Majority Voting và Info-Asymmetric Masking trước mọi thứ khác.

Giai đoạn 1 & 2:
  Rẽ nhánh chiến lược dựa trên kết quả Giai đoạn 0.

  PATH A:
    Nếu Delta > 0.10 trên masked data.
    Tập trung tối ưu text-channel collaboration.

  PATH B:
    Nếu Delta ~= 0 hoặc A→B ~= voting.
    Chuyển sang latent/differentiable architectures.

Giai đoạn 3:
  Kiến trúc đột phá và scale.
  Chạy Cognitive Oscillator nếu nhánh latent có tín hiệu.
```

---

# Giai Đoạn 0 - Chẩn Đoán Nền Tảng

Đây là giai đoạn **bắt buộc chạy đầu tiên**. Mục tiêu không phải đạt accuracy cao nhất, mà là xác định dự án đang đo đúng hiện tượng “cộng tác” hay chỉ đang đo hiệu ứng voting/ensemble.

## D12_0 - Majority Voting Baseline

### Mục tiêu

Kiểm tra pipeline A→B hiện tại có thật sự tạo thêm giá trị so với việc chạy hai agent độc lập rồi vote hay không.

### Ý tưởng

Với mỗi bài toán:

```text
1. Agent A giải độc lập.
2. Agent B giải độc lập.
3. Trích xuất final answer từ cả hai.
4. Nếu hai đáp án giống nhau -> lấy đáp án đó.
5. Nếu khác nhau -> đánh dấu disagreement hoặc dùng rule vote/abstain.
6. So sánh với A→B của D11.0.
```

### Files cần có

```text
configs/d12_0_voting_baseline.yaml
scripts/evaluate_voting_baseline.py
scripts/analyze_voting_baseline.py
notes/D12_0_voting_results.md

outputs/D12_0_voting/
  generations/voting_predictions.jsonl
  metrics/voting_metrics.json
  analysis/interesting_examples.md
```

### Evaluation modes

```text
agent_A_alone
agent_B_alone
A_then_B
B_then_A
majority_vote
oracle_vote
```

### Metrics chính

```yaml
agent_A_alone_accuracy: ...
agent_B_alone_accuracy: ...
A_then_B_accuracy: ...
B_then_A_accuracy: ...
majority_vote_accuracy: ...
oracle_vote_accuracy: ...
agreement_rate: ...
disagreement_rate: ...
A_then_B_minus_vote: ...
atb_correct_vote_wrong: ...
vote_correct_atb_wrong: ...
both_wrong_atb_correct: ...
```

### Tiêu chí diễn giải

```text
Nếu majority_vote ~= A_then_B:
  D11 có thể chỉ là expensive majority voting.

Nếu A_then_B > majority_vote rõ rệt:
  Có tín hiệu interaction A→B tạo thêm giá trị.
```

---

## D14.0 - Info-Asymmetric Masking

### Mục tiêu

Ép cộng tác về mặt cấu trúc bằng cách khiến mỗi agent chỉ thấy một phần dữ kiện.

### Ý tưởng

Che 50% số liệu trong đề bài cho mỗi agent:

```text
Agent A thấy một nửa số liệu.
Agent B thấy nửa còn lại.
Không agent nào đủ thông tin để tự giải chắc chắn.
Muốn đúng, hai agent phải chia sẻ hoặc tổng hợp thông tin.
```

### Vì sao đây là benchmark chỉnh sửa bắt buộc

GSM8K gốc quá dễ đối với single agent sau LoRA. Vì vậy, accuracy A→B cao chưa đủ chứng minh cộng tác. Masked GSM8K tạo ra một benchmark trung gian:

```text
GSM8K gốc:
  đo continuity với D11.

Masked GSM8K:
  đo cộng tác bắt buộc.

Sau đó mới cân nhắc benchmark mới:
  ZebraLogic split-clue, BBEH split-condition.
```

### Files cần có

```text
configs/d14_0_info_asymmetric_masking.yaml
src/masking_utils.py

scripts/bootstrap_info_asymmetric_traces.py
scripts/build_info_asymmetric_sft_data.py
scripts/train_info_asymmetric_lora_sft.py
scripts/evaluate_info_asymmetric_agents.py
scripts/analyze_info_asymmetric.py

notes/D14_0_info_asymmetric_masking_results.md

outputs/D14_0_info_asymmetric_masking/
  adapters/
  generations/
  metrics/
  analysis/
```

### Masking policy ban đầu

```yaml
masking:
  mask_token: "[HIDDEN]"
  split_mode: deterministic_alternating
  mask_ratio: 0.5
  min_visible_numbers_per_agent: 1
  avoid_masking_question_target: true
```

### Evaluation modes

```text
agent_A_partial_only
agent_B_partial_only
A_B_then_final_with_full_problem
A_B_then_final_without_full_problem
```

Mode quan trọng nhất:

```text
A_B_then_final_without_full_problem
```

Vì nếu final agent vẫn thấy full problem, hệ thống có thể đúng mà không cần cộng tác thật.

### Metrics chính

```yaml
agent_A_partial_accuracy: ...
agent_B_partial_accuracy: ...
combined_with_full_problem_accuracy: ...
combined_without_full_problem_accuracy: ...
delta_vs_best_partial: ...
collaboration_essential_count: ...
collaboration_essential_rate: ...
mask_failure_rate: ...
```

### Tiêu chí rẽ nhánh

```text
Nếu delta_vs_best_partial > 0.10:
  PATH A - Cộng tác thật khi bị ép buộc.

Nếu delta_vs_best_partial ~= 0:
  PATH B - Cộng tác giả hoặc chưa học được cơ chế chia sẻ thông tin.

Nếu A→B ~= majority_vote từ D12_0:
  PATH B - pipeline hiện tại giống ensemble hơn là collaboration.
```

---

# Giai Đoạn 1 & 2 - Phân Nhánh Chiến Lược

Sau Giai đoạn 0, roadmap rẽ thành hai nhánh.

---

# PATH A - Nếu Có Cộng Tác Thật

Điều kiện vào PATH A:

```text
Delta > 0.10 trên masked data
và combined_without_full_problem > max(agent_A_partial, agent_B_partial)
và A→B tốt hơn majority voting rõ rệt
```

Mục tiêu PATH A:

```text
Tối ưu các kênh giao tiếp bằng ngôn ngữ.
Giữ pipeline tương đối đơn giản.
Tận dụng việc model đã có khả năng cộng tác khi bị ép.
```

---

## D15.0 - Adversarial Compression Game

### Mục tiêu

Ép Agent A nén bài toán thành một đoạn code cực ngắn dạng `key:value`, để Agent B giải từ compressed code.

### Ý tưởng

```text
Agent A đọc problem đầy đủ.
Agent A sinh compressed code 3-8 items.
Agent B không thấy đề gốc trong chế độ blind.
Agent B phải giải bằng compressed code.
```

Ví dụ:

```text
Compressed code:
ducks:16, eggs_each:3, used:3+4, price:2
```

### Files cần có

```text
configs/d15_0_compression_game.yaml

scripts/bootstrap_compression_traces.py
scripts/build_compression_sft_data.py
scripts/train_compression_lora_sft.py
scripts/evaluate_compression_agents.py
scripts/analyze_compression_quality.py

notes/D15_0_compression_game_results.md
```

### Evaluation modes

```text
base_single
solver_full_no_code
solver_blind_code_only
solver_full_with_code
solver_with_random_code
solver_with_shuffled_code
```

### Metrics chính

```yaml
solver_full_no_code_accuracy: ...
solver_full_with_code_accuracy: ...
blind_code_only_accuracy: ...
random_code_accuracy: ...
code_help_delta: ...
gold_leakage_rate: ...
info_retention_rate: ...
avg_code_items: ...
```

### Tiêu chí thành công

```text
solver_full_with_code > solver_full_no_code
gold_leakage_rate thấp
solver_with_random_code thấp hơn solver_full_with_code rõ rệt
```

---

## D15.1 - Curriculum Compression

Bản nâng cấp trực tiếp của D15.0.

### Mục tiêu

Giảm dần dung lượng compressed code để ép Agent A học chọn thông tin cốt lõi.

```yaml
compression:
  stages:
    - max_code_items: 10
    - max_code_items: 8
    - max_code_items: 6
    - max_code_items: 4
```

### Câu hỏi cần trả lời

```text
Giảm dần code length có làm A học compression tốt hơn không?
Hay chỉ làm mất thông tin và giảm accuracy?
```

---

## D15.2 - Anti-Leak Compression

Bản kiểm soát an toàn của D15.0/D15.1.

### Mục tiêu

Đảm bảo compressed code không mã hóa trực tiếp final answer.

### Checks

```text
gold answer xuất hiện trong code
final intermediate result xuất hiện trong code
rule-based leakage check
linear probe từ code -> answer
randomized code ablation
```

### Metrics

```yaml
gold_answer_in_code_rate: ...
final_result_in_code_rate: ...
probe_answer_accuracy: ...
real_code_vs_random_code_gap: ...
```

---

## D16.0 - Debate Framework

### Mục tiêu

Cho hai agent giải độc lập trước. Nếu đáp án khác nhau, kích hoạt một vòng tranh luận để sửa lỗi.

### Ý tưởng

```text
Agent A giải lần 1.
Agent B giải lần 1.
Nếu cùng đáp án:
  giữ đáp án.
Nếu khác đáp án:
  A/B xem reasoning của nhau.
  Một agent refine lại lời giải.
```

### Files cần có

```text
configs/d16_0_debate_one_round.yaml

scripts/bootstrap_debate_traces.py
scripts/build_debate_sft_data.py
scripts/train_debate_lora_sft.py
scripts/evaluate_debate_agents.py
scripts/analyze_debate.py

notes/D16_0_debate_one_round_results.md
```

### Evaluation modes

```text
agent_A_initial
agent_B_initial
debate_A_to_B
debate_B_to_A
debate_only_when_disagree
```

### Metrics chính

```yaml
agreement_rate: ...
initial_disagreement_rate: ...
debate_fix_rate: ...
debate_break_rate: ...
debate_net_benefit: ...
debate_no_change_rate: ...
```

### Tiêu chí thành công

```text
debate_fix_rate > debate_break_rate
debate_net_benefit dương rõ rệt
debate không làm giảm accuracy khi hai agent ban đầu đã đúng
```

---

## D16.1 - Multi-Round / Symmetric Debate

Bản nâng cấp của D16.0.

### Mục tiêu

Kiểm tra nhiều vòng debate có giúp sửa lỗi thêm không hay chỉ tăng noise.

### Evaluation modes

```text
debate_2_rounds
debate_3_rounds
symmetric_debate
debate_with_confidence_filter
```

### Metrics thêm

```yaml
round_1_fix_rate: ...
round_2_fix_rate: ...
round_3_fix_rate: ...
roundwise_break_rate: ...
accuracy_vs_round_count: ...
```

---

## D17.0 - Specialized Decomposition

### Mục tiêu

Dùng cơ chế “chia để trị”: tách bài toán thành sub-problems, hai agent giải từng phần, sau đó merge.

### Ý tưởng

```text
Problem -> decomposer -> subproblems
Agent A giải subset subproblems
Agent B giải subset subproblems
Merge module tổng hợp đáp án cuối
```

### Files cần có

```text
configs/d17_0_decomposition_rule_teacher.yaml
src/decomposition_utils.py

scripts/bootstrap_decomposition_traces.py
scripts/build_decomposition_sft_data.py
scripts/train_decomposition_lora_sft.py
scripts/evaluate_decomposition_agents.py
scripts/analyze_decomposition.py

notes/D17_0_decomposition_results.md
```

### Evaluation modes

```text
base_single
teacher_decompose_A_B_merge
rule_decompose_A_B_merge
agent_A_subtasks_only
agent_B_subtasks_only
A_B_subtasks_then_merge
```

### Metrics chính

```yaml
subproblem_accuracy_A: ...
subproblem_accuracy_B: ...
merge_accuracy: ...
end_to_end_accuracy: ...
error_source_decomposition: ...
error_source_subsolve: ...
error_source_merge: ...
```

### Tiêu chí thành công

```text
A_B_subtasks_then_merge > base_single
subproblem accuracy cao
merge error không chiếm phần lớn lỗi
```

---

## D17.1 - Learned Decomposer

Bản nâng cấp của D17.0.

### Mục tiêu

Thay teacher/rule decomposition bằng student decomposer để inference không phụ thuộc teacher.

### Files

```text
configs/d17_1_learned_decomposer.yaml
scripts/train_learned_decomposer.py
scripts/evaluate_learned_decomposer_pipeline.py
scripts/analyze_learned_decomposer.py
notes/D17_1_learned_decomposer_results.md
```

---

# PATH B - Nếu Chỉ Là Cộng Tác Giả

Điều kiện vào PATH B:

```text
Delta ~= 0 trên masked data
hoặc A→B ~= majority_vote
hoặc combined_without_full_problem không vượt partial-only
```

Mục tiêu PATH B:

```text
Chuyển sang kiến trúc latent/differentiable.
Giảm nhiễu từ natural language.
Buộc gradient học một kênh giao tiếp thật sự giữa A và B.
```

---

## D12.0 - Soft Message GSM8K Gold-Only V1

D12 là Soft Message, đúng với folder:

```text
d12-soft-message-gsm8k-gold-only
```

### Mục tiêu

Thay text message bằng continuous vector có chiều thấp.

### Ý tưởng

```text
Agent A đọc problem.
Lấy hidden state tại <MSG>.
Đưa qua BottleneckMLP.
Tạo 1 soft token.
Chèn soft token vào input embedding của Agent B.
Agent B sinh final answer.
Gradient truyền ngược end-to-end từ B về A.
```

### Files cần có

```text
d12-soft-message-gsm8k-gold-only/
  requirements.txt
  RUN.md
  notes.md

gsm8k-dual-agent-finetune/
  configs/d12_0_soft_message_gsm8k_gold_only.yaml
  src/soft_message.py

  scripts/train_soft_message_e2e.py
  scripts/evaluate_soft_message_agents.py
  scripts/probe_soft_message_leakage.py
  scripts/analyze_soft_message.py

  notes/D12_0_soft_message_gsm8k_gold_only_results.md
```

### Config chính

```yaml
experiment_name: D12_0_soft_message_gsm8k_gold_only

student_model_name: Qwen/Qwen2.5-1.5B-Instruct
require_cuda: true

soft_message:
  msg_token: "<MSG>"
  num_soft_tokens: 1
  bottleneck_dim: 32
  noise_std: 0.0

training:
  mode: end_to_end
  max_seq_length: 1536
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_train_epochs: 1
```

### Evaluation modes

```text
base_single
agent_B_alone
D11_text_A_then_B
D12_soft_A_then_B
D12_soft_on_masked_gsm8k
```

### Metrics chính

```yaml
base_single_accuracy: ...
agent_B_alone_accuracy: ...
D11_text_A_then_B_accuracy: ...
D12_soft_A_then_B_accuracy: ...
soft_vs_text_delta: ...
soft_vs_B_alone_delta: ...
masked_soft_accuracy: ...
probe_leakage_accuracy: ...
```

### Tiêu chí thành công

```text
Soft A→B > B alone
Soft A→B >= text A→B
Probe leakage thấp
Không collapse message thành vector gần như hằng số
```

---

## D12.1 - Soft Message Bottleneck / Noise Ablation

Bản nâng cấp của D12.0.

### Mục tiêu

Kiểm tra kênh soft message nhạy thế nào với bottleneck size, số soft token và Gaussian noise.

### Ablations

```yaml
soft_message:
  bottleneck_dim: [16, 32, 64]
  num_soft_tokens: [1, 2, 4]
  noise_std: [0.0, 0.05, 0.1]
```

### Metrics

```yaml
accuracy_by_bottleneck_dim: ...
accuracy_by_num_soft_tokens: ...
accuracy_by_noise_std: ...
message_norm_mean: ...
message_variance: ...
collapse_rate: ...
```

---

## D12.2 - Soft Message Leakage Probe

Bản chẩn đoán an toàn của D12.0.

### Mục tiêu

Kiểm tra soft vector có mã hóa trực tiếp đáp án không.

### Probe

```text
soft_message_vector -> linear probe -> gold answer
```

### Metrics

```yaml
probe_accuracy: ...
probe_mse: ...
gold_answer_decodable_rate: ...
leakage_warning: true/false
```

### Diễn giải

```text
Accuracy tăng + probe cao:
  khả năng representation leakage.

Accuracy tăng + probe thấp:
  tín hiệu soft message đáng tin hơn.
```

---

## D12.3 - Hybrid Given/Need + Soft Message

Bản nâng cấp kết hợp D11.4 và D12.0.

### Mục tiêu

Kiểm tra soft token có thêm giá trị ngoài compact text notes không.

### Input cho Agent B

```text
Problem
Given/Need notes
Soft token(s)
```

### Metrics

```yaml
D11_4_given_need_accuracy: 0.67
D12_0_soft_only_accuracy: ...
D12_3_hybrid_accuracy: ...
hybrid_gain_over_given_need: ...
hybrid_gain_over_soft_only: ...
```

---

## D18.0 - Spectral Reasoning Fusion

### Mục tiêu

Hợp nhất “phổ suy luận” của hai agent qua hidden states và cross-attention.

### Ý tưởng

```text
Agent A: ưu tiên early layers -> abstract/strategic representation
Agent B: ưu tiên late layers -> detail/computation representation
Fusion module: cross-attention giữa hai representation
Decoder: sinh final answer
```

### Files cần có

```text
configs/d18_0_spectral_fusion_fixed.yaml
src/spectral_fusion.py

scripts/train_spectral_fusion.py
scripts/evaluate_spectral_fusion.py
scripts/analyze_spectral_fusion.py

notes/D18_0_spectral_fusion_results.md
```

### Evaluation modes

```text
base_single
agent_A_alone
agent_B_alone
spectral_fused
spectral_fused_random_partner
```

### Metrics chính

```yaml
fused_accuracy: ...
delta_vs_best_single: ...
spectral_diversity_score: ...
random_partner_drop: ...
layer_group_usage_A: ...
layer_group_usage_B: ...
```

---

## D18.1 - Learnable Spectral Fusion

Bản nâng cấp của D18.0.

### Mục tiêu

Không hard-code early/late layer. Cho model tự học trọng số layer.

### Metrics thêm

```yaml
layer_weight_entropy_A: ...
layer_weight_entropy_B: ...
dominant_layer_group_A: ...
dominant_layer_group_B: ...
```

---

## D18.2 - Spectral Fusion + Diversity Loss

Bản regularized.

### Mục tiêu

Ép hai agent không collapse về cùng representation.

### Loss

```text
L_total = L_task + lambda_diversity * L_spectral_diversity
```

### Metrics

```yaml
fused_accuracy_with_diversity: ...
fused_accuracy_without_diversity: ...
diversity_score: ...
accuracy_diversity_tradeoff: ...
```

---

# Giai Đoạn 3 - Đột Phá Kiến Trúc Và Scale

Giai đoạn này chỉ nên chạy sau khi một trong hai điều kiện sau đúng:

```text
PATH A cho thấy text-channel collaboration có tín hiệu mạnh.
hoặc PATH B cho thấy latent channel như Soft Message/Spectral Fusion có tín hiệu.
```

---

## D19.0 - Cognitive Oscillator T=2

### Mục tiêu

Thay vì giao tiếp một chiều, cho hai agent luân phiên refine hidden state.

### Ý tưởng

```text
h0 = encode(problem)

Agent A refine h0 -> h1
Gated residual -> h1'
Agent B refine h1' -> h2
Gated residual -> h2'
Decode h2' -> final answer
```

Mỗi vòng là một “dao động” nhận thức.

### Files cần có

```text
configs/d19_0_cognitive_oscillator_t2.yaml
src/oscillator.py

scripts/train_cognitive_oscillator.py
scripts/evaluate_cognitive_oscillator.py
scripts/analyze_cognitive_oscillator.py

notes/D19_0_cognitive_oscillator_results.md
```

### Evaluation modes

```text
base_single
oscillation_T1
oscillation_T2
oscillation_T3_if_possible
```

### Metrics chính

```yaml
T1_accuracy: ...
T2_accuracy: ...
T3_accuracy: ...
delta_vs_base_single: ...
compute_multiplier: ...
convergence_score: ...
```

---

## D19.1 - Adaptive Halt Oscillator

Bản nâng cấp của D19.0.

### Mục tiêu

Cho hệ thống tự dừng khi hidden state đã hội tụ.

### Config

```yaml
oscillator:
  adaptive_halt: true
  convergence_threshold: 0.01
```

### Metrics

```yaml
adaptive_accuracy: ...
avg_halt_step: ...
early_halt_rate: ...
accuracy_per_compute: ...
```

---

## D19.2 - Deep Supervision Oscillator

Bản nâng cấp mạnh hơn.

### Mục tiêu

Thêm auxiliary loss ở từng vòng để tránh gradient yếu và giúp hệ thống hội tụ.

### Loss

```text
L_total = L_final + alpha * sum(L_aux_t)
```

### Metrics

```yaml
deep_supervision_accuracy: ...
no_deep_supervision_accuracy: ...
convergence_rate: ...
stability_score: ...
```

---

# Thứ Tự Triển Khai Khuyến Nghị

## Giai đoạn 0 - Bắt buộc

```text
D12_0 Majority Voting Baseline
D14.0 Info-Asymmetric Masking
```

Đây là hai thí nghiệm rẻ nhất và quan trọng nhất để quyết định hướng đi.

## Nếu PATH A: cộng tác thật

```text
D15.0 Compression Game
D15.1 Curriculum Compression
D15.2 Anti-Leak Compression

D16.0 One-Round Debate
D16.1 Multi-Round Debate

D17.0 Specialized Decomposition
D17.1 Learned Decomposer
```

## Nếu PATH B: cộng tác giả

```text
D12.0 Soft Message GSM8K Gold-Only
D12.1 Soft Message Ablation
D12.2 Leakage Probe
D12.3 Hybrid Given/Need + Soft

D18.0 Spectral Fusion
D18.1 Learnable Spectral Fusion
D18.2 Diversity Loss
```

## Giai đoạn 3

```text
D19.0 Cognitive Oscillator T=2
D19.1 Adaptive Halt
D19.2 Deep Supervision
```

---

# Quyết Định Benchmark

Có, phải chỉnh benchmark, nhưng làm theo thứ tự:

```text
1. Giữ GSM8K gốc:
   để so sánh với D11.0 và D11.4.

2. Thêm Masked GSM8K:
   bắt buộc từ D14.0 để ép cộng tác.

3. Nếu Masked GSM8K có tín hiệu:
   mở sang ZebraLogic split-clue hoặc BBEH split-condition.

4. Không chuyển benchmark quá sớm:
   vì nếu đổi benchmark + đổi architecture cùng lúc,
   ta sẽ không biết kết quả đến từ đâu.
```

---

# Bảng Chốt Các D

```text
D12 = Soft Message
  D12.0 Soft Message GSM8K Gold-Only V1
  D12.1 Bottleneck / Noise Ablation
  D12.2 Leakage Probe
  D12.3 Hybrid Given/Need + Soft Message

D12_0 = Voting Baselines
  D12_0 Majority Voting Baseline

D14 = Info-Asymmetric Benchmarking
  D14.0 Masked GSM8K
  D14.1 Split-Clue ZebraLogic / BBEH

D15 = Adversarial Compression Game
  D15.0 Text Compression Game
  D15.1 Curriculum Compression
  D15.2 Anti-Leak Compression

D16 = Debate Framework
  D16.0 One-Round Debate
  D16.1 Multi-Round / Symmetric Debate

D17 = Specialized Decomposition
  D17.0 Teacher / Rule-Based Decomposition
  D17.1 Learned Decomposer

D18 = Spectral Reasoning Fusion
  D18.0 Fixed Spectral Fusion
  D18.1 Learnable Spectral Fusion
  D18.2 Diversity Loss

D19 = Cognitive Oscillator
  D19.0 Oscillator T=2
  D19.1 Adaptive Halt
  D19.2 Deep Supervision
```

Điểm mấu chốt: **D12 vẫn là Soft Message**, nhưng về thứ tự chạy thực nghiệm thì **D12_0 và D14.0 phải chạy trước hoặc song song** để biết liệu Soft Message đang giải đúng vấn đề hay chỉ đang được áp lên một benchmark chưa đủ sức đo cộng tác thật.