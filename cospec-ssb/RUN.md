# Run Commands

Các lệnh dưới đây dùng cho pipeline LoRA SFT dual-agent với config hiện tại.

## Smoke / 100 Examples

Chuẩn bị dữ liệu 100 mẫu train/test:

```bash
python scripts/prepare_gsm8k.py --max-train-examples 100 --max-test-examples 100
```

Bootstrap reasoning traces:

```bash
python scripts/bootstrap_traces.py --config configs/d11_qwen05b.yaml --max-examples 100
```

Train LoRA alternating agents:

```bash
python scripts/train_alternating_lora.py --config configs/d11_qwen05b.yaml --max-train-examples 100
```

Evaluate LoRA 4-way trên 100 câu test:

```bash
python scripts/evaluate_two_agent.py \
  --config configs/d11_qwen05b.yaml \
  --max-examples 100 \
  --output-metrics outputs/metrics/eval_metrics_lora_4way_qwen25_15b_100.json \
  --output-predictions outputs/generations/eval_predictions_lora_4way_qwen25_15b_100.jsonl
```

## Wider / Full Run

Chuẩn bị dữ liệu rộng hơn theo config/default:

```bash
python scripts/prepare_gsm8k.py
```

Bootstrap reasoning traces cho toàn bộ tập train được config chọn:

```bash
python scripts/bootstrap_traces.py --config configs/d11_qwen05b.yaml
```

Train LoRA alternating agents:

```bash
python scripts/train_alternating_lora.py --config configs/d11_qwen05b.yaml
```

Evaluate LoRA 4-way trên toàn bộ tập test được config chọn:

```bash
python scripts/evaluate_two_agent.py \
  --config configs/d11_qwen05b.yaml \
  --output-metrics outputs/metrics/eval_metrics_lora_4way_qwen25_15b_full.json \
  --output-predictions outputs/generations/eval_predictions_lora_4way_qwen25_15b_full.jsonl
```

## Notes

- `evaluate_two_agent.py` hiện chỉ evaluate LoRA agents; không còn `--base-only`.
- Nếu muốn đổi model, sửa `student_model_name` và các hyperparameter tương ứng trong config, hoặc tạo config mới rồi thay đường dẫn `--config`.
- Các ID được sample sẽ được ghi vào `outputs/metrics/sampled_ids.json`.

## D11.2 Latent Collaborative SFT Smoke Test

Bootstrap collaborative traces với 10 examples:

```bash
python scripts/bootstrap_collaborative_traces.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-examples 10 \
  --num-candidates 1 \
  --max-new-tokens 512
```

Build collaborative SFT data:

```bash
python scripts/build_collaborative_sft_data.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml
```

Train với 10 examples:

```bash
python scripts/train_collaborative_lora_sft.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-train-examples 10 \
  --variant alternating
```

Evaluate với 10 examples:

```bash
python scripts/evaluate_collaborative_agents.py \
  --config configs/d11_2_qwen_math7b_teacher.yaml \
  --max-examples 10 \
  --sampling-mode first_n
```

Analyze contributions:

```bash
python scripts/analyze_contributions.py \
  --predictions outputs/D11_2_latent_collaborative_sft/generations/eval_predictions.jsonl
```

## D11.3 Useful Notes + Final Decider SFT

Bootstrap compact useful-note traces:

```bash
python scripts/bootstrap_useful_notes_traces.py \
  --config configs/d11_3_useful_notes_decider.yaml \
  --max-examples 10 \
  --num-candidates 1 \
  --max-new-tokens 384
```

Build D11.3 SFT data:

```bash
python scripts/build_useful_notes_sft_data.py \
  --config configs/d11_3_useful_notes_decider.yaml
```

Train Agent A as notes generator and Agent B as final decider:

```bash
python scripts/train_useful_notes_lora_sft.py \
  --config configs/d11_3_useful_notes_decider.yaml \
  --max-train-examples 10
```

Evaluate D11.3:

```bash
python scripts/evaluate_useful_notes_agents.py \
  --config configs/d11_3_useful_notes_decider.yaml \
  --max-examples 10 \
  --sampling-mode first_n
```

For a wider run, use `--max-examples 200` for bootstrap/train and `--max-examples 100` for evaluation.

## D11.4 Compact Given/Need + Final Decider SFT

D11.4 keeps D11.3 intact and changes Agent A into a compact semantic compressor. Agent A outputs only `Given` and `Need`; Agent B solves from the problem plus those notes.

Bootstrap Given/Need notes with 14B Instruct and solutions with 7B Math:

```bash
python scripts/bootstrap_given_need_traces.py \
  --config configs/d11_4_compact_given_need_decider.yaml \
  --max-examples 200
```

Build D11.4 SFT data:

```bash
python scripts/build_given_need_sft_data.py \
  --config configs/d11_4_compact_given_need_decider.yaml
```

Train Agent A as Given/Need compressor and Agent B as final decider:

```bash
python scripts/train_given_need_lora_sft.py \
  --config configs/d11_4_compact_given_need_decider.yaml \
  --max-train-examples 200
```

Evaluate D11.4 on 100 test examples:

```bash
python scripts/evaluate_given_need_agents.py \
  --config configs/d11_4_compact_given_need_decider.yaml \
  --max-examples 100 \
  --sampling-mode first_n
```
## V02 controlled training and official external evaluation

Build and validate the controlled synthetic training benchmark:

```bash
python scripts/V02_generate_benchmark.py \
  --config configs/v02_multifamily_benchmark.yaml
python scripts/V02_validate_benchmark.py \
  --config configs/v02_multifamily_benchmark.yaml
```

Build the evaluation-only official holdout and verify provenance:

```bash
python scripts/V02_build_official_external.py \
  --config configs/v02_official_external.yaml
python scripts/V02_validate_official_external.py \
  --config configs/v02_official_external.yaml
```

Train all matched-budget controls and the latent split channel on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/V02_train_single_baselines.py \
  --config configs/v02_split_vs_single.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/V02_train_split_latent.py \
  --config configs/v02_split_vs_single.yaml
```

Run the controlled test followed by the untouched official external test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/V02_evaluate_split_vs_single.py \
  --config configs/v02_split_vs_single.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/V02_evaluate_split_vs_single.py \
  --config configs/v02_split_vs_single.yaml \
  --external-config configs/v02_official_external.yaml
```
