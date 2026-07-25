# D12.1 Info-Asymmetric Masking

## Smoke test (2 examples)

```bash
python scripts/bootstrap_d12_1_masked_traces.py \
  --config configs/d12_1_info_asymmetric.yaml \
  --max-examples 2

python scripts/build_d12_1_masked_sft_data.py \
  --config configs/d12_1_info_asymmetric.yaml

python scripts/train_d12_1_masked_lora.py \
  --config configs/d12_1_info_asymmetric.yaml \
  --max-train-examples 2

python scripts/evaluate_d12_1_masked_agents.py \
  --config configs/d12_1_info_asymmetric.yaml \
  --max-examples 2 \
  --sampling-mode first_n
```

## Wider run (200 train / 100 eval)

```bash
python scripts/bootstrap_d12_1_masked_traces.py \
  --config configs/d12_1_info_asymmetric.yaml

python scripts/build_d12_1_masked_sft_data.py \
  --config configs/d12_1_info_asymmetric.yaml

python scripts/train_d12_1_masked_lora.py \
  --config configs/d12_1_info_asymmetric.yaml

python scripts/evaluate_d12_1_masked_agents.py \
  --config configs/d12_1_info_asymmetric.yaml \
  --max-examples 100 \
  --sampling-mode first_n
```

## Optimization Note
The teacher model is configured to load in 4-bit quantization by default (`load_in_4bit: true` in config). This avoids heavy RAM offloading to CPU/SSD on 12GB VRAM cards (like RTX 3060) and provides a **50x speedup** for bootstrap generation. You can toggle this setting in `configs/d12_1_info_asymmetric.yaml` if needed.
