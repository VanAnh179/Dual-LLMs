# D12_0 Voting Baseline — Run Commands

## Step 1: Retrain D11.0 Adapters (Required)

Due to truncation of the baseline adapter weights in the workspace (corrupted to exactly 768 KB / 1 MB), we need to retrain the D11.0 Agent A and Agent B adapters first.

Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

Run retraining script (this runs LoRA SFT alternating training for 1 round on 100 examples):
```bash
python scripts/train_alternating_lora.py \
  --config configs/d11_retrain.yaml \
  --max-train-examples 100 \
  --num-rounds 1
```

This will save the regenerated adapters to:
- `outputs/D11_0_adapters_regenerated/adapters/agent_A_round_1`
- `outputs/D11_0_adapters_regenerated/adapters/agent_B_round_1`

---

## Step 2: D12_0 Majority Voting Baseline Evaluation

Once retraining is done, run the independent evaluation of the two agents and compute the Majority and Oracle vote accuracy:

```bash
python scripts/evaluate_d12_0_voting_baseline.py \
  --config configs/d12_0_voting_baseline.yaml \
  --max-examples 100 \
  --sampling-mode first_n
```

### Outputs:

- Predictions: `outputs/D12_0_voting/generations/voting_predictions.jsonl`
- Metrics: `outputs/D12_0_voting/metrics/d12_0_metrics.json`
- Notes: `notes/D12_0_voting_results.md`

### Verification/Diagnostics:

To print the final metrics on the console, you can run:
```bash
python -c "import json; m=json.load(open('outputs/D12_0_voting/metrics/d12_0_metrics.json')); [print(f'{k}: {v}') for k,v in m.items()]"
```
