# FedShield MNIST demo runbook

Run every command from `fedshield/` after `pip install -r requirements.txt` (torch 2.7.0, torchvision 0.22.0). MNIST downloads automatically on the first run. `--device auto` (the default) uses a CUDA GPU when one is available and otherwise falls back to CPU. A 10-round Level 3 run takes about 2 minutes on a GTX 1080 Ti and about 5 minutes on CPU.

All runs use 10 clients, 10 rounds, and 1 local epoch per round (the course setting). Client IDs are zero-indexed; rounds are one-indexed.

## 1. Main defended demo

```bash
python3 level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3
```

What to look for in the output:

- `malicious client 3: honest until round 1, then malicious_client.generate_malicious_update`
- From round 2 on: `FLAGGED client 3 in round r` and `accepted=9` each round
- At the end: `Clients flagged at least once: [3]` and the detection metrics line

Expected result (seed 42, IID, GPU, torch 2.7.0): 9/9 malicious updates rejected, 0 false-positive exclusions, final verdict client 3, final (round-10) model 98.71% test accuracy, best-validation model (round 9) 98.65%.

The round-10 model is saved to `checkpoints/fedshield_mnist_last.pt`. This is the model the course evaluates after live training. The lowest-validation-loss model is also saved, to `checkpoints/fedshield_mnist_best.pt`, for analysis.

## 2. Code substitution of the malicious client test case

The malicious client's logic lives only in `malicious_client.py`, behind the course signature:

```python
def generate_malicious_update(client_model_state_dict, local_dataset, device):
    ...
    return malicious_state_dict
```

`# === MALICIOUS CLIENT LOGIC HERE ===` marks the attack body inside `malicious_client.py`. That is the only file to change; `level3_fedshield.py` just calls the function.

Steps:

1. Replace the body of `generate_malicious_update` in `malicious_client.py`, or replace the whole file with the different logic.
2. Run `python3 level3_fedshield.py --onset 2 --malicious-id 3`. Choose any onset from 2 to 10 and any client ID from 0 to 9.

How the runner treats the replacement:

- It calls the function once per attack round with a copy of the global model on `device` and the malicious client's own data. Before `--onset`, that client trains honestly.
- If the file defines only `generate_malicious_update` (no `ATTACK_MODES`/`configure()`), the run prints `malicious_client.py has no configure(); using its generate_malicious_update() as provided`, and the attack is named `custom`.
- Returned tensors may be on any device and of any dtype; they are moved to CPU and cast to the model's dtype. Missing keys, wrong shapes, or NaN/Inf values make the upload invalid, so it is excluded. If the function raises an exception, that round's upload counts as missing and training continues.

Verified: replacing the file with the course template (`param + torch.randn_like(param) * 10.0`) ran unchanged. Client 3 was flagged in rounds 2–10 (9/9 rejected, 0 false positives), the final verdict was client 3, and the final model reached 98.71%.

## 3. Attack modes for rehearsal

```bash
python3 level3_fedshield.py --attack <mode> --onset 2 --malicious-id 3
```

| Mode | Malicious upload |
| --- | --- |
| `noise` | global model + Gaussian noise (std 1.0) |
| `scale` | honest update × 10 |
| `sign_flip` | honest update × −5 |
| `free_ride` | global model unchanged |
| `label_flip` | trained for 1 epoch on labels y → 9 − y |
| `zero_weights` | all-zero model |
| `random_weights` | freshly initialized model |
| `nan` | honest update with layer `fc2` filled with NaN |
| `grad_ascent` | trained to increase the loss (gradient norm capped at 1.0) |
| `layer_only` | honest update with layer `fc2`'s update × −10 |
| `none` | no attacker (benign control) |

Attack strengths and the targeted layer are set in `ATTACK_CONFIG` at the top of `malicious_client.py`.

## 4. Edge and control cases

```bash
python3 level3_fedshield.py --attack label_flip --onset 10 --malicious-id 3   # attack only in the last round
python3 level3_fedshield.py --attack nan --onset 2 --malicious-id 3           # malformed uploads
python3 level3_fedshield.py --attack none                                     # benign control
```

Each run writes `logs/fedshield_<attack>_<partition>_seed<seed>_onset<onset>.json` (summary) and `.jsonl` (per-round detector log). These scenario runs also overwrite `checkpoints/fedshield_mnist_{best,last}.pt` unless you pass `--best-ckpt` and `--last-ckpt`.

Observed results (seed 42, GPU, torch 2.7.0):

- **Label flip, onset 10:** client 3 was flagged and excluded in round 10 (1/1 rejected). Client 5 was also falsely flagged in round 2. The final verdict was empty because it currently requires flags in at least two rounds; client 3 still appears in `Clients flagged at least once: [3, 5]`. Final model: 98.76%.
- **NaN:** all 9 malicious uploads were rejected as invalid (`nan_or_inf`), and training continued without errors. Client 5 was again falsely flagged in round 2, and the final verdict was client 3. Final model: 98.72%.
- **Benign:** the earlier recorded run also flagged client 5 in round 2. This false positive is the reason the detection threshold is still being tuned.

## 5. Baseline and ablation conditions

```bash
python3 level3_fedshield.py --attack sign_flip --defense off      # plain FedAvg, no detector
python3 level3_fedshield.py --attack sign_flip --no-validation    # layer-aware scores only
python3 level3_fedshield.py --attack sign_flip --no-clip          # no norm clipping
```

Each condition writes its own checkpoints (`checkpoints/fedshield_mnist_nodefense_{best,last}.pt`, `_noval_`, `_noclip_`) and logs (`..._onset2_nodefense.json`, `_noval`, `_noclip`). `--defense off` cannot be combined with the ablation switches. The checkpoint name does not include the attack, so two runs of the same condition overwrite each other's checkpoints; pass `--best-ckpt`/`--last-ckpt` to keep both.

Observed (seed 42, GPU, torch 2.7.0, malicious client 3, onset 2):

| Condition | Attack | Final test acc | Malicious rejected | False positives |
| --- | --- | --- | --- | --- |
| FedShield (full) | `sign_flip` | 98.71% | 9/9 | 0 |
| Plain FedAvg (`--defense off`) | `sign_flip` | 96.41% | — | — |
| No validation evidence | `sign_flip` | 98.71% | 9/9 | 0 |
| No clipping | `sign_flip` | 98.72% | 9/9 | 0 |
| FedShield (full) | `label_flip` | 98.71% | 9/9 | 0 |
| No validation evidence | `label_flip` | 98.71% | 9/9 | 0 |
| Plain FedAvg (`--defense off`) | `nan` | 9.80% | — | — |

Without the defense, a sign-flip attacker costs 2.3 points of accuracy, and one NaN upload makes the global model unusable from round 2 on. With these two attacks, the layer-aware scores alone already catch every malicious update. Clipping at `--clip-norm 5.0` barely changes the result because honest updates are much smaller than 5.0. Report these as single-seed observations.

## 6. Reload and resume

```bash
python3 level3_fedshield.py --eval-only        # test accuracy of the best and last checkpoints
python3 level3_fedshield.py --eval-only \
  --best-ckpt checkpoints/fedshield_mnist_best.pt \
  --last-ckpt checkpoints/fedshield_mnist_last.pt
```

To continue an interrupted run, repeat the original command with `--resume`. The options must match the saved run configuration. Verified: stopping after round 3 and resuming to round 4 reproduced the uninterrupted run's round-4 detector scores and validation loss exactly.

## 7. Recorded results and limitations

- `results/level3_cpu_runs.json`, `plots/fedshield_results.png`, `FIRST_RUN_REPORT.md`, and the committed `logs/fedshield_*` files come from the first CPU run set. That set used PyTorch 2.15.0 and the earlier attack code built into `level3_fedshield.py`. Its numbers are close to, but not identical with, runs of the current code on torch 2.7.0. Regenerate these files with the current code before final reporting.
- The final verdict misses attacks that appear in only one round (for example, onset 10), and the benign false positive on client 5 in round 2 is still open. Both are being addressed in the detector settings.
- All results come from a single seed and an IID split. Run multiple seeds, attack types, and malicious-client IDs before making broad performance claims.
