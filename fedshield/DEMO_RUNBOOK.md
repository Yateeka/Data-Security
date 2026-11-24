# FedShield MNIST demo runbook

Run every command from `fedshield/` after `pip install -r requirements.txt` (torch 2.7.0, torchvision 0.22.0). MNIST downloads automatically on the first run. `--device auto` (the default) uses a CUDA GPU when one is available and otherwise falls back to CPU. A 10-round Level 3 run takes about 2 minutes on a GTX 1080 Ti and about 5.5 minutes on an 8-core CPU (measured Nov 24 from a fresh clone). On CPU the detection is identical (client 3 flagged in rounds 2–10, same verdict), but floating-point differences give a final accuracy of 98.69% instead of 98.71%.

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
2. Run `python3 level3_fedshield.py --onset 2 --malicious-id 3`. Choose any onset from 2 to 10 and any client ID from 0 to 9, or `--malicious-id random` to draw one from the seed.

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

Observed results (seed 42, GPU, torch 2.7.0, calibrated threshold 14.0):

- **Label flip, onset 10:** client 3 was flagged and excluded in round 10 (1/1 rejected), no other client was flagged, and the final verdict was client 3 (strong single-round evidence). Final model: 98.75%.
- **Sign flip, onset 10:** same outcome: client 3 flagged in round 10 only, final verdict client 3. Final model: 98.75%.
- **Benign:** no client was flagged in any round, and the final verdict was "no malicious client identified". Final model: 98.76%.
- **NaN:** all 9 malicious uploads were rejected as invalid (`nan_or_inf`), training continued without errors, and the final verdict was client 3. Final model: 98.71%.

### How the threshold and verdict rule were set

Full calibration logs, tables, and a script that recomputes the numbers: [`results/calibration/`](results/calibration/README.md) (`python calibrate_threshold.py`).

- **Threshold 14.0:** calibrated on benign development runs only (seeds 101–105, never seed 42 or test results). Those runs used `--threshold 1e9` so nothing was excluded, which recorded every honest client's score in 50 rounds. The highest was 11.23; `detection_metrics.calibrate_threshold` multiplies by a 1.25 safety margin, giving 14.04. The earlier placeholder 8.0 flagged client 5 in round 2 (score 10.2) as a false positive.
- **Verdict:** the top-suspicion client is named malicious if it was excluded in at least 2 rounds, or excluded once with strong evidence (score ≥ 2 × threshold = 28, or a malformed upload). On development seed 101, the attacker's weakest score across all 8 attacks was 45.6 (`layer_only`), while honest clients stayed at or below 5.1. So an attack in round 10 alone is still identified, while an honest client that only just crosses the threshold is not.

## 5. Baseline and ablation conditions

```bash
python3 level3_fedshield.py --attack sign_flip --defense off      # plain FedAvg, no detector
python3 level3_fedshield.py --attack sign_flip --no-validation    # layer-aware scores only
python3 level3_fedshield.py --attack sign_flip --no-clip          # no norm clipping
```

Each condition writes its own checkpoints (`checkpoints/fedshield_mnist_nodefense_{best,last}.pt`, `_noval_`, `_noclip_`) and logs (`..._onset2_nodefense.json`, `_noval`, `_noclip`). `--defense off` cannot be combined with the ablation switches. The checkpoint name does not include the attack, so two runs of the same condition overwrite each other's checkpoints; pass `--best-ckpt`/`--last-ckpt` to keep both.

Observed in the final sweep (seed 42, GPU, torch 2.7.0, threshold 14.0, malicious client 3, onset 2; full tables in [`RESULTS.md`](RESULTS.md)):

| Attack | Plain FedAvg (`--defense off`) | FedShield | No validation | No clipping |
| --- | ---: | ---: | ---: | ---: |
| `noise` | 75.67% | 98.71% | | |
| `layer_only` | 93.09% | 98.71% | 98.71% | 98.72% |
| `sign_flip` | 96.41% | 98.71% | | |
| `scale` | 96.81% | 98.71% | | |
| `label_flip` | 97.71% | 98.71% | 98.71% | 98.72% |
| `free_ride` | 98.61% | 98.71% | | |
| none (benign) | 98.71% | 98.76% | | |

FedShield rejected 9/9 malicious uploads with 0 false positives in every defended run. All defended attack runs end at the same 98.71%: client 3 is excluded from round 2 on, so each round averages the same 9 honest updates. Removing validation evidence or clipping changes nothing for these attacks, because the layer-aware scores alone catch them, and the clip bound of 5.0 is far above honest update norms. With `nan` uploads, plain FedAvg is ruined from the first attack round on (9.80% final accuracy, chance level), while FedShield rejects all 9 as invalid and ends at 98.71%.

To rerun every scenario: `python run_sweep.py` (82 runs, ~2.5 h on GPU; checkpoints go to `checkpoints/sweep/`). Weaker attacks (`--attack-setting`, e.g. `scale_factor=3`) do show what validation adds: without it, `scale` ×3 is accepted in all 9 rounds, with it all 9 are rejected; see [`RESULTS.md`](RESULTS.md) §6. `--whole-model --threshold 13.8` scores the model as one vector instead of per layer. Without validation, it rejects the single-layer `layer_only` attack in only 5 of 9 rounds, while per-layer scoring rejects all 9 (§7).

## 6. Reload and resume

```bash
python3 level3_fedshield.py --eval-only        # test accuracy of the best and last checkpoints
python3 level3_fedshield.py --eval-only \
  --best-ckpt checkpoints/fedshield_mnist_best.pt \
  --last-ckpt checkpoints/fedshield_mnist_last.pt
```

To continue an interrupted run, repeat the original command with `--resume`. The options must match the saved run configuration. Verified: stopping after round 3 and resuming to round 4 reproduced the uninterrupted run's round-4 detector scores and validation loss exactly.

## 7. Recorded results and limitations

- Final results: [`RESULTS.md`](RESULTS.md), [`results/level3_runs.json`](results/level3_runs.json) (82 runs from `run_sweep.py`), and the plots `plots/fedshield_results.png` and `plots/fedshield_rounds.png` (`python plot_results.py`). The `logs/fedshield_*` files listed in `results/level3_runs.json` come from this sweep.
- The threshold was calibrated on 5 benign IID runs. A non-IID split or a weaker, stealthier attack (score between 14 and 28, appearing only in the last round) would be excluded but not named in the final verdict.
- Results cover 3 seeds (42, 7, 123), an IID split, and malicious clients 3, 9, and 7. Adaptive attacks that tune themselves to stay under the threshold were not tested.
