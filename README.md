# FedShield

**A layer-aware, validation-guided defense against malicious clients in federated learning**

FedShield is a CSC 8370 Data Security project investigating how a central server can detect and limit harmful client updates during federated learning. We use a ten-client simulation on MNIST to study whether examining model updates layer by layer, together with trusted server-side validation data, can improve resilience while preserving model accuracy.

## Quick start

All commands run from the `fedshield/` folder; checkpoint paths are relative to it.

### 1. Set up the environment (Python 3.12)

```bash
cd fedshield
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MNIST downloads automatically to `fedshield/data/mnist/` on the first run. The code uses a CUDA GPU when one is available and falls back to CPU otherwise.

### 2. Test the saved models (Levels 1 and 2, inference only)

```bash
python level1_centralized.py --eval-only     # loads checkpoints/centralized_mnist_best.pt 
python level2_fedavg.py --eval-only          # loads checkpoints/fedavg_mnist_best.pt
```

Quick checks without training or MNIST (each takes seconds):

```bash
python test_detector_synthetic.py            # detector, data split, calibration
python test_malicious_client.py              # every attack mode in malicious_client.py
```

### 3. Retrain (optional)

```bash
python level1_centralized.py --epochs 5 --seed 42
python level2_fedavg.py                      # 10 clients, 10 rounds, 1 local epoch
```

### 4. Level 3 (live training)

```bash
python level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3   # 10 rounds, 1 local epoch
python level3_fedshield.py --attack none                                   # benign control
python level3_fedshield.py --eval-only       # test checkpoints/fedshield_mnist_{best,last}.pt
```

Ten clients train for 10 rounds with 1 local epoch each. The malicious client (`--malicious-id`, zero-indexed) trains honestly until `--onset` (round 2 or later) and from then on uploads whatever `malicious_client.generate_malicious_update()` returns. Each round prints `FLAGGED client k in round r` for excluded updates, and the run ends with the list of all flagged clients and the final verdict. The round-10 global model is saved to `checkpoints/fedshield_mnist_last.pt`; this is the model the course evaluates after live training.

`--attack` selects a mode from [`fedshield/malicious_client.py`](fedshield/malicious_client.py):

| Mode | Malicious upload |
| --- | --- |
| `noise` | global model + Gaussian noise |
| `scale` / `sign_flip` | honest update multiplied by +10 / -5 |
| `free_ride` | global model unchanged |
| `label_flip` | trained on wrong labels (y → 9 − y) |
| `zero_weights` / `random_weights` | all-zero model / freshly initialized model |
| `nan` | one layer filled with NaN (malformed upload) |
| `grad_ascent` | trained to increase the loss |
| `layer_only` | honest update except one layer (`fc2`) scaled by -10 |
| `none` | no attacker (benign control) |

### 5. Baseline and ablation runs (for the report)

```bash
python level3_fedshield.py --attack sign_flip --defense off      # plain FedAvg with the attacker
python level3_fedshield.py --attack sign_flip --no-validation    # detector without validation evidence
python level3_fedshield.py --attack sign_flip --no-clip          # defense without norm clipping
python level3_fedshield.py --attack sign_flip --malicious-id random   # attacker drawn from --seed (differs per attack)
python level3_fedshield.py --attack scale --attack-setting scale_factor=3   # weaker attack (any ATTACK_CONFIG key)
python level3_fedshield.py --attack layer_only --whole-model --threshold 13.8   # one score for the whole model
```

`--defense off` runs plain FedAvg over every upload with no detector. `--no-validation` and `--no-clip` each remove one defense component. Each condition saves to its own files (`checkpoints/fedshield_mnist_nodefense_*.pt`, `_noval_`, `_noclip_`, and matching `logs/` names), so the main defended checkpoint is never overwritten.

### 6. Replacing the malicious client (demo day)

The malicious client's logic is isolated in `fedshield/malicious_client.py` behind the course signature:

```python
def generate_malicious_update(client_model_state_dict, local_dataset, device):
    ...
    return malicious_state_dict
```

The attack logic is marked with `# === MALICIOUS CLIENT LOGIC HERE ===` inside that function; it is the only file to change. To substitute your own attack, replace the function body (or the whole file) and run:

```bash
python level3_fedshield.py --onset 2 --malicious-id 3
```

If the replacement file defines only `generate_malicious_update`, the script runs it as the `custom` attack. The script decides when the attack starts (`--onset`), so the function does not need a round number. If the function raises an exception or returns a malformed state dict, that upload is treated as invalid and excluded; training continues.

Final results (88 runs: defense on/off, ablations, 3 seeds, random attacker, weak attacks, per-layer vs whole-model scoring): [`fedshield/RESULTS.md`](fedshield/RESULTS.md). Detailed scenarios and expected outputs: [`fedshield/DEMO_RUNBOOK.md`](fedshield/DEMO_RUNBOOK.md). Developer notes and work split: [`fedshield/handoff_notes.md`](fedshield/handoff_notes.md).

## TA evaluation

From the extracted project folder, enter `fedshield/`, install the pinned requirements, and run the saved-checkpoint evaluations. These commands do not retrain the models:

```bash
cd fedshield
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python level1_centralized.py --eval-only
python level2_fedavg.py --eval-only
python level3_fedshield.py --eval-only
```

The Level 3 command reports both checkpoints. Use `checkpoints/fedshield_mnist_last.pt` for the final round-10 model; `fedshield_mnist_best.pt` is the best-validation checkpoint from round 9. The first run downloads MNIST if it is not already present. The scripts select CUDA when available and otherwise run on CPU.

If the TA supplies a separate test DataLoader, it can be passed directly to the shared evaluator. It must yield `(images, labels)` batches in the model's expected input format:

```python
from common import evaluate, get_device, load_checkpoint

device = get_device()
model, metadata = load_checkpoint(
    "checkpoints/fedshield_mnist_last.pt", device=device
)
loss, accuracy = evaluate(model, ta_test_loader, device)
print(f"Test accuracy: {accuracy:.2%}")
```

`ta_test_loader` is the DataLoader provided by the TA. This evaluates the saved round-10 model without retraining.

## What we are building

The project covers three levels:

1. **Centralized baseline:** train a compact CNN on MNIST as a reference for model accuracy.
2. **Federated baseline:** divide the MNIST training data among ten clients. Each client trains locally, and the server combines updates with Federated Averaging (FedAvg) over ten global rounds, with one local epoch per client per round.
3. **FedShield defense:** detect suspicious updates during training, exclude flagged or invalid updates from that round's aggregation, and track evidence across rounds to identify a likely malicious client.

## How the defense works

For each submitted client update, the server checks that its parameters have valid shapes and finite values. It then compares the update with the other clients' updates, separately for each model layer. Magnitude and direction measurements help surface unusual updates that a whole-model score could hide. Near-zero updates, such as a client returning the unchanged global model, are checked directly.

The server also holds out a small clean subset of MNIST training examples. It measures how applying each candidate update changes validation loss and can use that evidence alongside the layer-level scores. Suspicion accumulates over rounds, while any update flagged in a round is excluded immediately. Accepted updates are norm-clipped before defended aggregation to limit their influence.

The threat model assumes nine benign clients and at most one malicious client. The four proposal attacks (random noise, scaling or sign flipping, free-riding, and label flipping) are implemented in `malicious_client.py`, along with extra robustness modes: zero or random weights, NaN uploads, gradient ascent, and a single-layer attack. The detector may miss stealthy or benign-looking malicious updates; the project aims to measure those limits rather than claim complete protection.

## Evaluation

We compare three conditions using the same data partitions, attack scenarios, and random seeds:

- Benign federated learning with FedAvg
- Attacked federated learning with plain FedAvg
- Attacked federated learning with FedShield

We measure final global-model test accuracy and malicious-client detection. Supporting metrics include detection delay, malicious updates rejected, false-positive exclusions of benign clients, and per-round accuracy. Ablations compare layer-aware scoring alone, layer-aware scoring with validation evidence, and the full defense with clipping. The held-out validation set is used for detection and tuning; test data is reserved for final evaluation.

## Repository contents

The implementation is in [`fedshield/`](fedshield/), including the centralized MNIST baseline, ten-client FedAvg baseline, defended Level 3 runner, malicious-client attack module, data split utilities, detector, metrics, and synthetic checks. See the Quick start above for run instructions, [`fedshield/DEMO_RUNBOOK.md`](fedshield/DEMO_RUNBOOK.md) for demo scenarios, [`fedshield/handoff_notes.md`](fedshield/handoff_notes.md) for developer notes, and [`fedshield/RESULTS.md`](fedshield/RESULTS.md) for the final Level 3 results.
