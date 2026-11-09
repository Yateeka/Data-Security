# FedShield

**A layer-aware, validation-guided defense against malicious clients in federated learning**

FedShield is a CSC 8370 Data Security project investigating how a central server can detect and limit harmful client updates during federated learning. We use a ten-client simulation on MNIST to study whether examining model updates layer by layer, together with trusted server-side validation data, can improve resilience while preserving model accuracy.

## What we are building

The project covers three levels:

1. **Centralized baseline:** train a compact CNN on MNIST as a reference for model accuracy.
2. **Federated baseline:** divide the MNIST training data among ten clients. Each client trains locally, and the server combines updates with Federated Averaging (FedAvg) over ten global rounds, with one local epoch per client per round.
3. **FedShield defense:** detect suspicious updates during training, exclude flagged or invalid updates from that round's aggregation, and track evidence across rounds to identify a likely malicious client.

## How the defense works

For each submitted client update, the server checks that its parameters have valid shapes and finite values. It then compares the update with the other clients' updates, separately for each model layer. Magnitude and direction measurements help surface unusual updates that a whole-model score could hide. Near-zero updates, such as a client returning the unchanged global model, are checked directly.

The server also holds out a small clean subset of MNIST training examples. It measures how applying each candidate update changes validation loss and can use that evidence alongside the layer-level scores. Suspicion accumulates over rounds, while any update flagged in a round is excluded immediately. Accepted updates are norm-clipped before defended aggregation to limit their influence.

The threat model assumes nine benign clients and at most one malicious client. Planned attacks include random noise, scaling or sign flipping, free-riding, and label flipping. The detector may miss stealthy or benign-looking malicious updates; the project aims to measure those limits rather than claim complete protection.

## Evaluation

We compare three conditions using the same data partitions, attack scenarios, and random seeds:

- Benign federated learning with FedAvg
- Attacked federated learning with plain FedAvg
- Attacked federated learning with FedShield

We measure final global-model test accuracy and malicious-client detection. Supporting metrics include detection delay, malicious updates rejected, false-positive exclusions of benign clients, and per-round accuracy. Ablations compare layer-aware scoring alone, layer-aware scoring with validation evidence, and the full defense with clipping. The held-out validation set is used for detection and tuning; test data is reserved for final evaluation.

## Repository contents

The implementation is in [`fedshield/`](fedshield/), including the centralized MNIST baseline, the ten-client FedAvg baseline, data split utilities, detector, metrics, and synthetic checks. See [`fedshield/README.md`](fedshield/README.md) for setup and run instructions.
