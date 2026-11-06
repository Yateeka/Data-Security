# FedShield project handoff

This folder contains the Level 1 MNIST baseline and the components for a ten-client federated learning experiment with malicious-client detection and defense. The work is split between Yateeka and Jessie as described below. The baseline and synthetic checks have been run locally; the end-to-end federated workflow has not yet been verified.

## Progress and next milestone

Yateeka has run the following checks locally on a Mac:

- `python3 test_detector_synthetic.py` — all listed checks passed: five synthetic detector scenarios, late-onset exclusion, benign false-positive check, invalid-update rejection, validation evidence, threshold calibration, and reproducible split/no-leakage check. These are component-level synthetic checks, not evidence of performance on a federated training run.
- `python3 level1_centralized.py --epochs 5 --seed 42` — completed with **99.04% final test accuracy**. The checkpoint saved to `checkpoints/centralized_mnist.pt` and the reload accuracy check passed.

The main next milestone is the **end-to-end ten-client federated run**. Jessie should prioritize wiring local client training and 10-round FedAvg to the detector, then run the three proposal conditions (benign FedAvg, attacked FedAvg, defended FedAvg) with matching client partitions and seeds. Report final test accuracy alongside detection delay, rejected malicious updates, benign false positives, and final client-verdict accuracy. The current synthetic detector results are encouraging, but they do not establish federated attack performance.

Before calling the attack suite complete, add the proposal's **label-flipping** attack and test it through the actual simulator. Current synthetic tests cover noise, sign flip, scaling, free-riding, and a layer-only perturbation; they do not test label flipping or local training under the proposed attack interface. Also include late attack onset and benign runs in the integrated evaluation.

For evaluation hygiene, use the held-out validation set for training-time checks and detector scoring, and reserve test accuracy for the final evaluation. `level1_centralized.py` currently prints test accuracy after each epoch; avoid using those intermediate test results to choose settings or checkpoints. The reported 99.04% is the final epoch's test result from the current run.

## Work completed by Yateeka so far

The following Yateeka-owned modules currently have implementation in the workspace:

- **Foundations — `common.py`:** reproducible Python/NumPy/PyTorch seeding, device selection, the shared `SmallCNN`, model evaluation, and checkpoint save/load helpers.
- **Data preparation — `data_utils.py`:** seeded held-out validation split, split-index saving, loader construction using the instructor template, and a train/validation leakage check. The validation set is intended for server-side scoring; test data is kept for final accuracy.
- **Level 1 — `level1_centralized.py`:** centralized MNIST training/evaluation, validation tracking, checkpoint saving and reload check. It uses `data_utils.py` for the reproducible data split.
- **Security and defense — `fedshield_detector.py`:** parameter-delta validation; per-layer magnitude and cosine anomaly scores; optional validation-loss evidence; single-client round flagging; immediate exclusion of flagged/invalid updates; suspicion history and final repeated-evidence verdict.
- **Testing and reporting — `detection_metrics.py`:** per-run detection/false-positive metrics, benign-run threshold calibration, attack/onset/identity/seed sweep support, and per-attack summaries.
- **Synthetic tests — `test_detector_synthetic.py`:** synthetic attack, benign false-positive, malformed-update, validation-evidence, calibration, and data-split checks; all currently pass in the reported local run. Label flipping and simulator integration remain to be tested.
- **Instructor template — `dataloader4level1.py`:** present in the folder. Keep the course-provided file unchanged.

**Status:** the Yateeka-owned modules listed above have implementations in the workspace. Integration and tests still need to be run and any issues fixed; the simulator and end-to-end federated workflow remain part of Jessie's workstream.

## Jessie's workstream: implementation checklist

Jessie owns the federated-learning simulator and the parts that connect it to the detector. The detector API and model definitions below are the intended integration contract.

1. **Ten-client simulator and local training**
   - Build a ten-client simulation using the shared `SmallCNN` and data setup.
   - Implement each client's local training loop and the ten-round run configuration.
   - Use the same seed and model/checkpoint conventions from `common.py` so runs can be reproduced and checkpoints can be exchanged.

2. **FedAvg aggregation**
   - Implement standard FedAvg for the baseline and the guarded aggregation path for defended runs.
   - For defended runs, take only client updates in `RoundResult.accepted_ids` from `FedShieldDefense.inspect_round`.
   - Apply accepted-update norm clipping, then aggregate. If no client updates are accepted, retain the previous global model rather than averaging an empty set.

3. **Server-side validation-loss evidence**
   - For each uploaded client state, measure validation loss after applying that update and subtract the current global model's validation loss.
   - Pass those per-client deltas as `val_deltas` to `inspect_round(round_idx, global_state, client_states, val_deltas)`.
   - Use the held-out validation set from `data_utils.py`; do not use the test set to score clients or calibrate the detector.

4. **Malicious-client attack module**
   - Implement four configurable attack modes and make attack identity, onset round, and malicious client ID configurable.
   - Keep attack behavior reproducible under the run seed. Coordinate the mode names and update transformations with the synthetic scenarios in `test_detector_synthetic.py`.

5. **Round-by-round detector integration**
   - Keep one `FedShieldDefense` instance per run and call `inspect_round` once for each round, supplying the global state, one uploaded state (or `None`) per client, and validation deltas when enabled.
   - Exclude both flagged and invalid updates; the result exposes `accepted_ids`, `flagged_ids`, `invalid_ids`, `scores`, and `reasons`.
   - After training, call `final_verdict()` to get the repeated-evidence malicious-client decision.
   - Note: the detector supports a single-attacker assumption and flags at most one scored client per round by default.

6. **Evaluation, tests, and demo**
   - Compare centralized/baseline and federated accuracy, and compare undefended and defended runs.
   - Add attack unit tests, accuracy comparisons, checkpoint save/reload tests, and demo replacement tests.
   - Produce baseline/accuracy plots and a demo runbook. Include false positives on benign runs, detection delay, attack rejection, final verdict accuracy, and failure cases in reported results.
   - Ensure the complete run is reproducible from documented commands and seeds.

## Shared responsibilities

Yateeka and Jessie should jointly review the scoring interface and security-test changes, inspect the final plots, agree on conclusions, and rehearse the end-to-end demonstration. Keep work in separate feature branches and review each other's pull requests. Both owners should understand how to run the complete experiment.

## Current file map

| File | Purpose / current state |
| --- | --- |
| `common.py` | Shared seed, CNN, evaluation, and checkpoint helpers; implemented draft. |
| `data_utils.py` | Seeded held-out validation split, index saving, loader helper, and leakage check; implemented draft. |
| `level1_centralized.py` | Level 1 centralized MNIST training; implemented draft, depends on `data_utils.py`. |
| `fedshield_detector.py` | Layer-aware detection and defense interface; implemented draft for simulator integration. |
| `detection_metrics.py` | Detection metrics, threshold calibration, and sweep helpers; implemented draft. |
| `test_detector_synthetic.py` | Synthetic detector and split checks; passed in the reported local run. Does not replace integrated FL/label-flipping evaluation. |
| `dataloader4level1.py` | Course-provided data loader; preserve unchanged. |
| `checkpoints/`, `logs/` | Output directories for model checkpoints and run logs. |

## Setup and first checks

From this directory, install the project dependencies once:

```bash
pip install torch torchvision numpy
```

Run the synthetic checks from this folder:

```bash
python test_detector_synthetic.py
```

The first MNIST load creates `data/mnist/` automatically. Do not create it by hand. The current synthetic tests avoid downloading MNIST by substituting a small in-memory dataset.

To train the centralized baseline:

```bash
python level1_centralized.py --epochs 5 --seed 42
```

The simulator entry point, attack module, and plotting/runbook commands are still to be supplied by the integration work; agree on their filenames and command-line interface before documenting final demo commands.
