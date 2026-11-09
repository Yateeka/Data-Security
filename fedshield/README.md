# FedShield project handoff

This folder contains the Level 1 MNIST baseline, the Level 2 FedAvg baseline, and the components for a ten-client federated learning experiment with malicious-client detection and defense. The work is split between Yateeka and Jessie as described below. The Level 1 baseline, the Level 2 federated baseline (10 clients, FedAvg), and the synthetic detector checks have been run locally. The Level 3 defended workflow (attack module + detector wired into the simulator) has not yet been built or verified.

## Progress and next milestone

Yateeka has run the following checks locally on a Mac:

- `python3 test_detector_synthetic.py` — all listed checks passed: five synthetic detector scenarios, late-onset exclusion, benign false-positive check, invalid-update rejection, validation evidence, threshold calibration, and reproducible split/no-leakage check. These are component-level synthetic checks, not evidence of performance on a federated training run.
- `python3 level1_centralized.py --epochs 5 --seed 42` — completed with **99.04% final test accuracy**. The checkpoint saved to `checkpoints/centralized_mnist.pt` and the reload accuracy check passed.

Jessie has run Level 2 on Linux (CPU only, Python 3.12.3, torch 2.7.0, torchvision 0.22.0, NumPy 2.5.4):

- `python3 level2_fedavg.py` (10 clients, IID split, 10 rounds, 1 local epoch, seed 42) — about 5 minutes on CPU. Validation loss fell from 0.206 after round 1 to 0.034 by round 9 and stayed there in round 10. **Final global model: 98.72% test accuracy.** Best checkpoint by validation loss (round 9): **98.62% test accuracy**, saved to `checkpoints/fedavg_mnist_best.pt`.
- `python3 level2_fedavg.py --eval-only` — reloaded the best checkpoint and reproduced 98.62%.
- `--resume` was checked by stopping after round 2 and resuming to round 3; round 3 matched the uninterrupted run exactly.

The main next milestone is **Level 3: the defended ten-client run**. The FedAvg simulator from Level 2 now exists (`fl_core.py`, `level2_fedavg.py`); the remaining work is wiring it to the detector and an attack module, then running the three proposal conditions (benign FedAvg, attacked FedAvg, defended FedAvg) with matching client partitions and seeds. Report final test accuracy alongside detection delay, rejected malicious updates, benign false positives, and final client-verdict accuracy. The current synthetic detector results are encouraging, but they do not establish federated attack performance.

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

**Status:** the Yateeka-owned modules listed above have implementations in the workspace. Jessie's Level 2 simulator (`fl_core.py`, `level2_fedavg.py`, below) already uses `common.py` and `data_utils.py`. Integrating the detector with that simulator and the end-to-end defended workflow remain to be done.

## Work completed by Jessie so far: Level 2 federated baseline

The following Jessie-owned modules are implemented and tested in the workspace:

- **FL building blocks — `fl_core.py`:** reusable for Level 3.
  - `partition_clients()` splits the 59,500-image training pool (validation images already removed by `data_utils.py`) into 10 disjoint client shards. `iid` (default) gives 5,950 shuffled images per client; `dirichlet` gives a label-skewed non-IID split controlled by `--alpha`. The partition is seeded and saved to `checkpoints/client_partition_<mode>_seed<seed>.json`, and an assertion checks shards are disjoint and cover the pool.
  - `local_train()` is the client side: load the global weights, train for the given number of local epochs (SGD, lr 0.01, momentum 0.9, batch size 50 as in the instructor loader, fresh optimizer each round), and return only the new `state_dict`. Shuffling is seeded per (round, client).
  - `fedavg()` is the server side: weighted average `w = Σ (n_k / n) · w_k` using client sample counts. Non-float buffers are copied from the first client.
- **Level 2 entry point — `level2_fedavg.py`:** each round sends the global model to all 10 clients, trains them locally, aggregates with FedAvg, and scores the new global model on the server validation set.
  - The model with the **lowest validation loss** is saved to `checkpoints/fedavg_mnist_best.pt` (the model for TA testing). Validation loss is used instead of accuracy because 500-image validation accuracy moves in 0.2% steps and picked a weaker round in an earlier run.
  - The latest round is saved to `checkpoints/fedavg_mnist_last.pt` with the round history, so `--resume` continues training.
  - The test set is used only after training ends (best and final model), never to choose rounds or settings.
  - Run history and test results are written to `logs/level2_history_<partition>.json`.

Level 2 does not use a separate course-provided Level 2 dataloader; it builds on `data_utils.py`, which wraps `dataloader4level1.py`. If the course's Level 2/3 dataloader prescribes a specific client split, switch `partition_clients()` to match it.

**Status:** Jessie's Level 2 modules are implemented and tested (see the run results under *Progress and next milestone*). Jessie's next step is Level 3: the guarded aggregation path, attack module, and detector integration in the checklist below.

## Jessie's workstream: implementation checklist

Jessie owns the federated-learning simulator and the parts that connect it to the detector. The detector API and model definitions below are the intended integration contract. Jessie has completed item 1 and the standard-FedAvg part of item 2 for Level 2; items 2 (guarded path) through 6 are the remaining Level 3 work.

1. **Ten-client simulator and local training** — *done for Level 2 (`fl_core.py`, `level2_fedavg.py`).*
   - ~~Build a ten-client simulation using the shared `SmallCNN` and data setup.~~
   - ~~Implement each client's local training loop and the ten-round run configuration.~~
   - ~~Use the same seed and model/checkpoint conventions from `common.py` so runs can be reproduced and checkpoints can be exchanged.~~

2. **FedAvg aggregation** — *standard FedAvg done (`fl_core.fedavg`); guarded path still to do.*
   - ~~Implement standard FedAvg for the baseline~~ and the guarded aggregation path for defended runs.
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

| File | Owner | Purpose / current state |
| --- | --- | --- |
| `common.py` | Yateeka | Shared seed, CNN, evaluation, and checkpoint helpers; implemented draft. |
| `data_utils.py` | Yateeka | Seeded held-out validation split, index saving, loader helper, and leakage check; implemented draft. |
| `level1_centralized.py` | Yateeka | Level 1 centralized MNIST training; implemented draft, depends on `data_utils.py`. |
| `fl_core.py` | Jessie | Client partitioning (IID / Dirichlet), client local training, and weighted FedAvg; tested in the Level 2 run, intended for reuse in Level 3. |
| `level2_fedavg.py` | Jessie | Level 2 ten-client FedAvg training with best/last checkpoints, `--resume`, and `--eval-only`; tested. |
| `fedshield_detector.py` | Yateeka | Layer-aware detection and defense interface; implemented draft for simulator integration. |
| `detection_metrics.py` | Yateeka | Detection metrics, threshold calibration, and sweep helpers; implemented draft. |
| `test_detector_synthetic.py` | Yateeka | Synthetic detector and split checks; passed in the reported local run. Does not replace integrated FL/label-flipping evaluation. |
| `dataloader4level1.py` | Course-provided | Course-provided data loader; preserve unchanged. |
| `checkpoints/`, `logs/` | Shared | Output directories for model checkpoints and run logs. Level 1 (Yateeka): `centralized_mnist.pt`. Level 2 (Jessie): `fedavg_mnist_best.pt` (submit/test this), `fedavg_mnist_last.pt` (resume), `client_partition_iid_seed42.json`. |

## Setup and first checks

Tested environment for Level 2 (Jessie's run): Python 3.12.3, torch 2.7.0, torchvision 0.22.0, NumPy 2.5.4 (CPU, Linux). Level 1 was run on Yateeka's Mac; record its versions here too. From this directory, install the project dependencies once:

```bash
pip install torch torchvision numpy
```

Run the synthetic checks from this folder:

```bash
python test_detector_synthetic.py
```

The first MNIST load creates `data/mnist/` automatically. Do not create it by hand. The current synthetic tests avoid downloading MNIST by substituting a small in-memory dataset.

To train the Level 1 centralized baseline (Yateeka):

```bash
python level1_centralized.py --epochs 5 --seed 42
```

To train and test the Level 2 federated baseline (Jessie):

```bash
python level2_fedavg.py                                # 10 clients, IID, 10 rounds, 1 local epoch
python level2_fedavg.py --eval-only                    # reload checkpoints/fedavg_mnist_best.pt and test
python level2_fedavg.py --rounds 15 --resume           # continue training from fedavg_mnist_last.pt
python level2_fedavg.py --partition dirichlet --alpha 0.5   # non-IID experiment
```

Other options: `--clients`, `--local-epochs`, `--lr`, `--momentum`, `--seed`, `--val-size`. Note that a non-IID run writes to the same checkpoint files as the IID run unless you pass `--best-ckpt` / `--last-ckpt`.

The Level 3 entry point, `malicious_client.py` attack module, and plotting/runbook commands are still to be supplied by the integration work; agree on their filenames and command-line interface before documenting final demo commands.
