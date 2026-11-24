# FedShield project handoff

This folder contains the Level 1 MNIST baseline, the Level 2 FedAvg baseline, and a ten-client Level 3 federated learning experiment with malicious-client detection and defense. The work is split between Yateeka and Jessie as described below. Level 1, Level 2, synthetic detector checks, and multiple Level 3 runs have completed. Level 3 with the integrated `malicious_client.py` has been verified on a CUDA GPU with PyTorch 2.7.0 (see `DEMO_RUNBOOK.md`).

## Progress and next milestone

Yateeka has run the following checks locally on a Mac:

- `python3 test_detector_synthetic.py` — all listed checks passed: five synthetic detector scenarios, late-onset exclusion, benign false-positive check, invalid-update rejection, validation evidence, threshold calibration, and reproducible split/no-leakage check. These are component-level synthetic checks, not evidence of performance on a federated training run.
- `python3 level1_centralized.py --epochs 5 --seed 42` — first run (Mac, CPU): 99.04% test accuracy from the last epoch. Rerun on Nov 20 (Linux, GTX 1080 Ti, torch 2.7.0) after switching to best-epoch selection: the lowest validation loss was at epoch 4, and that checkpoint scores **98.97% test accuracy** (last epoch: 98.98%). It is saved to `checkpoints/centralized_mnist_best.pt`, and `--eval-only` reproduces 98.97%.

Jessie has run Level 2 on Linux (CPU only, Python 3.12.3, torch 2.7.0, torchvision 0.22.0, NumPy 2.5.4):

- `python3 level2_fedavg.py` (10 clients, IID split, 10 rounds, 1 local epoch, seed 42) — about 5 minutes on CPU. Validation loss fell from 0.206 after round 1 to 0.034 by round 9 and stayed there in round 10. **Final global model: 98.72% test accuracy.** Best checkpoint by validation loss (round 9): **98.62% test accuracy**, saved to `checkpoints/fedavg_mnist_best.pt`.
- `python3 level2_fedavg.py --eval-only` — reloaded the best checkpoint and reproduced 98.62%.
- `--resume` was checked by stopping after round 2 and resuming to round 3; round 3 matched the uninterrupted run exactly.

The Level 3 defended runner is implemented and integrated in `level3_fedshield.py`. It connects local client training, detector validation-loss scoring, guarded aggregation, clipping, metrics, and best/last checkpoints. The malicious client's upload comes from `malicious_client.generate_malicious_update()` (the course-required interface), which the runner calls from `--onset` on. Results from the sign-flip, label-flip, late-onset, and benign CPU runs are summarized below and in `DEMO_RUNBOOK.md`.

The first integrated evaluation (threshold placeholder 8.0) found a limitation: the seeded benign run flagged client 5 in round 2, and the round-10 attacks were caught but got no final verdict because the verdict required two flagged rounds. Both were fixed on Nov 24: the threshold is now 14.0, calibrated on benign development seeds 101–105, and a single flag with strong evidence (score ≥ 2 × threshold, or a malformed upload) also produces a verdict. With these settings, the seed-42 benign run has no flags, and round-10 sign-flip and label-flip attacks are identified (see `DEMO_RUNBOOK.md` §4). The default sign-flip run (onset round 2) had no false-positive flags and rejected all nine malicious updates. These results should be reported as observed; they do not establish general performance across seeds or attack identities.

See [`FIRST_RUN_REPORT.md`](FIRST_RUN_REPORT.md) for the complete first-run findings, [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) for reproducible commands, [`results/level3_cpu_runs.json`](results/level3_cpu_runs.json) for the recorded results, and [`plots/fedshield_results.png`](plots/fedshield_results.png) for the accuracy/detection summary. Regenerate the plot with `python3 plot_results.py` (requires Matplotlib).

For evaluation hygiene, use the held-out validation set for training-time checks and detector scoring, and reserve test accuracy for the final evaluation. `level1_centralized.py` now evaluates the test set only after training and selects its saved model by validation loss, like Level 2.

## Work completed by Yateeka so far

The following Yateeka-owned modules currently have implementation in the workspace:

- **Foundations — `common.py`:** reproducible Python/NumPy/PyTorch seeding, device selection, the shared `SmallCNN`, model evaluation, and checkpoint save/load helpers.
- **Data preparation — `data_utils.py`:** seeded held-out validation split, split-index saving, loader construction using the instructor template, and a train/validation leakage check. The validation set is intended for server-side scoring; test data is kept for final accuracy.
- **Level 1 — `level1_centralized.py`:** centralized MNIST training/evaluation, validation tracking, best-epoch checkpoint (lowest validation loss) plus a last-epoch checkpoint for `--resume`, `--eval-only`, and `--device`. It uses `data_utils.py` for the reproducible data split.
- **Security and defense — `fedshield_detector.py`:** parameter-delta validation; per-layer magnitude and cosine anomaly scores; optional validation-loss evidence; single-client round flagging; immediate exclusion of flagged/invalid updates; suspicion history, per-round `FLAGGED client k in round r` output, a summary of all clients flagged at least once, and the final verdict (repeated evidence, or one flag with strong evidence). Threshold 14.0 calibrated on benign development runs.
- **Testing and reporting — `detection_metrics.py`:** per-run detection/false-positive metrics, benign-run threshold calibration, attack/onset/identity/seed sweep support, and per-attack summaries.
- **Synthetic tests — `test_detector_synthetic.py`:** synthetic attack, benign false-positive, malformed-update, validation-evidence, calibration, and data-split checks; all pass. Integrated label-flip and simulator runs are recorded in the Level 3 results.
- **Instructor template — `dataloader4level1.py`:** present in the folder. Keep the course-provided file unchanged.

**Status:** the defended training entry point is `level3_fedshield.py`. CPU integration runs (PyTorch 2.15.0, earlier inline attack code) are complete for benign, sign-flip onset 2, sign-flip onset 10, and label-flip onset 10. GPU runs with PyTorch 2.7.0 and the integrated `malicious_client.py` are recorded in `DEMO_RUNBOOK.md`. The multi-seed/attack sweep remains outstanding.

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

**Status:** Jessie's Level 2 modules are implemented and tested. Level 3 reuses them through `level3_fedshield.py`.

## Jessie's workstream: implementation checklist

Jessie owns the federated-learning simulator and the parts that connect it to the detector. The detector API and model definitions below are the intended integration contract. Items 1 through 5 are implemented and have completed CPU integration runs; item 6 still needs broader multi-seed comparisons and target-GPU verification.

1. **Ten-client simulator and local training** — *done for Level 2 (`fl_core.py`, `level2_fedavg.py`).*
   - ~~Build a ten-client simulation using the shared `SmallCNN` and data setup.~~
   - ~~Implement each client's local training loop and the ten-round run configuration.~~
   - ~~Use the same seed and model/checkpoint conventions from `common.py` so runs can be reproduced and checkpoints can be exchanged.~~

2. **FedAvg aggregation** — *standard FedAvg and defended guarded aggregation implemented.*
   - ~~Implement standard FedAvg for the baseline~~ and the guarded aggregation path for defended runs.
   - For defended runs, take only client updates in `RoundResult.accepted_ids` from `FedShieldDefense.inspect_round`.
   - Apply accepted-update norm clipping, then aggregate. If no client updates are accepted, retain the previous global model rather than averaging an empty set.

3. **Server-side validation-loss evidence** — *implemented in `level3_fedshield.py`.*
   - For each uploaded client state, measure validation loss after applying that update and subtract the current global model's validation loss.
   - Pass those per-client deltas as `val_deltas` to `inspect_round(round_idx, global_state, client_states, val_deltas)`.
   - Use the held-out validation set from `data_utils.py`; do not use the test set to score clients or calibrate the detector.

4. **Malicious-client attack module** — *implemented in `malicious_client.py` and called by `level3_fedshield.py`.*
   - `generate_malicious_update(client_model_state_dict, local_dataset, device)` uses the course signature and contains the `# === MALICIOUS CLIENT LOGIC HERE ===` marker. That file is the only place the TA needs to change; `level3_fedshield.py` simply calls the function.
   - Modes: the four proposal attacks (`noise`, `scale`/`sign_flip`, `free_ride`, `label_flip`) plus robustness modes `zero_weights`, `random_weights`, `nan`, `grad_ascent`, and `layer_only`. The runner sets the mode, seed, and optimizer settings through `malicious_client.configure()`; strengths live in `ATTACK_CONFIG`.
   - The runner controls onset and malicious client ID. Uploads are moved to CPU and cast to the model's dtype; malformed uploads or exceptions count as invalid updates. A TA replacement that defines only `generate_malicious_update` runs as the `custom` attack.
   - Attacks are seeded per call, and `--resume` continues the same seed sequence.

5. **Round-by-round detector integration** — *implemented in `level3_fedshield.py`.*
   - Keep one `FedShieldDefense` instance per run and call `inspect_round` once for each round, supplying the global state, one uploaded state (or `None`) per client, and validation deltas when enabled.
   - Exclude both flagged and invalid updates; the result exposes `accepted_ids`, `flagged_ids`, `invalid_ids`, `scores`, and `reasons`.
   - After training, call `final_verdict()` to get the malicious-client decision (flagged in ≥ 2 rounds, or once with score ≥ 2 × threshold / a malformed upload).
   - Note: the detector supports a single-attacker assumption and flags at most one scored client per round by default.

6. **Evaluation, tests, and demo** — *partially complete.*
   - Centralized, FedAvg, benign, and defended CPU outcomes are recorded; the Level 3 plot and demo runbook are in the repo.
   - Synthetic tests and checkpoint reload checks pass. Run attack/benign sweeps across more seeds and review detector failure cases before drawing broad conclusions.
   - GPU/PyTorch 2.7.0 runs with the integrated attack module, the course-template substitution test, and a resume check are recorded in `DEMO_RUNBOOK.md`. The joint demo rehearsal remains to be done.

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
| `fedshield_detector.py` | Yateeka | Layer-aware detection, per-round flag output, all-flag summary, and final verdict; threshold 14.0 calibrated on benign runs; synthetic checks passed. |
| `detection_metrics.py` | Yateeka | Detection metrics, threshold calibration, and sweep helpers; integrated runs recorded. |
| `test_detector_synthetic.py` | Yateeka | Synthetic detector and split checks; passed. Integrated checks are recorded separately. |
| `level3_fedshield.py` | Shared | Level 3 defended MNIST runner: calls `malicious_client.py`, validation evidence, detector, clipping, guarded aggregation, and best/last checkpoints; CPU and GPU runs completed. |
| `malicious_client.py` | Jessie | Course-signature `generate_malicious_update` with the demo-day marker; four proposal attacks plus five robustness modes; TA-replaceable. |
| `DEMO_RUNBOOK.md` | Shared | Reproducible demo commands and observed outcomes. |
| `FIRST_RUN_REPORT.md` | Shared | Report page for the first integrated Level 3 run set. |
| `plot_results.py`, `results/level3_cpu_runs.json` | Shared | Reproducible plot source and run metrics. |
| `calibrate_threshold.py`, `results/calibration/` | Shared | Threshold calibration: benign development logs (seeds 101–105), attacker-score check (seed 101), seed-42 checks, and a script that recomputes the threshold from the logs. |
| `plots/fedshield_results.png` | Shared | Summary plot generated from saved run results. |
| `dataloader4level1.py` | Course-provided | Course-provided data loader; preserve unchanged. |
| `checkpoints/`, `logs/` | Shared | Output directories for model checkpoints and run logs. Level 1: `centralized_mnist_best.pt` (best epoch), `centralized_mnist_last.pt` (resume); Level 2: `fedavg_mnist_best.pt`, `fedavg_mnist_last.pt`; Level 3: `fedshield_mnist_best.pt`, `fedshield_mnist_last.pt`, plus named benign/late-onset scenario checkpoints and JSON summaries. |

## Setup and first checks

Tested environment for Level 2 (Jessie's run): Python 3.12.3, torch 2.7.0, torchvision 0.22.0, NumPy 2.5.4 (CPU, Linux). Level 1 was run on Yateeka's Mac; record its versions here too. From this directory, install the pinned project dependencies once:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins torch 2.7.0, torchvision 0.22.0, NumPy 2.5.4, and Matplotlib 3.10.8. Python 3.12.3 is the tested interpreter version (pip cannot pin the Python version itself). On Linux, the default PyPI `torch==2.7.0` wheel includes CUDA 12.6 support and also runs on CPU-only machines.

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

Device: `--device auto` (default) uses a CUDA GPU when one is available and falls back to CPU; `--device cpu` forces CPU; `--device cuda` requires a GPU and exits with an error if none is found. Other options: `--clients`, `--local-epochs`, `--lr`, `--momentum`, `--seed`, `--val-size`. Note that a non-IID run writes to the same checkpoint files as the IID run unless you pass `--best-ckpt` / `--last-ckpt`.

To run the defended experiment (from `fedshield/`):

```bash
python3 level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3 --device auto
python3 level3_fedshield.py --attack label_flip --onset 10 --malicious-id 3 --device cuda
python3 level3_fedshield.py --attack none --device auto
python3 level3_fedshield.py --eval-only
```

Supported attacks are `none` plus every mode in `malicious_client.ATTACK_MODES`: `noise`, `scale`, `sign_flip`, `free_ride`, `label_flip`, `zero_weights`, `random_weights`, `nan`, `grad_ascent`, and `layer_only`. The malicious client trains honestly before `--onset`, then uploads `malicious_client.generate_malicious_update()`. Client IDs are zero-indexed and rounds are one-indexed; attack onset must be round 2 or later. `--device auto` selects CUDA when available. Random tensor generation is compatible with the project’s PyTorch 2.7.0 pin. The best defended model is selected by server validation loss and saved to `checkpoints/fedshield_mnist_best.pt`; the latest round, training history, and detector state are saved to `checkpoints/fedshield_mnist_last.pt` for `--resume`; this round-10 model is the one the course evaluates after live training. Test accuracy is reported after training. Logs and a JSON run summary are written under `logs/`.
