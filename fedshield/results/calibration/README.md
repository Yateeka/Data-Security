# Detector threshold calibration

This folder records how the detector threshold (14.0) and the final-verdict rule in `fedshield_detector.py` were chosen. The threshold was set from **benign development runs only**: seeds 101–105, separate from the evaluation seed 42, and never from test accuracy.

To recompute every number below from the saved logs (no training needed), run from `fedshield/`:

```bash
python calibrate_threshold.py      # also writes calibration_summary.json
```

## 1. Benign development runs → threshold

Command (one per seed, no attacker, nothing excluded so every honest score is logged):

```bash
python level3_fedshield.py --attack none --threshold 1e9 --seed <101..105>
```

Highest honest-client anomaly score in each of the 10 rounds:

| Seed | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 2.5 | 2.1 | 2.7 | 3.2 | 3.2 | 5.0 | 2.6 | 3.5 | 3.7 | 2.7 |
| 102 | 2.4 | 2.6 | 5.3 | 4.2 | 3.8 | 2.9 | 2.4 | 3.6 | 3.4 | 4.5 |
| 103 | 2.5 | 2.2 | 1.6 | 2.2 | 2.5 | 3.0 | 3.6 | 2.7 | 3.7 | 4.3 |
| 104 | **11.2** | 4.3 | 6.0 | 2.1 | 3.6 | 5.1 | 2.5 | 5.4 | 4.0 | 3.9 |
| 105 | 2.3 | 4.1 | 3.3 | 1.7 | 4.7 | 4.4 | 2.4 | 3.6 | 3.6 | 2.8 |

- Highest honest score in 50 rounds: **11.23** (seed 104, round 1). Most rounds stay between 2 and 6.
- Threshold = 11.23 × 1.25 safety margin (`detection_metrics.calibrate_threshold`) = 14.04 → **14.0**.
- The earlier placeholder threshold of 8.0 would have flagged an honest client in 1 of these 50 rounds (seed 104, round 1), and on seed 42 it flagged client 5 in round 2 (score 10.2).

Logs: `benign_dev_runs/` (`.json` run summary and `.jsonl` per-round detector log for each seed).

## 2. Attack check → strong-evidence verdict rule

Command (development seed 101, malicious client 3, attack from round 2, new threshold):

```bash
python level3_fedshield.py --attack <mode> --onset 2 --threshold 14 --seed 101
```

| Attack | Attacker score (min) | Attacker score (median) | Highest honest score | Rejected | False positives |
| --- | --- | --- | --- | --- | --- |
| `layer_only` | 45.6 | 95.6 | 4.35 | 9/9 | 0 |
| `zero_weights` | 98.0 | 208.8 | 4.35 | 9/9 | 0 |
| `sign_flip` | 118.8 | 252.9 | 4.35 | 9/9 | 0 |
| `label_flip` | 324.8 | 787.9 | 4.68 | 9/9 | 0 |
| `scale` | 957.5 | 2147.8 | 4.75 | 9/9 | 0 |
| `free_ride` | 1000.0 | 1000.0 | 5.11 | 9/9 | 0 |
| `noise` | 748697.8 | 1700255.2 | 4.35 | 9/9 | 0 |
| `grad_ascent` | 1812643.9 | 3125817.0 | 5.11 | 9/9 | 0 |

The weakest attacker score (45.6) is far above the highest honest score in these runs (5.11) and above the calibration maximum (11.23). This gap is the basis of the verdict rule: the top-suspicion client is named malicious if it was flagged in at least 2 rounds, **or** flagged once with strong evidence (score ≥ 2 × threshold = 28, or a malformed upload). An attack that appears only in round 10 is therefore still identified, while an honest client that only just crosses the threshold is not.

Logs: `attack_dev_runs/`. These runs were made just before the strong-evidence rule was added; every attacker was flagged in all 9 attack rounds, so the verdicts are the same under the new rule.

## 3. Checks on seed 42 with the final settings

| Run | Clients flagged | Final verdict | Final test acc |
| --- | --- | --- | --- |
| Benign (`--attack none`) | none | none | 98.76% |
| `sign_flip`, onset 2 | 3 (rounds 2–10) | client 3 | 98.71% |
| `sign_flip`, onset 10 | 3 (round 10) | client 3 | 98.75% |
| `label_flip`, onset 10 | 3 (round 10) | client 3 | 98.75% |

With the old settings (threshold 8.0, verdict only after 2 flagged rounds), the benign run flagged client 5 in round 2, and both onset-10 attacks got no final verdict.

Logs: `seed42_checks/`.

## Limitations

- Calibrated on 5 benign runs with an IID split. A non-IID split (`--partition dirichlet`) makes honest updates less alike and may need recalibration.
- An attack scoring between 14 and 28 that appears only in the last round would be excluded but not named in the final verdict.

All runs: 10 clients, 10 rounds, 1 local epoch, IID, GTX 1080 Ti GPU, PyTorch 2.7.0, Nov 24, 2026.
