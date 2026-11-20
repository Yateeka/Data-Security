# FedShield MNIST demo runbook

Run commands from `fedshield/` after installing `../requirements.txt` and ensuring MNIST is available (the loader downloads it on first run).

## Main defended demo

```bash
python3 level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3 --device auto
```

Expected behavior on the recorded seed-42 CPU run: client 3 is flagged from round 2 onward, its updates are excluded, and the summary reports 9/9 malicious updates rejected with zero false-positive exclusions. The best and last models are saved to `checkpoints/fedshield_mnist_best.pt` and `checkpoints/fedshield_mnist_last.pt`.

## Edge and control cases

```bash
python3 level3_fedshield.py --attack label_flip --onset 10 --malicious-id 3 --device auto
python3 level3_fedshield.py --attack sign_flip --onset 10 --malicious-id 3 --device auto
python3 level3_fedshield.py --attack none --device auto
```

Each run writes a separate summary and detector log whose filename includes the attack, partition, seed, and onset. In the recorded runs, both round-10 attacks were caught in round 10 and rejected (1/1), but client 5 was also falsely flagged in round 2. The benign run also flagged client 5 in round 2. Because the final malicious-client verdict requires repeated evidence, the one-round attack runs correctly show no final verdict despite detecting client 3 in its attack round.

## Reload checkpoints

Pass the scenario's checkpoint paths to `--eval-only`; for example:

```bash
python3 level3_fedshield.py --eval-only \
  --best-ckpt checkpoints/fedshield_mnist_best.pt \
  --last-ckpt checkpoints/fedshield_mnist_last.pt
```

For custom `--best-ckpt` and `--last-ckpt` paths, repeat those options with the same values. The `--eval-only` path reports test accuracy for both saved models.

## Recorded evaluation and limitation

The current CPU results are summarized in `results/level3_cpu_runs.json` and plotted in `plots/fedshield_results.png`. These runs used seed 42, IID clients, 10 rounds, one local epoch, and PyTorch 2.15.0. This machine had no CUDA device; the target GPU/PyTorch 2.7.0 run remains to be completed in that environment. Run multiple seeds and attack types before making broad performance claims.
