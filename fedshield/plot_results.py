"""Render the CPU experiment summary in results/level3_cpu_runs.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
with (ROOT / "results/level3_cpu_runs.json").open() as f:
    runs = json.load(f)["runs"]

names = [run["name"] for run in runs]
x = np.arange(len(runs))
width = 0.36
fig, (accuracy_ax, detection_ax) = plt.subplots(
    1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.5, 1]}
)

best = [100 * run["best_test_accuracy"] for run in runs]
last = [100 * run["last_test_accuracy"] for run in runs]
accuracy_ax.bar(x - width / 2, best, width, label="Best checkpoint", color="#4C78A8")
accuracy_ax.bar(x + width / 2, last, width, label="Last checkpoint", color="#72B7B2")
accuracy_ax.set_ylabel("Test accuracy (%)")
accuracy_ax.set_title("Final model accuracy")
accuracy_ax.set_xticks(x, names, rotation=18, ha="right")
accuracy_ax.set_ylim(97.8, 99.0)
accuracy_ax.grid(axis="y", alpha=0.25)
accuracy_ax.legend(frameon=False)

rejected = [run["malicious_updates_rejected"] or 0 for run in runs]
false_positives = [run["false_positive_exclusions"] for run in runs]
detection_ax.bar(x - width / 2, rejected, width, label="Malicious updates rejected", color="#54A24B")
detection_ax.bar(x + width / 2, false_positives, width, label="False-positive exclusions", color="#E45756")
detection_ax.set_ylabel("Updates / flags")
detection_ax.set_title("Detection outcomes")
detection_ax.set_xticks(x, names, rotation=18, ha="right")
detection_ax.grid(axis="y", alpha=0.25)
detection_ax.legend(frameon=False, fontsize=8)

fig.suptitle("FedShield Level 3 MNIST — seed 42, IID, CPU", fontsize=14)
fig.text(
    0.5, 0.01,
    "Round-10 attacks were detected and rejected, but each had one earlier false-positive flag on client 5.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
output = ROOT / "plots/fedshield_results.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
