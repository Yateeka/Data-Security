"""
plot_results.py - draw the Level 3 figures from results/level3_runs.json.

Run after run_sweep.py (reads only the saved summary, trains nothing):
    python plot_results.py

Writes:
  plots/fedshield_results.png  final test accuracy per attack, FedAvg vs FedShield (seed 42)
  plots/fedshield_rounds.png   validation accuracy per round, one panel per attack
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results/level3_runs.json"
PLOTS = ROOT / "plots"

# Colors: slots 1-2 of the default categorical palette; recessive ink and grid.
FEDSHIELD, FEDAVG = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

ATTACK_ORDER = ["noise", "layer_only", "sign_flip", "scale", "label_flip", "free_ride", "none"]
LABELS = {"none": "no attacker", "noise": "noise", "layer_only": "layer_only",
          "sign_flip": "sign_flip", "scale": "scale", "label_flip": "label_flip",
          "free_ride": "free_ride"}


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK_2, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _main_runs(runs):
    """Seed 42, onset 2, no ablation: {(attack, condition): row}."""
    return {(r["attack"], r["condition"]): r for r in runs
            if r["seed"] == 42 and r["onset"] == 2 and r.get("malicious_id", 3) == 3
            and not r.get("attack_settings") and r["condition"] in ("FedShield", "FedAvg (no defense)")}


def plot_final_accuracy(plt, runs, setup):
    main = _main_runs(runs)
    attacks = [a for a in ATTACK_ORDER if (a, "FedShield") in main]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, a in enumerate(attacks):
        y = len(attacks) - 1 - i
        shield = 100 * main[(a, "FedShield")]["final_test_acc"]
        avg = 100 * main[(a, "FedAvg (no defense)")]["final_test_acc"]
        ax.plot([avg, shield], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([avg], [y], s=70, color=FEDAVG, edgecolor=SURFACE, lw=2, zorder=3,
                   label="FedAvg, no defense" if i == 0 else None)
        ax.scatter([shield], [y], s=70, color=FEDSHIELD, edgecolor=SURFACE, lw=2, zorder=3,
                   label="FedShield" if i == 0 else None)
        if shield - avg > 0.5:   # label the undefended value only where the gap is visible
            ax.annotate(f"{avg:.2f}%", (avg, y), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=9, color=INK_2)
        ax.annotate(f"{shield:.2f}%", (shield, y), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(attacks)), [LABELS[a] for a in reversed(attacks)])
    ax.set_xlim(70, 101.5)
    ax.set_xlabel("Final (round-10) test accuracy (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", frameon=False)
    hidden = [f"{LABELS[a]} {100 * main[(a, 'FedAvg (no defense)')]['final_test_acc']:.2f}%"
              for a in attacks
              if abs(main[(a, "FedShield")]["final_test_acc"]
                     - main[(a, "FedAvg (no defense)")]["final_test_acc"]) <= 0.005]
    if hidden:
        fig.text(0.01, -0.02, "FedAvg dot hidden behind FedShield where they are within "
                 "0.5 points: " + ", ".join(hidden), fontsize=8.5, color=INK_2)
    ax.set_title("Final test accuracy with one malicious client (client "
                 f"{setup['malicious_id']}, attack from round 2, seed 42)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    out = PLOTS / "fedshield_results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rounds(plt, runs):
    main = _main_runs(runs)
    attacks = [a for a in ATTACK_ORDER if a != "none" and (a, "FedShield") in main]
    ncols = 3
    nrows = (len(attacks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), sharex=True, sharey=False)
    for ax, a in zip(axes.flat, attacks):
        for cond, color, name in (("FedAvg (no defense)", FEDAVG, "FedAvg, no defense"),
                                  ("FedShield", FEDSHIELD, "FedShield")):
            acc = [100 * v for v in main[(a, cond)]["val_acc_per_round"]]
            ax.plot(range(1, len(acc) + 1), acc, color=color, lw=2, marker="o", ms=4,
                    label=name, zorder=3)
        # noise wrecks plain FedAvg (round 2: ~34%); the other panels zoom in.
        ax.set_ylim((30, 100) if a == "noise" else (92, 100))
        ax.axvline(2, color=GRID, lw=1, zorder=1)
        ax.set_title(LABELS[a], loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(attacks):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
    for row in axes:
        row[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Validation accuracy per round (500 held-out images, seed 42; attack starts "
                 "in round 2, grey line). Note: the noise panel has a wider y-axis.",
                 x=0.01, y=0.995, ha="left", fontsize=11, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = PLOTS / "fedshield_rounds.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with SUMMARY.open() as f:
        data = json.load(f)
    _style(plt)
    PLOTS.mkdir(parents=True, exist_ok=True)
    print(plot_final_accuracy(plt, data["runs"], data["setup"]))
    print(plot_rounds(plt, data["runs"]))


if __name__ == "__main__":
    main()
