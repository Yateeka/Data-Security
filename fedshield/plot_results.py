"""
plot_results.py - draw the Level 3 figures from results/level3_runs.json.

Run after run_sweep.py (reads only the saved summary, trains nothing):
    python plot_results.py

Writes, for seed 42 with malicious client 3 (the main runs):
  plots/fedshield_results.png  final test accuracy per attack, FedAvg vs FedShield
  plots/fedshield_rounds.png   validation accuracy per round, one panel per attack
and the same two figures with a randomly drawn malicious client per attack
(block F, --malicious-id random; each attack has its own client):
  plots/fedshield_results_random.png, plots/fedshield_rounds_random.png
"""
import math
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


def _main_runs(runs, mal=3):
    """Seed 42, onset 2, no ablation: {(attack, condition): row}. `mal` is 3 (fixed
    client, blocks A/B) or "random" (one drawn client per attack, block F)."""
    def picked(r):
        if mal == "random":
            return r.get("malicious_choice") == "random"
        return r.get("malicious_choice", "fixed") == "fixed" and r.get("malicious_id", 3) == mal
    return {(r["attack"], r["condition"]): r for r in runs
            if r["seed"] == 42 and r["onset"] == 2 and picked(r)
            and not r.get("attack_settings") and r["condition"] in ("FedShield", "FedAvg (no defense)")}


def _suffix(mal):
    return "" if mal == 3 else f"_{mal}"


def _who(mal):
    return f"client {mal}" if mal == 3 else "a different random client per attack"


def _label(main, a, mal):
    """Attack name, plus its malicious client when each attack has its own."""
    if mal != "random" or a == "none":
        return LABELS[a]
    return f"{LABELS[a]} (client {main[(a, 'FedShield')]['malicious_id']})"


def plot_final_accuracy(plt, runs, mal=3):
    main = _main_runs(runs, mal)
    attacks = [a for a in ATTACK_ORDER
               if (a, "FedShield") in main and (a, "FedAvg (no defense)") in main]
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
    ax.set_yticks(range(len(attacks)), [_label(main, a, mal) for a in reversed(attacks)])
    ax.set_xlim(70, 101.5)
    ax.set_xlabel("Final (round-10) test accuracy (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", frameon=False)
    hidden = [f"{_label(main, a, mal)} {100 * main[(a, 'FedAvg (no defense)')]['final_test_acc']:.2f}%"
              for a in attacks
              if abs(main[(a, "FedShield")]["final_test_acc"]
                     - main[(a, "FedAvg (no defense)")]["final_test_acc"]) <= 0.005]
    if hidden:
        fig.text(0.01, -0.02, "FedAvg value not labeled where it is within 0.5 points of "
                 "FedShield (dot may be hidden): " + ", ".join(hidden), fontsize=8.5, color=INK_2)
    ax.set_title(f"Final test accuracy with one malicious client ({_who(mal)}, "
                 "attack from round 2, seed 42)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    out = PLOTS / f"fedshield_results{_suffix(mal)}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rounds(plt, runs, mal=3):
    main = _main_runs(runs, mal)
    attacks = [a for a in ATTACK_ORDER if a != "none"
               and (a, "FedShield") in main and (a, "FedAvg (no defense)") in main]
    wide = []
    ncols = 3
    nrows = (len(attacks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), sharex=True, sharey=False)
    for ax, a in zip(axes.flat, attacks):
        lowest = 100.0
        for cond, color, name in (("FedAvg (no defense)", FEDAVG, "FedAvg, no defense"),
                                  ("FedShield", FEDSHIELD, "FedShield")):
            acc = [100 * v for v in main[(a, cond)]["val_acc_per_round"]]
            lowest = min(lowest, min(acc))
            ax.plot(range(1, len(acc) + 1), acc, color=color, lw=2, marker="o", ms=4,
                    label=name, zorder=3)
        # Panels zoom in to 92-100%; one that drops lower (noise) gets a wider axis.
        if lowest < 92:
            wide.append(LABELS[a])
            ax.set_ylim(min(30, 10 * math.floor(lowest / 10)), 100)
        else:
            ax.set_ylim(92, 100)
        ax.axvline(2, color=GRID, lw=1, zorder=1)
        ax.set_title(_label(main, a, mal), loc="left", fontsize=10, color=INK)
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
    note = (f" Note: the {' and '.join(wide)} panel{'s have' if len(wide) > 1 else ' has'} "
            "a wider y-axis." if wide else "")
    title = ("Validation accuracy per round (500 held-out images, seed 42; attack starts "
             "in round 2, grey line)." + note)
    if mal != 3:
        title = title.replace("seed 42;", "seed 42, random malicious client per attack;")
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=11, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = PLOTS / f"fedshield_rounds{_suffix(mal)}.png"
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
    for mal in (3, "random"):
        m = _main_runs(data["runs"], mal)
        if not any((a, "FedShield") in m and (a, "FedAvg (no defense)") in m for a in ATTACK_ORDER):
            print(f"malicious client {mal}: no seed-42 FedShield + FedAvg pairs yet; skipped")
            continue
        print(plot_final_accuracy(plt, data["runs"], mal))
        print(plot_rounds(plt, data["runs"], mal))


if __name__ == "__main__":
    main()
