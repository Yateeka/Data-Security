"""
run_sweep.py - run the final Level 3 evaluation set and collect the results.

Runs every scenario by calling level3_fedshield.main() in this process, one
after another (about 2 minutes each on a GPU). Each run writes its usual
logs/fedshield_<attack>_<partition>_seed<seed>_onset<onset><variant>.json(l);
sweep checkpoints go to checkpoints/sweep/ (git-ignored) so the main demo
checkpoints are never overwritten.

Re-running skips scenarios whose log already exists AND was produced by the
current code and threshold, so an interrupted sweep can simply be restarted.

Blocks:
  A  6 attacks x {defense on, off}, seed 42, onset 2        (main comparison)
  B  benign (no attacker) x {on, off}, seed 42               (baseline / false positives)
  C  sign_flip, label_flip at onset 5 and 10, defended       (late attacks)
  D  layer_only, label_flip with --no-validation / --no-clip (ablation)
  E  block A's defended runs + benign on seeds 7 and 123      (seed spread)

Usage:
    python run_sweep.py              # run everything not done yet, then summarize
    python run_sweep.py --summary    # only rebuild results/level3_runs.json
    python run_sweep.py --dry-run    # list the planned runs
"""
import argparse
import json
import os
import statistics
import time

from fedshield_detector import DetectorConfig

ATTACKS = ["noise", "sign_flip", "scale", "free_ride", "label_flip", "layer_only"]
MAIN_SEED, EXTRA_SEEDS = 42, [7, 123]
MALICIOUS_ID = 3
SUMMARY_PATH = "results/level3_runs.json"


def plan():
    runs = []

    def add(block, attack, seed=MAIN_SEED, onset=2, defense="on", ablation=None):
        runs.append(dict(block=block, attack=attack, seed=seed, onset=onset,
                         defense=defense, ablation=ablation))

    for a in ATTACKS:                                   # A
        for d in ("on", "off"):
            add("A", a, defense=d)
    for d in ("on", "off"):                             # B
        add("B", "none", defense=d)
    for a in ("sign_flip", "label_flip"):               # C
        for onset in (5, 10):
            add("C", a, onset=onset)
    for a in ("layer_only", "label_flip"):              # D
        for abl in ("no_validation", "no_clip"):
            add("D", a, ablation=abl)
    for s in EXTRA_SEEDS:                               # E
        for a in ATTACKS + ["none"]:
            add("E", a, seed=s)
    return runs


def run_name(r):
    tag = ""
    if r["defense"] == "off":
        tag = "_nodefense"
    elif r["ablation"] == "no_validation":
        tag = "_noval"
    elif r["ablation"] == "no_clip":
        tag = "_noclip"
    return f"fedshield_{r['attack']}_iid_seed{r['seed']}_onset{r['onset']}{tag}"


def argv_for(r):
    name = run_name(r)
    argv = ["--attack", r["attack"], "--seed", str(r["seed"]), "--onset", str(r["onset"]),
            "--malicious-id", str(MALICIOUS_ID), "--defense", r["defense"], "--device", "auto",
            "--best-ckpt", f"checkpoints/sweep/{name}_best.pt",
            "--last-ckpt", f"checkpoints/sweep/{name}_last.pt"]
    if r["ablation"] == "no_validation":
        argv.append("--no-validation")
    if r["ablation"] == "no_clip":
        argv.append("--no-clip")
    return argv


def load_current(r):
    """The run's summary if it exists and came from the current code/threshold."""
    path = f"logs/{run_name(r)}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    if "defense" not in cfg or cfg.get("threshold") != DetectorConfig().threshold:
        return None  # produced by older code or the old 8.0 threshold
    return d


def condition(r):
    if r["defense"] == "off":
        return "FedAvg (no defense)"
    return {None: "FedShield", "no_validation": "FedShield, no validation",
            "no_clip": "FedShield, no clipping"}[r["ablation"]]


def summarize(runs):
    rows = []
    for r in runs:
        d = load_current(r)
        if d is None:
            continue
        m = d.get("metrics") or {}
        rows.append(dict(
            block=r["block"], name=run_name(r), condition=condition(r), attack=r["attack"],
            seed=r["seed"], onset=r["onset"],
            final_test_acc=d["final_test_acc"], best_test_acc=d["best_test_acc"],
            val_acc_per_round=[h["val_acc"] for h in d["history"]],
            malicious_rejected=m.get("malicious_updates_rejected"),
            attack_rounds=m.get("attack_rounds"),
            first_detection_delay=m.get("first_detection_delay"),
            false_positive_exclusions=m.get("false_positive_exclusions"),
            verdict=(d.get("verdict") or {}).get("malicious_client"),
            verdict_correct=m.get("verdict_correct")))

    # Mean/stdev of final accuracy for defended runs over all seeds (blocks A, B, E).
    by_attack = {}
    for row in rows:
        if row["condition"] == "FedShield" and row["onset"] == 2:
            by_attack.setdefault(row["attack"], []).append(row["final_test_acc"])
    seed_summary = {a: dict(n=len(v), mean=round(statistics.mean(v), 4),
                            stdev=round(statistics.stdev(v), 4) if len(v) > 1 else 0.0)
                    for a, v in by_attack.items()}

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(dict(
            setup=dict(clients=10, rounds=10, local_epochs=1, partition="iid",
                       malicious_id=MALICIOUS_ID, threshold=DetectorConfig().threshold,
                       main_seed=MAIN_SEED, extra_seeds=EXTRA_SEEDS),
            runs=rows, fedshield_final_acc_over_seeds=seed_summary), f, indent=2)

    print(f"\n{len(rows)}/{len(runs)} runs collected -> {SUMMARY_PATH}")
    print(f"{'block':<5} {'condition':<26} {'attack':<11} {'seed':>4} {'onset':>5} "
          f"{'final':>7} {'rejected':>8} {'FP':>3} {'verdict':>7}")
    for row in rows:
        rej = (f"{row['malicious_rejected']}/{row['attack_rounds']}"
               if row["attack_rounds"] else "-")
        fp = "-" if row["false_positive_exclusions"] is None else row["false_positive_exclusions"]
        print(f"{row['block']:<5} {row['condition']:<26} {row['attack']:<11} {row['seed']:>4} "
              f"{row['onset']:>5} {row['final_test_acc']:>7.4f} {rej:>8} {fp:>3} "
              f"{str(row['verdict']):>7}")
    if seed_summary:
        print("\nFedShield final test accuracy over seeds (onset 2):")
        for a, s in seed_summary.items():
            print(f"  {a:<11} n={s['n']}  mean={s['mean']:.4f}  stdev={s['stdev']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Run the final Level 3 evaluation sweep")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary")
    ap.add_argument("--dry-run", action="store_true", help="list planned runs")
    args = ap.parse_args()
    runs = plan()

    if args.dry_run:
        for i, r in enumerate(runs, 1):
            status = "done" if load_current(r) else "todo"
            print(f"{i:2d} [{status}] {r['block']} {run_name(r)}")
        return
    if not args.summary:
        import level3_fedshield
        os.makedirs("checkpoints/sweep", exist_ok=True)
        todo = [r for r in runs if load_current(r) is None]
        print(f"{len(runs)} planned runs, {len(runs) - len(todo)} already done, "
              f"{len(todo)} to run")
        for i, r in enumerate(todo, 1):
            print(f"\n===== [{i}/{len(todo)}] {run_name(r)} =====", flush=True)
            t0 = time.time()
            try:
                level3_fedshield.main(argv_for(r))
            except (Exception, SystemExit) as exc:  # keep going; report at the end
                print(f"!!! run failed: {type(exc).__name__}: {exc}")
            print(f"===== done in {time.time() - t0:.0f}s =====", flush=True)
    summarize(runs)


if __name__ == "__main__":
    main()
