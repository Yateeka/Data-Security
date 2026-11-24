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
  A  6 attacks (+ nan uploads) x {defense on, off}, seed 42, onset 2   (main comparison)
  B  benign (no attacker) x {on, off}, seed 42               (baseline / false positives)
  C  sign_flip, label_flip at onset 5 and 10, defended       (late attacks)
  D  layer_only, label_flip with --no-validation / --no-clip (ablation)
  E  block A's defended runs + benign on seeds 7 and 123      (seed spread)
  F  the 6 attacks on seeds 42, 7, 123 with --malicious-id random, defended
     (attacker drawn from the seed instead of client 3; logs end in _mal<k>)
  G  weak attacks chosen on development seed 101 (results/weak_attack_probe/), seed 42:
     scale x3 (layer scores miss it, validation catches it) under full / no
     validation / no clipping / no defense; layer_only x-1 (missed by both) and
     sign_flip x-1 (caught by layer scores) under full / no defense
  W0 whole-model scoring, benign, development seeds 101-105, --threshold 1e9
     (calibrates the whole-model threshold the same way as the main one)
  W  per-layer vs whole-model scoring, seed 42: the 6 attacks scored without
     validation evidence in both modes (geometry only), whole-model with
     validation for layer_only/noise, and a whole-model benign run

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
DEV_SEEDS = [101, 102, 103, 104, 105]
# Whole-model threshold: max benign score in block W0 x 1.25, like the main one
# (python run_sweep.py --calibrate-whole recomputes it from the W0 logs).
WHOLE_MODEL_THRESHOLD = 13.8   # benign max 11.04 (seed 104, round 1) x 1.25 = 13.80
MALICIOUS_ID = 3
SUMMARY_PATH = "results/level3_runs.json"


def plan():
    runs = []

    def add(block, attack, seed=MAIN_SEED, onset=2, defense="on", ablation=None,
            malicious=str(MALICIOUS_ID), settings=None, whole=False, threshold=None):
        runs.append(dict(block=block, attack=attack, seed=seed, onset=onset,
                         defense=defense, ablation=ablation, malicious=malicious,
                         settings=settings or {}, whole=whole,
                         threshold=threshold or DetectorConfig().threshold))

    for a in ATTACKS:                                   # A
        for d in ("on", "off"):
            add("A", a, defense=d)
    for d in ("on", "off"):                             # A: malformed (NaN) uploads
        add("A", "nan", defense=d)
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
    for s in [MAIN_SEED] + EXTRA_SEEDS:                 # F
        for a in ATTACKS:
            add("F", a, seed=s, malicious="random")
    scale3 = {"scale_factor": 3.0}                      # G
    add("G", "scale", settings=scale3)
    add("G", "scale", settings=scale3, ablation="no_validation")
    add("G", "scale", settings=scale3, ablation="no_clip")
    add("G", "scale", settings=scale3, defense="off")
    for a, st in (("layer_only", {"layer_factor": -1.0}), ("sign_flip", {"flip_factor": -1.0})):
        add("G", a, settings=st)
        add("G", a, settings=st, defense="off")
    for s in DEV_SEEDS:                                 # W0
        add("W0", "none", seed=s, whole=True, threshold=1e9)
    if WHOLE_MODEL_THRESHOLD is not None:               # W
        wt = WHOLE_MODEL_THRESHOLD
        for a in ATTACKS:
            if a not in ("layer_only", "label_flip"):   # those two already in block D
                add("W", a, ablation="no_validation")
            add("W", a, ablation="no_validation", whole=True, threshold=wt)
        for a in ("layer_only", "noise"):
            add("W", a, whole=True, threshold=wt)
        add("W", "none", whole=True, threshold=wt)
    return runs


def malicious_id(r):
    """The attacker's client index (resolves 'random' exactly as the runner does)."""
    from level3_fedshield import _pick_malicious_id
    return _pick_malicious_id(r["malicious"], r["seed"], 10)


def run_name(r):
    if r["defense"] == "off":
        tag = "_nodefense"
    else:  # same order as level3_fedshield._variant_tag
        tag = (("_whole" if r["whole"] else "")
               + {None: "", "no_validation": "_noval", "no_clip": "_noclip"}[r["ablation"]])
    mid = malicious_id(r)
    mal = f"_mal{mid}" if r["attack"] != "none" and mid != MALICIOUS_ID else ""
    from level3_fedshield import _settings_tag
    return (f"fedshield_{r['attack']}_iid_seed{r['seed']}_onset{r['onset']}{mal}"
            f"{_settings_tag(r['settings'])}{tag}")


def argv_for(r):
    name = run_name(r)
    argv = ["--attack", r["attack"], "--seed", str(r["seed"]), "--onset", str(r["onset"]),
            "--malicious-id", r["malicious"], "--defense", r["defense"], "--device", "auto",
            "--best-ckpt", f"checkpoints/sweep/{name}_best.pt",
            "--last-ckpt", f"checkpoints/sweep/{name}_last.pt"]
    for k, v in sorted(r["settings"].items()):
        argv += ["--attack-setting", f"{k}={v}"]
    if r["ablation"] == "no_validation":
        argv.append("--no-validation")
    if r["ablation"] == "no_clip":
        argv.append("--no-clip")
    if r["whole"]:
        argv.append("--whole-model")
    if r["threshold"] != DetectorConfig().threshold:
        argv += ["--threshold", str(r["threshold"])]
    return argv


def load_current(r):
    """The run's summary if it exists and came from the current code/threshold."""
    path = f"logs/{run_name(r)}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    if "defense" not in cfg or cfg.get("threshold") != r["threshold"]:
        return None  # produced by older code or the old 8.0 threshold
    return d


def condition(r):
    if r["defense"] == "off":
        return "FedAvg (no defense)"
    if r["whole"]:
        return {None: "Whole-model", "no_validation": "Whole-model, no validation"}[r["ablation"]]
    return {None: "FedShield", "no_validation": "FedShield, no validation",
            "no_clip": "FedShield, no clipping"}[r["ablation"]]


def calibrate_whole(runs):
    """Whole-model threshold from the W0 benign development logs (max x 1.25)."""
    from detection_metrics import calibrate_threshold
    hist = [load_current(r)["history"] for r in runs if r["block"] == "W0" and load_current(r)]
    if len(hist) < len(DEV_SEEDS):
        raise SystemExit(f"only {len(hist)}/{len(DEV_SEEDS)} W0 runs found; run --block W0 first")
    maxes = [round(max(h["scores"]), 2) for run in hist for h in run]
    raw = calibrate_threshold(hist, margin=1.25)
    print(f"whole-model benign max score {max(maxes)} over {len(maxes)} rounds "
          f"-> x1.25 = {raw:.2f} -> threshold {round(raw, 1)}")
    return round(raw, 1)


def summarize(runs):
    rows = []
    for r in runs:
        d = load_current(r)
        if d is None:
            continue
        m = d.get("metrics") or {}
        rows.append(dict(
            block=r["block"], name=run_name(r), condition=condition(r), attack=r["attack"],
            seed=r["seed"], onset=r["onset"], malicious_id=d["config"]["malicious_id"],
            malicious_choice="random" if r["malicious"] == "random" else "fixed",
            scoring="whole-model" if r["whole"] else "per-layer", threshold=r["threshold"],
            attack_settings=r["settings"],
            final_test_acc=d["final_test_acc"], best_test_acc=d["best_test_acc"],
            val_acc_per_round=[h["val_acc"] for h in d["history"]],
            malicious_rejected=m.get("malicious_updates_rejected"),
            attack_rounds=m.get("attack_rounds"),
            first_detection_delay=m.get("first_detection_delay"),
            false_positive_exclusions=m.get("false_positive_exclusions"),
            verdict=(d.get("verdict") or {}).get("malicious_client"),
            verdict_correct=m.get("verdict_correct")))

    # Mean/stdev of defended final accuracy over seeds, onset 2: client 3 (A, B, E)
    # and the random attacker (F) kept apart.
    def over_seeds(choice):
        by_attack = {}
        for row in rows:
            if (row["condition"] == "FedShield" and row["onset"] == 2
                    and row["malicious_choice"] == choice and not row["attack_settings"]):
                by_attack.setdefault(row["attack"], []).append(row["final_test_acc"])
        return {a: dict(n=len(v), mean=round(statistics.mean(v), 4),
                        stdev=round(statistics.stdev(v), 4) if len(v) > 1 else 0.0)
                for a, v in by_attack.items()}
    seed_summary, random_summary = over_seeds("fixed"), over_seeds("random")

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(dict(
            setup=dict(clients=10, rounds=10, local_epochs=1, partition="iid",
                       malicious_id=MALICIOUS_ID, threshold=DetectorConfig().threshold,
                       main_seed=MAIN_SEED, extra_seeds=EXTRA_SEEDS),
            runs=rows, fedshield_final_acc_over_seeds=seed_summary,
            fedshield_final_acc_over_seeds_random_attacker=random_summary), f, indent=2)

    print(f"\n{len(rows)}/{len(runs)} runs collected -> {SUMMARY_PATH}")
    print(f"{'block':<5} {'condition':<26} {'attack':<11} {'seed':>4} {'onset':>5} {'mal':>3} "
          f"{'final':>7} {'rejected':>8} {'FP':>3} {'verdict':>7}")
    for row in rows:
        rej = (f"{row['malicious_rejected']}/{row['attack_rounds']}"
               if row["attack_rounds"] else "-")
        fp = "-" if row["false_positive_exclusions"] is None else row["false_positive_exclusions"]
        print(f"{row['block']:<5} {row['condition']:<26} {row['attack'] + ('*' if row['attack_settings'] else ''):<11} {row['seed']:>4} "
              f"{row['onset']:>5} {row['malicious_id'] if row['attack'] != 'none' else '-':>3} "
              f"{row['final_test_acc']:>7.4f} {rej:>8} {fp:>3} "
              f"{str(row['verdict']):>7}")
    print("(* = weakened attack, see attack_settings)")
    for title, summ in (("client 3", seed_summary), ("random attacker", random_summary)):
        if summ:
            print(f"\nFedShield final test accuracy over seeds (onset 2, {title}):")
            for a, s in summ.items():
                print(f"  {a:<11} n={s['n']}  mean={s['mean']:.4f}  stdev={s['stdev']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Run the final Level 3 evaluation sweep")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary")
    ap.add_argument("--dry-run", action="store_true", help="list planned runs")
    ap.add_argument("--calibrate-whole", action="store_true",
                    help="print the whole-model threshold from the W0 logs")
    ap.add_argument("--block", action="append",
                    help="only run these blocks (e.g. --block G); the summary covers all")
    args = ap.parse_args()
    runs = plan()
    selected = [r for r in runs if not args.block or r["block"] in args.block]

    if args.calibrate_whole:
        calibrate_whole(runs)
        return
    if args.dry_run:
        for i, r in enumerate(selected, 1):
            status = "done" if load_current(r) else "todo"
            print(f"{i:2d} [{status}] {r['block']} {run_name(r)}")
        return
    if not args.summary:
        import level3_fedshield
        os.makedirs("checkpoints/sweep", exist_ok=True)
        todo = [r for r in selected if load_current(r) is None]
        print(f"{len(selected)} planned runs, {len(selected) - len(todo)} already done, "
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
