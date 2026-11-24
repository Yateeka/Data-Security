"""
calibrate_threshold.py - recompute the detector threshold from saved run logs.

The threshold in fedshield_detector.py (DetectorConfig.threshold) was chosen
from BENIGN development runs only, never from seed 42 or test accuracy:

  1. Benign runs (no attacker) on development seeds 101-105, with
     --threshold 1e9 so no client is excluded and every honest score is logged:
         python level3_fedshield.py --attack none --threshold 1e9 --seed <s>
  2. threshold = max honest score over all rounds x 1.25 safety margin
     (detection_metrics.calibrate_threshold).
  3. Sanity check: attacker scores on development seed 101 (8 attacks, onset 2,
     threshold 14) must sit well above the threshold and above the
     strong-evidence level (2 x threshold) used by the final verdict.

This script only reads the saved logs in results/calibration/ and writes
results/calibration/calibration_summary.json; it does not train anything.

Usage:
    python calibrate_threshold.py
"""
import glob
import json
import os

from detection_metrics import calibrate_threshold

CAL_DIR = "results/calibration"
MARGIN = 1.25
STRONG_FACTOR = 2.0


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    benign_files = sorted(glob.glob(f"{CAL_DIR}/benign_dev_runs/fedshield_none_*.json"))
    if not benign_files:
        raise SystemExit(f"no benign runs found in {CAL_DIR}/benign_dev_runs/")

    histories, per_seed = [], {}
    for f in benign_files:
        run = _load(f)
        hist = run["history"]
        assert run["config"]["attack"] == "none", f
        histories.append(hist)
        per_seed[run["config"]["seed"]] = [round(max(r["scores"]), 3) for r in hist]

    all_round_max = [m for seed in per_seed.values() for m in seed]
    raw = calibrate_threshold(histories, margin=MARGIN)
    threshold = round(raw, 1)

    print(f"Benign development runs: {len(benign_files)} seeds x 10 rounds "
          f"= {len(all_round_max)} rounds")
    print("Highest client score per round (honest clients only):")
    for seed, maxes in per_seed.items():
        print(f"  seed {seed}: " + " ".join(f"{m:5.1f}" for m in maxes))
    print(f"Max honest score: {max(all_round_max):.2f}  ->  x{MARGIN} = {raw:.2f}"
          f"  ->  threshold {threshold}")

    attacks = {}
    for f in sorted(glob.glob(f"{CAL_DIR}/attack_dev_runs/*.json")):
        run = _load(f)
        cfg, hist = run["config"], run["history"]
        mal = cfg["malicious_id"]
        attack_rounds = [r for r in hist if r["round"] >= cfg["onset"]]
        mal_scores = [r["scores"][mal] for r in attack_rounds]
        honest_max = max(s for r in hist for i, s in enumerate(r["scores"]) if i != mal)
        attacks[cfg["attack"]] = dict(
            attacker_score_min=round(min(mal_scores), 1),
            attacker_score_median=round(sorted(mal_scores)[len(mal_scores) // 2], 1),
            honest_score_max=round(honest_max, 2),
            rejected=f"{run['metrics']['malicious_updates_rejected']}/{len(attack_rounds)}",
            false_positives=run["metrics"]["false_positive_exclusions"],
            final_test_acc=run["final_test_acc"])

    if attacks:
        strong = STRONG_FACTOR * threshold
        print(f"\nAttack check (seed 101, onset 2, threshold {threshold}; "
              f"strong evidence >= {strong}):")
        print(f"  {'attack':<13} {'min score':>11} {'median':>11} {'honest max':>10}  rejected  FP")
        for name, a in attacks.items():
            print(f"  {name:<13} {a['attacker_score_min']:>11.1f} {a['attacker_score_median']:>11.1f}"
                  f" {a['honest_score_max']:>10.2f}  {a['rejected']:>8}  {a['false_positives']}")
        weakest = min(attacks.items(), key=lambda kv: kv[1]["attacker_score_min"])
        print(f"Weakest attacker score: {weakest[1]['attacker_score_min']} ({weakest[0]}), "
              f"{'above' if weakest[1]['attacker_score_min'] >= strong else 'BELOW'} "
              f"the strong-evidence level {strong}")

    summary = dict(
        method=("threshold = max honest client score in benign development runs "
                f"(seeds {sorted(per_seed)}, --threshold 1e9, nothing excluded) x {MARGIN}"),
        margin=MARGIN,
        benign_rounds=len(all_round_max),
        benign_round_max_scores=per_seed,
        max_honest_score=round(max(all_round_max), 4),
        calibrated_threshold_raw=round(raw, 4),
        threshold=threshold,
        strong_evidence_factor=STRONG_FACTOR,
        strong_evidence_level=STRONG_FACTOR * threshold,
        attack_check_seed101=attacks)
    out = os.path.join(CAL_DIR, "calibration_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
