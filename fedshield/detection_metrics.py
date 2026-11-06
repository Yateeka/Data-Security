"""
detection_metrics.py - detection metrics, benign FP checks, calibration, sweeps.

Owner: Yateeka (Testing workstream). Works on the `history` list produced by
`FedShieldDefense` (also saved as logs/fedshield_flags.jsonl). Rounds are
1-indexed; `onset` is the first round in which the attacker actually attacks.
"""
import csv
import itertools
import os
from typing import Callable, Dict, Iterable, List, Optional


def run_metrics(history: List[dict], malicious_id: Optional[int], onset: Optional[int],
                verdict: Optional[dict] = None) -> dict:
    """Metrics for ONE run. Pass malicious_id=None for benign (no-attacker) runs."""
    excluded = lambda rec: set(rec["flagged"]) | set(rec["invalid"])
    out = dict(rounds=len(history))

    # False positives: benign clients excluded (any round, incl. before onset)
    fp = [(rec["round"], i) for rec in history for i in excluded(rec) if i != malicious_id]
    out["false_positive_exclusions"] = len(fp)
    out["false_positive_rounds"] = len({r for r, _ in fp})

    if malicious_id is not None:
        active = [rec for rec in history if rec["round"] >= onset]
        caught = [rec["round"] for rec in active if malicious_id in excluded(rec)]
        out["attack_rounds"] = len(active)
        out["malicious_updates_rejected"] = len(caught)
        # delay in rounds after onset; 0 = caught in the very first attack round
        out["first_detection_delay"] = (caught[0] - onset) if caught else None
        out["detected_during_training"] = bool(caught)
        # Rounds where the attacker slipped through (benign-looking update or miss)
        out["missed_attack_rounds"] = len(active) - len(caught)
        # Flags BEFORE onset are false positives on the (still honest) attacker
        out["pre_onset_flags_on_attacker"] = sum(
            1 for rec in history if rec["round"] < onset and malicious_id in excluded(rec))
    if verdict is not None:
        pred = verdict["malicious_client"]
        out["verdict"] = pred
        out["verdict_correct"] = (pred == malicious_id)  # benign run: correct iff None
    return out


def calibrate_threshold(benign_histories: Iterable[List[dict]], margin: float = 1.25,
                        quantile: float = 1.0) -> float:
    """Pick a threshold from BENIGN development runs only (never from test outcomes).

    Takes the `quantile` (default max) of the per-round maximum client score over
    all benign rounds/seeds, times a safety margin. Run the detector with a very
    high threshold (e.g. 1e9) on benign runs so nothing is excluded while logging
    scores, then call this.
    """
    round_max = sorted(max(rec["scores"]) for h in benign_histories for rec in h)
    if not round_max:
        raise ValueError("no benign rounds supplied")
    idx = min(len(round_max) - 1, int(quantile * (len(round_max) - 1)))
    return round_max[idx] * margin


def run_sweep(run_fn: Callable[..., dict], attacks: List[str], onsets: List[int],
              identities: List[int], seeds: List[int],
              out_csv: str = "logs/sweep_results.csv") -> List[dict]:
    """Grid over attack x onset x malicious identity x seed.

    `run_fn(attack=..., onset=..., malicious_id=..., seed=...)` is supplied by the
    integration code (Jessie's simulator + attack module) and must return
        {"history": [...], "verdict": {...}, "final_test_acc": float}
    Failures are kept in the CSV on purpose (report failures, not just successes).
    """
    rows = []
    for attack, onset, mid, seed in itertools.product(attacks, onsets, identities, seeds):
        res = run_fn(attack=attack, onset=onset, malicious_id=mid, seed=seed)
        m = run_metrics(res["history"], mid, onset, res.get("verdict"))
        m.update(attack=attack, onset=onset, malicious_id=mid, seed=seed,
                 final_test_acc=res.get("final_test_acc"))
        rows.append(m)
    if rows:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize(rows: List[dict]) -> Dict[str, dict]:
    """Per-attack summary: detection rate, mean delay, FP exclusions, verdict acc."""
    out: Dict[str, dict] = {}
    for atk in sorted({r["attack"] for r in rows}):
        rs = [r for r in rows if r["attack"] == atk]
        delays = [r["first_detection_delay"] for r in rs
                  if r.get("first_detection_delay") is not None]
        out[atk] = dict(
            runs=len(rs),
            detected=sum(bool(r.get("detected_during_training")) for r in rs) / len(rs),
            mean_delay=(sum(delays) / len(delays)) if delays else None,
            fp_exclusions=sum(r["false_positive_exclusions"] for r in rs),
            verdict_acc=sum(bool(r.get("verdict_correct")) for r in rs) / len(rs),
        )
    return out