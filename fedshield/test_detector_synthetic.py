"""
test_detector_synthetic.py - run with:  python test_detector_synthetic.py

Checks the detector WITHOUT needing the FL simulator or MNIST:
benign clients share a common update direction + noise; attacks are injected
into one client. Also checks the data split (with a fake dataset, no download).
"""
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import SmallCNN, set_seed
from detection_metrics import calibrate_threshold, run_metrics
from fedshield_detector import DetectorConfig, FedShieldDefense

N, MAL = 10, 3


def make_round(global_state, attack=None, scale=0.05, gen=None):
    """Return list of N client state_dicts. Client MAL is attacked if `attack`."""
    shared = {k: torch.randn_like(v, generator=gen) for k, v in global_state.items()}
    states = []
    for i in range(N):
        delta = {k: scale * (shared[k] + 0.5 * torch.randn_like(v, generator=gen))
                 for k, v in global_state.items()}
        if i == MAL and attack:
            if attack == "noise":
                delta = {k: scale * 1.5 * torch.randn_like(v, generator=gen) for k, v in delta.items()}
            elif attack == "sign_flip":
                delta = {k: -5 * v for k, v in delta.items()}
            elif attack == "scale":
                delta = {k: 10 * v for k, v in delta.items()}
            elif attack == "free_ride":
                delta = {k: torch.zeros_like(v) for k, v in delta.items()}
            elif attack == "layer_only":     # only fc1 is corrupted (layer-aware case)
                delta["fc1.weight"] = -3 * delta["fc1.weight"]
        states.append({k: global_state[k] + delta[k] for k in global_state})
    return states


def run_scenario(attack, onset, rounds=10, cfg=None, seed=0):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    gs = SmallCNN().state_dict()
    d = FedShieldDefense(N, cfg or DetectorConfig(use_validation=False), log_path=None)
    for r in range(1, rounds + 1):
        states = make_round(gs, attack if (attack and r >= onset) else None, gen=gen)
        d.inspect_round(r, gs, states)
    return d


def test_attacks_detected():
    for atk in ["noise", "sign_flip", "scale", "free_ride", "layer_only"]:
        d = run_scenario(atk, onset=2)
        m = run_metrics(d.history, MAL, 2, d.final_verdict())
        print(f"{atk:10s} delay={m['first_detection_delay']} rejected="
              f"{m['malicious_updates_rejected']}/{m['attack_rounds']} "
              f"FP={m['false_positive_exclusions']} verdict={m['verdict']}")
        assert m["detected_during_training"], atk
        assert m["verdict_correct"], atk
        assert m["false_positive_exclusions"] == 0, atk


def test_late_onset_immediate_exclusion():
    d = run_scenario("sign_flip", onset=9)
    assert MAL in d.history[8]["flagged"], "late attack must be excluded immediately"
    assert MAL not in d.history[8]["accepted"]


def test_benign_no_false_positives():
    fp = 0
    for seed in range(5):
        d = run_scenario(None, onset=99, seed=seed)
        fp += run_metrics(d.history, None, None, d.final_verdict())["false_positive_exclusions"]
        assert d.final_verdict()["malicious_client"] is None
    assert fp == 0, f"{fp} benign false positives"
    print("benign runs: 0 false positives, no verdict (as intended)")


def test_invalid_updates_rejected_before_scoring():
    gs = SmallCNN().state_dict()
    states = make_round(gs, None)
    states[1]["fc2.bias"] = states[1]["fc2.bias"] * float("nan")        # NaN
    states[2]["conv1.weight"] = torch.zeros(3, 3)                       # wrong shape
    d = FedShieldDefense(N, DetectorConfig(use_validation=False), log_path=None)
    res = d.inspect_round(1, gs, states)
    assert set(res.invalid_ids) == {1, 2} and not ({1, 2} & set(res.accepted_ids))


def test_validation_evidence_adds_score():
    set_seed(0)
    gs = SmallCNN().state_dict()
    states = make_round(gs, None)
    cfg = DetectorConfig(use_validation=True, threshold=5.0)
    base = [-0.05] * N
    base[MAL] = 0.8      # applying this update raises val loss a lot
    d = FedShieldDefense(N, cfg, log_path=None)
    res = d.inspect_round(1, gs, states, val_deltas=base)
    assert res.flagged_ids == [MAL], res.flagged_ids


def test_calibration():
    hs = [run_scenario(None, 99, seed=s).history for s in range(3)]
    thr = calibrate_threshold(hs)
    print(f"calibrated threshold from benign runs: {thr:.2f}")
    assert thr > 0


def test_data_split_no_leakage():
    import data_utils
    from dataloader4level1 import dataloader
    x = torch.randn(2000, 1, 28, 28)
    y = torch.randint(0, 10, (2000,))
    xt = torch.randn(300, 1, 28, 28)
    yt = torch.randint(0, 10, (300,))
    tr, te = TensorDataset(x, y), TensorDataset(xt, yt)
    data_utils.instructor_load_data = lambda: dataloader(tr, te)   # avoid download
    pool, val, test = data_utils.make_splits(seed=1, val_size=100)
    assert len(pool) == 1900 and len(val) == 100 and len(test) == 300
    assert data_utils.check_no_leakage(pool, val)
    pool2, val2, _ = data_utils.make_splits(seed=1, val_size=100)
    assert sorted(val.indices) == sorted(val2.indices), "split must be reproducible"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)