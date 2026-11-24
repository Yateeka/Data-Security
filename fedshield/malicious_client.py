"""
malicious_client.py - simulated malicious client for Level 3 (owner: Jessie).

Built-in attacks (choose with configure() or the FEDSHIELD_ATTACK env variable):

    noise       replace the local update with Gaussian noise
    scale       train honestly, then multiply the update by a large factor (+10)
    sign_flip   train honestly, then multiply the update by a negative factor (-5)
    free_ride   upload the global model unchanged (no training at all)
    label_flip  train locally on deliberately wrong labels (y -> 9 - y)

Extra attack modes (beyond the four proposal attacks):

    zero_weights    upload an all-zero model
    random_weights  upload a freshly initialized (untrained) model
    nan             train honestly, then fill one layer with NaN (malformed upload)
    grad_ascent     train to INCREASE the loss on local data
    layer_only      train honestly, then corrupt only one layer (default fc2)

The function signature has no round number or attack argument, so the attack
settings live in the module-level ATTACK_CONFIG dictionary below.
"""
import copy
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import SmallCNN
from fl_core import BATCH_SIZE, local_train

ATTACK_MODES = ("noise", "scale", "sign_flip", "free_ride", "label_flip",
                "zero_weights", "random_weights", "nan", "grad_ascent", "layer_only")

# Module-level attack settings. Change them with configure(...) from the
# simulator, or set FEDSHIELD_ATTACK=<mode> before running.
ATTACK_CONFIG = {
    "mode": os.environ.get("FEDSHIELD_ATTACK", "label_flip"),
    "noise_std": 1.0,        # std of the Gaussian noise update ("noise")
    "scale_factor": 10.0,    # multiplier for the honest update ("scale")
    "flip_factor": -5.0,     # multiplier for the honest update ("sign_flip")
    "bad_value": float("nan"),  # value written into the target layer ("nan")
    "target_layer": "fc2",   # layer attacked by "nan" and "layer_only"
    "layer_factor": -10.0,   # multiplier for the target layer's update ("layer_only")
    "ascent_clip": 1.0,      # max gradient norm per step ("grad_ascent")
    "local_epochs": 1,       # course rule: 1 local epoch per round
    "lr": 0.01,              # same local optimizer settings as benign clients
    "momentum": 0.9,
    "seed": 42,              # base seed; each call uses seed + call counter
}

_call_count = 0  # makes every call's randomness different but reproducible


def configure(**settings) -> Dict:
    """Update ATTACK_CONFIG, e.g. configure(mode="noise", seed=7)."""
    unknown = set(settings) - set(ATTACK_CONFIG)
    if unknown:
        raise KeyError(f"unknown attack setting(s): {sorted(unknown)}")
    if "mode" in settings and settings["mode"] not in ATTACK_MODES:
        raise ValueError(f"unknown attack mode {settings['mode']!r}; "
                         f"choose one of {ATTACK_MODES}")
    ATTACK_CONFIG.update(settings)
    return dict(ATTACK_CONFIG)


def reset_call_count(value: int = 0) -> None:
    """Restart the per-call seed sequence (call at the start of every run).

    When resuming a run, pass the number of attack calls already made so the
    remaining rounds use the same seeds as an uninterrupted run.
    """
    global _call_count
    _call_count = value


class FlippedLabels(Dataset):
    """Wraps a client dataset and returns (x, 9 - y): every label is wrong."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, 9 - int(y)


def _cpu_copy(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _honest_train(global_state, dataset, device, seed):
    """Train exactly like a benign client (same optimizer, 1 local epoch)."""
    model = SmallCNN().to(device)
    return local_train(model, global_state, dataset,
                       epochs=ATTACK_CONFIG["local_epochs"], lr=ATTACK_CONFIG["lr"],
                       momentum=ATTACK_CONFIG["momentum"], seed=seed, device=device)


def _scaled_update(global_state, trained_state, factor):
    """global + factor * (trained - global), applied to float tensors only."""
    return {k: g + factor * (trained_state[k] - g) if g.is_floating_point() else g.clone()
            for k, g in global_state.items()}


def noise_attack(global_state, local_dataset, device, seed):
    gen = torch.Generator().manual_seed(seed)
    std = ATTACK_CONFIG["noise_std"]
    return {k: g + std * torch.randn(g.shape, generator=gen, dtype=g.dtype)
            if g.is_floating_point() else g.clone()
            for k, g in global_state.items()}


def scale_attack(global_state, local_dataset, device, seed):
    trained = _honest_train(global_state, local_dataset, device, seed)
    return _scaled_update(global_state, trained, ATTACK_CONFIG["scale_factor"])


def sign_flip_attack(global_state, local_dataset, device, seed):
    trained = _honest_train(global_state, local_dataset, device, seed)
    return _scaled_update(global_state, trained, ATTACK_CONFIG["flip_factor"])


def free_ride_attack(global_state, local_dataset, device, seed):
    return _cpu_copy(global_state)


def label_flip_attack(global_state, local_dataset, device, seed):
    return _honest_train(global_state, FlippedLabels(local_dataset), device, seed)


def zero_weights_attack(global_state, local_dataset, device, seed):
    return {k: torch.zeros_like(g) if g.is_floating_point() else g.clone()
            for k, g in global_state.items()}


def random_weights_attack(global_state, local_dataset, device, seed):
    # fork_rng: seed the layer initialization without disturbing the
    # simulator's global random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _cpu_copy(SmallCNN().state_dict())


def _target_keys(state):
    layer = ATTACK_CONFIG["target_layer"]
    keys = [k for k in state if k.split(".")[0] == layer and state[k].is_floating_point()]
    if not keys:
        raise ValueError(f"target_layer {layer!r} not found in the model")
    return keys


def nan_attack(global_state, local_dataset, device, seed):
    state = _honest_train(global_state, local_dataset, device, seed)
    for k in _target_keys(state):
        state[k] = torch.full_like(state[k], ATTACK_CONFIG["bad_value"])
    return state


def grad_ascent_attack(global_state, local_dataset, device, seed):
    """Like honest local training, but steps UP the loss gradient."""
    model = SmallCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(local_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
    opt = torch.optim.SGD(model.parameters(), lr=ATTACK_CONFIG["lr"],
                          momentum=ATTACK_CONFIG["momentum"])
    for _ in range(ATTACK_CONFIG["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            (-F.cross_entropy(model(x), y)).backward()  # minus sign = ascent
            # Ascent has no natural stopping point; without a cap the weights
            # overflow to NaN and this would just duplicate the "nan" attack.
            torch.nn.utils.clip_grad_norm_(model.parameters(), ATTACK_CONFIG["ascent_clip"])
            opt.step()
    return _cpu_copy(model.state_dict())


def layer_only_attack(global_state, local_dataset, device, seed):
    """Honest update everywhere except one layer, whose update is scaled/flipped."""
    state = _honest_train(global_state, local_dataset, device, seed)
    for k in _target_keys(state):
        g = global_state[k]
        state[k] = g + ATTACK_CONFIG["layer_factor"] * (state[k] - g)
    return state


_ATTACKS = {
    "noise": noise_attack,
    "scale": scale_attack,
    "sign_flip": sign_flip_attack,
    "free_ride": free_ride_attack,
    "label_flip": label_flip_attack,
    "zero_weights": zero_weights_attack,
    "random_weights": random_weights_attack,
    "nan": nan_attack,
    "grad_ascent": grad_ascent_attack,
    "layer_only": layer_only_attack,
}

def generate_malicious_update(client_model_state_dict, local_dataset, device):
    """
    Args:
        client_model_state_dict (dict): A copy of the current global model's state_dict().
        local_dataset (Dataset): The local dataset assigned to this client.
        device (torch.device): CPU or CUDA device.
    Returns:
        malicious_state_dict (dict): The modified model parameters (state_dict) to be uploaded to the server.
    """
    global _call_count
    _call_count += 1
    # Work on our own CPU copy so the caller's dict is never modified.
    global_state = _cpu_copy(copy.deepcopy(client_model_state_dict))

    # === MALICIOUS CLIENT LOGIC HERE ===
    # TA: replace everything from here to the `return` with your own attack.
    # Input: `global_state` (CPU copy of the global model), `local_dataset`,
    # `device`. Output: a state_dict with the same keys and shapes.
    mode = ATTACK_CONFIG["mode"]
    if mode not in _ATTACKS:
        raise ValueError(f"unknown attack mode {mode!r}; choose one of {ATTACK_MODES}")
    seed = ATTACK_CONFIG["seed"] * 10_000 + _call_count
    malicious_state_dict = _ATTACKS[mode](global_state, local_dataset, device, seed)
    # === END MALICIOUS CLIENT LOGIC ===

    return malicious_state_dict
