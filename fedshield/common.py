"""common.py - shared utilities for FedShield (seed control, model, eval, checkpoints).

Owner: Yateeka (Foundations). Jessie's FL simulator should import `SmallCNN`,
`set_seed`, `evaluate`, and the checkpoint helpers from here so that both parts
of the project use the exact same model and layer names.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    """Compact CNN for 28x28 MNIST.

    Layer names (conv1, conv2, fc1, fc2) matter: the detector groups each
    layer's weight+bias into one vector and scores layers independently.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)    # 28 -> 24 -> pool 12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)   # 12 -> 8  -> pool 4
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device=None):
    """Return (mean cross-entropy loss, accuracy) of `model` over `loader`."""
    device = device or next(model.parameters()).device
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return total_loss / n, correct / n


def save_checkpoint(path: str, model: nn.Module, **meta) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, model: nn.Module = None, device=None):
    """Load a checkpoint saved by `save_checkpoint`. Returns (model, meta)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model = model or SmallCNN()
    model.load_state_dict(ckpt["model_state"])
    return model.to(device), ckpt.get("meta", {})