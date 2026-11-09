"""
fl_core.py - reusable federated-learning building blocks (Level 2, reused by Level 3).

    partition_clients()  split the training pool into N client shards (IID or non-IID)
    local_train()        one client's local training, starting from the global model
    fedavg()             server-side weighted Federated Averaging of client states

The server never sees client data - only the state_dicts returned by local_train().
"""
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

BATCH_SIZE = 50  # same batch size as the instructor's dataloader4level1.py


def _targets_of(dataset) -> np.ndarray:
    """Labels of a (possibly nested) torch Subset of MNIST, without loading images."""
    if isinstance(dataset, Subset):
        return _targets_of(dataset.dataset)[np.asarray(dataset.indices)]
    return np.asarray(dataset.targets)


def partition_clients(train_pool, num_clients: int = 10, seed: int = 42,
                      mode: str = "iid", alpha: float = 0.5,
                      save_path: str = None) -> List[Subset]:
    """Split `train_pool` into `num_clients` disjoint Subsets.

    mode="iid":       shuffle, then cut into equal-size shards (default).
    mode="dirichlet": label-skewed non-IID split; for every class, its samples
                      are divided among clients with proportions ~ Dir(alpha).
                      Smaller alpha = more skew.
    """
    rng = np.random.default_rng(seed)
    n = len(train_pool)

    if mode == "iid":
        perm = rng.permutation(n)
        shards = [s.tolist() for s in np.array_split(perm, num_clients)]
    elif mode == "dirichlet":
        labels = _targets_of(train_pool)
        shards = [[] for _ in range(num_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].extend(part.tolist())
        for s in shards:
            rng.shuffle(s)
    else:
        raise ValueError(f"unknown partition mode: {mode}")

    # Shards must be disjoint and cover the whole pool.
    flat = [i for s in shards for i in s]
    assert len(flat) == n and len(set(flat)) == n, "client partitions overlap"

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"mode": mode, "seed": seed, "alpha": alpha,
                       "shards": [sorted(int(i) for i in s) for s in shards]}, f)
    return [Subset(train_pool, s) for s in shards]


def local_train(model: nn.Module, global_state: Dict[str, torch.Tensor], dataset,
                epochs: int = 1, lr: float = 0.01, momentum: float = 0.9,
                seed: int = 0, device=None) -> Dict[str, torch.Tensor]:
    """Client side: load the global weights, train on local data, return new weights.

    A fresh optimizer is created every round (clients keep no state between rounds).
    `seed` fixes this client's shuffling order so runs are reproducible.
    """
    device = device or next(model.parameters()).device
    model.load_state_dict(global_state)
    model.to(device).train()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fedavg(client_states: Sequence[Dict[str, torch.Tensor]],
           weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Server side: w_global = sum_k (n_k / n) * w_k.

    `weights` are usually the clients' sample counts n_k. Non-float buffers
    (e.g. BatchNorm's num_batches_tracked) are copied from the first client.
    """
    if not client_states:
        raise ValueError("fedavg needs at least one client state")
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    out = {}
    for key, ref in client_states[0].items():
        if ref.is_floating_point():
            stacked = torch.stack([s[key].double() for s in client_states])
            out[key] = (w.view(-1, *[1] * ref.dim()) * stacked).sum(0).to(ref.dtype)
        else:
            out[key] = ref.clone()
    return out
