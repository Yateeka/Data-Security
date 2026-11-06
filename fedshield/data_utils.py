"""
data_utils.py - MNIST split built on the instructor's dataloader4level1.py.

We reuse the instructor's `load_data()` (same transforms, same download path)
and `dataloader()` (same batch size / loader settings). On top of that we carve
a small CLEAN validation set out of the 60k training images for the server.

    60,000 MNIST train images
        |-- server_val  (default 500)  -> ONLY the server's validation-loss scoring
        |-- train_pool  (59,500)       -> centralized training (Level 1) and the
                                          10 client partitions (Level 2/3, Jessie)
    10,000 MNIST test images           -> final accuracy only, never used for tuning

The split is determined entirely by `seed`, so it is reproducible.
"""
import json
import os

import torch
from torch.utils.data import DataLoader, random_split

# Instructor-provided template (keep dataloader4level1.py next to this file).
from dataloader4level1 import dataloader as instructor_dataloader
from dataloader4level1 import load_data as instructor_load_data


def make_splits(seed: int = 42, val_size: int = 500, save_dir: str = None):
    """Return (train_pool, server_val, test_dataset) as torch Datasets.

    `train_pool` excludes every validation index, so hand THIS (not the raw
    MNIST train set) to Jessie's client partitioner.
    """
    # load_data() returns loaders; the underlying datasets are on `.dataset`.
    full_train_loader, test_loader = instructor_load_data()
    full_train = full_train_loader.dataset
    test_dataset = test_loader.dataset

    g = torch.Generator().manual_seed(seed)
    train_pool, server_val = random_split(
        full_train, [len(full_train) - val_size, val_size], generator=g
    )

    if save_dir:  # record the held-out indices for reproducibility / auditing
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"val_indices_seed{seed}.json"), "w") as f:
            json.dump(sorted(server_val.indices), f)
    return train_pool, server_val, test_dataset


def make_loaders(seed: int = 42, val_size: int = 500, save_dir: str = None):
    """Convenience wrapper for Level 1.

    Returns (train_loader, val_loader, test_loader, train_pool, server_val).
    train/test loaders come from the instructor's `dataloader()` so batch size
    and shuffling match the course template exactly.
    """
    train_pool, server_val, test_dataset = make_splits(seed, val_size, save_dir)
    train_loader, test_loader = instructor_dataloader(train_pool, test_dataset)
    val_loader = DataLoader(server_val, batch_size=250, shuffle=False)
    return train_loader, val_loader, test_loader, train_pool, server_val


def check_no_leakage(train_pool, server_val) -> bool:
    """Sanity check: no validation index appears in the training pool."""
    overlap = set(train_pool.indices) & set(server_val.indices)
    assert not overlap, f"{len(overlap)} validation samples leaked into training!"
    return True