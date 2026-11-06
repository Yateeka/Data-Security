"""
level1_centralized.py - Level 1: centralized MNIST CNN baseline.

Usage:
    python level1_centralized.py --epochs 5 --seed 42
    python level1_centralized.py --eval-only          # reload checkpoint & re-test
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from common import (SmallCNN, evaluate, get_device, load_checkpoint,
                    save_checkpoint, set_seed)
from data_utils import check_no_leakage, make_loaders

CKPT = "checkpoints/centralized_mnist.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader, pool, val = make_loaders(
        args.seed, args.val_size, save_dir="checkpoints")
    check_no_leakage(pool, val)

    if args.eval_only:
        model, meta = load_checkpoint(args.ckpt, device=device)
        _, acc = evaluate(model, test_loader, device)
        print(f"[reloaded] test acc = {acc:.4f}  (saved: {meta.get('test_acc')})")
        return

    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        val_loss, val_acc = evaluate(model, val_loader, device)
        _, test_acc = evaluate(model, test_loader, device)
        history.append(dict(epoch=ep, train_loss=running / seen,
                            val_loss=val_loss, val_acc=val_acc, test_acc=test_acc))
        print(f"epoch {ep}/{args.epochs}  train_loss={running/seen:.4f}  "
              f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    save_checkpoint(args.ckpt, model, seed=args.seed, epochs=args.epochs,
                    test_acc=history[-1]["test_acc"], val_size=args.val_size)
    os.makedirs("logs", exist_ok=True)
    with open("logs/level1_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Reload check (deliverable: checkpoints must be reloadable)
    reloaded, _ = load_checkpoint(args.ckpt, device=device)
    _, acc2 = evaluate(reloaded, test_loader, device)
    assert abs(acc2 - history[-1]["test_acc"]) < 1e-9, "checkpoint reload mismatch"
    print(f"Saved {args.ckpt}; reload check OK (test acc {acc2:.4f})")


if __name__ == "__main__":
    main()