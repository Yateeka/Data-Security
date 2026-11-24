"""
level1_centralized.py - Level 1: centralized MNIST CNN baseline.

The epoch with the lowest server-validation loss is saved as the best model
(checkpoints/centralized_mnist_best.pt, the model the TA tests). The latest epoch is
saved separately for --resume. The test set is used only after training.

Usage:
    python level1_centralized.py --epochs 5 --seed 42
    python level1_centralized.py --eval-only          # reload best checkpoint & test
    python level1_centralized.py --epochs 8 --resume  # continue from the last epoch
    python level1_centralized.py --device cpu         # force CPU (default: GPU if available)
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from common import (SmallCNN, evaluate, get_device, load_checkpoint,
                    save_checkpoint, set_seed)
from data_utils import check_no_leakage, make_loaders

CKPT = "checkpoints/centralized_mnist_best.pt"            # best epoch (for the TA)
LAST_CKPT = "checkpoints/centralized_mnist_last.pt"  # latest epoch (for --resume)


def resolve_device(choice: str) -> torch.device:
    """'auto' -> GPU if available else CPU; 'cuda' fails loudly if no GPU."""
    if choice == "auto":
        return get_device()
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available; "
                         "use --device cpu or --device auto")
    return torch.device(choice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--ckpt", default=CKPT, help="best-epoch checkpoint")
    ap.add_argument("--last-ckpt", default=LAST_CKPT, help="latest-epoch checkpoint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = GPU if available, else CPU (default)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    train_loader, val_loader, test_loader, pool, val = make_loaders(
        args.seed, args.val_size, save_dir="checkpoints")
    check_no_leakage(pool, val)

    if args.eval_only:
        model, meta = load_checkpoint(args.ckpt, device=device)
        _, acc = evaluate(model, test_loader, device)
        print(f"[reloaded] epoch {meta.get('epoch')}  test acc = {acc:.4f}")
        return

    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    history, best_val, start_epoch = [], float("inf"), 1

    if args.resume:
        if not os.path.exists(args.last_ckpt):
            ap.error(f"cannot resume: checkpoint not found: {args.last_ckpt}")
        model, meta = load_checkpoint(args.last_ckpt, model, device=device)
        opt.load_state_dict(meta["optimizer"])
        torch.set_rng_state(meta["rng_state"].cpu())  # same shuffling as an uninterrupted run
        history, best_val = meta["history"], meta["best_val_loss"]
        start_epoch = meta["epoch"] + 1
        print(f"resumed after epoch {meta['epoch']} (best val loss {best_val:.4f})")

    for ep in range(start_epoch, args.epochs + 1):
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
        history.append(dict(epoch=ep, train_loss=running / seen,
                            val_loss=val_loss, val_acc=val_acc))

        # Select by validation loss (never by test accuracy).
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(args.ckpt, model, epoch=ep, val_loss=val_loss, val_acc=val_acc,
                            seed=args.seed, val_size=args.val_size)
        save_checkpoint(args.last_ckpt, model, epoch=ep, history=history,
                        best_val_loss=best_val, optimizer=opt.state_dict(),
                        rng_state=torch.get_rng_state(), seed=args.seed)
        print(f"epoch {ep}/{args.epochs}  train_loss={running/seen:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
              f"{'  *best*' if improved else ''}")

    # Final evaluation: the test set is touched only here, after training.
    best_model, best_meta = load_checkpoint(args.ckpt, device=device)
    _, best_test = evaluate(best_model, test_loader, device)
    _, last_test = evaluate(model, test_loader, device)
    print(f"last epoch ({history[-1]['epoch']}) test acc = {last_test:.4f}")
    print(f"best epoch ({best_meta['epoch']}) test acc = {best_test:.4f} -> saved at {args.ckpt}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/level1_history.json", "w") as f:
        json.dump(dict(history=history, best_epoch=best_meta["epoch"],
                       best_test_acc=best_test, last_test_acc=last_test,
                       device=str(device), torch=torch.__version__,
                       args=vars(args)), f, indent=2)


if __name__ == "__main__":
    main()