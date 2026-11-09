"""
level2_fedavg.py - Level 2: federated learning on MNIST with 10 clients + FedAvg.

Each round:
  1. the server sends the current global weights to all 10 clients
  2. each client trains locally on its own shard for `--local-epochs` epochs
  3. clients upload their weights; the server aggregates them with FedAvg
  4. the server scores the new global model on its held-out validation set
     and keeps the best-so-far model (lowest validation loss, never by test)

Usage:
    python level2_fedavg.py                          # 10 rounds, 1 local epoch, IID
    python level2_fedavg.py --partition dirichlet --alpha 0.5   # non-IID experiment
    python level2_fedavg.py --resume                 # continue from last checkpoint
    python level2_fedavg.py --eval-only              # test the saved best model
"""
import argparse
import json
import os

from common import (SmallCNN, evaluate, get_device, load_checkpoint,
                    save_checkpoint, set_seed)
from data_utils import check_no_leakage, make_loaders
from fl_core import fedavg, local_train, partition_clients

BEST_CKPT = "checkpoints/fedavg_mnist_best.pt"   # best global model (for the TA)
LAST_CKPT = "checkpoints/fedavg_mnist_last.pt"   # latest round (for --resume)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--best-ckpt", default=BEST_CKPT)
    ap.add_argument("--last-ckpt", default=LAST_CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    _, val_loader, test_loader, pool, val = make_loaders(
        args.seed, args.val_size, save_dir="checkpoints")
    check_no_leakage(pool, val)

    if args.eval_only:
        model, meta = load_checkpoint(args.best_ckpt, device=device)
        _, acc = evaluate(model, test_loader, device)
        print(f"[reloaded] round {meta.get('round')}  test acc = {acc:.4f}")
        return

    clients = partition_clients(
        pool, args.clients, args.seed, args.partition, args.alpha,
        save_path=f"checkpoints/client_partition_{args.partition}_seed{args.seed}.json")
    sizes = [len(c) for c in clients]
    print(f"{args.clients} clients, samples per client: {sizes}")

    model = SmallCNN().to(device)
    global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    start_round, history, best_val = 1, [], float("inf")

    if args.resume and os.path.exists(args.last_ckpt):
        model, meta = load_checkpoint(args.last_ckpt, model, device=device)
        global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        start_round = meta["round"] + 1
        history, best_val = meta["history"], meta["best_val_loss"]
        print(f"resumed after round {meta['round']} (best val loss {best_val:.4f})")

    for rnd in range(start_round, args.rounds + 1):
        client_states = []
        for cid, data in enumerate(clients):
            # Seed differs per (round, client) so shuffling is reproducible but not repeated.
            state = local_train(model, global_state, data, args.local_epochs,
                                args.lr, args.momentum,
                                seed=args.seed * 10_000 + rnd * 100 + cid, device=device)
            client_states.append(state)

        global_state = fedavg(client_states, sizes)
        model.load_state_dict(global_state)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(dict(round=rnd, val_loss=val_loss, val_acc=val_acc))

        # Val loss is a smoother selection signal than accuracy on 500 images.
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(args.best_ckpt, model, round=rnd, val_loss=val_loss, val_acc=val_acc,
                            seed=args.seed, partition=args.partition,
                            clients=args.clients, local_epochs=args.local_epochs)
        save_checkpoint(args.last_ckpt, model, round=rnd, history=history,
                        best_val_loss=best_val, seed=args.seed)
        print(f"round {rnd}/{args.rounds}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}{'  *best*' if improved else ''}")

    # Final evaluation: test set is touched only here, after training is over.
    best_model, meta = load_checkpoint(args.best_ckpt, device=device)
    _, best_test = evaluate(best_model, test_loader, device)
    _, last_test = evaluate(model, test_loader, device)
    print(f"final global model test acc = {last_test:.4f}")
    print(f"best model (round {meta['round']}) test acc = {best_test:.4f} "
          f"-> saved at {args.best_ckpt}")

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/level2_history_{args.partition}.json", "w") as f:
        json.dump(dict(history=history, best_round=meta["round"],
                       best_test_acc=best_test, final_test_acc=last_test,
                       args=vars(args)), f, indent=2)


if __name__ == "__main__":
    main()
