"""Level 3: FedShield defended federated training on MNIST.

Examples (run from this directory):
    python level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3
    python level3_fedshield.py --attack label_flip --onset 10 --device cuda
    python level3_fedshield.py --attack none
    python level3_fedshield.py --eval-only

The held-out validation set drives detector evidence and best-checkpoint
selection. The test set is evaluated only after training is complete.
"""
import argparse
import json
import os
from typing import Dict

import torch
from torch.utils.data import Dataset

from common import SmallCNN, evaluate, get_device, load_checkpoint, save_checkpoint, set_seed
from data_utils import check_no_leakage, make_loaders
from fedshield_detector import DetectorConfig, FedShieldDefense
from fl_core import fedavg, local_train, partition_clients


BEST_CKPT = "checkpoints/fedshield_mnist_best.pt"
LAST_CKPT = "checkpoints/fedshield_mnist_last.pt"
ATTACKS = ("none", "noise", "sign_flip", "scale", "free_ride", "label_flip")


class LabelFlipDataset(Dataset):
    """Client view that maps every MNIST label y to 9-y during local training."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, 9 - int(y)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return get_device()
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable; use --device cpu or auto")
    return torch.device(choice)


def _clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _attack_update(global_state, trained_state, attack: str, seed: int):
    """Transform a malicious client's uploaded model update reproducibly."""
    if attack == "none" or attack == "label_flip":
        return _clone_state(trained_state)
    if attack == "free_ride":
        return _clone_state(global_state)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    floating = [k for k, v in global_state.items() if torch.is_floating_point(v)]
    if attack == "noise":
        delta_norm_sq = sum(
            float((trained_state[k].detach().cpu().float() - global_state[k].detach().cpu().float()).pow(2).sum())
            for k in floating
        )
        count = sum(global_state[k].numel() for k in floating)
        noise_std = max((delta_norm_sq ** 0.5) / max(count, 1) ** 0.5 * 1.5, 1e-4)
    for key, g in global_state.items():
        if not torch.is_floating_point(g):
            out[key] = trained_state[key].detach().cpu().clone()
            continue
        g_cpu = g.detach().cpu()
        delta = trained_state[key].detach().cpu() - g_cpu
        if attack == "sign_flip":
            new_delta = -5.0 * delta
        elif attack == "scale":
            new_delta = 10.0 * delta
        elif attack == "noise":
            noise = torch.randn(delta.shape, generator=gen, dtype=delta.dtype, device="cpu") * noise_std
            new_delta = noise
        else:
            raise ValueError(f"unsupported attack: {attack}")
        out[key] = (g_cpu + new_delta).to(dtype=g.dtype)
    return out


def _clip_update(global_state, client_state, max_norm: float):
    """L2-clip the full floating-point model delta, preserving integer buffers."""
    keys = [k for k, v in global_state.items() if torch.is_floating_point(v)]
    norm_sq = sum(
        float((client_state[k].detach().cpu().float() - global_state[k].detach().cpu().float()).pow(2).sum())
        for k in keys
    )
    norm = norm_sq ** 0.5
    factor = min(1.0, max_norm / max(norm, 1e-12))
    clipped = {}
    for key, g in global_state.items():
        if torch.is_floating_point(g):
            g_cpu = g.detach().cpu()
            delta = client_state[key].detach().cpu() - g_cpu
            clipped[key] = (g_cpu + factor * delta).to(dtype=g.dtype)
        else:
            clipped[key] = client_state[key].detach().cpu().clone()
    return clipped


@torch.no_grad()
def _validation_delta(model, global_state, client_state, val_loader, device, base_loss):
    """Measure validation loss change after applying one candidate update."""
    try:
        model.load_state_dict(client_state)
        loss, _ = evaluate(model, val_loader, device)
        return loss - base_loss
    except (RuntimeError, KeyError, ValueError):
        return float("inf")
    finally:
        model.load_state_dict(global_state)


def _detector_state(defense):
    return dict(suspicion=defense.suspicion, flag_counts=defense.flag_counts,
                history=defense.history)


def _restore_detector(defense, state):
    defense.suspicion = list(state["suspicion"])
    defense.flag_counts = list(state["flag_counts"])
    defense.history = list(state["history"])


def main():
    ap = argparse.ArgumentParser(description="Run the FedShield defended MNIST experiment")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--attack", choices=ATTACKS, default="sign_flip")
    ap.add_argument("--onset", type=int, default=2, help="1-indexed first attack round (>=2 for attacks)")
    ap.add_argument("--malicious-id", type=int, default=3, help="0-indexed malicious client")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--best-ckpt", default=BEST_CKPT)
    ap.add_argument("--last-ckpt", default=LAST_CKPT)
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    if args.clients < 3:
        ap.error("--clients must be at least 3 for layer-aware scoring")
    if args.attack != "none" and args.onset < 2:
        ap.error("attacks must start at round 2 or later")
    if not 0 <= args.malicious_id < args.clients:
        ap.error("--malicious-id must be a valid zero-based client index")
    if args.onset > args.rounds and args.attack != "none":
        ap.error("--onset cannot be later than --rounds")
    if args.clip_norm <= 0:
        ap.error("--clip-norm must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device} (PyTorch {torch.__version__})")
    _, val_loader, test_loader, pool, val = make_loaders(
        args.seed, args.val_size, save_dir="checkpoints")
    check_no_leakage(pool, val)
    partition_path = f"checkpoints/client_partition_{args.partition}_seed{args.seed}.json"
    clients = partition_clients(pool, args.clients, args.seed, args.partition,
                                args.alpha, save_path=partition_path)
    sizes = [len(c) for c in clients]

    model = SmallCNN().to(device)
    start_state = _clone_state(model.state_dict())
    start_round, history, best_val = 1, [], float("inf")
    config = dict(attack=args.attack, onset=args.onset, malicious_id=args.malicious_id,
                  clients=args.clients, partition=args.partition, alpha=args.alpha,
                  seed=args.seed, local_epochs=args.local_epochs, lr=args.lr,
                  momentum=args.momentum, clip_norm=args.clip_norm,
                  threshold=args.threshold, val_size=args.val_size)

    if args.eval_only:
        best_model, best_meta = load_checkpoint(args.best_ckpt, device=device)
        _, best_acc = evaluate(best_model, test_loader, device)
        last_model, last_meta = load_checkpoint(args.last_ckpt, device=device)
        _, last_acc = evaluate(last_model, test_loader, device)
        print(f"best checkpoint round {best_meta.get('round')}: test accuracy={best_acc:.4f}")
        print(f"last checkpoint round {last_meta.get('round')}: test accuracy={last_acc:.4f}")
        return

    log_path = args.log_path or (
        f"logs/fedshield_{args.attack}_{args.partition}_seed{args.seed}_onset{args.onset}.jsonl"
    )
    detector_cfg = DetectorConfig(threshold=args.threshold, use_validation=True)
    # Do not truncate a prior run's detector log when resuming from its last
    # checkpoint. New runs still start with a fresh config header.
    defense = FedShieldDefense(args.clients, detector_cfg,
                               log_path=None if args.resume else log_path)
    global_state = start_state

    if args.resume:
        if not os.path.exists(args.last_ckpt):
            ap.error(f"cannot resume: checkpoint not found: {args.last_ckpt}")
        model, meta = load_checkpoint(args.last_ckpt, model=model, device=device)
        if meta.get("run_config") != config:
            ap.error("resume options do not match the run configuration saved in the last checkpoint")
        global_state = _clone_state(model.state_dict())
        start_round = int(meta["round"]) + 1
        history, best_val = meta["history"], meta["best_val_loss"]
        _restore_detector(defense, meta["detector_state"])
        if not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                f.write(json.dumps({"config": detector_cfg.__dict__, "resumed": True}) + "\n")
        defense.log_path = log_path
        print(f"resumed after round {meta['round']} (best val loss {best_val:.4f})")

    for rnd in range(start_round, args.rounds + 1):
        client_states = []
        is_attack_round = args.attack != "none" and rnd >= args.onset
        for cid, data in enumerate(clients):
            client_data = data
            if is_attack_round and cid == args.malicious_id and args.attack == "label_flip":
                client_data = LabelFlipDataset(data)
            uploaded = local_train(
                model, global_state, client_data, args.local_epochs, args.lr, args.momentum,
                seed=args.seed * 10_000 + rnd * 100 + cid, device=device)
            if is_attack_round and cid == args.malicious_id and args.attack != "label_flip":
                uploaded = _attack_update(global_state, uploaded, args.attack,
                                          args.seed * 1_000_000 + rnd * 100 + cid)
            client_states.append(uploaded)

        model.load_state_dict(global_state)
        base_val_loss, _ = evaluate(model, val_loader, device)
        val_deltas = [
            _validation_delta(model, global_state, st, val_loader, device, base_val_loss)
            for st in client_states
        ]
        result = defense.inspect_round(rnd, global_state, client_states, val_deltas)

        if result.accepted_ids:
            accepted = [_clip_update(global_state, client_states[i], args.clip_norm)
                        for i in result.accepted_ids]
            accepted_sizes = [sizes[i] for i in result.accepted_ids]
            global_state = fedavg(accepted, accepted_sizes)
        else:
            print(f"round {rnd}: no accepted updates; retaining previous global model")

        model.load_state_dict(global_state)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(dict(round=rnd, val_loss=val_loss, val_acc=val_acc,
                            accepted=result.accepted_ids, flagged=result.flagged_ids,
                            invalid=result.invalid_ids, scores=result.scores,
                            reasons=result.reasons))
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(args.best_ckpt, model, round=rnd, val_loss=val_loss,
                            val_acc=val_acc, run_config=config, attack=args.attack,
                            detector_state=_detector_state(defense))
        save_checkpoint(args.last_ckpt, model, round=rnd, history=history,
                        best_val_loss=best_val, run_config=config,
                        detector_state=_detector_state(defense))
        print(f"round {rnd}/{args.rounds} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
              f" accepted={len(result.accepted_ids)} flagged={result.flagged_ids}"
              f"{' *best*' if improved else ''}")

    verdict = defense.final_verdict()
    from detection_metrics import run_metrics
    metrics = run_metrics(defense.history,
                          args.malicious_id if args.attack != "none" else None,
                          args.onset if args.attack != "none" else None,
                          verdict)
    best_model, best_meta = load_checkpoint(args.best_ckpt, device=device)
    _, best_test_acc = evaluate(best_model, test_loader, device)
    _, final_test_acc = evaluate(model, test_loader, device)
    print(f"best checkpoint (round {best_meta['round']}): test accuracy={best_test_acc:.4f}")
    print(f"last checkpoint (round {args.rounds}): test accuracy={final_test_acc:.4f}")
    print(f"detection metrics: {json.dumps(metrics, sort_keys=True)}")

    os.makedirs("logs", exist_ok=True)
    result_path = (
        f"logs/fedshield_{args.attack}_{args.partition}_seed{args.seed}_onset{args.onset}.json"
    )
    with open(result_path, "w") as f:
        json.dump(dict(config=config, history=history, verdict=verdict,
                       metrics=metrics, best_round=best_meta["round"],
                       best_test_acc=best_test_acc, final_test_acc=final_test_acc), f, indent=2)
    print(f"run summary: {result_path}")


if __name__ == "__main__":
    main()
