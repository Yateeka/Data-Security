"""Level 3: FedShield defended federated training on MNIST.

Examples (run from this directory):
    python level3_fedshield.py --attack sign_flip --onset 2 --malicious-id 3
    python level3_fedshield.py --attack label_flip --onset 10 --device cuda
    python level3_fedshield.py --attack sign_flip --malicious-id random   # attacker drawn from --seed
    python level3_fedshield.py --attack none
    python level3_fedshield.py --eval-only
    python level3_fedshield.py --attack sign_flip --defense off      # plain FedAvg baseline
    python level3_fedshield.py --attack sign_flip --no-validation    # ablation: layer scores only
    python level3_fedshield.py --attack sign_flip --no-clip          # ablation: no norm clipping

The held-out validation set drives detector evidence and best-checkpoint
selection. The test set is evaluated only after training is complete.

Conditions: --defense on (default) runs the full FedShield pipeline (detector
+ exclusion + clipping). --defense off is the undefended baseline: plain FedAvg
over every upload, no detector. --no-validation and --no-clip remove one
defense component each for the ablation study. Non-default conditions get
their own checkpoint and log names so they never overwrite the main run.

The malicious client's behavior comes ONLY from malicious_client.py
(generate_malicious_update). This script decides WHEN it attacks: the malicious
client trains honestly before --onset and calls generate_malicious_update from
round --onset on. On demo day the TA can replace malicious_client.py; if the
replacement has no ATTACK_MODES/configure(), run with the default --attack.
"""
import argparse
import json
import os
import random
from typing import Dict, Optional

import torch

from common import SmallCNN, evaluate, get_device, load_checkpoint, save_checkpoint, set_seed
from data_utils import check_no_leakage, make_loaders
from fedshield_detector import DetectorConfig, FedShieldDefense
from fl_core import fedavg, local_train, partition_clients
import malicious_client


CKPT_PREFIX = "checkpoints/fedshield_mnist"   # + "<variant>_best.pt" / "_last.pt"
# Attack modes come from malicious_client.py. A TA replacement file may only
# define generate_malicious_update(), so fall back to a single "custom" mode.
_MODES = tuple(getattr(malicious_client, "ATTACK_MODES", ()))
ATTACKS = ("none",) + (_MODES or ("custom",))
DEFAULT_ATTACK = "sign_flip" if "sign_flip" in _MODES else ATTACKS[1]
# Attack settings as shipped in malicious_client.py, restored before every run so
# an --attack-setting from one in-process run (run_sweep.py) never leaks into the next.
_ATTACK_DEFAULTS = dict(getattr(malicious_client, "ATTACK_CONFIG", {}))


def _parse_attack_settings(items):
    """['flip_factor=-1', ...] -> {'flip_factor': -1.0}, typed like the default."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or key not in _ATTACK_DEFAULTS or key in ("mode", "seed"):
            raise ValueError(item)
        default = _ATTACK_DEFAULTS[key]
        out[key] = type(default)(value) if not isinstance(default, str) else value
    return out


def _settings_tag(settings) -> str:
    """{'flip_factor': -1.0} -> '_flip_factor-1.0' (part of the log name)."""
    return "".join(f"_{k}{v}" for k, v in sorted(settings.items()))


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return get_device()
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable; use --device cpu or auto")
    return torch.device(choice)


def _clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _sanitize_upload(state, global_state) -> Optional[Dict[str, torch.Tensor]]:
    """Turn whatever the malicious client returned into a CPU state_dict.

    Keeps only the model's keys and casts tensors to the global dtype. Anything
    unusable becomes None, which the detector rejects as an invalid update
    (missing keys, wrong shapes and NaN/Inf are also rejected there).
    """
    if not isinstance(state, dict):
        return None
    clean = {}
    for key, g in global_state.items():
        if key not in state:
            continue  # detector reports missing_key
        try:
            v = torch.as_tensor(state[key])
            clean[key] = v.detach().cpu().to(dtype=g.dtype).clone()
        except (TypeError, RuntimeError, ValueError):
            return None
    return clean


def _usable_for_fedavg(state, global_state) -> bool:
    """Plain FedAvg has no defense, but it can only average uploads with the
    right keys and shapes. NaN/Inf values are NOT filtered (no defense)."""
    return state is not None and all(
        k in state and tuple(state[k].shape) == tuple(g.shape)
        for k, g in global_state.items())


def _variant_tag(args) -> str:
    """'' for the full defense; otherwise a suffix naming the condition."""
    if args.defense == "off":
        return "_nodefense"
    return ("_noval" if args.no_validation else "") + ("_noclip" if args.no_clip else "")


DEFAULT_MALICIOUS_ID = 3


def _pick_malicious_id(value: str, seed: int, n_clients: int) -> int:
    """--malicious-id: a client index, or 'random' = drawn from the seed with a
    private RNG (same seed -> same client; the training RNG is not touched)."""
    if value == "random":
        return random.Random(f"malicious-client-{seed}").randrange(n_clients)
    return int(value)


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
    if client_state is None:
        return float("inf")
    try:
        model.load_state_dict(client_state)
        loss, _ = evaluate(model, val_loader, device)
        return loss - base_loss
    except (RuntimeError, KeyError, ValueError):
        return float("inf")
    finally:
        model.load_state_dict(global_state)


def _detector_state(defense):
    if defense is None:
        return None
    return dict(suspicion=defense.suspicion, flag_counts=defense.flag_counts,
                history=defense.history)


def _restore_detector(defense, state):
    defense.suspicion = list(state["suspicion"])
    defense.flag_counts = list(state["flag_counts"])
    defense.history = list(state["history"])


def main(argv=None):
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
    ap.add_argument("--attack", choices=ATTACKS, default=DEFAULT_ATTACK,
                    help="mode passed to malicious_client.configure(); 'none' = no attacker")
    ap.add_argument("--onset", type=int, default=2, help="1-indexed first attack round (>=2 for attacks)")
    ap.add_argument("--malicious-id", default=str(DEFAULT_MALICIOUS_ID),
                    help="0-indexed malicious client, or 'random' (chosen from --seed)")
    ap.add_argument("--attack-setting", action="append", metavar="KEY=VALUE",
                    help="override one ATTACK_CONFIG value in malicious_client.py for this "
                         "run, e.g. --attack-setting flip_factor=-1 (repeatable)")
    ap.add_argument("--threshold", type=float, default=DetectorConfig().threshold,
                    help="detector flag threshold (default: calibrated value in fedshield_detector.py)")
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--defense", choices=["on", "off"], default="on",
                    help="on = FedShield (default); off = plain FedAvg baseline, no detector")
    ap.add_argument("--no-validation", action="store_true",
                    help="ablation: detector uses layer-aware scores only")
    ap.add_argument("--no-clip", action="store_true",
                    help="ablation: skip norm clipping of accepted updates")
    ap.add_argument("--best-ckpt", default=None,
                    help="default: checkpoints/fedshield_mnist<variant>_best.pt")
    ap.add_argument("--last-ckpt", default=None,
                    help="default: checkpoints/fedshield_mnist<variant>_last.pt")
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args(argv)

    if args.clients < 3:
        ap.error("--clients must be at least 3 for layer-aware scoring")
    if args.attack != "none" and args.onset < 2:
        ap.error("attacks must start at round 2 or later")
    try:
        attack_settings = _parse_attack_settings(args.attack_setting)
    except ValueError as exc:
        ap.error(f"bad --attack-setting {exc}; use KEY=VALUE with a key from "
                 f"malicious_client.ATTACK_CONFIG other than mode/seed")
    if attack_settings and (args.attack == "none" or not _ATTACK_DEFAULTS):
        ap.error("--attack-setting needs an attack and a malicious_client.py with ATTACK_CONFIG")
    malicious_choice = args.malicious_id
    try:
        args.malicious_id = _pick_malicious_id(args.malicious_id, args.seed, args.clients)
    except ValueError:
        ap.error("--malicious-id must be a client index or 'random'")
    if not 0 <= args.malicious_id < args.clients:
        ap.error("--malicious-id must be a valid zero-based client index")
    if malicious_choice == "random" and args.attack != "none":
        print(f"--malicious-id random: seed {args.seed} picked client {args.malicious_id}")
    if args.onset > args.rounds and args.attack != "none":
        ap.error("--onset cannot be later than --rounds")
    if args.clip_norm <= 0:
        ap.error("--clip-norm must be positive")
    if args.defense == "off" and (args.no_validation or args.no_clip):
        ap.error("--no-validation/--no-clip are ablations of the defense; "
                 "do not combine them with --defense off")
    tag = _variant_tag(args)
    args.best_ckpt = args.best_ckpt or f"{CKPT_PREFIX}{tag}_best.pt"
    args.last_ckpt = args.last_ckpt or f"{CKPT_PREFIX}{tag}_last.pt"
    defended = args.defense == "on"

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
                  threshold=args.threshold, val_size=args.val_size,
                  defense=args.defense, use_validation=not args.no_validation,
                  clip=not args.no_clip)
    if attack_settings:   # only present when used, so older checkpoints still --resume
        config["attack_settings"] = attack_settings

    if args.eval_only:
        best_model, best_meta = load_checkpoint(args.best_ckpt, device=device)
        _, best_acc = evaluate(best_model, test_loader, device)
        last_model, last_meta = load_checkpoint(args.last_ckpt, device=device)
        _, last_acc = evaluate(last_model, test_loader, device)
        print(f"best checkpoint round {best_meta.get('round')}: test accuracy={best_acc:.4f}")
        print(f"last checkpoint round {last_meta.get('round')}: test accuracy={last_acc:.4f}")
        return

    # Runs with a non-default attacker get their own log names (..._mal7).
    mal_tag = (f"_mal{args.malicious_id}"
               if args.attack != "none" and args.malicious_id != DEFAULT_MALICIOUS_ID else "")
    run_name = (f"fedshield_{args.attack}_{args.partition}_seed{args.seed}"
                f"_onset{args.onset}{mal_tag}{_settings_tag(attack_settings)}{tag}")
    log_path = args.log_path or f"logs/{run_name}.jsonl"
    print(f"condition: " + ("FedShield defense" if defended else "plain FedAvg (no defense)")
          + (" | no validation evidence" if args.no_validation else "")
          + (" | no clipping" if args.no_clip else ""))
    defense = None
    if defended:
        detector_cfg = DetectorConfig(threshold=args.threshold,
                                      use_validation=not args.no_validation)
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
        if defended:
            _restore_detector(defense, meta["detector_state"])
            if not os.path.exists(log_path):
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                with open(log_path, "w") as f:
                    f.write(json.dumps({"config": detector_cfg.__dict__, "resumed": True}) + "\n")
            defense.log_path = log_path
        print(f"resumed after round {meta['round']} (best val loss {best_val:.4f})")

    if args.attack != "none":
        if hasattr(malicious_client, "configure"):
            malicious_client.configure(**{
                **_ATTACK_DEFAULTS, **attack_settings, "mode": args.attack,
                "seed": args.seed, "local_epochs": args.local_epochs, "lr": args.lr,
                "momentum": args.momentum})
        else:
            print("malicious_client.py has no configure(); using its "
                  "generate_malicious_update() as provided")
        if hasattr(malicious_client, "reset_call_count"):
            # Attack calls already made before start_round (non-zero on --resume).
            malicious_client.reset_call_count(max(0, start_round - args.onset))
        if attack_settings:
            print(f"attack settings: {attack_settings}")
        print(f"malicious client {args.malicious_id}: honest until round "
              f"{args.onset - 1}, then malicious_client.generate_malicious_update")

    for rnd in range(start_round, args.rounds + 1):
        client_states = []
        is_attack_round = args.attack != "none" and rnd >= args.onset
        for cid, data in enumerate(clients):
            if is_attack_round and cid == args.malicious_id:
                # Malicious upload: the attack itself lives in malicious_client.py.
                global_copy = {k: v.clone().to(device) for k, v in global_state.items()}
                try:
                    raw = malicious_client.generate_malicious_update(global_copy, data, device)
                except Exception as exc:  # a crashing attack must not stop training
                    print(f"round {rnd}: malicious client {cid} raised "
                          f"{type(exc).__name__}: {exc}; treating upload as missing")
                    raw = None
                uploaded = _sanitize_upload(raw, global_state)
            else:
                uploaded = local_train(
                    model, global_state, data, args.local_epochs, args.lr, args.momentum,
                    seed=args.seed * 10_000 + rnd * 100 + cid, device=device)
            client_states.append(uploaded)

        if defended:
            val_deltas = None
            if not args.no_validation:
                model.load_state_dict(global_state)
                base_val_loss, _ = evaluate(model, val_loader, device)
                val_deltas = [
                    _validation_delta(model, global_state, st, val_loader, device, base_val_loss)
                    for st in client_states
                ]
            result = defense.inspect_round(rnd, global_state, client_states, val_deltas)
            accepted_ids, flagged_ids, invalid_ids = (
                result.accepted_ids, result.flagged_ids, result.invalid_ids)
            scores, reasons = result.scores, result.reasons
        else:
            # Plain FedAvg: average every upload it can; nothing is flagged.
            accepted_ids = [i for i, st in enumerate(client_states)
                            if _usable_for_fedavg(st, global_state)]
            invalid_ids = [i for i in range(len(client_states)) if i not in accepted_ids]
            flagged_ids, scores, reasons = [], None, {}

        if accepted_ids:
            accepted = [client_states[i] if args.no_clip or not defended
                        else _clip_update(global_state, client_states[i], args.clip_norm)
                        for i in accepted_ids]
            global_state = fedavg(accepted, [sizes[i] for i in accepted_ids])
        else:
            print(f"round {rnd}: no accepted updates; retaining previous global model")

        model.load_state_dict(global_state)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(dict(round=rnd, val_loss=val_loss, val_acc=val_acc,
                            accepted=accepted_ids, flagged=flagged_ids,
                            invalid=invalid_ids, scores=scores, reasons=reasons))
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
              f" accepted={len(accepted_ids)} flagged={flagged_ids}"
              f"{' *best*' if improved else ''}")

    verdict, metrics = None, None
    if defended:
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
    if defended:
        print(f"detection metrics: {json.dumps(metrics, sort_keys=True)}")
    else:
        print("detection metrics: none (no defense)")

    os.makedirs("logs", exist_ok=True)
    result_path = f"logs/{run_name}.json"
    with open(result_path, "w") as f:
        json.dump(dict(config=config, history=history, verdict=verdict,
                       metrics=metrics, best_round=best_meta["round"],
                       best_test_acc=best_test_acc, final_test_acc=final_test_acc), f, indent=2)
    print(f"run summary: {result_path}")


if __name__ == "__main__":
    main()
