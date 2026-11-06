"""
fedshield_detector.py - layer-aware anomaly detection + defense integration.

Owner: Yateeka (Security + Defense workstreams).

Pipeline for every round (call `FedShieldDefense.inspect_round`):

 1. Extract each client's update  delta_i = uploaded_params_i - global_params
    from the ACTUAL uploaded state_dicts (never self-reported statistics).
 2. Reject updates with NaN/Inf, missing keys, or wrong shapes BEFORE scoring.
 3. Per layer (weight+bias grouped, e.g. conv1/conv2/fc1/fc2) compare each
    update with the coordinate-wise median of the OTHER clients' updates:
        - magnitude : |log ||u_il|| - median| / robust spread   (large OR small)
        - direction : (median cosine - cos(u_il, median)) / robust spread
      A near-zero norm (free-riding) is flagged directly by the magnitude check
      and cosine is NOT computed for that layer.
 4. Optionally add Jessie's server-side validation evidence (loss increase after
    applying the update), also turned into a robust within-round z-score.
 5. Flag AT MOST ONE scored client per round (the max score, if > threshold).
    Flagged + invalid updates are excluded immediately from aggregation.
 6. Accumulate suspicion across rounds; at the end the top-suspicion client is
    reported only if it was flagged in >= `min_flag_rounds` rounds.

Interface for Jessie (see bottom of file for a usage example):
    defense.inspect_round(round_idx, global_state, client_states, val_deltas)
        -> RoundResult(accepted_ids=..., flagged_ids=..., ...)
    `val_deltas[i]` = val_loss(global + delta_i) - val_loss(global)   (optional)
"""
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a std-dev for Gaussian data


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    # Flag threshold on the combined anomaly score. PLACEHOLDER: choose it from
    # benign development runs with `detection_metrics.calibrate_threshold`
    # (never from test-set outcomes).
    threshold: float = 8.0
    use_validation: bool = True      # ablation switch: layer-only vs layer+val
    val_weight: float = 1.0          # weight of the validation z-score
    # Near-zero norm rule: ||u_il|| < zero_norm_frac * median(norm of others)
    zero_norm_frac: float = 1e-3
    # Safeguards for near-zero dispersion (minimum robust spread):
    log_norm_floor: float = 0.10     # ~10% relative norm tolerance (log scale)
    cos_floor: float = 0.05
    val_floor: float = 0.01          # absolute val-loss units
    # Suspicion tracking
    direct_flag_score: float = 1e3   # score given to near-zero-norm updates
    suspicion_cap: float = 50.0      # per-round cap so one huge score can't dominate
    suspicion_floor_frac: float = 0.5  # only score above this*threshold accumulates
    min_flag_rounds: int = 2         # needed before a final verdict is issued
    max_flags_per_round: int = 1     # single-attacker assumption


@dataclass
class RoundResult:
    round: int
    accepted_ids: List[int]
    flagged_ids: List[int]           # flagged by anomaly score
    invalid_ids: List[int]           # NaN/Inf/shape problems
    scores: List[float]              # combined anomaly score per client
    reasons: Dict[int, str] = field(default_factory=dict)

    @property
    def excluded_ids(self) -> List[int]:
        return sorted(set(self.flagged_ids) | set(self.invalid_ids))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _layer_of(key: str) -> str:
    """'conv1.weight' / 'conv1.bias' -> 'conv1'."""
    return key.rsplit(".", 1)[0]


def _flat_layers(delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Group a parameter-delta dict into one flat vector per layer."""
    groups: Dict[str, list] = {}
    for k, v in delta.items():
        groups.setdefault(_layer_of(k), []).append(v.flatten().float())
    return {l: torch.cat(vs) for l, vs in groups.items()}


def _robust_z(x: torch.Tensor, floor: float, signed: bool = False) -> torch.Tensor:
    """Within-round robust z-score: (x - median) / max(1.4826*MAD, floor).

    The floor is the safeguard for near-zero dispersion (e.g. all benign
    updates almost identical late in training).
    """
    med = x.median()
    mad = (x - med).abs().median() * MAD_TO_SIGMA
    z = (x - med) / torch.clamp(mad, min=floor)
    return z if signed else z.abs()


def extract_delta(global_state, client_state):
    """Return (delta_dict, None) or (None, reason) if the upload is unusable."""
    if client_state is None:
        return None, "missing_update"
    delta = {}
    for k, g in global_state.items():
        if not torch.is_floating_point(g):
            continue  # ignore integer buffers (e.g. BN num_batches_tracked)
        if k not in client_state:
            return None, f"missing_key:{k}"
        c = client_state[k]
        if tuple(c.shape) != tuple(g.shape):
            return None, f"bad_shape:{k}"
        if not torch.isfinite(c).all():
            return None, f"nan_or_inf:{k}"
        delta[k] = c.detach().cpu().float() - g.detach().cpu().float()
    return delta, None


# --------------------------------------------------------------------------- #
# Layer-aware scoring (Security workstream)
# --------------------------------------------------------------------------- #
def layer_aware_scores(deltas: List[Dict[str, torch.Tensor]], cfg: DetectorConfig):
    """Score m valid updates (m >= 3).

    Returns dict with per-client, per-layer matrices:
        norm_z [m,L], cos_z [m,L], direct [m,L] (bool), layers (names)
    """
    m = len(deltas)
    layers = list(deltas[0].keys())
    L = len(layers)
    norm = torch.zeros(m, L)
    cos = torch.full((m, L), float("nan"))
    direct = torch.zeros(m, L, dtype=torch.bool)

    for l, name in enumerate(layers):
        V = torch.stack([d[name] for d in deltas])           # [m, D]
        norm[:, l] = V.norm(dim=1)
        for a in range(m):
            others_idx = [j for j in range(m) if j != a]
            med_norm = norm[others_idx, l].median()
            # Magnitude check first: near-zero update => direct flag, no cosine.
            if med_norm > 1e-12 and norm[a, l] < cfg.zero_norm_frac * med_norm:
                direct[a, l] = True
                continue
            med_vec = V[others_idx].median(dim=0).values      # coordinate-wise median
            denom = V[a].norm() * med_vec.norm()
            if denom > 1e-12:
                cos[a, l] = torch.dot(V[a], med_vec) / denom

    # Magnitude z-score on log scale: catches both too-large and too-small.
    log_norm = torch.log(norm + 1e-12)
    norm_z = torch.stack(
        [_robust_z(log_norm[:, l], cfg.log_norm_floor) for l in range(L)], dim=1)

    # Direction z-score: only LOW cosine is suspicious (one-sided).
    cos_z = torch.zeros(m, L)
    for l in range(L):
        col = cos[:, l].clone()
        if torch.isnan(col).all():
            continue
        col[torch.isnan(col)] = torch.nanmedian(col)          # neutral fill
        cos_z[:, l] = torch.clamp(-_robust_z(col, cfg.cos_floor, signed=True), min=0)
        cos_z[direct[:, l], l] = 0.0                          # cosine not computed
    return dict(norm_z=norm_z, cos_z=cos_z, direct=direct, layers=layers)


def validation_scores(val_deltas: List[float], cfg: DetectorConfig) -> torch.Tensor:
    """One-sided robust z of validation-loss increase (higher = more harmful)."""
    v = torch.tensor(val_deltas, dtype=torch.float32)
    v = torch.nan_to_num(v, nan=1e3, posinf=1e3, neginf=-1e3)
    return torch.clamp(_robust_z(v, cfg.val_floor, signed=True), min=0)


# --------------------------------------------------------------------------- #
# Defense integration (Defense workstream)
# --------------------------------------------------------------------------- #
class FedShieldDefense:
    def __init__(self, num_clients: int = 10, cfg: Optional[DetectorConfig] = None,
                 log_path: Optional[str] = "logs/fedshield_flags.jsonl"):
        self.n = num_clients
        self.cfg = cfg or DetectorConfig()
        self.suspicion = [0.0] * num_clients
        self.flag_counts = [0] * num_clients
        self.history: List[dict] = []
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:  # fresh log per run, config on line 1
                f.write(json.dumps({"config": asdict(self.cfg)}) + "\n")

    def inspect_round(self, round_idx: int, global_state, client_states,
                      val_deltas: Optional[List[float]] = None) -> RoundResult:
        cfg, n = self.cfg, self.n
        assert len(client_states) == n, "need one state_dict (or None) per client"

        # 1-2. extract updates; reject malformed ones before any scoring
        deltas, invalid = {}, {}
        for i, st in enumerate(client_states):
            d, why = extract_delta(global_state, st)
            if d is None:
                invalid[i] = why
            else:
                deltas[i] = _flat_layers(d)
        valid_ids = [i for i in range(n) if i not in invalid]

        scores = [0.0] * n
        reasons = {i: why for i, why in invalid.items()}
        flagged: List[int] = []

        if len(valid_ids) >= 3:
            # 3. layer-aware norm/cosine scoring
            s = layer_aware_scores([deltas[i] for i in valid_ids], cfg)
            norm_score = s["norm_z"].max(dim=1)
            cos_score = s["cos_z"].max(dim=1)
            layer_score = torch.maximum(norm_score.values, cos_score.values)
            total = layer_score.clone()

            # 4. validation evidence (optional)
            val_z = torch.zeros(len(valid_ids))
            if cfg.use_validation and val_deltas is not None:
                val_z = validation_scores([val_deltas[i] for i in valid_ids], cfg)
                total = total + cfg.val_weight * val_z

            direct_any = s["direct"].any(dim=1)
            total[direct_any] = cfg.direct_flag_score

            # per-client score + dominant reason
            for a, i in enumerate(valid_ids):
                scores[i] = float(total[a])
                if direct_any[a]:
                    l = int(s["direct"][a].nonzero()[0])
                    reasons[i] = f"near_zero_norm:{s['layers'][l]}"
                else:
                    cands = {
                        f"norm:{s['layers'][int(norm_score.indices[a])]}": float(norm_score.values[a]),
                        f"cos:{s['layers'][int(cos_score.indices[a])]}": float(cos_score.values[a]),
                        "val": float(cfg.val_weight * val_z[a]),
                    }
                    reasons[i] = max(cands, key=cands.get)

            # 5. single-client flag logic (at most `max_flags_per_round`)
            order = sorted(valid_ids, key=lambda i: scores[i], reverse=True)
            for i in order[:cfg.max_flags_per_round]:
                if scores[i] > cfg.threshold:
                    flagged.append(i)
        else:
            for i in valid_ids:
                reasons.setdefault(i, "too_few_valid_updates_to_score")

        invalid_ids = sorted(invalid)
        excluded = set(flagged) | set(invalid_ids)
        accepted = [i for i in range(n) if i not in excluded]

        # 6. suspicion history (only evidence well above the benign range counts)
        base = cfg.suspicion_floor_frac * cfg.threshold
        for i in range(n):
            sc = cfg.suspicion_cap if i in invalid else min(scores[i], cfg.suspicion_cap)
            self.suspicion[i] += max(0.0, sc - base) / cfg.threshold
        for i in excluded:
            self.flag_counts[i] += 1

        res = RoundResult(round=round_idx, accepted_ids=accepted, flagged_ids=flagged,
                          invalid_ids=invalid_ids, scores=scores,
                          reasons={i: reasons[i] for i in excluded if i in reasons})
        rec = dict(round=round_idx, accepted=accepted, flagged=flagged,
                   invalid=invalid_ids, scores=[round(x, 4) for x in scores],
                   reasons=res.reasons,
                   suspicion=[round(x, 4) for x in self.suspicion])
        self.history.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return res

    def final_verdict(self) -> dict:
        """Report the malicious client only with repeated evidence."""
        top = max(range(self.n), key=lambda i: self.suspicion[i])
        ok = self.flag_counts[top] >= self.cfg.min_flag_rounds
        return dict(malicious_client=top if ok else None, top_candidate=top,
                    suspicion=list(self.suspicion), flag_counts=list(self.flag_counts))


# --------------------------------------------------------------------------- #
# Usage example 
# --------------------------------------------------------------------------- #
# defense = FedShieldDefense(num_clients=10)
# for r in range(1, 11):
#     client_states = [train_client(i, global_model) for i in range(10)]
#     val_deltas = [val_loss_after_applying(global_model, cs) - base_val_loss
#                   for cs in client_states]                       # Jessie's part
#     res = defense.inspect_round(r, global_model.state_dict(), client_states, val_deltas)
#     accepted_states = [client_states[i] for i in res.accepted_ids]
#     # -> norm-clip accepted updates, then guarded FedAvg (Jessie's part);
#     #    if accepted_states is empty, keep the previous global model.
# print(defense.final_verdict())
