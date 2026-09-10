"""Combination: proper GROWING bank (BankV2) + intra-class compactness gate.

This composes the two strongest representation-level ideas from the
representation-upgrade study:

* **BankV2** (``models/bank_v2_pta.py``) — a per-class GROWING bank of K
  prototypes (anchor + threshold-gated growth + least-used recycling) that
  fixes the v1 bank's rich-get-richer collapse.
* **Compactness gate** (``models/compact_pta.py``) — each write's EMA weight
  is modulated by the geometric distance ``d = 1 - cos(x, proto)`` to the
  current prototype (soft attenuation or hard drop), because writes far from
  the class prototype are disproportionately wrong (M2 pre-validation).

How they combine: the growing-bank routing is unchanged (nearest active
prototype; if the best similarity is below ``create_threshold`` the feature is
a *new mode* and grows/recycles a slot **ungated** — creating a new mode is a
deliberate action).  When the feature is *absorbed* into an existing slot
``k*``, the compactness gate modulates the EMA weight ``w_c`` by the distance
to that slot:

    d = 1 - cos(x, P[c, k*])
    soft: w_c *= exp(-d^2 / (2 sigma_c^2))   (sigma_c = per-class running EMA)
    hard: w_c = 0 if d > mean_c + hard_k * std_c else w_c

So the gate tames the one remaining drag in the bank — an outlier that is
close enough to be absorbed but far enough to pull the prototype off its mode
— while the new-mode branch still captures genuinely distinct modes.  A slot
that is still all-zero (never written) is written ungated with full weight and
contributes no stats (distance undefined), exactly as in ``compact_pta``.

Score = max-of-active cosine; fusion = exact PTA mirror.  K=1 with
``gate_mode=off`` reproduces base PTA exactly.

``K``, ``create_threshold``, and ``gate_mode`` are selected via ``--override``.
"""

import math
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.records import (
    write_record_header,
    write_record,
    write_summary,
    write_convergence,
)

GATE_SOFT = "soft"
GATE_HARD = "hard"
GATE_OFF = "off"
_VALID_GATE_MODES = (GATE_SOFT, GATE_HARD, GATE_OFF)
ZERO_PROTO_NORM = 1e-6


def refine_bank(text_norm, prototype_state, alpha):
    """Per-prototype refined text (base PTA blend); zero rows -> text anchor."""
    blended = alpha * text_norm.unsqueeze(1) + (1.0 - alpha) * prototype_state
    norms = blended.norm(dim=-1, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    return blended / safe


def init_bank(text_features, K):
    """Growing-bank init: zero prototypes + active=1 + zero usage."""
    C, D = text_features.shape
    P = torch.zeros(C, K, D, dtype=torch.float32, device=text_features.device)
    active = torch.ones(C, dtype=torch.int32, device=text_features.device)
    usage = torch.zeros(C, K, dtype=torch.float32, device=text_features.device)
    return P, active, usage


def init_gate_stats(num_classes, sigma_init):
    """Per-class compactness-gate stat containers (index = class)."""
    stats = []
    for _ in range(num_classes):
        stats.append({
            "n_gated": 0,
            "n_modified": 0,
            "sigma": float(sigma_init),
            "mean": float(sigma_init),
            "ema_sq": float(sigma_init) ** 2,
        })
    return stats


def bank_compact_write(image_feature, clip_logits, text_norm, P, active, usage,
                       alpha, T, create_threshold, gate_mode, sigma_min,
                       stats_ema, hard_k, gate_stats):
    """Growing-bank write with a compactness gate on the absorb branch.

    Routing is identical to ``bank_v2_write`` (nearest active prototype; new
    mode grows/recycles a slot **ungated**).  When the feature is absorbed
    into slot ``k*``, the EMA weight is modulated by the distance to that slot
    (soft attenuation or hard drop); a never-written (zero) slot is written
    ungated with full weight and contributes no stats.

    Returns:
        ``writes`` — list of ``(class_idx, slot, w_used, action)`` where
        ``action`` is ``"absorb" | "grow" | "recycle"``.
    """
    w = F.softmax(clip_logits, dim=-1).float().squeeze(0)   # (C)
    mask = w >= 1e-1                                        # (C) bool
    w_new = torch.zeros_like(w)                             # (C)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)               # (C)

    x = image_feature.squeeze(0).float()                    # (D)
    C, K, _ = P.shape
    writes = []
    for c in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        a = int(active[c])
        refine_c = refine_bank(text_norm[c:c + 1], P[c, :a].unsqueeze(0), alpha)[0]
        sims = x @ refine_c.T                               # (a,)
        k_star = int(sims.argmax())
        if K > 1 and float(sims[k_star]) < create_threshold:
            # New mode: grow or recycle (UNGATED — deliberate creation).
            if a < K:
                P[c, a] = x
                active[c] = a + 1
                usage[c, a] = 1.0
                writes.append((c, a, float(w_new[c]), "grow"))
            else:
                k_rec = 1 + int(usage[c, 1:K].argmin())
                P[c, k_rec] = x
                usage[c, k_rec] = 1.0
                writes.append((c, k_rec, float(w_new[c]), "recycle"))
            continue

        # Absorb branch: modulate the EMA weight with the compactness gate.
        w_c = float(w_new[c])
        proto_c = P[c, k_star]
        if float(proto_c.norm()) < ZERO_PROTO_NORM:
            # Never written: distance undefined -> ungated full-weight write.
            d = 1.0
        else:
            st = gate_stats[c]
            d = float(1.0 - F.cosine_similarity(x, proto_c.float(), dim=0).item())
            st["n_gated"] += 1
            if gate_mode == GATE_SOFT:
                sigma = max(float(st["sigma"]), sigma_min)
                factor = math.exp(-(d * d) / (2.0 * sigma * sigma))
                if factor < 0.99:
                    st["n_modified"] += 1
                w_c = w_c * factor
                st["sigma"] = max(
                    (1.0 - stats_ema) * st["sigma"] + stats_ema * d, sigma_min
                )
            elif gate_mode == GATE_HARD:
                mean_c = float(st["mean"])
                ema_sq_c = float(st["ema_sq"])
                std_c = math.sqrt(max(ema_sq_c - mean_c * mean_c, 0.0))
                if d > mean_c + hard_k * std_c:
                    w_c = 0.0
                    st["n_modified"] += 1
                st["mean"] = (1.0 - stats_ema) * mean_c + stats_ema * d
                st["ema_sq"] = (1.0 - stats_ema) * ema_sq_c + stats_ema * d * d
        P[c, k_star] = (1.0 - w_c) * P[c, k_star] + w_c * x
        usage[c, k_star] += 1.0
        writes.append((c, k_star, w_c, "absorb"))
    return writes


def bank_compact_score(image_feature, text_norm, P, active, alpha):
    """Max-of-active base-PTA-style cosine score per class (float32)."""
    x = image_feature.squeeze(0).float()
    C = P.shape[0]
    logits = torch.zeros(C, dtype=torch.float32, device=P.device)
    for c in range(C):
        a = int(active[c])
        refine_c = refine_bank(text_norm[c:c + 1], P[c, :a].unsqueeze(0), alpha)[0]
        logits[c] = float((x @ refine_c.T).max())
    return logits


class BankCompactPTAAdapter(BaseAdapter):
    """Proper growing bank + intra-class compactness write gate (combination).

    K=1 with gate_mode=off reproduces base PTA exactly.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        self.K = int(cfg.get("K", 3))
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))
        self.create_threshold = float(cfg.get("create_threshold", 0.85))
        self.gate_mode = str(cfg.get("gate_mode", GATE_SOFT)).lower()
        if self.gate_mode == "none":
            self.gate_mode = GATE_OFF
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ValueError(
                "bank_compact_pta: unknown gate_mode {!r}".format(self.gate_mode)
            )
        self.sigma_init = float(cfg.get("sigma_init", 0.1))
        self.sigma_min = float(cfg.get("sigma_min", 0.05))
        self.stats_ema = float(cfg.get("stats_ema", 0.05))
        self.hard_k = float(cfg.get("hard_k", 0.5))

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "BankCompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            text_features = text_embeddings.t()
            P, active, usage = init_bank(text_features, self.K)
            text_norm = F.normalize(text_features.float(), dim=-1)

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes
            gate_stats = init_gate_stats(num_classes, self.sigma_init)

            n_updates = [0] * num_classes
            k_usage = [[0] * self.K for _ in range(num_classes)]
            n_grow = [0] * num_classes
            n_recycle = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[BankCompactPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                x = image_features.squeeze(0).float()
                writes = bank_compact_write(
                    x.unsqueeze(0), clip_logits.float(), text_norm, P, active,
                    usage, self.alpha, self.T, self.create_threshold,
                    self.gate_mode, self.sigma_min, self.stats_ema, self.hard_k,
                    gate_stats,
                )

                logit_c = bank_compact_score(x, text_norm, P, active, self.alpha)
                image_proto_logits = logit_c.half().unsqueeze(0)
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)
                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                for c, k_slot, _w_c, action in writes:
                    n_updates[c] += 1
                    k_usage[c][k_slot] += 1
                    if action == "grow":
                        n_grow[c] += 1
                    elif action == "recycle":
                        n_recycle[c] += 1

                if record_dir:
                    proto_stats = {}
                    proto_stats["K"] = self.K
                    proto_stats["create_threshold"] = self.create_threshold
                    proto_stats["gate_mode"] = self.gate_mode
                    for c in range(num_classes):
                        st = gate_stats[c]
                        cos_to_text = P[c] @ text_norm[c]
                        proto_stats["class_{}".format(c)] = {
                            "n_updates": int(n_updates[c]),
                            "active_k": int(active[c]),
                            "moved_k": int((cos_to_text < 0.99).sum()),
                            "n_grow": int(n_grow[c]),
                            "n_recycle": int(n_recycle[c]),
                            "gate_fire_rate": (
                                st["n_modified"] / st["n_gated"]
                            ) if st["n_gated"] else 0.0,
                            "mean_d": (
                                st["sigma"] if self.gate_mode == GATE_SOFT
                                else st["mean"]
                            ),
                        }
                    write_record(
                        batch_idx=i,
                        target=int(target.item()),
                        pred=int(final_logits.argmax(dim=-1).item()),
                        correct=bool(acc),
                        conf=float(F.softmax(final_logits, dim=-1).max().item()),
                        quality_gate=None,
                        gate_mode=self.gate_mode,
                        proto_alpha=None,
                        logits={
                            "clip": clip_logits.squeeze(0).float().cpu().tolist(),
                            "image_proto": image_proto_logits.squeeze(0).float().cpu().tolist(),
                            "patch_proto": None,
                            "final": final_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats=proto_stats,
                        write_gate=None,
                        write_occurred=bool(len(writes) > 0),
                    )

                if i % 1000 == 0:
                    print(
                        f"---- BankCompactPTA(K={self.K},gate={self.gate_mode}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- BankCompactPTA(K={self.K},gate={self.gate_mode}) "
            f"test accuracy: {final_acc:.2f}. ----\n"
        )

        if record_dir:
            per_class = {}
            for c in range(num_classes):
                total_c = cls_total[c]
                correct_c = cls_correct[c]
                per_class["class_{}".format(c)] = {
                    "total": total_c,
                    "correct": correct_c,
                    "acc": (100.0 * correct_c / total_c) if total_c else 0.0,
                }
            write_summary(
                method=os.environ.get("RESULT_LABEL", "BankCompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "BankCompactPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> BankCompactPTAAdapter:
    """Factory: instantiate a BankCompactPTAAdapter with the given config."""
    return BankCompactPTAAdapter(cfg)
