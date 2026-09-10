"""Combination: per-class Gaussian prototype (GaussPTA) + compactness gate.

Composes the two representation-level ideas:

* **GaussPTA** (``models/gauss_pta.py``) — a per-class Gaussian (L2-normalized
  mean + per-dim variance) scored by a shifted Mahalanobis distance
  (variance-aware; the TCA-style upgrade of base PTA's single-mean cosine).
* **Compactness gate** (``models/compact_pta.py``) — each write's EMA weight
  is modulated by the distance ``d = 1 - cos(x, mu_c)`` to the current class
  mean (soft attenuation or hard drop), because writes far from the class
  prototype are disproportionately wrong.

The write rule is base PTA's MULTI-CLASS write (every class with
``softmax(clip)[c] >= 0.1`` receives ``w_c = 1 - exp(-p_c / T)``); before the
Gaussian EMA update, ``w_c`` is modulated by the compactness gate on the
distance to the current mean ``mu_c``:

    d = 1 - cos(x, mu_c)
    soft: w_c *= exp(-d^2 / (2 sigma_c^2))   (sigma_c = per-class running EMA)
    hard: w_c = 0 if d > mean_c + hard_k * std_c else w_c

The gated ``w_c`` drives the same Gaussian update as ``gauss_pta``:
``mu_c <- normalize((1-w_c) mu_c + w_c x)``, ``v_c <- (1-w_c) v_c + w_c (x-mu_c)^2``,
``n_c <- n_c + w_c``.  Score and fusion are identical to ``gauss_pta``
(shifted Mahalanobis, rescaled to cosine's scale).  ``gate_mode=off``
reproduces ``gauss_pta`` exactly.

``shrink`` and ``gate_mode`` are selected per-run via ``--override``.
"""

import math
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.gauss_pta import (
    init_gauss_state,
    mahalanobis_logits,
    fuse_mahalanobis,
)
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


def gauss_compact_write(image_feature, clip_logits, mu, v, n, T,
                        gate_mode, sigma_min, stats_ema, hard_k, gate_stats):
    """Base PTA multi-class write onto the Gaussian state, with a gate.

    For every class c with ``softmax(clip)[c] >= 0.1``:
        w_c = 1 - exp(-p_c / T)
        d   = 1 - cos(x, mu_c)
        gate: soft -> w_c *= exp(-d^2/2sigma^2); hard -> w_c = 0 if d > mean+k*std
        mu_c <- normalize((1-w_c) mu_c + w_c x)
        v_c  <- (1-w_c) v_c + w_c (x - mu_c)^2   (deviation from UPDATED mean)
        n_c  <- n_c + w_c

    Args:
        image_feature: (1, D) L2-normalized CLIP image embedding.
        clip_logits:   (1, C) zero-shot logits.
        mu, v:         (C, D) float32 Gaussian state.
        n:             (C,)   float32 effective counts.
        T:             EMA temperature.
        gate_mode:     "soft" | "hard" | "off".
        sigma_min:     floor on the soft-gate running sigma.
        stats_ema:     EMA rate for the per-class distance stats.
        hard_k:        hard-gate threshold multiplier (in std units).
        gate_stats:    per-class stat containers from ``init_gate_stats``.

    Returns:
        (mu_new, v_new, n_new) — new float32 tensors (inputs not mutated).
    """
    x = image_feature.reshape(-1).float()                    # (D,) fp32
    probs = F.softmax(clip_logits.reshape(-1).float(), dim=0)  # (C,) fp32

    w = torch.zeros_like(probs)                              # (C,)
    mask = probs >= 1e-1                                     # (C,) bool
    w[mask] = 1.0 - torch.exp(-probs[mask] / T)              # (K,)

    # ── Compactness gate on the per-class write weight (float32) ───────────
    if gate_mode in (GATE_SOFT, GATE_HARD):
        for c in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            st = gate_stats[c]
            d = float(1.0 - F.cosine_similarity(x, mu[c].float(), dim=0).item())
            st["n_gated"] += 1
            if gate_mode == GATE_SOFT:
                sigma = max(float(st["sigma"]), sigma_min)
                factor = math.exp(-(d * d) / (2.0 * sigma * sigma))
                if factor < 0.99:
                    st["n_modified"] += 1
                w[c] = w[c] * factor
                st["sigma"] = max(
                    (1.0 - stats_ema) * st["sigma"] + stats_ema * d, sigma_min
                )
            else:  # GATE_HARD
                mean_c = float(st["mean"])
                ema_sq_c = float(st["ema_sq"])
                std_c = math.sqrt(max(ema_sq_c - mean_c * mean_c, 0.0))
                if d > mean_c + hard_k * std_c:
                    w[c] = 0.0
                    st["n_modified"] += 1
                st["mean"] = (1.0 - stats_ema) * mean_c + stats_ema * d
                st["ema_sq"] = (1.0 - stats_ema) * ema_sq_c + stats_ema * d * d

    mu_new = mu.clone()
    v_new = v.clone()
    n_new = n.clone()
    if bool(mask.any()):
        m = w[mask]                                          # (K,)
        mu_sel = (1.0 - m.unsqueeze(1)) * mu[mask] + m.unsqueeze(1) * x
        mu_sel = mu_sel / mu_sel.norm(dim=-1, keepdim=True)
        dev2 = (x - mu_sel) ** 2                             # (K, D)
        v_sel = (1.0 - m.unsqueeze(1)) * v[mask] + m.unsqueeze(1) * dev2
        mu_new[mask] = mu_sel
        v_new[mask] = v_sel
        n_new[mask] = n_new[mask] + m
    return mu_new, v_new, n_new


class GaussCompactPTAAdapter(BaseAdapter):
    """Per-class Gaussian prototype + intra-class compactness write gate.

    gate_mode=off reproduces gauss_pta exactly.
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
        self.tau_text = float(fusion_cfg["tau_text"])
        self.tau_image_proto = float(fusion_cfg["tau_image_proto"])

        self.shrink = float(cfg.get("shrink", 0.1))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))
        self.gate_mode = str(cfg.get("gate_mode", GATE_SOFT)).lower()
        if self.gate_mode == "none":
            self.gate_mode = GATE_OFF
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ValueError(
                "gauss_compact_pta: unknown gate_mode {!r}".format(self.gate_mode)
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
                method=os.environ.get("RESULT_LABEL", "GaussCompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            mu, v, n = init_gauss_state(text_embeddings.t())
            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes
            gate_stats = init_gate_stats(num_classes, self.sigma_init)

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[GaussCompactPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                n_before = n
                mu, v, n = gauss_compact_write(
                    image_features, clip_logits, mu, v, n, self.T,
                    self.gate_mode, self.sigma_min, self.stats_ema,
                    self.hard_k, gate_stats,
                )

                logit, _ = mahalanobis_logits(image_features, mu, v, self.shrink)
                final_logits = fuse_mahalanobis(
                    clip_logits, logit, self.tau_text, self.tau_image_proto
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)
                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                if record_dir:
                    proto_stats = {}
                    proto_stats["shrink"] = self.shrink
                    proto_stats["gate_mode"] = self.gate_mode
                    for c in range(num_classes):
                        st = gate_stats[c]
                        n_c = float(n[c].item())
                        v_max = float(v[c].max().item()) if n_c > 0 else 0.0
                        var_aniso = (
                            (v_max - float(v[c].min().item())) / (v_max + 1e-8)
                            if n_c > 0 else 0.0
                        )
                        proto_stats["class_{}".format(c)] = {
                            "n": n_c,
                            "var_aniso": var_aniso,
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
                            "image_proto": logit.float().cpu().tolist(),
                            "patch_proto": None,
                            "final": final_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats=proto_stats,
                        write_gate=None,
                        write_occurred=bool((n > n_before).any().item()),
                    )

                if i % 1000 == 0:
                    print(
                        f"---- GaussCompactPTA(shrink={self.shrink},"
                        f"gate={self.gate_mode}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- GaussCompactPTA(shrink={self.shrink},gate={self.gate_mode}) "
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
                method=os.environ.get("RESULT_LABEL", "GaussCompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "GaussCompactPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> GaussCompactPTAAdapter:
    """Factory: instantiate a GaussCompactPTAAdapter with the given config."""
    return GaussCompactPTAAdapter(cfg)
