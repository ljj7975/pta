"""Prototype-Based TTA with an intra-class compactness write gate.

The write *rule* is identical to base PTA (the multi-class write of
``models/image_level/pta_image.py``): for every class ``c`` with
``softmax(clip)[c] >= 0.1`` the class prototype is EMA-updated with
``w_c = 1 - exp(-p_c / T)``.  The only difference from base PTA is that each
write's EMA weight is modulated by the geometric distance between the image
feature and the *current* class prototype:

    d_c = 1 - cos(x, prototype_state[c])

Pre-validation on archived base-PTA records showed write correctness is
strongly predicted by this distance (purity 71% -> 13% across distance
deciles on dtd; 87% -> 12% on flowers; 89% -> 38% on pets), so writes far
from the class prototype are disproportionately wrong and should be
down-weighted (soft) or dropped (hard).  This is the MCP-style intra-class
compactness filter.

Gate modes (``gate_mode``, selected per-run via ``--override``):

* **soft**: ``w_c *= exp(-d_c^2 / (2 sigma_c^2))`` where ``sigma_c`` is a
  per-class running EMA of ``d_c`` over prior writes to ``c`` (rate
  ``stats_ema``, init ``sigma_init``, floor ``sigma_min``).
* **hard**: ``w_c = 0`` if ``d_c > mean_c + hard_k * std_c`` else ``w_c``,
  where ``mean_c`` / ``ema_sq_c`` are running EMAs (rate ``stats_ema``) of
  ``d_c`` / ``d_c^2`` (init ``sigma_init`` / ``sigma_init^2``) and
  ``std_c = sqrt(max(ema_sq_c - mean_c^2, 0))``.
* **off**: gate disabled — reproduces base PTA exactly (control).

A class whose prototype row is still all-zero (never written, norm < 1e-6)
is written *ungated* with the full ``w_c`` and contributes no stats update
(the distance is undefined for a zero prototype).  Stats are updated with
``d_c`` for every gated write (kept or zeroed), after the gate decision.
All gate math runs in float32 even though the prototype flow is fp16 on GPU.

Prediction and fusion are identical to base PTA:
``refined_text = alpha*text_features + (1-alpha)*prototype_state`` L2-
normalized, ``image_proto_logits = image_features @ refined_text.T`` (same
fp16 matmul as ``models/image_level/base.py``), and
``final = WeightedFusion(tau_text=1.0, tau_image_proto=100.0,
tau_patch_proto=0.0)``.  Image-level only — no patch vote, no patch bank.
"""

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.records import (
    write_record_header,
    write_record,
    write_summary,
    write_convergence,
)

# Gate modes
GATE_SOFT = "soft"
GATE_HARD = "hard"
GATE_OFF = "off"
_VALID_GATE_MODES = (GATE_SOFT, GATE_HARD, GATE_OFF)

# Prototype rows with norm below this are treated as "never written".
ZERO_PROTO_NORM = 1e-6


def init_gate_stats(num_classes, gate_mode, sigma_init):
    """Return per-class gate-stat containers (list of dicts, index = class).

    Each entry tracks:
        n_writes_pre   — writes where the class was in the write mask
                         (w_c > 0 before the gate)
        n_writes_post  — writes with w_c > 0 after the gate
        n_gated        — gated writes (prototype already written)
        n_modified     — gated writes whose weight the gate changed
                         (soft: factor < 0.99; hard: zeroed)
        sigma          — soft mode: running EMA of d_c (init sigma_init)
        mean           — hard mode: running EMA of d_c (init sigma_init)
        ema_sq         — hard mode: running EMA of d_c^2 (init sigma_init^2)
    """
    stats = []
    for _ in range(num_classes):
        stats.append({
            "n_writes_pre": 0,
            "n_writes_post": 0,
            "n_gated": 0,
            "n_modified": 0,
            "sigma": float(sigma_init),
            "mean": float(sigma_init),
            "ema_sq": float(sigma_init) ** 2,
        })
    return stats


def compact_ema_write(
    image_feature,
    clip_logits,
    text_features,
    prototype_state,
    alpha,
    T,
    gate_mode,
    sigma_min,
    stats_ema,
    hard_k,
    gate_stats,
):
    """Base PTA multi-class EMA write with an intra-class compactness gate.

    Write rule (identical to ``PTAImageLevel.update_prototypes`` in
    models/image_level/pta_image.py): for every class ``c`` with
    ``softmax(clip_logits)[c] >= 0.1``, ``w_c = 1 - exp(-p_c / T)`` and
    ``prototype_state[c] = (1 - w_c) * prototype_state[c] + w_c * x``.

    Gate (applied per written class before the EMA update, float32 math):
        d_c = 1 - cos(x, prototype_state[c])
        * prototype row all-zero (norm < 1e-6): ungated full-weight write,
          no stats update;
        * soft: ``w_c *= exp(-d_c^2 / (2 sigma_c^2))``, sigma floored at
          ``sigma_min``;
        * hard: ``w_c = 0`` if ``d_c > mean_c + hard_k * std_c`` else kept;
        * stats updated with d_c after the gate decision.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding.
        clip_logits:     (1, C) zero-shot logits.
        text_features:   (C, D) current refined text features.
        prototype_state: (C, D) running prototype bank (mutated in-place).
        alpha:           Weight on original text features.
        T:               EMA temperature.
        gate_mode:       "soft" | "hard" | "off" (off = base PTA exactly).
        sigma_min:       Floor on the soft-gate running sigma.
        stats_ema:       EMA rate for the per-class distance stats.
        hard_k:          Hard-gate threshold multiplier (in std units).
        gate_stats:      Per-class stat containers from ``init_gate_stats``.

    Returns:
        (refined_text, prototype_state, compact_d, written) where
        ``compact_d`` maps written class -> d_c (float32; 1.0 for
        never-written classes; empty when the gate is off) and ``written``
        is the list of written class indices.
    """
    # ── Base PTA multi-class write rule ──────────────────────────────
    probs = F.softmax(clip_logits, dim=-1)          # (1, C)
    w = probs.squeeze(0)                             # (C)
    w_new = torch.zeros_like(w)                      # (C)
    mask = w >= 1e-1                                 # (C) bool
    w_new[mask] = 1 - torch.exp(-w[mask] / T)        # (C)

    written = mask.nonzero(as_tuple=True)[0].tolist()

    # ── Intra-class compactness gate (float32 math) ──────────────────
    compact_d = {}
    if gate_mode in (GATE_SOFT, GATE_HARD):
        x = image_feature.squeeze(0)
        for c in written:
            st = gate_stats[c]
            st["n_writes_pre"] += 1
            proto_c = prototype_state[c]
            if float(proto_c.norm()) < ZERO_PROTO_NORM:
                # Never written: distance undefined -> ungated full-weight
                # write, no stats update.
                compact_d[c] = 1.0
                st["n_writes_post"] += 1
                continue
            d_c = float(
                1.0
                - F.cosine_similarity(x.float(), proto_c.float(), dim=0).item()
            )
            compact_d[c] = d_c
            st["n_gated"] += 1
            if gate_mode == GATE_SOFT:
                sigma = max(float(st["sigma"]), sigma_min)
                factor = math.exp(-(d_c * d_c) / (2.0 * sigma * sigma))
                if factor < 0.99:
                    st["n_modified"] += 1
                w_new[c] = w_new[c] * factor
                # Stats update after the gate decision (floor applied).
                st["sigma"] = max(
                    (1.0 - stats_ema) * st["sigma"] + stats_ema * d_c,
                    sigma_min,
                )
            else:  # GATE_HARD
                mean_c = float(st["mean"])
                ema_sq_c = float(st["ema_sq"])
                std_c = math.sqrt(max(ema_sq_c - mean_c * mean_c, 0.0))
                if d_c > mean_c + hard_k * std_c:
                    w_new[c] = 0.0
                    st["n_modified"] += 1
                # Stats update after the gate decision.
                st["mean"] = (1.0 - stats_ema) * mean_c + stats_ema * d_c
                st["ema_sq"] = (1.0 - stats_ema) * ema_sq_c + stats_ema * d_c * d_c
            if float(w_new[c]) > 0.0:
                st["n_writes_post"] += 1
    else:
        # gate_mode == "off": base PTA exactly, no gate math.
        for c in written:
            st = gate_stats[c]
            st["n_writes_pre"] += 1
            st["n_writes_post"] += 1

    # ── EMA update (identical to base PTA) ────────────────────────────
    w_new = w_new.unsqueeze(1)                       # (C, 1)
    prototype_state[mask] = (
        (1 - w_new[mask]) * prototype_state[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )

    # ── Refined text (identical to base PTA) ──────────────────────────
    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

    return refined_text, prototype_state, compact_d, written


class CompactPTAAdapter(BaseAdapter):
    """PTA with an intra-class compactness write gate.

    Prediction rule and fusion are identical to base PTA; the write rule is
    base PTA's multi-class EMA write with each write's weight modulated by
    the distance to the current class prototype (soft attenuation or hard
    drop).  With ``gate_mode=off`` this reproduces base PTA exactly.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── Backward-compatible nested config ──────────────────────────────
        # Flat configs (alpha, T at root level) are still used by some
        # callers.  Propagate them into the nested "image_level" sub-dict
        # so the PTAImageLevel component can find them.
        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        # ── Image-level prototype component ────────────────────────────────
        self.image_level = create_image_level(cfg)

        # ── Fusion component (identical to base PTA) ───────────────────────
        # Defaults: tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0
        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        # ── Compactness gate parameters (flat config, --override friendly) ─
        self.gate_mode = str(cfg.get("gate_mode", GATE_SOFT)).lower()
        if self.gate_mode == "none":
            self.gate_mode = GATE_OFF
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ValueError(
                "compact_pta: unknown gate_mode {!r} "
                "(expected 'soft', 'hard', or 'off')".format(self.gate_mode)
            )
        self.sigma_init = float(cfg.get("sigma_init", 0.1))
        self.sigma_min = float(cfg.get("sigma_min", 0.05))
        self.stats_ema = float(cfg.get("stats_ema", 0.05))
        self.hard_k = float(cfg.get("hard_k", 0.5))
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        # ── ENV-GATED PER-SAMPLE RECORDING (behavior-neutral) ─────────────
        # When RECORD_DIR is set, write the JSONL header + per-sample records
        # + summary + convergence via utils.records. When unset/empty the
        # writers are strict no-ops; everything below is skipped entirely so
        # no tensor, RNG, or loop-order state is perturbed. RESULT_LABEL /
        # SEED only affect the recorded metadata, never the computation.
        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "CompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            refine_feature = text_embeddings.t()            # [C, D] fp16
            target_prototype = self.image_level.init_state(refine_feature)

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            # Per-class compactness-gate stats (fresh per run).
            gate_stats = init_gate_stats(
                num_classes, self.gate_mode, self.sigma_init
            )

            # Support early termination via MAX_BATCHES env var
            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[CompactPTA] {dataset_name}")
            ):
                # Early termination check
                if max_batches is not None and i >= max_batches:
                    break

                # ── ZERO-SHOT PREDICTION ───────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── WRITE (rule = base PTA multi-class; each write's weight
                #    gated by the distance to the current class prototype) ──
                refine_feature, target_prototype, compact_d, written = (
                    compact_ema_write(
                        image_features,
                        clip_logits,
                        refine_feature,
                        target_prototype,
                        self.alpha,
                        self.T,
                        self.gate_mode,
                        self.sigma_min,
                        self.stats_ema,
                        self.hard_k,
                        gate_stats,
                    )
                )

                # ── FUSED PREDICTION (identical to base PTA) ───────────────
                image_proto_logits = self.image_level.compute_logits(
                    image_features, refine_feature
                )
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                # ── MEASURE ACCURACY ───────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # Per-class accuracy accumulation (consumed by summary below).
                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                # ── RECORD PER-SAMPLE (env-gated, behavior-neutral) ───────
                if record_dir:
                    per_class_stats = {}
                    for c in range(num_classes):
                        st = gate_stats[c]
                        pre = st["n_writes_pre"]
                        post = st["n_writes_post"]
                        mean_d = (
                            st["sigma"] if self.gate_mode == GATE_SOFT else st["mean"]
                        )
                        per_class_stats["class_{}".format(c)] = {
                            "n_writes_pre": pre,
                            "n_writes_post": post,
                            "write_rate": (post / pre) if pre else 0.0,
                            "mean_d": mean_d,
                            "gate_fire_rate": (
                                st["n_modified"] / st["n_gated"]
                            ) if st["n_gated"] else 0.0,
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
                        # write_record (utils/records.py) has a fixed signature
                        # with no compact_d kwarg, so the per-written-class
                        # distances for this sample ride along in proto_stats.
                        proto_stats={
                            "true": None,
                            "pred": None,
                            **per_class_stats,
                            "compact_d": {
                                str(c): float(d) for c, d in compact_d.items()
                            },
                        },
                        write_gate=None,
                        write_occurred=len(written) > 0,
                    )

                # Periodic logging (every 1000 samples)
                if i % 1000 == 0:
                    print(
                        f"---- CompactPTA(gate_mode={self.gate_mode}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        # ── FINAL RESULTS ──────────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- CompactPTA(gate_mode={self.gate_mode}) "
            f"test accuracy: {final_acc:.2f}. ----\n"
        )

        # ── SUMMARY (env-gated, behavior-neutral) ──────────────────────────
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
                method=os.environ.get("RESULT_LABEL", "CompactPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        # Append results to output file (append mode, multiple runs accumulate)
        label = os.environ.get("RESULT_LABEL", "CompactPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> CompactPTAAdapter:
    """Factory: instantiate a CompactPTAAdapter with the given config."""
    return CompactPTAAdapter(cfg)
