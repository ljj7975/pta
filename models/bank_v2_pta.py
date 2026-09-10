"""Prototype-Based TTA with a per-class GROWING bank of prototypes (v2).

This is the *proper* re-implementation of the per-class prototype bank
(``models/bank_pta.py``).  The v1 bank collapsed to a single active
prototype per class (rich-get-richer): it zero-initialized all K slots, so
every slot's refined text was the *same* class-text anchor and
``k* = argmax_k cos(x, refine[c,k])`` was a permanent tie won by slot 0; the
always-write-nearest rule then meant no data could ever reach slots 1..K-1.

The fix follows the anti-collapse pattern already proven in this repo's
patch-level bank (``utils/kmeans.py::_incremental_kmeans_step`` +
``models/patch_level/gaussian_patch.py``): a point may only *join* an
existing prototype if it is similar enough; otherwise it *spawns* a new one,
and when the bank is full the least-used slot is recycled.  Concretely, each
class keeps K slots:

* **slot 0 — the anchor**: starts at ZERO (exactly like base PTA's
  prototype), is writable (absorbed into), and is NEVER recycled.  It plays
  the role of base PTA's single prototype (the dominant mode, text-anchored).
* **slots 1..K-1 — specialized modes**: created on demand when an incoming
  feature is dissimilar to every active prototype, EMA-updated like the
  anchor, and recyclable (least-used slot is reset when the bank is full).

Write rule — base PTA's MULTI-CLASS write (``PTAImageLevel.update_prototypes``):
every class ``c`` with ``softmax(clip)[c] >= 0.1`` receives
``w_c = 1 - exp(-p_c / T)``.  For each written class the feature is routed:

    a  = active[c]                      # number of created slots (1..K)
    refine[c, :a] = normalize(alpha*text + (1-alpha)*P[c, :a])
    k* = argmax_k cos(x, refine[c, k])
    if K > 1 and cos(x, refine[c, k*]) < create_threshold:
        # genuinely new mode
        if a < K:   P[c, a] = x;  active[c] = a + 1;  usage[c, a] = 1   # grow
        else:       k_r = 1 + argmin(usage[c, 1:K]); P[c, k_r] = x; usage[c, k_r] = 1
    else:
        P[c, k*] = (1 - w_c) * P[c, k*] + w_c * x;  usage[c, k*] += 1   # absorb

A newly created / recycled slot is set to the raw feature ``x`` (full
strength), mirroring the repo's "new center = mean of the unmatched group"
rule (a single-member group is the member itself).  The anchor (slot 0) is
excluded from recycling so the class always retains its text-anchored default.

Score: ``logit_c = max_k cos(x, refine[c, k])`` over the ``active[c]`` slots
(float32, cast to fp16 for the fusion/record path).

**K=1 reproduces base PTA exactly**: with K=1 the ``K > 1`` growth/recycle
branch never fires, so every write absorbs into the single slot 0
(``P[c,0] = (1-w_c) P[c,0] + w_c x``) and the score is
``cos(x, normalize(0.01*text + 0.99*P[c,0]))`` — bit-for-bit base PTA
dynamics (modulo the fp16/fp32 EMA precision of the GPU path, identical to
v1's ~1pp K=1 gap).

Fusion: exact mirror of base PTA — ``final = 1.0 * clip + 100.0 * logit_c``.

``K`` and ``create_threshold`` are selected per-run via ``--override``.
"""

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


def refine_bank(text_norm, prototype_state, alpha):
    """Per-prototype refined text, exactly mirroring base PTA's blend.

    For every (c, k): ``refine[c, k] = normalize(alpha * text_norm[c] +
    (1 - alpha) * P[c, k])``.  A zero (never-written) row reduces to
    ``text_norm[c]`` — a never-written prototype scores exactly like the
    class text embedding (base PTA's initial state).

    Args:
        text_norm:       (C, D) L2-normalized class text embeddings (fp32).
        prototype_state: (C, K, D) prototype bank (fp32, may contain zeros).
        alpha:           weight on the original text (base PTA default 0.01).

    Returns:
        (C, K, D) fp32 L2-normalized refined-text tensor.
    """
    blended = alpha * text_norm.unsqueeze(1) + (1.0 - alpha) * prototype_state
    norms = blended.norm(dim=-1, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    return blended / safe


def init_bank_v2(text_features, K):
    """Initialize the growing bank: zero prototypes + active=1 + zero usage.

    Slot 0 (the anchor) starts at ZERO exactly like base PTA's prototype;
    ``active[c]=1`` means only the anchor exists initially and slots 1..K-1
    are created on demand.  ``usage`` counts writes per slot (drives the
    least-used recycling).

    Args:
        text_features: (C, D) class text embeddings (``text_embeddings.t()``).
        K:             number of prototype slots per class.

    Returns:
        (P, active, usage):
            P:      (C, K, D) float32 zeros.
            active: (C,)     int32, all 1 (only the anchor is active).
            usage:  (C, K)   float32 zeros (per-slot write counts).
    """
    C, D = text_features.shape
    P = torch.zeros(C, K, D, dtype=torch.float32, device=text_features.device)
    active = torch.ones(C, dtype=torch.int32, device=text_features.device)
    usage = torch.zeros(C, K, dtype=torch.float32, device=text_features.device)
    return P, active, usage


def bank_v2_write(image_feature, clip_logits, text_norm, P, active, usage,
                  alpha, T, create_threshold):
    """Multi-class growing-bank write (anchor + threshold-gated growth).

    Write rule mirrors ``PTAImageLevel.update_prototypes`` (every class with
    ``softmax >= 0.1`` writes with ``w_c = 1 - exp(-p_c / T)``).  Each written
    class routes its feature to the nearest active prototype (measured against
    the base-PTA-style refine); if that best similarity is below
    ``create_threshold`` the feature is a *new mode* and either grows a fresh
    slot (if room) or recycles the least-used specialized slot.  Otherwise the
    nearest slot is EMA-updated (no re-normalization — base PTA's prototype is
    a raw EMA; normalization happens inside ``refine`` at scoring time).

    Args:
        image_feature:    (1, D) L2-normalized CLIP image embedding.
        clip_logits:      (1, C) zero-shot logits (before softmax).
        text_norm:        (C, D) L2-normalized class text embeddings (fp32).
        P:                (C, K, D) prototype bank (mutated in-place, fp32).
        active:           (C,) int32 active-slot counts (mutated in-place).
        usage:            (C, K) fp32 per-slot write counts (mutated in-place).
        alpha:            weight on the original text (base PTA default 0.01).
        T:                EMA temperature.
        create_threshold: best-similarity below which a feature is a new mode.

    Returns:
        ``writes`` — list of ``(class_idx, slot, w_c, action)`` tuples where
        ``action`` is one of ``"absorb" | "grow" | "recycle"``.
    """
    w = F.softmax(clip_logits, dim=-1).float().squeeze(0)   # (C)
    mask = w >= 1e-1                                        # (C) bool
    w_new = torch.zeros_like(w)                             # (C)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)               # (C)

    x = image_feature.squeeze(0).float()                    # (D)
    C, K, _ = P.shape
    writes = []
    for c in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        a = int(active[c])                                  # active slots (1..K)
        refine_c = refine_bank(text_norm[c:c + 1], P[c, :a].unsqueeze(0), alpha)[0]
        sims = x @ refine_c.T                               # (a,)
        k_star = int(sims.argmax())
        if K > 1 and float(sims[k_star]) < create_threshold:
            # Genuinely new mode relative to every active prototype.
            if a < K:
                # Grow: create a fresh specialized slot = the feature itself.
                P[c, a] = x
                active[c] = a + 1
                usage[c, a] = 1.0
                writes.append((c, a, float(w_new[c]), "grow"))
            else:
                # Bank full: recycle the least-used SPECIALIZED slot (1..K-1).
                # The anchor (slot 0) is never recycled.
                k_rec = 1 + int(usage[c, 1:K].argmin())
                P[c, k_rec] = x
                usage[c, k_rec] = 1.0
                writes.append((c, k_rec, float(w_new[c]), "recycle"))
        else:
            # Absorb into the nearest active slot (base PTA EMA, no re-norm).
            P[c, k_star] = (1.0 - w_new[c]) * P[c, k_star] + w_new[c] * x
            usage[c, k_star] += 1.0
            writes.append((c, k_star, float(w_new[c]), "absorb"))
    return writes


def bank_v2_score(image_feature, text_norm, P, active, alpha):
    """Max-of-active base-PTA-style cosine score per class.

    ``logit_c = max_k cos(x, refine[c, k])`` over the ``active[c]`` slots.
    For K=1 this is exactly base PTA's ``cos(x, refined_text)``.  Computed in
    float32; the caller casts to fp16 for the fusion/record path.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding.
        text_norm:       (C, D) L2-normalized class text embeddings (fp32).
        P:               (C, K, D) prototype bank (fp32).
        active:          (C,) int32 active-slot counts.
        alpha:           weight on the original text (base PTA default 0.01).

    Returns:
        (C,) float32 tensor of per-class max-of-active cosine scores.
    """
    x = image_feature.squeeze(0).float()                    # (D)
    C = P.shape[0]
    logits = torch.zeros(C, dtype=torch.float32, device=P.device)
    for c in range(C):
        a = int(active[c])
        refine_c = refine_bank(text_norm[c:c + 1], P[c, :a].unsqueeze(0), alpha)[0]
        logits[c] = float((x @ refine_c.T).max())
    return logits


class BankV2PTAAdapter(BaseAdapter):
    """PTA with a per-class GROWING bank of K prototypes (proper v2).

    Slot 0 is a text-anchored anchor (writable, never recycled); slots 1..K-1
    are specialized modes created on threshold-gated growth and recycled
    (least-used) when the bank is full.  Write = base PTA multi-class EMA;
    score = max-of-active cosines; fusion = exact PTA mirror.  K=1 reproduces
    base PTA exactly.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── Backward-compatible nested config ──────────────────────────────
        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        # ── Fusion component (exact PTA mirror) ─────────────────────────────
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

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "BankV2PTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            text_features = text_embeddings.t()            # [C, D] fp16
            P, active, usage = init_bank_v2(text_features, self.K)
            text_norm = F.normalize(text_features.float(), dim=-1)  # [C, D] fp32

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            # Per-class action counters for the per-sample records.
            n_updates = [0] * num_classes
            k_usage = [[0] * self.K for _ in range(num_classes)]
            n_grow = [0] * num_classes
            n_recycle = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[BankV2PTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Write (growing bank; state is float32, mutated in-place) ─
                x = image_features.squeeze(0).float()      # (D,) fp32
                writes = bank_v2_write(
                    x.unsqueeze(0), clip_logits.float(), text_norm, P,
                    active, usage, self.alpha, self.T, self.create_threshold,
                )

                # ── Score (max-of-active cosine) + fused prediction ─────────
                logit_c = bank_v2_score(x, text_norm, P, active, self.alpha)
                image_proto_logits = logit_c.half().unsqueeze(0)  # [1, C] fp16
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
                    for c in range(num_classes):
                        cos_to_text = P[c] @ text_norm[c]  # (K,)
                        proto_stats["class_{}".format(c)] = {
                            "n_updates": int(n_updates[c]),
                            "active_k": int(active[c]),
                            "moved_k": int((cos_to_text < 0.99).sum()),
                            "n_grow": int(n_grow[c]),
                            "n_recycle": int(n_recycle[c]),
                            "top_k_usage": (
                                int(max(range(self.K), key=lambda k: k_usage[c][k]))
                                if n_updates[c]
                                else 0
                            ),
                        }
                    write_record(
                        batch_idx=i,
                        target=int(target.item()),
                        pred=int(final_logits.argmax(dim=-1).item()),
                        correct=bool(acc),
                        conf=float(F.softmax(final_logits, dim=-1).max().item()),
                        quality_gate=None,
                        gate_mode=None,
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
                        f"---- BankV2PTA(K={self.K},ct={self.create_threshold}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- BankV2PTA(K={self.K},ct={self.create_threshold}) "
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
                method=os.environ.get("RESULT_LABEL", "BankV2PTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "BankV2PTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> BankV2PTAAdapter:
    """Factory: instantiate a BankV2PTAAdapter with the given config."""
    return BankV2PTAAdapter(cfg)
