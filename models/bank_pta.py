"""Prototype-Based TTA with a per-class bank of K prototypes.

Replaces base PTA's single-mean per-class prototype with a bank of K
prototypes per class (online k-means style). Pre-validation on a one-off
feature dump (a favorable full-stream proxy) showed a K-representative bank
scored by max-of-K cosine beats the single-mean cosine on all 3 datasets:

    K=3: dtd +12.4pp, flowers +2.1pp, pets +1.5pp
    K=5: dtd +18.0pp, flowers +2.4pp, pets +3.0pp

because CLIP features within a class are multi-modal (e.g. texture
orientations on dtd) and a single mean is a blurry, poor representative.

Write rule — identical to base PTA's MULTI-CLASS write
(``PTAImageLevel.update_prototypes``, models/image_level/pta_image.py):
every class ``c`` with ``softmax(clip_logits)[c] >= 0.1`` receives
``w_c = 1 - exp(-p_c / T)`` (``T`` from config, default 20.0). The only
difference from base PTA is where the weight lands: instead of blending into
one mean prototype, the written feature updates ONLY the nearest prototype
of that class, where "nearest" is measured against the base-PTA-style
refined text ``refine[c, k] = normalize(alpha*text + (1-alpha)*P[c, k])``:

    k* = argmax_k cos(x, refine[c, k])
    P[c, k*] = (1 - w_c) * P[c, k*] + w_c * x      (NO re-normalization)

(all other prototypes of ``c`` unchanged).

State: ``P`` of shape ``[C, K, D]`` (float32), initialized at ZERO — exactly
like base PTA's prototype (``PTAImageLevel.init_state`` returns zeros). The
text embedding enters only through the ``refine`` blend at scoring time, so a
never-written prototype scores like the class text: this is what makes K=1
reproduce base PTA bit-for-bit in dynamics (the earlier text-initialized
variant did NOT match base PTA and scored ~25% on dtd vs 47.7%).

Score: ``logit_c = max_k cos(x, refine[c, k])`` (computed in float32, cast
to fp16 for the fusion/record path). For K=1 this is exactly base PTA's
``cos(x, refined_text)``.

Fusion: an exact mirror of base PTA — ``final = 1.0 * clip + 100.0 *
logit_c`` (``WeightedFusion`` with tau_text=1.0, tau_image_proto=100.0,
tau_patch_proto=0.0). No patch vote, no patch bank, no patch-level signal.

``K`` is selected per-run via ``--override`` (e.g. ``--override K=5``).
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
    (1 - alpha) * P[c, k])``.  When ``P[c, k]`` is the zero vector this
    reduces to ``normalize(alpha * text_norm[c]) = text_norm[c]`` — i.e. a
    never-written prototype scores exactly like the class text embedding,
    which is precisely base PTA's initial state (its prototype is zero and
    ``refined_text = alpha * text`` normalizes to the text embedding).

    Args:
        text_norm:       (C, D) L2-normalized class text embeddings (fp32).
        prototype_state: (C, K, D) prototype bank (fp32, may contain zeros).
        alpha:           weight on the original text (base PTA default 0.01).

    Returns:
        (C, K, D) fp32 L2-normalized refined-text tensor.
    """
    blended = alpha * text_norm.unsqueeze(1) + (1.0 - alpha) * prototype_state
    # Rows that are still all-zero (never written) normalize to zero; guard
    # against divide-by-zero by normalizing only non-zero rows.
    norms = blended.norm(dim=-1, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    return blended / safe


def init_bank(text_features, K):
    """Initialize the prototype bank at ZERO, matching base PTA's prototype.

    Base PTA's ``PTAImageLevel.init_state`` returns a zero prototype; the
    text embedding enters only through the ``refine = normalize(alpha*text +
    (1-alpha)*proto)`` blend at scoring time.  Initializing here at zero (not
    at the text embedding) is what makes K=1 reproduce base PTA exactly.

    Args:
        text_features: (C, D) class text embeddings (``text_embeddings.t()``).
        K:             number of prototypes per class.

    Returns:
        (C, K, D) float32 zero tensor.
    """
    C, D = text_features.shape
    return torch.zeros(C, K, D, dtype=torch.float32, device=text_features.device)


def bank_ema_write(image_feature, clip_logits, text_norm, prototype_state, alpha, T):
    """Multi-class nearest-prototype EMA write (online k-means style).

    Write rule mirrors ``PTAImageLevel.update_prototypes``
    (models/image_level/pta_image.py): every class ``c`` with
    ``softmax(clip_logits)[c] >= 0.1`` receives
    ``w_c = 1 - exp(-p_c / T)``. Each written class updates ONLY its nearest
    prototype, where "nearest" is measured against the base-PTA-style
    refined text ``refine[c, k] = normalize(alpha*text + (1-alpha)*P[c, k])``:

        k* = argmax_k cos(x, refine[c, k])
        P[c, k*] = (1 - w_c) * P[c, k*] + w_c * x      (NO re-normalization)

    The prototype is NOT re-normalized after the update — base PTA's
    prototype is a raw EMA and the normalization happens inside the
    ``refine`` blend at scoring time.  All other prototypes of the written
    class (and all prototypes of unwritten classes) are left unchanged.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding.
        clip_logits:     (1, C) zero-shot logits (before softmax).
        text_norm:       (C, D) L2-normalized class text embeddings (fp32).
        prototype_state: (C, K, D) prototype bank (mutated in-place, fp32).
        alpha:           weight on the original text (base PTA default 0.01).
        T:               EMA temperature.

    Returns:
        ``writes`` — a list of ``(class_idx, k_star, w_c)`` tuples for the
        written classes. The bank itself is mutated in-place (the caller
        keeps holding it).
    """
    w = F.softmax(clip_logits, dim=-1).float().squeeze(0)   # (C)
    mask = w >= 1e-1                                        # (C) bool

    # Update weights: w_new = 1 - exp(-w / T)  for w >= 0.1
    w_new = torch.zeros_like(w)                             # (C)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)               # (C)

    x = image_feature.squeeze(0).float()                    # (D)
    writes = []
    for c in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        # Nearest prototype, measured against the base-PTA-style refine.
        refine_c = refine_bank(text_norm[c:c + 1], prototype_state[c:c + 1], alpha)[0]
        k_star = int((x @ refine_c.T).argmax())
        prototype_state[c, k_star] = (
            (1.0 - w_new[c]) * prototype_state[c, k_star]
            + w_new[c] * x
        )
        writes.append((c, k_star, float(w_new[c])))
    return writes


def bank_score(image_feature, text_norm, prototype_state, alpha):
    """Max-of-K base-PTA-style cosine score per class.

    ``logit_c = max_k cos(x, refine[c, k])`` where
    ``refine[c, k] = normalize(alpha*text + (1-alpha)*P[c, k])``.  For K=1
    this is exactly base PTA's ``cos(x, refined_text)``.  Computed in
    float32; the caller casts to fp16 for the fusion/record path.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding.
        text_norm:       (C, D) L2-normalized class text embeddings (fp32).
        prototype_state: (C, K, D) prototype bank (fp32).
        alpha:           weight on the original text (base PTA default 0.01).

    Returns:
        (C,) float32 tensor of per-class max-of-K cosine scores.
    """
    x = image_feature.squeeze(0).float()                    # (D)
    refine = refine_bank(text_norm, prototype_state, alpha)  # (C, K, D)
    sims = torch.einsum("d,ckd->ck", x, refine)             # (C, K)
    return sims.max(dim=-1).values                          # (C,)


class BankPTAAdapter(BaseAdapter):
    """PTA with a per-class bank of K prototypes (nearest-prototype EMA).

    Write rule is base PTA's multi-class EMA (every class with
    ``softmax(clip) >= 0.1`` writes with weight ``1 - exp(-p / T)``); the
    written feature updates only the nearest of the class's K prototypes.
    Score is the max of the K cosines; prediction is
    ``final = clip + 100 * logit_c`` (exact mirror of base PTA).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── Backward-compatible nested config ──────────────────────────────
        # Flat configs (alpha, T at root level) are still used by some
        # callers.  Propagate them into the nested "image_level" sub-dict
        # (same pattern as models/pta.py).  T is used by the write rule;
        # alpha is unused by this method but kept for config compatibility.
        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        # ── Fusion component ───────────────────────────────────────────────
        # Exact mirror of base PTA: tau_text=1.0, tau_image_proto=100.0,
        # tau_patch_proto=0.0 (no patch-level signal).
        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        self.K = int(cfg.get("K", 3))
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "BankPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            text_features = text_embeddings.t()            # [C, D] fp16
            P = init_bank(text_features, self.K)           # [C, K, D] fp32
            text_norm = F.normalize(text_features.float(), dim=-1)  # [C, D] fp32

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            # Per-class bank diagnostics for the per-sample records.
            n_updates = [0] * num_classes
            k_usage = [[0] * self.K for _ in range(num_classes)]

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(tqdm(loader, desc=f"[BankPTA] {dataset_name}")):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Write (base PTA multi-class rule, nearest-prototype EMA;
                #    bank state is float32; P is mutated in-place) ──────────
                x = image_features.squeeze(0).float()      # (D,) fp32
                writes = bank_ema_write(
                    x.unsqueeze(0), clip_logits.float(), text_norm, P,
                    self.alpha, self.T,
                )

                # ── Score (max-of-K base-PTA-style cosine) + fused prediction
                #    Compute in float32, cast to fp16 for the fusion/record.
                logit_c = bank_score(x, text_norm, P, self.alpha)  # (C,) fp32
                image_proto_logits = logit_c.half().unsqueeze(0)  # [1, C] fp16
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                for c, k_star, _w_c in writes:
                    n_updates[c] += 1
                    k_usage[c][k_star] += 1

                if record_dir:
                    # Per-class bank diagnostics. ``K`` is recorded as a
                    # top-level proto_stats key: write_record has a fixed
                    # signature (no **kwargs), so an extra kwarg channel is
                    # not available without touching utils/records.py.
                    proto_stats = {}
                    proto_stats["K"] = self.K
                    for c in range(num_classes):
                        cos_to_text = P[c] @ text_norm[c]  # (K,)
                        proto_stats["class_{}".format(c)] = {
                            "n_updates": int(n_updates[c]),
                            "active_k": int((cos_to_text < 0.99).sum()),
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
                        f"---- BankPTA(K={self.K}) test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- BankPTA(K={self.K}) test accuracy: {final_acc:.2f}. ----\n"
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
                method=os.environ.get("RESULT_LABEL", "BankPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "BankPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> BankPTAAdapter:
    """Factory: instantiate a BankPTAAdapter with the given config."""
    return BankPTAAdapter(cfg)
