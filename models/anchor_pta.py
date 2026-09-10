"""AnchorPTA — text-anchor-referenced EMA damping for base PTA.

Mechanism
---------
Base PTA's per-class prototype is a softmax-weighted EMA of CLIP's writes
(``w_new = 1 - exp(-softmax / T)`` for every class with softmax >= 0.1).
When a class's prototype has been captured by CLIP's error modes (wrong
writes accumulate), the class's *refined* vector ``refine[c] =
normalize(alpha*text[c] + (1-alpha)*proto[c])`` drifts away from its
IMMUTABLE text anchor ``text[c]`` (the original zero-shot class embedding,
which is fixed and never estimated from the stream).

AnchorPTA damps the EMA rate of exactly those classes, per write:

    sim_c      = cos(refine[c], text[c])          # impartial referee
    damp_c     = 1.0                              if sim_c >= damp_threshold
               = damp_floor + (1 - damp_floor) * sim_c / damp_threshold
                                                  otherwise (monotone in sim_c)
    w_new[c]  *= damp_c

Properties (deliberate, and the reason this is NOT a repeat of the failed
write-time gates):

* The referee is the TEXT ANCHOR — fixed a priori, never estimated from the
  biased stream. This satisfies the study's binding constraint ("a mechanism
  whose advantage does not depend on statistics that must be estimated from
  CLIP's own (biased) guesses").
* The gate is PER-CLASS ACCUMULATED-DRIFT, not per-sample confidence. Every
  sample still writes and writes are never removed — the write weight is
  merely *reduced* toward ``damp_floor`` (default 0.5 => at most a 2x cut).
  It therefore cannot starve a class the way per-sample skip/drop gates did.
* Healthy classes never damp: with alpha = 0.01 the refined vector stays
  within ~1% of its anchor under legitimate adaptation (sim_c ~ 0.97-0.99),
  well above the thresholds used (0.93 / 0.96). Only classes whose prototype
  has drifted far enough to indicate error capture are slowed, letting the
  text anchor reassert itself through the next refine step.
* ``damp_threshold = 0`` disables damping entirely and the adapter reduces to
  base PTA EXACTLY (a CPU unit test asserts this).

Config keys (root level; selected per-run via ``--override``):
    damp_threshold — cosine drift threshold below which writes are damped.
                     Default 0.0 (disabled = base PTA).  Wave settings 0.96/0.93.
    damp_floor     — minimum damp factor (never fully starves). Default 0.5.

Records: same schema as base PTA, plus per-class ``n_updates`` and
``n_damped`` (write-purity + damping analytics) in ``proto_stats``.
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.pta import PTAAdapter
from utils import cls_acc, get_clip_logits
from utils.records import (
    write_convergence,
    write_record,
    write_record_header,
    write_summary,
)


class AnchorPTAAdapter(PTAAdapter):
    """PTA + per-class EMA-rate damping referenced to the immutable text anchor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.damp_threshold = float(cfg.get("damp_threshold", 0.0))
        self.damp_floor = float(cfg.get("damp_floor", 0.5))
        self.damp_floor = max(0.0, min(self.damp_floor, 1.0))

    # ------------------------------------------------------------------
    # Damped prototype update (base PTA's rule + anchor drift damping)
    # ------------------------------------------------------------------
    def _damped_update(self, image_feature, clip_logits, text_ref, refine, proto):
        """EMA update identical to PTAImageLevel.update_prototypes, except the
        per-class EMA rate is scaled by ``damp_c`` (anchor drift referee).

        Args:
            image_feature: (1, D) fp32 L2-normalized ORIGINAL image embedding.
            clip_logits:   (1, C) pseudo-label logits (zero-shot scale).
            text_ref:      (C, D) fp32 IMMUTABLE original text anchors.
            refine:        (C, D) fp32 current refined features (mutated).
            proto:         (C, D) fp32 prototype bank (mutated in-place).

        Returns:
            (refined, prototype_state).
        """
        alpha = float(self.image_level._cfg.get("alpha", 0.01))
        T = float(self.image_level._cfg.get("T", 20.0))

        probs = F.softmax(clip_logits, dim=-1)          # (1, C)
        w = probs.squeeze(0)                            # (C)
        mask = w >= 1e-1                                # (C) bool
        w_new = torch.zeros_like(w)                     # (C)
        w_new[mask] = 1 - torch.exp(-w[mask] / T)       # (C)

        # ── Anchor-drift damping (impartial referee) ──────────────────────
        damp = torch.ones_like(w)
        if self.damp_threshold > 0.0:
            sim = F.cosine_similarity(refine.float(), text_ref.float(), dim=-1)
            below = sim < self.damp_threshold
            scale = (sim.clamp(min=0.0) / self.damp_threshold).clamp(max=1.0)
            damp[below] = (
                self.damp_floor + (1.0 - self.damp_floor) * scale[below]
            )
        w_new = (w_new * damp).unsqueeze(1)             # (C, 1)

        # ── EMA write (exact PTA form) ────────────────────────────────────
        proto[mask] = (
            (1 - w_new[mask]) * proto[mask]
            + w_new[mask] * image_feature.squeeze(0)
        )

        # ── Text blend + normalize (exact PTA form) ───────────────────────
        refined = alpha * refine + (1 - alpha) * proto
        refined = refined / refined.norm(dim=-1, keepdim=True)
        return refined, proto

    # ------------------------------------------------------------------
    # Run (base PTA loop with the damped update)
    # ------------------------------------------------------------------
    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "AnchorPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            # Immutable text anchors (the referee) — never mutated.
            text_ref = text_embeddings.t().float().clone()   # [C, D] fp32
            refine_feature = text_embeddings.t().float()     # [C, D] fp32
            target_prototype = self.image_level.init_state(refine_feature)

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes
            n_updates = [0] * num_classes
            n_damped = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[AnchorPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── DAMPED ONLINE UPDATE ──────────────────────────────────
                x = image_features.squeeze(0).float()        # (D,) fp32
                probs = F.softmax(clip_logits.float(), dim=-1).squeeze(0)
                write_mask = probs >= 1e-1

                sim_c = F.cosine_similarity(
                    refine_feature.float(), text_ref, dim=-1
                )
                damped_c = torch.zeros_like(sim_c)
                if self.damp_threshold > 0.0:
                    damped_c = sim_c < self.damp_threshold

                refine_feature, target_prototype = self._damped_update(
                    image_features, clip_logits, text_ref,
                    refine_feature, target_prototype,
                )

                for c in range(num_classes):
                    if bool(write_mask[c].item()):
                        n_updates[c] += 1
                        if bool(damped_c[c].item()):
                            n_damped[c] += 1

                # ── FUSED PREDICTION (unchanged math) ─────────────────────
                image_proto_logits = self.image_level.compute_logits(
                    image_features, refine_feature
                )
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                if record_dir:
                    proto_stats = {}
                    proto_stats["damp_threshold"] = self.damp_threshold
                    proto_stats["damp_floor"] = self.damp_floor
                    for c in range(num_classes):
                        proto_stats["class_{}".format(c)] = {
                            "n_updates": int(n_updates[c]),
                            "n_damped": int(n_damped[c]),
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
                        write_gate="anchor-damp" if self.damp_threshold > 0.0 else None,
                        write_occurred=bool(int(write_mask.sum().item()) > 0),
                    )

                if i % 1000 == 0:
                    print(
                        f"---- AnchorPTA accuracy: "
                        f"{sum(accuracies) / len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- AnchorPTA's test accuracy: {final_acc:.2f}. ----\n")

        if record_dir:
            per_class = {}
            for c in range(num_classes):
                per_class["class_{}".format(c)] = {
                    "total": cls_total[c],
                    "correct": cls_correct[c],
                    "acc": (100.0 * cls_correct[c] / cls_total[c]) if cls_total[c] else 0.0,
                }
            write_summary(
                method=os.environ.get("RESULT_LABEL", "AnchorPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
        write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "AnchorPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> AnchorPTAAdapter:
    """Factory: instantiate AnchorPTAAdapter (module-level, required by runner)."""
    return AnchorPTAAdapter(cfg)