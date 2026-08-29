"""Prototype-Based TTA with two-sided write re-weighting.

Part 4b study: the write *rule* is identical to base PTA — every image writes
``argmax(clip)`` into the image prototype via a single-class EMA (the same
top-1 write as the Part 4a ``baseline``). No write is ever dropped. The only
difference from base PTA is the *write weight*, which is scaled by a per-sample
factor along the trust axis:

* **trusted** = **confident** (``softmax(clip).top1 - .top2 >= conf_margin_thresh``)
  AND **agree** (the topk20 patch vote predicts ``argmax(clip)``): weight is
  multiplied by ``boost`` (up-weight, clamped <= 1.0).
* **untrusted** = **ambiguous** (low confidence) OR **disagree** (patch vote
  mispredicts): weight is multiplied by ``boost_down`` (down-weight, < 1.0).

With ``boost=1`` and ``boost_down=1`` this reproduces base PTA exactly (the
in-grid control). Scaling both sides widens (or narrows) the effective weight
spread across the trust axis. This is the two-sided counterpart to Part 4a's
hard gate (which *drops* untrusted writes) and the up-only Part 4b sweep (07):
it never drops a write, but reduces the prototype drift contributed by low
confidence / disagreeing samples while amplifying trusted ones.

The experiment also measures **convergence speed**: each run writes
``convergence.json`` (via ``utils.records.write_convergence``) with the
per-sample online-accuracy trajectory, so we can test whether re-weighting
changes *how fast* adaptation reaches its plateau, not just where it lands.

The agreement signal uses the stateless topk20 CLIP patch vote
(``utils.patch_vote.compute_patch_vote``) — it deliberately does NOT touch the
``patch_proto`` bank. Prediction and fusion are identical to PTA
(``WeightedFusion`` with tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0).
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.clip_inference import _safe_normalize
from utils.patch_vote import compute_patch_vote
from utils.records import (
    write_record_header,
    write_record,
    write_summary,
    write_convergence,
)

# write_gate config values (reused for labels/consistency with Part 4a)
GATE_BASELINE = "baseline"


def _boosted_top1_ema_write(
    image_feature,
    clip_logits,
    text_features,
    prototype_state,
    alpha,
    T,
    write_class,
    weight_scale,
):
    """Single-class EMA write of ``argmax(clip)``, weight scaled by a factor.

    Mirrors ``PTAImageLevel.update_prototypes`` (models/image_level/pta_image.py)
    — same EMA weight formula ``1 - exp(-w_top1 / T)``, fp16 flow, in-place
    prototype mutation, and ``refined = alpha*text + (1-alpha)*proto``
    L2-normalization. The write rule is identical to base PTA for the written
    class (single top-1 row); the only difference is ``w_new *= weight_scale``
    (clamped to 1.0), so a factor above 1 pulls trusted images in harder and a
    factor below 1 dampens the drift from untrusted ones.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding (fp16).
        clip_logits:     (1, C) zero-shot logits (fp16).
        text_features:   (C, D) current refined text features (fp16).
        prototype_state: (C, D) running prototype bank (mutated in-place).
        alpha:           Weight on original text features.
        T:               EMA temperature.
        write_class:     int class to write (argmax(clip)); always set here.
        weight_scale:    float factor to scale the EMA weight (trust-adjusted).

    Returns:
        (refined_text, prototype_state) — both (C, D).
    """
    if write_class is not None:
        w = F.softmax(clip_logits, dim=-1).squeeze(0)      # (C) fp16
        w_top1 = w[write_class]
        w_new = 1.0 - torch.exp(-w_top1 / T)               # scalar fp16
        w_new = (w_new * weight_scale).clamp(max=1.0)
        prototype_state[write_class] = (
            (1.0 - w_new) * prototype_state[write_class]
            + w_new * image_feature.squeeze(0)
        )

    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, prototype_state


class ReweightPTAAdapter(BaseAdapter):
    """PTA with two-sided write-weight re-scaling on the trust axis.

    Prediction rule ``final = clip + 100 * image_proto`` is identical across all
    re-weight settings; the write rule is always ``argmax(clip)`` (base PTA).
    Only the write weight differs: ``boost`` multiplies the EMA weight for
    trusted (confident AND agree) samples, ``boost_down`` multiplies it for
    untrusted (ambiguous OR disagree) samples. With ``boost=boost_down=1`` the
    write reduces to base PTA, so any accuracy or convergence delta across
    settings is due solely to the reweighting.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        image_level_cfg = cfg.get("image_level", {})
        self.image_level = create_image_level(cfg)

        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        self.boost = float(cfg.get("boost", 1.0))
        self.boost_down = float(cfg.get("boost_down", 1.0))
        self.conf_margin_thresh = float(cfg.get("conf_margin_thresh", 0.2))
        self.pv_aggregation = str(cfg.get("pv_aggregation", "topk20"))
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

        # The agree signal needs the patch vote for every run.
        self._needs_patch_vote = True

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "ReweightPTA"),
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

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(tqdm(loader, desc=f"[ReweightPTA] {dataset_name}")):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Signal extraction (causal: frozen clip logits + stateless
                #    patch vote, no bank read) ──────────────────────────────
                w = F.softmax(clip_logits, dim=-1).squeeze(0)   # (C) fp16
                top1 = int(w.argmax(dim=-1).item())
                top2_val = w.topk(min(2, w.shape[0])).values[-1].item()
                clip_margin = float(w[top1].item() - top2_val)

                patches_norm = _safe_normalize(
                    encoder.get_patch_embeddings(images).float()
                )
                patch_vote_pred, patch_vote_margin = compute_patch_vote(
                    patches_norm, text_embeddings, aggregation=self.pv_aggregation
                )

                confident = clip_margin >= self.conf_margin_thresh
                agree = patch_vote_pred == top1
                trusted = confident and agree
                applied_factor = self.boost if trusted else self.boost_down

                # ── Write (rule = base PTA: always top-1; weight scaled by the
                #    trust-adjusted factor) ──────────────────────────────────
                refine_feature, target_prototype = _boosted_top1_ema_write(
                    image_features,
                    clip_logits,
                    refine_feature,
                    target_prototype,
                    self.alpha,
                    self.T,
                    top1,
                    applied_factor,
                )

                # ── Fused prediction (identical across all boost levels) ───
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
                        proto_stats={"true": None, "pred": None},
                        write_gate=None,
                        write_occurred=True,
                        patch_vote_pred=patch_vote_pred,
                        patch_vote_margin=patch_vote_margin,
                        clip_margin=clip_margin,
                        boost=applied_factor,
                        boost_down=self.boost_down,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- ReweightPTA(boost={self.boost}, boost_down={self.boost_down}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- ReweightPTA(boost={self.boost}, boost_down={self.boost_down}) "
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
                method=os.environ.get("RESULT_LABEL", "ReweightPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "ReweightPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> ReweightPTAAdapter:
    """Factory: instantiate a ReweightPTAAdapter with the given config."""
    return ReweightPTAAdapter(cfg)
