"""Prototype-Based TTA with a write-gate on the image prototype.

A PTA-only study (Part 4): the prediction rule is fixed across all variants
(``final = clip + 100 * image_proto``, argmax), and only the *write gate* on
the image prototype differs:

    baseline             — always write argmax(clip)              (new top-1 PTA baseline)
    confident            — write iff softmax(clip).top1-top2 >= conf_margin_thresh
    agree                — write iff patch_vote_pred(topk20) == argmax(clip)
    confident_and_agree  — write iff both confident AND agree

The write is a single-class EMA of the top-1 ``argmax(clip)`` image feature
into the class's image prototype row (TPT-style hard top-1 write), following
the PTAImageLevel EMA formula. When the gate is closed no write occurs and the
prototype is unchanged; prediction is always computed from the current
prototype.

The agreement signal uses the stateless topk20 patch vote
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
from utils.records import write_record_header, write_record, write_summary

# write_gate config values
GATE_BASELINE = "baseline"
GATE_CONFIDENT = "confident"
GATE_AGREE = "agree"
GATE_CONFIDENT_AND_AGREE = "confident_and_agree"
_VALID_GATES = {GATE_BASELINE, GATE_CONFIDENT, GATE_AGREE, GATE_CONFIDENT_AND_AGREE}


def _gated_top1_ema_write(
    image_feature,
    clip_logits,
    text_features,
    prototype_state,
    alpha,
    T,
    write_class,
):
    """Single-class EMA write of ``argmax(clip)`` into the image prototype.

    Mirrors ``PTAImageLevel.update_prototypes`` (models/image_level/pta_image.py)
    — same EMA weight formula, fp16 flow, in-place prototype mutation, and
    ``refined = alpha*text + (1-alpha)*proto`` L2-normalization. The only
    difference: the write updates a single class row with weight
    ``1 - exp(-w_top1 / T)`` (no ``w >= 0.1`` threshold), where ``w_top1`` is the
    softmax probability of the written class.

    Args:
        image_feature:   (1, D) L2-normalized CLIP image embedding (fp16).
        clip_logits:     (1, C) zero-shot logits (fp16).
        text_features:   (C, D) current refined text features (fp16).
        prototype_state: (C, D) running prototype bank (mutated in-place).
        alpha:           Weight on original text features.
        T:               EMA temperature.
        write_class:     int class to write, or None to skip the write (gate closed).

    Returns:
        (refined_text, prototype_state) — both (C, D).
    """
    if write_class is not None:
        w = F.softmax(clip_logits, dim=-1).squeeze(0)          # (C) fp16
        w_top1 = w[write_class]
        w_new = 1.0 - torch.exp(-w_top1 / T)                    # scalar fp16
        prototype_state[write_class] = (
            (1.0 - w_new) * prototype_state[write_class]
            + w_new * image_feature.squeeze(0)
        )

    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, prototype_state


def _evaluate_gate(mode, clip_margin, patch_vote_pred, top1, conf_margin_thresh):
    """Return (gate_open, write_class) for the given write-gate mode.

    ``clip_margin`` is softmax(clip).top1 - .top2; ``patch_vote_pred`` is the
    topk20 patch-vote prediction (None when the agree signal is unused).
    """
    if mode == GATE_BASELINE:
        return True, top1
    if mode == GATE_CONFIDENT:
        return clip_margin >= conf_margin_thresh, top1
    if mode == GATE_AGREE:
        return patch_vote_pred == top1, top1
    if mode == GATE_CONFIDENT_AND_AGREE:
        return (clip_margin >= conf_margin_thresh) and (patch_vote_pred == top1), top1
    raise ValueError(f"Unknown write_gate mode: {mode}")


class WriteGatePTAAdapter(BaseAdapter):
    """PTA with a confidence/agreement-gated single-class image-prototype write.

    Prediction rule is identical across all write-gate modes:
    ``final = clip + 100 * image_proto`` (WeightedFusion, tau_image_proto=100).
    Only the write gate differs, so any accuracy delta between modes is due
    solely to accumulated prototype-bank differences.
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

        self.write_gate = str(
            cfg.get("write_gate", GATE_BASELINE)
        ).lower()
        if self.write_gate not in _VALID_GATES:
            raise ValueError(
                f"write_gate must be one of {sorted(_VALID_GATES)}, got {self.write_gate!r}"
            )
        self.conf_margin_thresh = float(cfg.get("conf_margin_thresh", 0.2))
        self.pv_aggregation = str(cfg.get("pv_aggregation", "topk20"))
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

        # The agree gate needs the patch vote for every mode that depends on it.
        self._needs_patch_vote = self.write_gate in (
            GATE_AGREE,
            GATE_CONFIDENT_AND_AGREE,
        )

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "WriteGatePTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            refine_feature = text_embeddings.t()                # [C, D] fp16
            target_prototype = self.image_level.init_state(refine_feature)

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(tqdm(loader, desc=f"[WriteGatePTA] {dataset_name}")):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Gate evaluation (causal: depends only on frozen clip logits
                #    + stateless patch vote, no bank read) ───────────────────
                w = F.softmax(clip_logits, dim=-1).squeeze(0)   # (C) fp16
                top1 = int(w.argmax(dim=-1).item())
                top2_val = w.topk(min(2, w.shape[0])).values[-1].item()
                clip_margin = float(w[top1].item() - top2_val)

                patch_vote_pred = None
                patch_vote_margin = None
                if self._needs_patch_vote:
                    patches_norm = _safe_normalize(
                        encoder.get_patch_embeddings(images).float()
                    )
                    patch_vote_pred, patch_vote_margin = compute_patch_vote(
                        patches_norm, text_embeddings, aggregation=self.pv_aggregation
                    )

                gate_open, write_class = _evaluate_gate(
                    self.write_gate,
                    clip_margin,
                    patch_vote_pred,
                    top1,
                    self.conf_margin_thresh,
                )
                write_class = write_class if gate_open else None

                # ── Gated single-class EMA write ────────────────────────────
                refine_feature, target_prototype = _gated_top1_ema_write(
                    image_features,
                    clip_logits,
                    refine_feature,
                    target_prototype,
                    self.alpha,
                    self.T,
                    write_class,
                )

                # ── Fused prediction (identical across all gate modes) ──────
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
                        write_gate=self.write_gate,
                        write_occurred=bool(gate_open),
                        patch_vote_pred=patch_vote_pred,
                        patch_vote_margin=patch_vote_margin,
                        clip_margin=clip_margin,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- WriteGatePTA({self.write_gate}) test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- WriteGatePTA({self.write_gate}) test accuracy: {final_acc:.2f}. ----\n")

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
                method=os.environ.get("RESULT_LABEL", "WriteGatePTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        label = os.environ.get("RESULT_LABEL", "WriteGatePTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> WriteGatePTAAdapter:
    """Factory: instantiate a WriteGatePTAAdapter with the given config."""
    return WriteGatePTAAdapter(cfg)
