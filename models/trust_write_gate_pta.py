"""Prototype-Based TTA with a write-gate driven by multi-view consistency
and/or prototype confusability (independently or combined).

Mirrors `models/write_gate_pta.py` (Part 4a of
`experimental_results/PatchModPTA_Purity_Separability_Trust_Analysis.md`)
exactly -- same single-class top-1 EMA write, same `baseline`/`confident`
structure -- but replaces the patch-vote `agree` signal with a choice of two
new signals (or their conjunction), each validated as a strong diagnostic
this round but never tested as a per-sample write-time gate:

* **view**: agreement between CLIP's zero-shot top-1 prediction on the
  original image and on `n_views-1` perturbed copies of it
  (`utils/augmentation.py:_augment_image`, the same computation used in
  `models/view_consistency_pta.py`). Unlike patch-vote, this isn't an
  alternative classifier with its own (often worse) standalone accuracy --
  it only tests CLIP's own robustness under perturbation.
* **confusability**: whether the class about to be written already has a
  prototype close to some other class's prototype (a `[C,C]` matmul on the
  already-computed bank, no extra CLIP forward passes). Evaluated fresh
  every sample from the CURRENT bank state (causal, before this sample's
  write) -- NOT the permanent per-class freeze used in
  `models/confusability_gated_pta.py`. If the bank's geometry later shifts
  so a class is no longer confusable, its next write proceeds normally.
* **both**: the class is only "clean" if BOTH conditions hold.

    write_gate | condition
    -----------|----------
    baseline              | always write argmax(clip) (signal-independent control)
    clean                 | write iff the selected signal_source's clean condition holds
    confident_and_clean   | confident AND clean

`confident`-alone (no clean condition) is signal-independent and identical
to Part 4a's own row -- intentionally not reproduced here; cite Part 4a's
numbers directly (dtd 42.48 / flowers 74.12 / pets 90.36).
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.clip_inference import _safe_normalize
from utils.augmentation import _augment_image
from utils.records import write_record_header, write_record, write_summary

GATE_BASELINE = "baseline"
GATE_CLEAN = "clean"
GATE_CONFIDENT_AND_CLEAN = "confident_and_clean"
_VALID_GATES = {GATE_BASELINE, GATE_CLEAN, GATE_CONFIDENT_AND_CLEAN}

SOURCE_VIEW = "view"
SOURCE_CONFUSABILITY = "confusability"
SOURCE_BOTH = "both"
_VALID_SOURCES = {SOURCE_VIEW, SOURCE_CONFUSABILITY, SOURCE_BOTH}


def _gated_top1_ema_write(
    image_feature, clip_logits, text_features, prototype_state, alpha, T, write_class,
):
    """Single-class EMA write of ``argmax(clip)`` into the image prototype.

    Verbatim copy of the Part 4a mechanism (models/write_gate_pta.py) --
    same EMA weight formula, fp16 flow, in-place prototype mutation.
    """
    if write_class is not None:
        w = F.softmax(clip_logits, dim=-1).squeeze(0)
        w_top1 = w[write_class]
        w_new = 1.0 - torch.exp(-w_top1 / T)
        prototype_state[write_class] = (
            (1.0 - w_new) * prototype_state[write_class]
            + w_new * image_feature.squeeze(0)
        )
    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, prototype_state


def _compute_view_clean(images, top1, encoder, text_embeddings, n_views, agreement_thresh):
    if n_views <= 1:
        return True, 1.0
    aug_views = [_augment_image(images) for _ in range(n_views - 1)]
    batched = torch.cat(aug_views, dim=0).cuda()
    aug_feats = _safe_normalize(encoder.encode_image(batched).float())
    aug_preds = (100.0 * aug_feats @ text_embeddings.float()).argmax(dim=-1)
    agreement = float((aug_preds == top1).float().mean().item())
    return agreement >= agreement_thresh, agreement


def _compute_proto_clean(prototype_state, top1, percentile):
    """Causal: uses the bank state BEFORE this sample's write. No memory
    across samples -- recomputed from scratch every call."""
    proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
    sims = proto_norm @ proto_norm.t()
    sims.fill_diagonal_(-1.0)
    nearest_other, _ = sims.max(dim=1)
    written = prototype_state.float().norm(dim=-1) > 1e-6
    n_written = int(written.sum().item())
    target_confusability = float(nearest_other[top1].item())
    if n_written >= 5 and bool(written[top1]):
        thresh = torch.quantile(nearest_other[written], percentile / 100.0)
        confusable = bool((nearest_other[top1] > thresh).item())
    else:
        confusable = False
    return (not confusable), target_confusability


class TrustWriteGatePTAAdapter(BaseAdapter):
    """PTA with a write-gate on the image prototype driven by view
    consistency, prototype confusability, or both (config-selected)."""

    def __init__(self, cfg):
        super().__init__(cfg)

        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        image_level_cfg = cfg.get("image_level", {})
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

        self.write_gate = str(cfg.get("write_gate", GATE_BASELINE)).lower()
        if self.write_gate not in _VALID_GATES:
            raise ValueError(f"write_gate must be one of {sorted(_VALID_GATES)}, got {self.write_gate!r}")

        self.signal_source = str(cfg.get("signal_source", SOURCE_VIEW)).lower()
        if self.signal_source not in _VALID_SOURCES:
            raise ValueError(f"signal_source must be one of {sorted(_VALID_SOURCES)}, got {self.signal_source!r}")

        self.conf_margin_thresh = float(cfg.get("conf_margin_thresh", 0.2))
        self.n_views = int(cfg.get("n_views", 4))
        self.agreement_thresh = float(cfg.get("agreement_thresh", 1.0))
        self.confusability_percentile = float(cfg.get("confusability_percentile", 66.0))

        # baseline never needs any signal -- skip all extra compute.
        self._needs_signal = self.write_gate != GATE_BASELINE
        self._needs_view = self._needs_signal and self.signal_source in (SOURCE_VIEW, SOURCE_BOTH)
        self._needs_proto = self._needs_signal and self.signal_source in (SOURCE_CONFUSABILITY, SOURCE_BOTH)

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "TrustWriteGatePTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            refine_feature = text_embeddings.t()  # [C, D] fp16
            num_classes = text_embeddings.shape[1]
            target_prototype = torch.zeros_like(refine_feature)

            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            gate_fired_count = 0
            gate_total_count = 0

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[TrustWriteGatePTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Signal evaluation (causal: uses the bank state as it
                #    exists BEFORE this sample's write) ────────────────────
                w = F.softmax(clip_logits, dim=-1).squeeze(0)
                top1 = int(w.argmax(dim=-1).item())
                top2_val = w.topk(min(2, w.shape[0])).values[-1].item()
                clip_margin = float(w[top1].item() - top2_val)
                confident = clip_margin >= self.conf_margin_thresh

                view_agreement = None
                target_confusability = None
                view_clean = True
                proto_clean = True
                if self._needs_view:
                    view_clean, view_agreement = _compute_view_clean(
                        images, top1, encoder, text_embeddings, self.n_views, self.agreement_thresh
                    )
                if self._needs_proto:
                    proto_clean, target_confusability = _compute_proto_clean(
                        target_prototype, top1, self.confusability_percentile
                    )

                if self.signal_source == SOURCE_VIEW:
                    clean = view_clean
                elif self.signal_source == SOURCE_CONFUSABILITY:
                    clean = proto_clean
                else:
                    clean = view_clean and proto_clean

                if self.write_gate == GATE_BASELINE:
                    gate_open = True
                elif self.write_gate == GATE_CLEAN:
                    gate_open = clean
                else:  # confident_and_clean
                    gate_open = confident and clean
                write_class = top1 if gate_open else None

                if self._needs_signal:
                    gate_total_count += 1
                    if gate_open:
                        gate_fired_count += 1

                # ── Gated single-class EMA write ────────────────────────────
                refine_feature, target_prototype = _gated_top1_ema_write(
                    image_features, clip_logits, refine_feature, target_prototype,
                    self.alpha, self.T, write_class,
                )

                # ── Fused prediction (identical across all settings) ───────
                image_proto_logits = image_features.half() @ refine_feature.half().T
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                t = int(target.item())
                cls_total[t] += 1
                if acc:
                    cls_correct[t] += 1

                if record_dir:
                    write_record(
                        batch_idx=i,
                        target=t,
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
                        clip_margin=clip_margin,
                        view_agreement=view_agreement,
                        target_confusability=target_confusability,
                        signal_source=self.signal_source if self._needs_signal else None,
                    )

                if i % 1000 == 0:
                    rate = (gate_fired_count / gate_total_count) if gate_total_count else 1.0
                    print(
                        f"---- TrustWriteGatePTA({self.write_gate},{self.signal_source}) test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. write_rate={rate:.2f} ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        rate = (gate_fired_count / gate_total_count) if gate_total_count else 1.0
        print(
            f"---- TrustWriteGatePTA({self.write_gate},{self.signal_source}) test accuracy: "
            f"{final_acc:.2f}. Final write_rate={rate:.2f} ----\n"
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
                method=os.environ.get("RESULT_LABEL", "TrustWriteGatePTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        label = os.environ.get("RESULT_LABEL", "TrustWriteGatePTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> TrustWriteGatePTAAdapter:
    """Factory: instantiate a TrustWriteGatePTAAdapter with the given config."""
    return TrustWriteGatePTAAdapter(cfg)
