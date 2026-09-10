"""Prototype-Based TTA with two-sided write re-weighting driven by
multi-view consistency and/or prototype confusability (independently or
combined).

Mirrors `models/reweight_pta.py` (Part 4b.5 of
`experimental_results/PatchModPTA_Purity_Separability_Trust_Analysis.md`)
exactly -- write rule is always `argmax(clip)`, single-class EMA, never
dropped; only the EMA weight scales with trust -- but replaces the
patch-vote-derived trust condition with a choice of two new signals (or
their conjunction), the same ones used in `models/trust_write_gate_pta.py`
(see that file's docstring for the full rationale and exact per-sample
signal computations, reused verbatim here).

    applied_weight_scale = boost       if confident AND clean (trusted)
                          = boost_down  otherwise (untrusted)

`clean` is `view_clean`, `proto_clean`, or `view_clean and proto_clean`,
depending on `signal_source`. `boost == boost_down` (e.g. the (1.0, 1.0)
control) skips all signal computation entirely, since the applied factor is
identical regardless of the trust condition.

Convergence is tracked via the unchanged `utils.records.write_convergence`
for the same early-fraction-accuracy / normalized-AUC comparison Part 4b.5
used.
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
from utils.records import write_record_header, write_record, write_summary, write_convergence

SOURCE_VIEW = "view"
SOURCE_CONFUSABILITY = "confusability"
SOURCE_BOTH = "both"
_VALID_SOURCES = {SOURCE_VIEW, SOURCE_CONFUSABILITY, SOURCE_BOTH}


def _boosted_top1_ema_write(
    image_feature, clip_logits, text_features, prototype_state, alpha, T, write_class, weight_scale,
):
    """Verbatim copy of the Part 4b mechanism (models/reweight_pta.py)."""
    if write_class is not None:
        w = F.softmax(clip_logits, dim=-1).squeeze(0)
        w_top1 = w[write_class]
        w_new = 1.0 - torch.exp(-w_top1 / T)
        w_new = (w_new * weight_scale).clamp(max=1.0)
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


class TrustReweightPTAAdapter(BaseAdapter):
    """PTA with two-sided write-weight re-scaling, trust condition driven by
    view consistency, prototype confusability, or both (config-selected)."""

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

        self.boost = float(cfg.get("boost", 1.0))
        self.boost_down = float(cfg.get("boost_down", 1.0))

        self.signal_source = str(cfg.get("signal_source", SOURCE_VIEW)).lower()
        if self.signal_source not in _VALID_SOURCES:
            raise ValueError(f"signal_source must be one of {sorted(_VALID_SOURCES)}, got {self.signal_source!r}")

        self.conf_margin_thresh = float(cfg.get("conf_margin_thresh", 0.2))
        self.n_views = int(cfg.get("n_views", 4))
        self.agreement_thresh = float(cfg.get("agreement_thresh", 1.0))
        self.confusability_percentile = float(cfg.get("confusability_percentile", 66.0))

        # boost == boost_down -> the trust condition never changes the
        # applied factor, so skip all signal computation (control setting).
        self._needs_signal = self.boost != self.boost_down
        self._needs_view = self._needs_signal and self.signal_source in (SOURCE_VIEW, SOURCE_BOTH)
        self._needs_proto = self._needs_signal and self.signal_source in (SOURCE_CONFUSABILITY, SOURCE_BOTH)

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "TrustReweightPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            refine_feature = text_embeddings.t()
            num_classes = text_embeddings.shape[1]
            target_prototype = torch.zeros_like(refine_feature)

            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[TrustReweightPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

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

                if not self._needs_signal:
                    clean = True
                elif self.signal_source == SOURCE_VIEW:
                    clean = view_clean
                elif self.signal_source == SOURCE_CONFUSABILITY:
                    clean = proto_clean
                else:
                    clean = view_clean and proto_clean

                trusted = confident and clean
                applied_factor = self.boost if trusted else self.boost_down

                refine_feature, target_prototype = _boosted_top1_ema_write(
                    image_features, clip_logits, refine_feature, target_prototype,
                    self.alpha, self.T, top1, applied_factor,
                )

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
                        write_gate=None,
                        write_occurred=True,
                        clip_margin=clip_margin,
                        boost=applied_factor,
                        boost_down=self.boost_down,
                        view_agreement=view_agreement,
                        target_confusability=target_confusability,
                        signal_source=self.signal_source if self._needs_signal else None,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- TrustReweightPTA(boost={self.boost},down={self.boost_down},"
                        f"src={self.signal_source}) test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- TrustReweightPTA(boost={self.boost},down={self.boost_down},"
            f"src={self.signal_source}) test accuracy: {final_acc:.2f}. ----\n"
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
                method=os.environ.get("RESULT_LABEL", "TrustReweightPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "TrustReweightPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> TrustReweightPTAAdapter:
    """Factory: instantiate a TrustReweightPTAAdapter with the given config."""
    return TrustReweightPTAAdapter(cfg)
