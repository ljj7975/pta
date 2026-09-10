"""Prototype-Based TTA with an explicit geometry-correction (repulsion) step,
applied in addition to -- never instead of -- the normal write.

Every write-time trust study so far this round (patch-vote / view-consistency
/ prototype-confusability, hard gate / two-sided reweight / permanent freeze
/ read-time reweight -- see `experimental_results/Phase7_...md`) is
defensive: it reduces or blocks the write when a signal says "don't trust
this sample," and every one of them nets negative, because blocking a write
throws away information the EMA needs regardless of whether that sample was
"clean." This adapter tests a mechanistically different family: the write
always fires, unmodified (`_gated_top1_ema_write`-equivalent, `weight_scale`
fixed at 1.0), and a small explicit repulsive correction is applied
afterward to push two confusable class prototypes apart along their own
connecting direction -- correcting geometry instead of filtering
information.

`repulsion_lr=0.0` is a hard no-op (the control setting, must reproduce the
established top-1-write reference exactly). `trigger="confusable"` only
fires the correction when the same tercile check from
`models/trust_write_gate_pta.py` / `models/trust_reweight_pta.py` flags the
target class as confusable; `trigger="always"` fires unconditionally
whenever the target class and its nearest-other class are both already
written, testing a continuous uniform-margin-enforcement hypothesis instead
of a triggered one.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.clip_inference import _safe_normalize
from utils.records import write_record_header, write_record, write_summary, write_convergence

TRIGGER_ALWAYS = "always"
TRIGGER_CONFUSABLE = "confusable"
_VALID_TRIGGERS = {TRIGGER_ALWAYS, TRIGGER_CONFUSABLE}


def _top1_ema_write(image_feature, clip_logits, text_features, prototype_state, alpha, T, write_class):
    """Unmodified single-class top-1 EMA write -- never gated, never reweighted."""
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


def _nearest_other(prototype_state):
    """Return (nearest_val [C], nearest_idx [C], written mask [C]) for the current bank."""
    proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
    sims = proto_norm @ proto_norm.t()
    sims.fill_diagonal_(-1.0)
    nearest_val, nearest_idx = sims.max(dim=1)
    written = prototype_state.float().norm(dim=-1) > 1e-6
    return nearest_val, nearest_idx, written


def _apply_repulsion(prototype_state, top1, repulsion_lr, percentile, trigger):
    """Push prototype[top1] and its nearest-other class apart, in place.

    Returns (applied: bool, target_confusability: float pre-repulsion,
    post_confusability: Optional[float] -- nearest-other similarity for
    top1 recomputed after the correction, only when applied).
    """
    nearest_val, nearest_idx, written = _nearest_other(prototype_state)
    target_confusability = float(nearest_val[top1].item())

    if repulsion_lr == 0.0 or not bool(written[top1]):
        return False, target_confusability, None
    other = int(nearest_idx[top1].item())
    if not bool(written[other]):
        return False, target_confusability, None

    if trigger == TRIGGER_CONFUSABLE:
        n_written = int(written.sum().item())
        if n_written < 5:
            return False, target_confusability, None
        thresh = torch.quantile(nearest_val[written], percentile / 100.0)
        if not bool((nearest_val[top1] > thresh).item()):
            return False, target_confusability, None
    # trigger == "always": fires unconditionally once top1 and its nearest
    # other class are both written.

    proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
    direction = proto_norm[top1] - proto_norm[other]
    unit_dir = direction / direction.norm().clamp(min=1e-8)
    prototype_state[top1] = (
        prototype_state[top1] + repulsion_lr * prototype_state[top1].norm().clamp(min=1e-8) * unit_dir
    )
    prototype_state[other] = (
        prototype_state[other] - repulsion_lr * prototype_state[other].norm().clamp(min=1e-8) * unit_dir
    )

    post_nearest_val, _, _ = _nearest_other(prototype_state)
    post_confusability = float(post_nearest_val[top1].item())
    return True, target_confusability, post_confusability


class RepulsivePTAAdapter(BaseAdapter):
    """PTA with an unmodified top-1 EMA write plus an explicit prototype-
    repulsion correction, triggered always or only when confusable."""

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

        self.repulsion_lr = float(cfg.get("repulsion_lr", 0.0))
        self.trigger = str(cfg.get("trigger", TRIGGER_CONFUSABLE)).lower()
        if self.trigger not in _VALID_TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(_VALID_TRIGGERS)}, got {self.trigger!r}")
        self.confusability_percentile = float(cfg.get("confusability_percentile", 66.0))

        # repulsion_lr == 0.0 -> the correction is a hard no-op, skip it entirely (control setting).
        self._needs_repulsion = self.repulsion_lr != 0.0

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "RepulsivePTA"),
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
                tqdm(loader, desc=f"[RepulsivePTA] {dataset_name}")
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

                refine_feature, target_prototype = _top1_ema_write(
                    image_features, clip_logits, refine_feature, target_prototype, self.alpha, self.T, top1,
                )

                repulsion_applied = False
                target_confusability = None
                post_repulsion_confusability = None
                if self._needs_repulsion:
                    repulsion_applied, target_confusability, post_repulsion_confusability = _apply_repulsion(
                        target_prototype, top1, self.repulsion_lr, self.confusability_percentile, self.trigger,
                    )
                    if repulsion_applied:
                        refine_feature = self.alpha * text_embeddings.t() + (1.0 - self.alpha) * target_prototype
                        refine_feature = refine_feature / refine_feature.norm(dim=-1, keepdim=True)

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
                        signal_source=self.trigger if self._needs_repulsion else None,
                        target_confusability=target_confusability,
                        repulsion_applied=repulsion_applied if self._needs_repulsion else None,
                        post_repulsion_confusability=post_repulsion_confusability,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- RepulsivePTA(lr={self.repulsion_lr},trigger={self.trigger}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- RepulsivePTA(lr={self.repulsion_lr},trigger={self.trigger}) "
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
                method=os.environ.get("RESULT_LABEL", "RepulsivePTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "RepulsivePTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> RepulsivePTAAdapter:
    """Factory: instantiate a RepulsivePTAAdapter with the given config."""
    return RepulsivePTAAdapter(cfg)
