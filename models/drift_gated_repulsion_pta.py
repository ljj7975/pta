"""Prototype-Based TTA with drift-gated repulsion: the same explicit
geometry-correction idea from `models/repulsive_pta.py` (Phase 8), but only
fires when the confusability signal is corroborated by a second, independent
signal -- whether the target class's own prototype is still actively moving
(drift) -- rather than firing on static closeness alone.

Phase 8 found the confusability-repulsion mechanism does exactly what it
claims (nearest-other-class similarity measurably drops every time it
fires) yet regresses `oxford_pets` monotonically at every dose. Root cause:
static closeness can't distinguish "this class's prototype recently drifted
toward another class" (repulsion is a legitimate fix) from "these two
classes are just genuinely similar in CLIP's embedding space" (repulsion
distorts an already-reasonable boundary -- plausible for oxford_pets'
fine-grained breeds). This adapter adds a per-class drift-velocity EMA and
gates repulsion on it:

    drift_confusable   -- fires iff confusable AND drift_ema[top1] is above
                           the median (drift_percentile) over written classes
    stable_confusable  -- fires iff confusable AND NOT drifting (diagnostic
                           counterfactual: deliberately targets the case the
                           hypothesis says repulsion should avoid)
    confusable         -- Phase 8's original behavior, drift-independent
                           (kept for a same-file sanity check; not swept)
    confusable_with_floor -- Phase 8's confusable check PLUS the repulsion
                           floor, no drift condition (the floor is the only
                           gate beyond confusability)
    drift_only         -- confusable AND drift-gated, with the repulsion
                           floor skipped (isolates the drift gate)
    bilateral_drift    -- confusable AND floor-gated, fires iff EITHER class
                           in the confused pair (top1 OR nearest-other) is
                           drifting above the percentile threshold

A repulsion floor (relative, `repulsion_floor_percentile`) is applied in all
non-zero-lr settings: once a pair is already among the more-separated
classes (relative to the current bank), stop re-firing on it -- targets
Phase 8's other failure mode, unbounded compounding at high dose.
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

TRIGGER_CONFUSABLE = "confusable"
TRIGGER_DRIFT_CONFUSABLE = "drift_confusable"
TRIGGER_STABLE_CONFUSABLE = "stable_confusable"
TRIGGER_CONFUSABLE_WITH_FLOOR = "confusable_with_floor"
TRIGGER_DRIFT_ONLY = "drift_only"
TRIGGER_BILATERAL_DRIFT = "bilateral_drift"
_VALID_TRIGGERS = {
    TRIGGER_CONFUSABLE,
    TRIGGER_DRIFT_CONFUSABLE,
    TRIGGER_STABLE_CONFUSABLE,
    TRIGGER_CONFUSABLE_WITH_FLOOR,
    TRIGGER_DRIFT_ONLY,
    TRIGGER_BILATERAL_DRIFT,
}


def _top1_ema_write(image_feature, clip_logits, text_features, prototype_state, alpha, T, write_class):
    """Verbatim copy of models/repulsive_pta.py's unmodified top-1 EMA write."""
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
    """Verbatim copy of models/repulsive_pta.py's helper."""
    proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
    sims = proto_norm @ proto_norm.t()
    sims.fill_diagonal_(-1.0)
    nearest_val, nearest_idx = sims.max(dim=1)
    written = prototype_state.float().norm(dim=-1) > 1e-6
    return nearest_val, nearest_idx, written


def _update_drift(prototype_state, top1, drift_ema, prev_dir, drift_decay):
    """EMA of the angular step size at each write to class `top1`, in place."""
    proto_dir = _safe_normalize(prototype_state[top1].unsqueeze(0), dim=-1).squeeze(0)
    if prev_dir[top1].norm() > 1e-6:
        angular_step = 1.0 - float((proto_dir * prev_dir[top1]).sum().item())
        drift_ema[top1] = drift_decay * drift_ema[top1] + (1.0 - drift_decay) * angular_step
    prev_dir[top1] = proto_dir.clone()


def _apply_repulsion_v2(
    prototype_state, top1, drift_ema, repulsion_lr, confusability_percentile,
    drift_percentile, trigger, repulsion_floor_percentile,
):
    """Returns (applied, target_confusability, post_repulsion_confusability, drift_velocity)."""
    nearest_val, nearest_idx, written = _nearest_other(prototype_state)
    target_confusability = float(nearest_val[top1].item())

    if repulsion_lr == 0.0 or not bool(written[top1]):
        return False, target_confusability, None, None
    other = int(nearest_idx[top1].item())
    if not bool(written[other]):
        return False, target_confusability, None, None

    n_written = int(written.sum().item())
    if n_written < 5:
        return False, target_confusability, None, None

    conf_thresh = torch.quantile(nearest_val[written], confusability_percentile / 100.0)
    if not bool((nearest_val[top1] > conf_thresh).item()):
        return False, target_confusability, None, float(drift_ema[top1].item())

    require_floor = trigger != TRIGGER_DRIFT_ONLY
    if require_floor:
        floor_thresh = torch.quantile(nearest_val[written], repulsion_floor_percentile / 100.0)
        if bool((nearest_val[top1] <= floor_thresh).item()):
            return False, target_confusability, None, float(drift_ema[top1].item())

    drift_velocity = float(drift_ema[top1].item())
    if trigger in (TRIGGER_DRIFT_CONFUSABLE, TRIGGER_STABLE_CONFUSABLE, TRIGGER_DRIFT_ONLY):
        drift_thresh = torch.quantile(drift_ema[written], drift_percentile / 100.0)
        is_drifting = bool((drift_ema[top1] > drift_thresh).item())
        if trigger in (TRIGGER_DRIFT_CONFUSABLE, TRIGGER_DRIFT_ONLY) and not is_drifting:
            return False, target_confusability, None, drift_velocity
        if trigger == TRIGGER_STABLE_CONFUSABLE and is_drifting:
            return False, target_confusability, None, drift_velocity
    elif trigger == TRIGGER_BILATERAL_DRIFT:
        # Fires iff either class in the confused pair is drifting.
        drift_thresh = torch.quantile(drift_ema[written], drift_percentile / 100.0)
        is_drifting_top1 = bool((drift_ema[top1] > drift_thresh).item())
        is_drifting_other = bool((drift_ema[other] > drift_thresh).item())
        if not (is_drifting_top1 or is_drifting_other):
            return False, target_confusability, None, drift_velocity
    # trigger in ("confusable", "confusable_with_floor"): no drift condition -- Phase 8 behavior.

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
    return True, target_confusability, post_confusability, drift_velocity


class DriftGatedRepulsionPTAAdapter(BaseAdapter):
    """PTA with an unmodified top-1 EMA write plus a repulsion correction
    gated on confusability AND (optionally) drift velocity."""

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
        self.trigger = str(cfg.get("trigger", TRIGGER_DRIFT_CONFUSABLE)).lower()
        if self.trigger not in _VALID_TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(_VALID_TRIGGERS)}, got {self.trigger!r}")
        self.confusability_percentile = float(cfg.get("confusability_percentile", 66.0))
        self.drift_percentile = float(cfg.get("drift_percentile", 50.0))
        self.drift_decay = float(cfg.get("drift_decay", 0.7))
        self.repulsion_floor_percentile = float(cfg.get("repulsion_floor_percentile", 20.0))

        self._needs_repulsion = self.repulsion_lr != 0.0

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "DriftGatedRepulsionPTA"),
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
            D = text_embeddings.shape[0]
            target_prototype = torch.zeros_like(refine_feature)
            drift_ema = torch.zeros(num_classes, dtype=torch.float32, device=target_prototype.device)
            prev_dir = torch.zeros(num_classes, D, dtype=torch.float32, device=target_prototype.device)

            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[DriftGatedRepulsionPTA] {dataset_name}")
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
                _update_drift(target_prototype, top1, drift_ema, prev_dir, self.drift_decay)

                repulsion_applied = False
                target_confusability = None
                post_repulsion_confusability = None
                drift_velocity = None
                if self._needs_repulsion:
                    repulsion_applied, target_confusability, post_repulsion_confusability, drift_velocity = (
                        _apply_repulsion_v2(
                            target_prototype, top1, drift_ema, self.repulsion_lr,
                            self.confusability_percentile, self.drift_percentile,
                            self.trigger, self.repulsion_floor_percentile,
                        )
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
                        drift_velocity=drift_velocity,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- DriftGatedRepulsionPTA(lr={self.repulsion_lr},trigger={self.trigger}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- DriftGatedRepulsionPTA(lr={self.repulsion_lr},trigger={self.trigger}) "
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
                method=os.environ.get("RESULT_LABEL", "DriftGatedRepulsionPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "DriftGatedRepulsionPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> DriftGatedRepulsionPTAAdapter:
    """Factory: instantiate a DriftGatedRepulsionPTAAdapter with the given config."""
    return DriftGatedRepulsionPTAAdapter(cfg)
