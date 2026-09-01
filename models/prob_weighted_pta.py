"""Prototype-Based TTA with a probability-weighted top-1 EMA write.

Every write-time trust study so far this round (patch-vote / view-consistency
/ prototype-confusability, hard gate / two-sided reweight / permanent freeze
/ read-time reweight -- see `experimental_results/Phase7_...md`) is
defensive: it reduces or blocks the write when a signal says "don't trust
this sample," and every one of them nets negative, because blocking a write
throws away information the EMA needs regardless of whether that sample was
"clean." This adapter tests a different, non-defensive mechanism: the write
always fires, unmodified in structure (single-class top-1 EMA write, never
gated, never thresholded), but the write *weight* is scaled by the top-1
softmax probability raised to a power `gamma`:

    w_new = (1.0 - torch.exp(-w_top1 / T)) * (w_top1 ** gamma)

`gamma=0.0` makes `w_top1 ** 0 = 1`, so `w_new` is exactly the canonical
value and the control is bit-identical to the established top-1-write
reference (dtd 46.85 / flowers 74.22 / pets 90.75). `gamma > 0` downweights
low-confidence top-1 writes (a high-confidence write is weighted ~as before,
a low-confidence one barely moves the prototype); `gamma < 1` would upweight
them and is rejected at construction (`gamma >= 0.0` enforced).

There is deliberately NO confidence threshold in this write path: the 0.1
gate lives only in the legacy multi-class writer
(`models/image_level/pta_image.py:63-67`), which is not used here.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.records import write_record_header, write_record, write_summary, write_convergence


def _top1_ema_write(image_feature, clip_logits, text_features, prototype_state, alpha, T, gamma, write_class):
    """Single-class top-1 EMA write, probability-weighted by `w_top1 ** gamma`.

    Verbatim copy of the canonical top-1 write
    (`models/repulsive_pta.py:43-54` / `models/drift_gated_repulsion_pta.py:48-59`)
    with exactly one modification: the `w_new` line is scaled by
    `w_top1 ** gamma`. `gamma=0.0` reproduces the canonical write bit-for-bit.

    Returns (refined_text, prototype_state, w_new_vec), where `w_new_vec` is
    the per-class write-weight vector (zeros everywhere except the written
    class) -- recorded so the top-1-only write rule is checkable from the
    record dump.
    """
    w = F.softmax(clip_logits, dim=-1).squeeze(0)
    w_top1 = w[write_class]
    w_new = (1.0 - torch.exp(-w_top1 / T)) * (w_top1 ** gamma)
    prototype_state[write_class] = (
        (1.0 - w_new) * prototype_state[write_class]
        + w_new * image_feature.squeeze(0)
    )
    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

    w_new_vec = torch.zeros_like(w)
    w_new_vec[write_class] = w_new
    return refined_text, prototype_state, w_new_vec


class ProbWeightedPTAAdapter(BaseAdapter):
    """PTA with an unmodified top-1 EMA write whose weight is scaled by
    `w_top1 ** gamma` (gamma=0.0 == canonical write, bit-identical)."""

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

        self.gamma = float(cfg.get("gamma", 0.0))
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0.0, got {self.gamma}")

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "ProbWeightedPTA"),
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
                tqdm(loader, desc=f"[ProbWeightedPTA] {dataset_name}")
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

                refine_feature, target_prototype, w_new_vec = _top1_ema_write(
                    image_features, clip_logits, refine_feature, target_prototype,
                    self.alpha, self.T, self.gamma, top1,
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
                        proto_alpha=w_new_vec,
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
                        signal_source=None,
                        target_confusability=None,
                        repulsion_applied=None,
                        post_repulsion_confusability=None,
                        drift_velocity=None,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- ProbWeightedPTA(gamma={self.gamma}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- ProbWeightedPTA(gamma={self.gamma}) "
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
                method=os.environ.get("RESULT_LABEL", "ProbWeightedPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "ProbWeightedPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> ProbWeightedPTAAdapter:
    """Factory: instantiate a ProbWeightedPTAAdapter with the given config."""
    return ProbWeightedPTAAdapter(cfg)