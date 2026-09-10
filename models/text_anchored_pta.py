"""Text-anchored PTA: the canonical top-1 EMA write, except the written
feature is anchored toward the current refined text row of the predicted
class instead of being the raw image feature alone.

The write rule is identical to the canonical top-1 EMA write
(``_top1_ema_write`` in ``models/drift_gated_repulsion_pta.py`` -- single
class, top-1, always writes, no confidence threshold) except that the
feature written into the prototype is a convex blend:

    blend = anchor_mix * image_feature + (1.0 - anchor_mix) * text_features[top1]

where ``text_features[top1]`` is the CURRENT refined text row (the same
``refined_text`` the canonical write re-derives at its line 57). The
hypothesis: anchoring each write toward the class's text embedding keeps the
prototype from drifting into a neighboring class's region on low-confidence
samples, at the cost of pulling the prototype toward the (static) text
anchor. ``anchor_mix=1.0`` is the canonical write and must reproduce the
established top-1-write reference (dtd 46.85 / flowers 74.22 / pets 90.75).
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.records import write_record_header, write_record, write_summary, write_convergence


def _top1_ema_write(image_feature, clip_logits, text_features, prototype_state, alpha, T, write_class, anchor_mix):
    """Canonical top-1 EMA write, except the written feature is a blend of the
    image feature and the current refined text row for the top-1 class."""
    w = F.softmax(clip_logits, dim=-1).squeeze(0)
    w_top1 = w[write_class]
    w_new = 1.0 - torch.exp(-w_top1 / T)
    blend = anchor_mix * image_feature.squeeze(0) + (1.0 - anchor_mix) * text_features[write_class]
    prototype_state[write_class] = (
        (1.0 - w_new) * prototype_state[write_class]
        + w_new * blend
    )
    refined_text = alpha * text_features + (1.0 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, prototype_state


class TextAnchoredPTAAdapter(BaseAdapter):
    """PTA with a top-1 EMA write whose written feature is anchored toward the
    current refined text row of the predicted class (``anchor_mix`` blend)."""

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

        self.anchor_mix = float(cfg.get("anchor_mix", 1.0))
        if not (0.0 <= self.anchor_mix <= 1.0):
            raise ValueError(
                f"anchor_mix must be in [0.0, 1.0], got {self.anchor_mix}"
            )

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "TextAnchoredPTA"),
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
                tqdm(loader, desc=f"[TextAnchoredPTA] {dataset_name}")
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
                    image_features, clip_logits, refine_feature, target_prototype,
                    self.alpha, self.T, top1, self.anchor_mix,
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
                    )

                if i % 1000 == 0:
                    print(
                        f"---- TextAnchoredPTA(anchor_mix={self.anchor_mix}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- TextAnchoredPTA(anchor_mix={self.anchor_mix}) "
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
                method=os.environ.get("RESULT_LABEL", "TextAnchoredPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "TextAnchoredPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> TextAnchoredPTAAdapter:
    """Factory: instantiate a TextAnchoredPTAAdapter with the given config."""
    return TextAnchoredPTAAdapter(cfg)