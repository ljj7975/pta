"""Prototype-Based TTA with a class-level confusability freeze on the image
prototype.

Phase 2 of the "structurally different directions" round (see
experimental_results/README.md and the plan this implements). Unlike every
previous lever in this codebase (all of which gate/reweight on a PER-SAMPLE
CLIP-confidence or patch-vote signal), this adapter watches the prototype
BANK'S OWN GEOMETRY. After each write, it computes each class's cosine
similarity to its nearest OTHER class's prototype ("confusability").
``scripts/check_prototype_confusability.py`` (Phase 2a) found this is a real
diagnostic signal purely from the bank's final state (+16.5pp / +16.3pp /
+9.2pp low-vs-high-confusability tercile accuracy gap on dtd / oxford_flowers
/ oxford_pets respectively) -- entirely independent of CLIP confidence or
patch content.

Raw cosine confusability values sit in a narrow, dataset-dependent band
(observed: dtd 0.85-0.98, oxford_flowers 0.00-1.00, oxford_pets 0.93-1.00 --
see outputs/prototype_confusability_report.md), so a single fixed absolute
cosine cutoff is not comparable across datasets. Instead the gate uses a
RUNNING PERCENTILE of the current per-class confusability array: a class is
permanently frozen (no further EMA writes for the rest of the run) once its
confusability exceeds the ``freeze_percentile``-th percentile of all classes'
CURRENT confusability values. This adapts automatically to each dataset's own
similarity scale, and only ever uses information available at that point in
the (causal, online) run.

``freeze_percentile=100`` is the control: the threshold equals the current
max and a strict ``>`` comparison means no class can ever exceed its own
array's maximum, so nothing freezes and this exactly reproduces base PTA.

Prediction rule is identical to base PTA in every setting (WeightedFusion,
tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0) -- only per-class
write eligibility differs, so any accuracy delta is due solely to the
resulting differences in the prototype bank.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.clip_inference import _safe_normalize
from utils.records import write_record_header, write_record, write_summary


def _confusability_gated_update(
    image_feature,
    clip_logits,
    text_features,
    prototype_state,
    alpha,
    T,
    frozen_mask,
):
    """Same math as PTAImageLevel.update_prototypes (models/image_level/pta_image.py),
    with frozen classes excluded from the write mask."""
    probs = F.softmax(clip_logits, dim=-1)          # (1, C)
    w = probs.squeeze(0)                             # (C)

    w_new = torch.zeros_like(w)
    mask = (w >= 1e-1) & (~frozen_mask)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)

    prototype_state[mask] = (
        (1 - w_new[mask]) * prototype_state[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )

    refined_text = alpha * text_features + (1 - alpha) * prototype_state
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, prototype_state


class ConfusabilityGatedPTAAdapter(BaseAdapter):
    """PTA with a class-level write freeze based on prototype self-confusability."""

    def __init__(self, cfg):
        super().__init__(cfg)

        image_level_cfg = cfg.get("image_level", {})
        self.alpha = float(image_level_cfg.get("alpha", cfg.get("alpha", 0.01)))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

        # 100 = off (control, exactly reproduces base PTA). Lower = more
        # aggressive freezing (freezes classes above a lower percentile of
        # the CURRENT confusability distribution).
        self.freeze_percentile = float(cfg.get("freeze_percentile", 100.0))

    def run(self, loader, encoder, text_embeddings, dataset_name):
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "ConfusabilityGatedPTA"),
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
            prototype_state = torch.zeros_like(refine_feature)
            frozen_mask = torch.zeros(num_classes, dtype=torch.bool, device=refine_feature.device)

            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            quantile_q = min(1.0, max(0.0, self.freeze_percentile / 100.0))

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[ConfusabilityGatedPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                refine_feature, prototype_state = _confusability_gated_update(
                    image_features, clip_logits, refine_feature, prototype_state,
                    self.alpha, self.T, frozen_mask,
                )

                # ── Recompute confusability from the just-updated bank and
                #    extend the freeze mask (monotonic: once frozen, stays
                #    frozen for the rest of the run) ────────────────────────
                # The percentile is computed only over ALREADY-WRITTEN classes
                # (nonzero prototype norm) -- including untouched zero-vector
                # classes skews the threshold low the moment more than
                # `(100-freeze_percentile)%` of classes are still unwritten,
                # causing spurious mass-freezing in the first few hundred
                # samples (observed empirically: p=70 froze 26/47 classes
                # within 300 samples before this fix). A minimum
                # written-class count avoids freezing off tiny-sample noise.
                proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
                sims = proto_norm @ proto_norm.t()
                sims.fill_diagonal_(-1.0)
                nearest_other, _ = sims.max(dim=1)
                written = prototype_state.float().norm(dim=-1) > 1e-6
                n_written = int(written.sum().item())
                if n_written >= 5:
                    thresh = torch.quantile(nearest_other[written], quantile_q)
                    newly_frozen = written & (nearest_other > thresh)
                    frozen_mask = frozen_mask | newly_frozen

                # ── Fused prediction (identical rule across all settings) ──
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
                        n_frozen=int(frozen_mask.sum().item()),
                        max_confusability=float(nearest_other.max().item()),
                    )

                if i % 1000 == 0:
                    print(
                        f"---- ConfusabilityGatedPTA(p={self.freeze_percentile}) test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. n_frozen={int(frozen_mask.sum().item())} ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- ConfusabilityGatedPTA(p={self.freeze_percentile}) test accuracy: "
            f"{final_acc:.2f}. Final n_frozen={int(frozen_mask.sum().item())}/{num_classes} ----\n"
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
                method=os.environ.get("RESULT_LABEL", "ConfusabilityGatedPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        label = os.environ.get("RESULT_LABEL", "ConfusabilityGatedPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> ConfusabilityGatedPTAAdapter:
    """Factory: instantiate a ConfusabilityGatedPTAAdapter with the given config."""
    return ConfusabilityGatedPTAAdapter(cfg)
