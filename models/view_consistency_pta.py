"""Prototype-Based TTA with a multi-view-consistency trust-adaptive
READ-time fusion weight.

Phase 3 of the "structurally different directions" round. This is the same
``TrustAdaptiveFusion`` lever tested in ``models/trust_fusion_pta.py``
(Phase 1 of that round, no-go), but with the trust signal swapped from the
patch-vote agreement to a fundamentally different source: agreement between
CLIP's zero-shot prediction on the ORIGINAL image and its predictions on
(N-1) randomly perturbed copies of the same image (rotation, affine,
brightness/contrast, occasional grayscale/edge-blend --
``utils/augmentation.py:_augment_image``, the same pipeline already used to
build the patch-level Gaussian bank).

``scripts/check_view_consistency_signal.py`` (Phase 3a) validated this
signal in isolation: a +16.6 to +27.4pp purity gap between agree/disagree
cases on the 3 dev datasets -- comparable to or stronger than the
already-validated patch-vote signal (+9 to +35pp). This adapter tests
whether a BETTER-VALIDATED trust signal unlocks the same read-time
reweighting lever that failed with the patch-vote signal, or whether
read-time reweighting is itself a dead end regardless of signal source.

Write rule is byte-identical to base PTA (``PTAImageLevel.update_prototypes``,
unmodified) -- only the fusion READ weight on image_proto is scaled per
sample:

    agreement = fraction of (n_views-1) augmented views whose top-1 matches
                the original image's top-1 zero-shot prediction
    trusted   = agreement >= agreement_thresh
    tau_eff   = tau_image_proto * (tau_scale_trusted if trusted else tau_scale_untrusted)
    final     = clip + tau_eff * image_proto

``(tau_scale_trusted, tau_scale_untrusted) == (1.0, 1.0)`` reproduces base
PTA exactly (in-grid control) regardless of ``n_views``/``agreement_thresh``,
since the trust value never affects the write side or the fusion weights
when both scales are 1.0.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.fusion import TrustAdaptiveFusion
from utils import get_clip_logits, cls_acc
from utils.clip_inference import _safe_normalize
from utils.augmentation import _augment_image
from utils.records import write_record_header, write_record, write_summary


class ViewConsistencyPTAAdapter(BaseAdapter):
    """PTA with an unmodified write rule and a multi-view-consistency
    trust-adaptive fusion weight.

    Write: identical to ``models.pta.PTAAdapter``. Read: ``TrustAdaptiveFusion``
    scales ``tau_image_proto`` per sample by the causal multi-view agreement
    signal computed here (no patch content, no CLIP-confidence margin).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg
        self.image_level = create_image_level(cfg)

        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = TrustAdaptiveFusion(cfg)

        self.n_views = int(cfg.get("n_views", 4))
        self.agreement_thresh = float(cfg.get("agreement_thresh", 1.0))

    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "ViewConsistencyPTA"),
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

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[ViewConsistencyPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── Write (byte-identical to base PTA) ──────────────────────
                refine_feature, target_prototype = (
                    self.image_level.update_prototypes(
                        image_features,
                        clip_logits,
                        refine_feature,
                        target_prototype,
                    )
                )

                # ── Causal multi-view consistency trust signal (frozen clip
                #    logits + independently perturbed forward passes; no
                #    bank read, no patch content) ─────────────────────────
                top1 = int(clip_logits.argmax(dim=-1).item())
                if self.n_views > 1:
                    aug_views = [_augment_image(images) for _ in range(self.n_views - 1)]
                    batched = torch.cat(aug_views, dim=0).cuda()
                    aug_feats = encoder.encode_image(batched)
                    aug_feats = _safe_normalize(aug_feats.float())
                    aug_logits = 100.0 * aug_feats @ text_embeddings.float()
                    aug_preds = aug_logits.argmax(dim=-1)
                    agreement = float((aug_preds == top1).float().mean().item())
                else:
                    agreement = 1.0

                trusted = agreement >= self.agreement_thresh

                # ── Fused prediction (trust-adaptive tau_image_proto) ───────
                image_proto_logits = self.image_level.compute_logits(
                    image_features, refine_feature
                )
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None, trusted=trusted
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                if record_dir:
                    tau_eff = self.fusion.tau_image_proto * (
                        self.fusion.tau_scale_trusted
                        if trusted
                        else self.fusion.tau_scale_untrusted
                    )
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
                        tau_eff=float(tau_eff),
                        trust_regime="trusted" if trusted else "untrusted",
                        view_agreement=agreement,
                    )

                if i % 1000 == 0:
                    print(
                        f"---- ViewConsistencyPTA test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- ViewConsistencyPTA test accuracy: {final_acc:.2f}. ----\n")

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
                method=os.environ.get("RESULT_LABEL", "ViewConsistencyPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        label = os.environ.get("RESULT_LABEL", "ViewConsistencyPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> ViewConsistencyPTAAdapter:
    """Factory: instantiate a ViewConsistencyPTAAdapter with the given config."""
    return ViewConsistencyPTAAdapter(cfg)
