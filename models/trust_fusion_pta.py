"""Prototype-Based TTA with a trust-adaptive READ-time fusion weight.

Part 4a/4b (see ``experimental_results/PatchModPTA_Purity_Separability_Trust_Analysis.md``)
showed that no scalar lever applied to the *write* rule of the image-level
prototype (hard gate, up-weight, down-weight, two-sided) improves on base PTA:
the confidence x patch-agreement trust signal is a real diagnostic but does
not identify *mistakes* well enough to safely gate or reweight what gets
written.

This adapter tests the complementary, previously-untried lever: leave the
write rule byte-identical to base PTA (``PTAImageLevel.update_prototypes``,
unmodified), and instead scale how much the *existing* prototype is trusted
at prediction time, per sample, via ``TrustAdaptiveFusion``
(``models/fusion.py``):

    trusted   = (clip_margin >= conf_margin_thresh) and (patch_vote_pred == argmax(clip))
    tau_eff   = tau_image_proto * (tau_scale_trusted if trusted else tau_scale_untrusted)
    final     = clip + tau_eff * image_proto

Rationale (Part 3.1/3.3): when CLIP and the patch vote agree and CLIP is
confident, CLIP alone is already highly reliable, so the prototype's
dissenting opinion should count for *less* there (`tau_scale_trusted < 1`).
When they disagree or CLIP is ambiguous, CLIP alone is markedly less
reliable, so the prototype's opinion should count for *more* there
(`tau_scale_untrusted > 1`). `(tau_scale_trusted, tau_scale_untrusted) ==
(1.0, 1.0)` reproduces base PTA exactly (in-grid control).

The agree signal uses the same stateless topk20 patch vote
(``utils.patch_vote.compute_patch_vote``) as the write-gate/reweight studies
-- it deliberately does NOT touch the ``patch_proto`` bank (there is none
here; ``patch_proto_logits`` is always ``None``).
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
from utils.patch_vote import compute_patch_vote
from utils.records import write_record_header, write_record, write_summary


class TrustFusionPTAAdapter(BaseAdapter):
    """PTA with an unmodified write rule and a trust-adaptive fusion weight.

    Write: identical to ``models.pta.PTAAdapter`` (``PTAImageLevel``'s
    multi-class soft EMA, ``w >= 0.1`` gate). Read: ``TrustAdaptiveFusion``
    scales ``tau_image_proto`` per sample by the causal confidence x
    patch-agreement trust signal computed here.
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

        self.conf_margin_thresh = float(cfg.get("conf_margin_thresh", 0.2))
        self.pv_aggregation = str(cfg.get("pv_aggregation", "topk20"))

    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "TrustFusionPTA"),
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
                tqdm(loader, desc=f"[TrustFusionPTA] {dataset_name}")
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

                # ── Causal trust signal (frozen clip logits + stateless
                #    patch vote, no bank read) ────────────────────────────
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

                trusted = (
                    clip_margin >= self.conf_margin_thresh
                    and patch_vote_pred == top1
                )

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
                        patch_vote_pred=patch_vote_pred,
                        patch_vote_margin=patch_vote_margin,
                        clip_margin=clip_margin,
                        tau_eff=float(tau_eff),
                        trust_regime="trusted" if trusted else "untrusted",
                    )

                if i % 1000 == 0:
                    print(
                        f"---- TrustFusionPTA test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- TrustFusionPTA test accuracy: {final_acc:.2f}. ----\n")

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
                method=os.environ.get("RESULT_LABEL", "TrustFusionPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        label = os.environ.get("RESULT_LABEL", "TrustFusionPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> TrustFusionPTAAdapter:
    """Factory: instantiate a TrustFusionPTAAdapter with the given config."""
    return TrustFusionPTAAdapter(cfg)
