"""
Patch-Modulated PTA: Three-way fusion with quality-gated modulation of
image-level prototype update rate.

This adapter composes independent components:
  - Image-level (PTA-style): per-class running prototype via EMA
  - Patch-level (Gaussian-style): per-class prototype centers + variance
  - Fusion: QualityGatedFusion with tunable tau weights

The patch-level Gaussian prototype quality (variance of raw_proto scores)
is computed BEFORE the image-level prototype update and used to modulate
the EMA update rate via an additional quality_gate parameter.

Key difference from the basic PTA adapter:
  Patch-level quality information flows back to influence the image-level
  adaptation rate, making the update sensitive to patch-level evidence.
"""
import os
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.patch_level import create as create_patch_level
from models.patch_level.base import _alpha_from_evidence, _safe_normalize
from models.fusion import QualityGatedFusion
from utils import cls_acc, get_clip_logits


# ------------------------------------------------------------------
# Quality-gated prototype update (mirrors
# models.exp12_patch_quality_modulation.update_text_features_with_quality)
# ------------------------------------------------------------------

def _update_text_features_with_quality(
    image_feature: torch.Tensor,
    probs: torch.Tensor,
    text_features: torch.Tensor,
    target_prototype: torch.Tensor,
    alpha: float = 0.01,
    T: float = 20.0,
    quality_gate: torch.Tensor = None,
    quality_modulation: float = 0.0,
):
    """
    Quality-gated EMA prototype update.

    When quality_gate is provided and quality_modulation > 0, the update
    weight for high-confidence classes is amplified:
        w_new *= (1 + quality_modulation * quality_gate)

    All other logic is identical to PTAImageLevel.update_prototypes.

    Args:
        image_feature:    (1, D) L2-normalised image embedding.
        probs:            (1, C) soft-max distribution over classes.
        text_features:    (C, D) original CLIP text embeddings.
        target_prototype: (C, D) running prototype bank (mutated in-place).
        alpha:            Weight on original text features.
        T:                Temperature controlling update rate.
        quality_gate:     Scalar tensor [0,1] from patch-level evidence quality.
        quality_modulation: How much quality_gate amplifies the update.

    Returns:
        refined_text:     (C, D) L2-normalised updated text features.
        target_prototype: (C, D) updated prototype bank.
    """
    # Extract soft probabilities [C] from batch dim
    w = probs.squeeze(0)                          # [C] — class confidence

    # Compute update weights via exponential decay: w_new = 1 - exp(-w / T)
    # Only apply to high-confidence classes (w >= 0.1)
    w_new = torch.zeros_like(w)                   # [C]
    mask = w >= 1e-1                              # [C] bool
    w_new[mask] = 1 - torch.exp(-w[mask] / T)     # [C]

    # Quality-gated modulation: amplify update when patch evidence is strong
    if quality_gate is not None and quality_modulation > 0:
        w_new[mask] *= (1 + quality_modulation * quality_gate)

    w_new = w_new.unsqueeze(1)                    # [C, 1]

    # EMA update: blend old prototype with new image feature
    target_prototype[mask] = (
        (1 - w_new[mask]) * target_prototype[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )

    # Form refined text features as a blend of original CLIP text + updated prototype
    refined_text = alpha * text_features + (1 - alpha) * target_prototype  # [C, D]

    # L2-normalize so cosine similarity = dot product
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)  # [C, D]

    return refined_text, target_prototype


class PatchModulatedPTAAdapter(BaseAdapter):
    """Three-way fusion with quality-gated modulation of image-level
    prototype update rate.

    Composes independent image-level, patch-level, and fusion components
    to replicate the Exp12 algorithm.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Create component sub-adapters (via factory functions that handle
        # flat-to-nested config wrapping)
        self.image_level = create_image_level(cfg)
        self.patch_level = create_patch_level(cfg)

        # Construct nested config for fusion so QualityGatedFusion reads
        # the flat top-level tau / quality keys rather than defaults
        fusion_cfg = {
            "fusion": {
                "tau_text":          float(cfg.get("tau_text", 1.0)),
                "tau_image_proto":   float(cfg.get("tau_image_proto", 100.0)),
                "tau_patch_proto":   float(cfg.get("tau_patch_proto", 20.0)),
                "quality_modulation": float(cfg.get("quality_modulation", 1.0)),
            }
        }
        self.fusion = QualityGatedFusion(fusion_cfg)

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(
        self,
        loader,
        clip_model,
        clip_weights,
        dataset_name: str,
    ) -> float:
        # ── Backbone check ──────────────────────────────────────────
        if not hasattr(clip_model.visual, "positional_embedding"):
            raise ValueError(
                "Exp12PatchQualityModulation requires a ViT backbone (ViT-B/16) "
                "for patch extraction. "
                f"Got: {type(clip_model.visual).__name__}"
            )

        # ── Config ──────────────────────────────────────────────────
        max_K              = int(self.cfg.get("max_K", 100))
        match_thresh       = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh        = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half             = float(self.cfg.get("n_half", 15.0))
        alpha_max          = float(self.cfg.get("proto_alpha_max", 0.2))

        # PTA-style image-level update params
        alpha_pta = float(self.cfg.get("alpha", 0.01))
        T         = float(self.cfg.get("T", 50.0))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(clip_weights.t().float())  # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        # ── Dual prototype systems ──────────────────────────────────
        # Image-level (PTA-style)
        refine_feature   = clip_weights.t().float()   # [C, D]
        target_prototype = self.image_level.init_state(refine_feature)

        # Patch-level (Gaussian-style)
        states = self.patch_level.init_state(refine_feature)

        max_batches = int(os.environ.get("MAX_BATCHES", "0"))
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[PatchModulatedPTA] {dataset_name}")
            ):
                if max_batches > 0 and i >= max_batches:
                    break
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # 1) CLIP forward
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, clip_model, clip_weights
                )
                feat      = image_features.squeeze(0).float()
                feat_norm = _safe_normalize(feat)

                # 2) Patch-level Gaussian prototype scores + quality gate
                #    (computed BEFORE image-level update so quality_gate
                #     can modulate the EMA rate)
                patch_proto_logits, quality_gate = (
                    self.patch_level.compute_patch_logits(images, clip_model, states)
                )

                # 3) Adaptive evidence weighting for patch-level contribution
                proto_alpha = torch.tensor(
                    [
                        min(alpha_max, _alpha_from_evidence(
                            states[c]["n_images"], n_half
                        ))
                        for c in range(C)
                    ],
                    device=device,
                )

                # 4) Update image-level prototype (WITH quality modulation)
                soft_logits = F.softmax(clip_logits, dim=-1)
                refine_feature, target_prototype = _update_text_features_with_quality(
                    image_features,
                    soft_logits.half(),
                    refine_feature,
                    target_prototype,
                    alpha=alpha_pta,
                    T=T,
                    quality_gate=quality_gate,
                    quality_modulation=self.fusion.quality_modulation,
                )

                # 5) Image-level proto logits
                image_proto_logits = (
                    image_features.half() @ refine_feature.half().T
                )  # [1, C]

                # 6) Three-way fusion with quality-gated patch modulation
                final_logits = self.fusion.forward(
                    clip_logits,
                    image_proto_logits,
                    patch_proto_logits,
                    quality_gate=quality_gate,
                    proto_alpha=proto_alpha,
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # 7) Online memory update (patch-level, binary top-1 gate)
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)
                top2_vals, top2_idx = pred_conf.topk(min(2, C))
                best_conf   = float(top2_vals[0].item())
                second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                conf_margin = best_conf - second_conf
                best_cls    = int(top2_idx[0].item())

                if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                    states[best_cls] = self.patch_level.update_state(
                        states[best_cls], images, clip_model, feat_norm,
                        is_high_confidence=True,
                    )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- PatchModulatedPTA {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- PatchModulatedPTA FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(
                f"PatchModulatedPTA's performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc

    def refine_with(self, clip_model, clip_weights, data_loader, dataset_name):
        """Convenience alias for run()."""
        return self.run(data_loader, clip_model, clip_weights, dataset_name)


def build(cfg: dict) -> PatchModulatedPTAAdapter:
    """Factory function — called by runner.py via dynamic import.

    Usage in runner.py:
        adapter_module = __import__('models.patch_modulated_pta', fromlist=['build'])
        adapter = adapter_module.build(cfg)
    """
    return PatchModulatedPTAAdapter(cfg)
