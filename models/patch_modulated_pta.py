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
from models.patch_level.base import _alpha_from_evidence
from utils.clip_inference import _safe_normalize
from models.fusion import QualityGatedFusion, ProtoAlphaFusion

# Map of fusion type strings to classes.
FUSION_REGISTRY = {
    "QualityGatedFusion": QualityGatedFusion,
    "ProtoAlphaFusion": ProtoAlphaFusion,
    "NoQualityGateFusion": ProtoAlphaFusion,  # backward compat alias
}
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

        # Read fusion params from nested "fusion" dict (standard),
        # falling back to top-level keys for backward compat with flat configs.
        _fusion_cfg = cfg.get("fusion", {})
        fusion_type = _fusion_cfg.get("type", "QualityGatedFusion")
        fusion_cfg = {
            "fusion": {
                "tau_text":          float(_fusion_cfg.get("tau_text", cfg.get("tau_text", 1.0))),
                "tau_image_proto":   float(_fusion_cfg.get("tau_image_proto", cfg.get("tau_image_proto", 100.0))),
                "tau_patch_proto":   float(_fusion_cfg.get("tau_patch_proto", cfg.get("tau_patch_proto", 20.0))),
            }
        }
        fusion_cls = FUSION_REGISTRY.get(fusion_type, QualityGatedFusion)
        self.fusion = fusion_cls(fusion_cfg)
        self.quality_modulation = float(_fusion_cfg.get("quality_modulation", cfg.get("quality_modulation", 1.0)))

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        # ── Backbone check ──────────────────────────────────────────
        if not hasattr(encoder.visual, "positional_embedding") and not hasattr(encoder.visual, "pos_embed"):
            raise ValueError(
                "Exp12PatchQualityModulation requires a ViT backbone (ViT-B/16) "
                "for patch extraction. "
                f"Got: {type(encoder.visual).__name__}"
            )

        # ── Config ──────────────────────────────────────────────────
        max_K              = int(self.cfg.get("max_K", 100))
        match_thresh       = float(self.cfg.get("match_threshold", 0.60))
        _pl_cfg            = self.cfg.get("patch_level", {})
        conf_thresh        = float(_pl_cfg.get("conf_threshold", self.cfg.get("conf_threshold", 0.5)))
        conf_margin_thresh = float(_pl_cfg.get("conf_margin_threshold", self.cfg.get("conf_margin_threshold", 0.05)))
        n_half             = float(self.cfg.get("n_half", 15.0))
        alpha_max          = float(self.cfg.get("proto_alpha_max", 0.2))
        conf_source        = str(self.cfg.get("conf_source", "text"))
        multi_gate         = bool(self.cfg.get("multi_gate", False))
        # Which logits to use for the confidence gate (step 7):
        #   "text"  — clip_logits (zero-shot CLIP text)
        #   "image" — image_proto_logits (image-level prototype only)
        #   "pta"   — clip_logits + tau_image_proto * image_proto_logits
        #   "full"  — final_logits (text + image + patch fusion)

        # PTA-style image-level update params (read from nested image_level
        # config; fall back to flat keys for backward compat)
        _il_cfg = self.cfg.get("image_level", {})
        alpha_pta = float(_il_cfg.get("alpha", self.cfg.get("alpha", 0.01)))
        T         = float(_il_cfg.get("T",     self.cfg.get("T",     20.0)))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(text_embeddings.t().float())  # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        # ── Dual prototype systems ──────────────────────────────────
        # Image-level (PTA-style)
        refine_feature   = text_embeddings.t().float()   # [C, D]
        target_prototype = self.image_level.init_state(refine_feature)

        # Patch-level (Gaussian-style)
        states = self.patch_level.init_state(refine_feature)

        # Initialise patch-level text context for filtering (no-op if patch_filter_mode=none)
        _filter_mode = self.cfg.get("patch_level", {}).get("patch_filter_mode", "none")
        if _filter_mode != "none":
            self.patch_level.set_text_context(text_embeddings, encoder, device)

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
                    images, encoder, text_embeddings
                )
                feat      = image_features.squeeze(0).float()
                feat_norm = _safe_normalize(feat)

                # 2) Patch-level Gaussian prototype scores + quality gate
                #    (computed BEFORE image-level update so quality_gate
                #     can modulate the EMA rate)
                patch_proto_logits, quality_gate = (
                    self.patch_level.compute_patch_logits(images, encoder, states)
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
                
                # print(f"alpha max: {alpha_max}, n_half: {n_half}, quality_gate: {quality_gate}, tau_patch_proto: {self.fusion.tau_patch_proto}")
                # coutns = [states[c]["n_images"] for c in range(C)]
                # alpha = proto_alpha.cpu().numpy()
                # for c in range(C):
                #     print(f"PatchModulatedPTA: class {c}: n_images = {coutns[c]}, proto_alpha = {round(alpha[c], 3)}")

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
                    quality_modulation=self.quality_modulation,
                )

                # 5) Image-level proto logits
                image_proto_logits = (
                    image_features.half() @ refine_feature.half().T
                )  # [1, C]

                final_logits = self.fusion.forward(
                    clip_logits,
                    image_proto_logits,
                    patch_proto_logits,
                    quality_gate=quality_gate,
                    proto_alpha=proto_alpha,
                )

                if conf_source == "text":
                    gate_logits = clip_logits
                elif conf_source == "image":
                    gate_logits = image_proto_logits
                elif conf_source == "pta":
                    gate_logits = clip_logits + self.fusion.tau_image_proto * image_proto_logits
                elif conf_source == "full":
                    gate_logits = final_logits
                else:
                    raise ValueError(f"Unknown conf_source: {conf_source}")

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                pred_conf = F.softmax(gate_logits, dim=-1).squeeze(0)

                if multi_gate:
                    above_thresh = (pred_conf > conf_thresh).nonzero(as_tuple=True)[0]
                    for cls_idx in above_thresh:
                        cls = int(cls_idx.item())
                        states[cls] = self.patch_level.update_state(
                            states[cls], images, encoder, feat_norm,
                            target_class_idx=cls,
                        )
                else:
                    top2_vals, top2_idx = pred_conf.topk(min(2, C))
                    best_conf   = float(top2_vals[0].item())
                    second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                    conf_margin = best_conf - second_conf
                    best_cls    = int(top2_idx[0].item())

                    if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                        states[best_cls] = self.patch_level.update_state(
                            states[best_cls], images, encoder, feat_norm,
                            target_class_idx=best_cls,
                        )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- PatchModulatedPTA {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- PatchModulatedPTA FINAL {final_acc:.2f}% ----")

        label = os.environ.get("RESULT_LABEL", "PatchModulatedPTA")
        with open("outputs/result.txt", "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc

    def refine_with(self, encoder, text_embeddings, data_loader, dataset_name):
        """Convenience alias for run()."""
        return self.run(data_loader, encoder, text_embeddings, dataset_name)


def build(cfg: dict) -> PatchModulatedPTAAdapter:
    """Factory function — called by runner.py via dynamic import.

    Usage in runner.py:
        adapter_module = __import__('models.patch_modulated_pta', fromlist=['build'])
        adapter = adapter_module.build(cfg)
    """
    return PatchModulatedPTAAdapter(cfg)
