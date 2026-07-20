"""
Adaptive Image Weight: Per-class adaptive image-level weighting.

This adapter composes independent components:
  - Image-level (PTA-style): per-class running prototype via EMA
  - Patch-level (Gaussian-style): per-class prototype centers + variance
  - Fusion: AdaptiveImageWeightFusion with per-class tau_image[c]

Key difference from PatchModulatedPTA:
  Instead of a single tau_image_proto scalar for all classes, the image-level
  weight is computed per-class from proto_alpha[c]:
      tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) * proto_alpha[c]

  When proto_alpha[c] ≈ 0 (few samples seen), tau_image[c] ≈ tau_image_max.
  When proto_alpha[c] ≈ alpha_max (many samples seen), tau_image[c] ≈ tau_image_min.
"""
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.patch_level import create as create_patch_level
from models.patch_level.base import _alpha_from_evidence
from utils.clip_inference import _safe_normalize
from models.fusion import AdaptiveImageWeightFusion
from utils import cls_acc, get_clip_logits


# ------------------------------------------------------------------
# Quality-gated prototype update (shared with PatchModulatedPTA)
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
    """Quality-gated EMA prototype update."""
    w = probs.squeeze(0)
    w_new = torch.zeros_like(w)
    mask = w >= 1e-1
    w_new[mask] = 1 - torch.exp(-w[mask] / T)

    if quality_gate is not None and quality_modulation > 0:
        w_new[mask] *= (1 + quality_modulation * quality_gate)

    w_new = w_new.unsqueeze(1)

    target_prototype[mask] = (
        (1 - w_new[mask]) * target_prototype[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )

    refined_text = alpha * text_features + (1 - alpha) * target_prototype
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

    return refined_text, target_prototype


class AdaptiveImageWeightAdapter(BaseAdapter):
    """Per-class adaptive image-level weighting with quality-gated modulation."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.image_level = create_image_level(cfg)
        self.patch_level = create_patch_level(cfg)

        # Read fusion params from nested "fusion" dict (standard),
        # falling back to top-level keys for backward compat with flat configs.
        _fusion_cfg = cfg.get("fusion", {})
        fusion_cfg = {
            "fusion": {
                "tau_text":          float(_fusion_cfg.get("tau_text", cfg.get("tau_text", 1.0))),
                "tau_image_proto":   float(_fusion_cfg.get("tau_image_proto", cfg.get("tau_image_proto", 100.0))),
                "tau_patch_proto":   float(_fusion_cfg.get("tau_patch_proto", cfg.get("tau_patch_proto", 20.0))),
                "tau_image_max":     float(_fusion_cfg.get("tau_image_max", cfg.get("tau_image_max", 100.0))),
                "tau_image_min":     float(_fusion_cfg.get("tau_image_min", cfg.get("tau_image_min", 50.0))),
            }
        }
        self.fusion = AdaptiveImageWeightFusion(fusion_cfg)
        self.quality_modulation = float(_fusion_cfg.get("quality_modulation", cfg.get("quality_modulation", 1.0)))

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        if not hasattr(encoder.visual, "positional_embedding") and not hasattr(encoder.visual, "pos_embed"):
            raise ValueError(
                "AdaptiveImageWeight requires a ViT backbone (ViT-B/16) "
                "for patch extraction. "
                f"Got: {type(encoder.visual).__name__}"
            )

        # ── Config ──────────────────────────────────────────────────
        max_K              = int(self.cfg.get("max_K", 100))
        match_thresh       = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh        = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half             = float(self.cfg.get("n_half", 15.0))
        alpha_max          = float(self.cfg.get("proto_alpha_max", 0.2))

        _il_cfg = self.cfg.get("image_level", {})
        alpha_pta = float(_il_cfg.get("alpha", self.cfg.get("alpha", 0.01)))
        T         = float(_il_cfg.get("T",     self.cfg.get("T",     20.0)))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(text_embeddings.t().float())  # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        # ── Dual prototype systems ──────────────────────────────────
        refine_feature   = text_embeddings.t().float()   # [C, D]
        target_prototype = self.image_level.init_state(refine_feature)
        states = self.patch_level.init_state(refine_feature)

        max_batches = int(os.environ.get("MAX_BATCHES", "0"))
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[AdaptiveImageWeight] {dataset_name}")
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

                # 6) Fusion with per-class adaptive image weight
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
                        states[best_cls], images, encoder, feat_norm,
                    )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- AdaptiveImageWeight {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- AdaptiveImageWeight FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(
                f"AdaptiveImageWeight's performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc

    def refine_with(self, encoder, text_embeddings, data_loader, dataset_name):
        """Convenience alias for run()."""
        return self.run(data_loader, encoder, text_embeddings, dataset_name)


def build(cfg: dict) -> AdaptiveImageWeightAdapter:
    """Factory function — called by runner.py via dynamic import."""
    return AdaptiveImageWeightAdapter(cfg)
