"""
PatchBoost: Isolate the effect of patch-level prototype logits in fusion,
WITHOUT quality gate or proto_alpha modulation.

This adapter is identical to PatchModulatedPTA except:
  1. Uses WeightedFusion instead of QualityGatedFusion
     → patch_proto_logits are NOT multiplied by quality_gate or proto_alpha
     → patch term = tau_patch_proto * patch_proto_logits (raw, no gating)
  2. Uses standard PTA EMA update (no quality-gated modulation of update rate)
     → quality_gate is computed (needed for patch_proto_logits) but discarded
  3. Still computes patch-level prototypes to get raw patch scores

Purpose: Find the tau_patch_proto setting where patch-level prototypes
actually contribute to prediction. The current PatchModulatedPTA uses
tau_patch_proto=20 but it's multiplied by proto_alpha (≤0.2) × quality_gate
(≤1.0), making the effective contribution ≤4 × [0-1 patch scores], which is
drowned out by tau_image_proto=100 × [10-30 image scores].

This adapter also supports two sweep modes via config:
  - Independent: tau_image_proto fixed, tau_patch_proto swept independently
  - Shared Budget: tau_image_proto + tau_patch_proto = total_budget,
    controlling the split ratio between image and patch prototypes.
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
from models.fusion import WeightedFusion
from utils import cls_acc, get_clip_logits


class PatchBoostAdapter(BaseAdapter):
    """Patch-level prototype boost without quality gate modulation."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.image_level = create_image_level(cfg)
        self.patch_level = create_patch_level(cfg)

        # ── Fusion: WeightedFusion (NO quality gate, NO proto_alpha) ──────
        # Key difference from PatchModulatedPTA: uses WeightedFusion instead
        # of QualityGatedFusion, so patch_proto_logits are not gated.
        #
        # Supports both sweep modes:
        #
        #   Independent mode (default):
        #     tau_image_proto = cfg["fusion"]["tau_image_proto"]  (default 100.0)
        #     tau_patch_proto = cfg["fusion"]["tau_patch_proto"]  (swept)
        #
        #   Shared budget mode (when "tau_total_budget" is set):
        #     tau_patch_proto = cfg["fusion"]["tau_patch_proto"]
        #     tau_image_proto = total_budget - tau_patch_proto
        fusion_src = cfg.get("fusion", {})
        total_budget = float(cfg.get("tau_total_budget", 0.0))
        tau_image_proto = float(fusion_src.get("tau_image_proto", 100.0))
        tau_patch_proto = float(fusion_src.get("tau_patch_proto", 0.0))

        if total_budget > 0:
            # Shared budget mode: tau_image + tau_patch = total_budget
            tau_image_proto = total_budget - tau_patch_proto
            if tau_image_proto < 0:
                tau_image_proto = 0.0

        fusion_cfg = {
            "fusion": {
                "tau_text":          float(fusion_src.get("tau_text", 1.0)),
                "tau_image_proto":   tau_image_proto,
                "tau_patch_proto":   tau_patch_proto,
            }
        }
        self.fusion = WeightedFusion(fusion_cfg)

        # Log the actual tau values being used
        self._tau_image = tau_image_proto
        self._tau_patch = tau_patch_proto

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        if not hasattr(encoder.visual, "positional_embedding") and not hasattr(encoder.visual, "pos_embed"):
            raise ValueError(
                "PatchBoost requires a ViT backbone (ViT-B/16) "
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

        # Log tau values at start
        tau_tag = f"τ_img={self._tau_image:.0f}_τ_pch={self._tau_patch:.0f}"

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[PatchBoost {tau_tag}] {dataset_name}")
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
                #    quality_gate is computed but DISCARDED in fusion
                patch_proto_logits, quality_gate = (
                    self.patch_level.compute_patch_logits(images, encoder, states)
                )

                # 3) Adaptive evidence weighting (computed for logging only)
                proto_alpha = torch.tensor(
                    [
                        min(alpha_max, _alpha_from_evidence(
                            states[c]["n_images"], n_half
                        ))
                        for c in range(C)
                    ],
                    device=device,
                )

                # 4) STANDARD PTA image-level prototype update
                #    NO quality-gated modulation — pure PTA-style EMA
                soft_logits = F.softmax(clip_logits, dim=-1)
                # Use the standard PTAImageLevel update (not quality-gated)
                refine_feature, target_prototype = self._update_pta_ema(
                    image_features,
                    soft_logits.half(),
                    refine_feature,
                    target_prototype,
                    alpha=alpha_pta,
                    T=T,
                )

                # 5) Image-level proto logits
                image_proto_logits = (
                    image_features.half() @ refine_feature.half().T
                )  # [1, C]

                # 6) Fusion with WeightedFusion — NO quality gate, NO proto_alpha
                final_logits = self.fusion.forward(
                    clip_logits,
                    image_proto_logits,
                    patch_proto_logits,  # raw, unmodulated
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # 7) Online memory update (patch-level, binary top-1 gate)
                #    Same as PatchModulatedPTA / PTA baseline
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
                    print(f"---- PatchBoost [{tau_tag}] {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- PatchBoost [{tau_tag}] FINAL {final_acc:.2f}% ----")

        label = self.cfg.get("result_label")
        if label is None:
            label = os.environ.get("RESULT_LABEL")
        if label is None:
            label = f"PatchBoost[{tau_tag}]"

        with open("outputs/result.txt", "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc

    # ------------------------------------------------------------------
    # Standard PTA-style EMA update (NO quality-gated modulation)
    # ------------------------------------------------------------------

    def _update_pta_ema(
        self,
        image_feature: torch.Tensor,
        probs: torch.Tensor,
        text_features: torch.Tensor,
        target_prototype: torch.Tensor,
        alpha: float = 0.01,
        T: float = 20.0,
    ):
        """
        Pure PTA-style EMA update — identical to PTAImageLevel.update_prototypes.
        No quality-gated modulation.
        """
        w = probs.squeeze(0)
        w_new = torch.zeros_like(w)
        mask = w >= 1e-1
        w_new[mask] = 1 - torch.exp(-w[mask] / T)
        w_new = w_new.unsqueeze(1)

        target_prototype[mask] = (
            (1 - w_new[mask]) * target_prototype[mask]
            + w_new[mask] * image_feature.squeeze(0)
        )

        refined_text = alpha * text_features + (1 - alpha) * target_prototype
        refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

        return refined_text, target_prototype

    def refine_with(self, encoder, text_embeddings, data_loader, dataset_name):
        """Convenience alias for run()."""
        return self.run(data_loader, encoder, text_embeddings, dataset_name)


def build(cfg: dict) -> PatchBoostAdapter:
    """Factory function — called by runner.py via dynamic import."""
    return PatchBoostAdapter(cfg)
