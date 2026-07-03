"""
Experiment 9: Entropy-Guided Patch Proto Modulation
=====================================================

Extension of Exp5 (Tunable Fusion) where tau_patch_proto is modulated
per-sample by normalized CLIP zero-shot entropy:

    eff_tau_patch = tau_patch_proto * (1 + entropy_boost * norm_entropy)

High-entropy (uncertain) samples get stronger patch-prototype influence;
low-entropy (confident) samples rely more on the original CLIP/text branch.

Three-way fusion of CLIP text, image-level prototype, and patch-level
Gaussian prototype:

    final_logits = tau_text       * clip_logits
                 + tau_image_proto * image_proto_logits
                 + eff_tau_patch   * proto_alpha * quality_gate * raw_proto

Same dual independent prototype systems as exp4/exp5:
  - Image-level (PTA-style): per-class running prototype via update_text_features
  - Patch-level (Gaussian-style): per-class prototype centers + variance via exp3
"""
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.exp3_gaussian_prototypes import (
    _alpha_from_evidence,
    _gaussian_score_for_class,
)
from models.multi_proto_pta_base import (
    _extract_patch_embeddings,
    _incremental_kmeans_step,
    _safe_normalize,
)
from models.pta import update_text_features
from utils import cls_acc, get_clip_logits


class Exp9EntropyFusionAdapter(BaseAdapter):
    """Three-way fusion with entropy-guided tau_patch_proto modulation.

    Reads configurable tau weights (tau_text, tau_image_proto,
    tau_patch_proto) from YAML, plus an entropy_boost multiplier that
    scales tau_patch_proto per-sample based on CLIP zero-shot entropy.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Read tunable tau weights from config with fallback defaults
        self.tau_text        = float(self.cfg.get("tau_text", 1.0))
        self.tau_image_proto = float(self.cfg.get("tau_image_proto", 100.0))
        self.tau_patch_proto = float(self.cfg.get("tau_patch_proto", 20.0))
        # Entropy modulation strength
        self.entropy_boost   = float(self.cfg.get("entropy_boost", 2.0))

    # ------------------------------------------------------------------
    # Patch-level prototype state (Gaussian, same as Exp3)
    # ------------------------------------------------------------------

    def _make_class_state(self, D: int, device: torch.device) -> Dict:
        return {
            "centers":    torch.empty(0, D, device=device),
            "variance":   torch.empty(0, D, device=device),
            "appearance": torch.empty(0,    device=device),
            "n_images":   0,
        }

    def _update_state(
        self,
        state: Dict,
        patches_norm: torch.Tensor,
        global_feat: torch.Tensor,
        match_threshold: float,
        max_K: int,
    ) -> Dict:
        """
        Same update logic as Exp3GaussianPrototypesAdapter._update_state().

        Update prototype centers, appearance counts AND variance via EMA.
        """
        centers    = state["centers"]
        apps       = state["appearance"]
        variances  = state["variance"]
        old_K      = centers.shape[0]

        gaussian_ema   = float(self.cfg.get("gaussian_ema", 0.1))
        variance_min   = float(self.cfg.get("variance_min", 0.001))
        variance_max   = float(self.cfg.get("variance_max", 1.0))
        default_new_var = float(variance_min * 10)

        grow_cap = max(max_K + int(patches_norm.shape[0]), max_K)

        # ── Run incremental K-means ──────────────────────────────────
        if old_K == 0:
            init = _safe_normalize(global_feat, dim=-1).unsqueeze(0)
            updated_centers, appeared, matched, best_clusters, new_groups = (
                _incremental_kmeans_step(init, patches_norm, match_threshold, grow_cap)
            )
        else:
            updated_centers, appeared, matched, best_clusters, new_groups = (
                _incremental_kmeans_step(centers, patches_norm, match_threshold, grow_cap)
            )

        total_K = updated_centers.shape[0]
        n_new = total_K - old_K if old_K > 0 else total_K - 1

        all_vars = []
        all_apps = []

        # ── Process "old" prototypes ─────────────────────────────────
        if old_K > 0:
            centers_old_norm = _safe_normalize(centers, dim=-1)
            for k in range(old_K):
                mask = matched & (best_clusters == k)
                if mask.any():
                    residuals = patches_norm[mask] - centers_old_norm[k]
                    batch_var = residuals.pow(2).mean(dim=0).clamp(variance_min, variance_max)
                    updated_var = (1 - gaussian_ema) * variances[k] + gaussian_ema * batch_var
                    all_vars.append(updated_var.clamp(variance_min, variance_max))
                else:
                    all_vars.append(variances[k])
                all_apps.append(apps[k] + (1.0 if appeared[k] else 0.0))
        else:
            # Seed prototype
            mask = matched & (best_clusters == 0)
            if mask.any():
                residuals = patches_norm[mask] - _safe_normalize(init)
                seed_var = residuals.pow(2).mean(dim=0).clamp(variance_min, variance_max)
            else:
                seed_var = torch.full(
                    (updated_centers.shape[1],), default_new_var,
                    device=updated_centers.device,
                )
            all_vars.append(seed_var)
            all_apps.append(1.0)

        # ── Process new prototypes ───────────────────────────────────
        if n_new > 0:
            for idx, group_idx in enumerate(new_groups):
                proto_idx = (old_K if old_K > 0 else 1) + idx
                group_patches = patches_norm[group_idx]
                group_center = _safe_normalize(updated_centers[proto_idx])
                if group_patches.shape[0] <= 1:
                    group_var = torch.full(
                        (updated_centers.shape[1],), default_new_var,
                        device=updated_centers.device,
                    )
                else:
                    residuals = group_patches - group_center
                    group_var = residuals.pow(2).mean(dim=0).clamp(variance_min, variance_max)
                all_vars.append(group_var)
                all_apps.append(1.0)

        updated_vars = torch.stack(all_vars, dim=0)
        updated_apps = torch.tensor(all_apps, device=apps.device, dtype=torch.float)

        # ── Prune to max_K by appearance weight ──────────────────────
        n_images_next = max(int(state.get("n_images", 0)) + 1, 1)
        if updated_centers.shape[0] > max_K:
            app_w = updated_apps / float(n_images_next)
            keep  = torch.argsort(app_w, descending=True)[:max_K]
            keep  = torch.sort(keep).values
            updated_centers = updated_centers[keep]
            updated_vars    = updated_vars[keep]
            updated_apps    = updated_apps[keep]

        state = dict(state)
        state["centers"]    = updated_centers
        state["variance"]   = updated_vars
        state["appearance"] = updated_apps
        state["n_images"]   = state["n_images"] + 1
        return state

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
                "Exp9EntropyFusion requires a ViT backbone (ViT-B/16) for patch extraction. "
                f"Got: {type(clip_model.visual).__name__}"
            )

        # ── Config ──────────────────────────────────────────────────
        max_K              = int(self.cfg.get("max_K", 100))
        match_thresh       = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh        = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half             = float(self.cfg.get("n_half", 15.0))
        alpha_max          = float(self.cfg.get("proto_alpha_max", 0.2))
        top_m              = int(self.cfg.get("soft_nn_top_m", 4))
        exclude_pos        = bool(self.cfg.get("exclude_pos", False))
        quality_eps        = float(self.cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self.cfg.get("patch_group_threshold", 0.9))
        variance_min       = float(self.cfg.get("variance_min", 0.001))
        variance_max       = float(self.cfg.get("variance_max", 1.0))

        # PTA-style image-level update params
        alpha_pta = float(self.cfg.get("alpha", 0.01))
        T         = float(self.cfg.get("T", 50.0))

        # Tunable fusion weights
        tau_text       = self.tau_text
        tau_image_proto = self.tau_image_proto
        tau_patch_proto = self.tau_patch_proto

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(clip_weights.t().float())  # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        # ── Dual prototype systems ──────────────────────────────────
        # Image-level (PTA-style)
        refine_feature   = clip_weights.t().float()   # [C, D]
        target_prototype = torch.zeros_like(refine_feature).to(device)

        # Patch-level (Gaussian-style)
        states = [self._make_class_state(D, device) for _ in range(C)]

        max_batches = int(os.environ.get("MAX_BATCHES", "0"))
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[Exp9] {dataset_name}")
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

                # 2) Update image-level prototype (PTA-style)
                soft_logits = F.softmax(clip_logits, dim=-1)
                refine_feature, target_prototype = update_text_features(
                    image_features,
                    soft_logits.half(),
                    refine_feature,
                    target_prototype,
                    alpha=alpha_pta,
                    T=T,
                )

                # 3) Image-level proto logits
                # NOTE: no hardcoded 100.0 scaling — tau_image_proto handles it
                image_proto_logits = (
                    image_features.half() @ refine_feature.half().T
                )  # [1, C]

                # 4) Patch-level Gaussian prototype scores
                patch_embs = _extract_patch_embeddings(
                    images, clip_model, exclude_pos=exclude_pos
                )
                patches_norm = _safe_normalize(patch_embs)

                raw_proto = torch.zeros(C, device=device)
                for c in range(C):
                    if states[c]["centers"].shape[0] > 0:
                        centers_norm = _safe_normalize(states[c]["centers"])
                        raw_proto[c] = _gaussian_score_for_class(
                            patches_norm,
                            centers_norm,
                            states[c]["variance"],
                            states[c]["appearance"],
                            update_samples=int(states[c]["n_images"]),
                            top_m=top_m,
                            patch_group_threshold=patch_group_threshold,
                            variance_min=variance_min,
                        )

                # 5) Adaptive evidence weighting (same as exp3/exp4)
                proto_alpha = torch.tensor(
                    [
                        min(alpha_max, _alpha_from_evidence(
                            states[c]["n_images"], n_half
                        ))
                        for c in range(C)
                    ],
                    device=device,
                )
                proto_var    = raw_proto.var()
                quality_gate = proto_var / (proto_var + quality_eps)

                # 6) Entropy-guided tau_patch_proto modulation
                # Compute CLIP zero-shot entropy per-sample
                probs = F.softmax(clip_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                max_entropy = math.log(C)
                norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)  # [0, 1]

                # Modulate tau_patch_proto per-sample
                eff_tau_patch = tau_patch_proto * (1 + self.entropy_boost * norm_entropy)

                # 7) Three-way fusion with entropy-modulated patch weight
                final_logits = (
                    tau_text * clip_logits.clone()
                    + tau_image_proto * image_proto_logits
                    + eff_tau_patch * proto_alpha * quality_gate * raw_proto.unsqueeze(0)
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # 8) Online memory update (patch-level, same confidence gate as exp3)
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)
                top2_vals, top2_idx = pred_conf.topk(min(2, C))
                best_conf   = float(top2_vals[0].item())
                second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                conf_margin = best_conf - second_conf
                best_cls    = int(top2_idx[0].item())

                if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                    states[best_cls] = self._update_state(
                        states[best_cls], patches_norm, feat_norm,
                        match_thresh, max_K,
                    )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- Exp9 {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- Exp9 FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(
                f"Exp9EntropyFusion's performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc

    def refine_with(self, clip_model, clip_weights, data_loader, dataset_name):
        """Convenience alias for run()."""
        return self.run(data_loader, clip_model, clip_weights, dataset_name)


def build(cfg: dict) -> Exp9EntropyFusionAdapter:
    """Factory function — called by runner.py via dynamic import."""
    return Exp9EntropyFusionAdapter(cfg)
