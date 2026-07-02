"""
Experiment 4: Full Fusion (Text + Image-Level Prototype + Patch-Level Gaussian Prototype)
========================================================================================

Fuses THREE logit branches in a fixed-weight manner:
  1. Text-based zero-shot CLIP logits
  2. Image-level prototype logits (PTA-style, EMA update per sample)
  3. Patch-level Gaussian prototype logits (Exp3-style, strict top-1 gate)

Both prototype systems are independent and updated with different gates:
  - Image-level (PTA permissive): updates EVERY sample for ANY class with
    softmax prob >= 0.1.
  - Patch-level (Exp3 strict): updates only when top-1 CLIP confidence exceeds
    conf_threshold AND the margin over the runner-up exceeds conf_margin_threshold.

Fusion is fixed-weight (no learned blending):
    final_logits = clip_logits + 100.0 * image_proto_logits
                   + tau_proto * proto_alpha * quality_gate * patch_proto_logits
"""
import os
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.exp3_gaussian_prototypes import _alpha_from_evidence, _gaussian_score_for_class
from models.multi_proto_pta_base import (
    _extract_patch_embeddings,
    _incremental_kmeans_step,
    _safe_normalize,
)
from models.pta import update_text_features
from utils import cls_acc, get_clip_logits


class Exp4FullFusionAdapter(BaseAdapter):
    """
    Full Fusion Adapter: combines three independent logit branches.

    Maintains TWO independent prototype systems:
      - Image-level (PTA-style): [C, D] running prototypes updated per sample
        with permissive gate (any class with softmax prob >= 0.1).
      - Patch-level (Exp3-style): per-class Gaussian prototypes (center + variance)
        with strict top-1 confidence + margin gate.
    """

    # ------------------------------------------------------------------
    # Patch-level state management (Exp3-style Gaussian prototypes)
    # ------------------------------------------------------------------

    def _make_class_state(self, D: int, device: torch.device) -> Dict:
        """Create an empty class state for patch-level Gaussian prototypes."""
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
        Update patch-level prototype bank: incremental K-means with Gaussian
        variance EMA update.  (Same implementation as Exp3GaussianPrototypes.)

        Steps:
          1. Run incremental K-means to match patches to existing prototypes
             or form new groups.
          2. For matched prototypes: EMA-update variance from per-patch residuals.
          3. For new prototypes: initialise variance from own group's scatter.
          4. Prune to max_K by appearance frequency (normalised by n_images).
        """
        centers    = state["centers"]        # [K, D]
        apps       = state["appearance"]     # [K]
        variances  = state["variance"]       # [K, D]
        old_K      = centers.shape[0]

        gaussian_ema    = float(self.cfg.get("gaussian_ema", 0.1))
        variance_min    = float(self.cfg.get("variance_min", 0.001))
        variance_max    = float(self.cfg.get("variance_max", 1.0))
        default_new_var = float(variance_min * 10)

        grow_cap = max(max_K + int(patches_norm.shape[0]), max_K)

        # ── Incremental K-means ──────────────────────────────────────────────
        if old_K == 0:
            init = _safe_normalize(global_feat, dim=-1).unsqueeze(0)
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                init, patches_norm, match_threshold, grow_cap
            )
        else:
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                centers, patches_norm, match_threshold, grow_cap
            )

        total_K = updated_centers.shape[0]
        n_new = total_K - old_K if old_K > 0 else total_K - 1

        # ── Build variance & appearance tensors ──────────────────────────────
        all_vars = []
        all_apps = []

        # Process "old" prototypes
        if old_K > 0:
            centers_old_norm = _safe_normalize(centers, dim=-1)  # [old_K, D]
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
            # Seed prototype (index 0 in updated_centers from init)
            mask = matched & (best_clusters == 0)
            if mask.any():
                residuals = patches_norm[mask] - _safe_normalize(init)
                seed_var = residuals.pow(2).mean(dim=0).clamp(variance_min, variance_max)
            else:
                seed_var = torch.full((updated_centers.shape[1],), default_new_var,
                                      device=updated_centers.device)
            all_vars.append(seed_var)
            all_apps.append(1.0)

        # Process new prototypes (from unmatched patch groups)
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

        # Stack into tensors
        updated_vars = torch.stack(all_vars, dim=0)   # [total_K, D]
        updated_apps = torch.tensor(all_apps, device=apps.device, dtype=torch.float)

        # ── Prune to max_K by appearance weight ──────────────────────────────
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
        """
        Run Exp4 Full Fusion on test set.

        Per-sample loop (EXACT ORDER):
          1. Get image features & CLIP zero-shot logits
          2. Update image-level prototypes (PTA permissive gate)
          3. Branch 1: text-based zero-shot (clip_logits)
          4. Branch 2: image-level prototype logits
          5. Extract & normalise patch embeddings
          6. Branch 3: patch-level Gaussian prototype scores
          7. Evidence gating per class (alpha from n_images)
          8. Quality gating (variance ratio)
          9. Fuse three branches
         10. Measure accuracy
         11. Update patch-level prototypes (Exp3 strict gate)
        """
        # ── Backbone check ──────────────────────────────────────────
        if not hasattr(clip_model.visual, "positional_embedding"):
            raise ValueError(
                "Exp4FullFusion requires a ViT backbone (ViT-B/16) for patch extraction. "
                f"Got: {type(clip_model.visual).__name__}"
            )

        # ── Config ──────────────────────────────────────────────────
        alpha               = float(self.cfg.get("alpha", 0.01))
        T                   = float(self.cfg.get("T", 50.0))
        max_K               = int(self.cfg.get("max_K", 100))
        match_thresh        = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh         = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh  = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half              = float(self.cfg.get("n_half", 15.0))
        proto_alpha_max     = float(self.cfg.get("proto_alpha_max", 0.2))
        top_m               = int(self.cfg.get("soft_nn_top_m", 4))
        exclude_pos         = bool(self.cfg.get("exclude_pos", False))
        tau_proto           = float(self.cfg.get("tau_proto", 20.0))
        quality_eps         = float(self.cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self.cfg.get("patch_group_threshold", 0.9))

        os.makedirs("outputs", exist_ok=True)

        # ── Setup ───────────────────────────────────────────────────────────
        device = clip_weights.device

        # Image-level prototypes (PTA-style)
        refine_feature = clip_weights.t()                           # [C, D]
        target_prototype = torch.zeros_like(refine_feature).to(device)   # [C, D]

        # Patch-level Gaussian prototypes (Exp3-style)
        text_proto = _safe_normalize(clip_weights.t().float())      # [C, D]
        C, D = text_proto.shape
        states = [self._make_class_state(D, device) for _ in range(C)]

        max_batches = int(os.environ.get("MAX_BATCHES", "0"))
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[Exp4FullFusion] {dataset_name}")
            ):
                if max_batches > 0 and i >= max_batches:
                    break
                # Handle list-of-augmented-tensors (OOD datasets)
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # ─────────────────────────────────────────────────────────────
                # Step 1: Get features
                # ─────────────────────────────────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, clip_model, clip_weights
                )

                # ─────────────────────────────────────────────────────────────
                # Step 2: Update image-level prototypes (PTA permissive gate)
                # ─────────────────────────────────────────────────────────────
                soft_logits = F.softmax(clip_logits, dim=-1)
                refine_feature, target_prototype = update_text_features(
                    image_features, soft_logits.half(), refine_feature, target_prototype,
                    alpha=alpha, T=T,
                )

                # ─────────────────────────────────────────────────────────────
                # Step 3: Branch 1 — text-based zero-shot
                #         clip_logits is already computed from step 1
                # ─────────────────────────────────────────────────────────────

                # ─────────────────────────────────────────────────────────────
                # Step 4: Branch 2 — image-level prototype logits
                # ─────────────────────────────────────────────────────────────
                image_proto_logits = 100.0 * image_features.half() @ refine_feature.half().T  # [1, C]

                # ─────────────────────────────────────────────────────────────
                # Step 5: Extract patch embeddings & global feature
                # ─────────────────────────────────────────────────────────────
                feat = image_features.squeeze(0).float()
                feat_norm = _safe_normalize(feat)

                patch_embs = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)
                patches_norm = _safe_normalize(patch_embs)

                # ─────────────────────────────────────────────────────────────
                # Step 6: Branch 3 — patch-level Gaussian prototype scores
                # ─────────────────────────────────────────────────────────────
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
                            variance_min=float(self.cfg.get("variance_min", 0.001)),
                        )

                # ─────────────────────────────────────────────────────────────
                # Step 7: Evidence gating (per-class alpha from n_images)
                # ─────────────────────────────────────────────────────────────
                proto_alpha = torch.tensor(
                    [min(proto_alpha_max, _alpha_from_evidence(states[c]["n_images"], n_half))
                     for c in range(C)],
                    device=device,
                )

                # ─────────────────────────────────────────────────────────────
                # Step 8: Quality gating (variance ratio across classes)
                # ─────────────────────────────────────────────────────────────
                proto_var = raw_proto.var()
                quality_gate = proto_var / (proto_var + quality_eps)

                # ─────────────────────────────────────────────────────────────
                # Step 9: Fuse all three branches
                #   final_logits = clip_logits + 100.0 * image_proto_logits
                #                  + tau_proto * proto_alpha * quality_gate * patch_proto_logits
                # ─────────────────────────────────────────────────────────────
                final_logits = (
                    clip_logits.clone()
                    + image_proto_logits
                    + tau_proto * proto_alpha * quality_gate * raw_proto.unsqueeze(0)
                )

                # ─────────────────────────────────────────────────────────────
                # Step 10: Measure accuracy
                # ─────────────────────────────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # ─────────────────────────────────────────────────────────────
                # Step 11: Update patch-level prototypes (Exp3 strict gate)
                # ─────────────────────────────────────────────────────────────
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)
                top2_vals, top2_idx = pred_conf.topk(min(2, C))
                best_conf = float(top2_vals[0].item())
                second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                conf_margin = best_conf - second_conf
                best_cls = int(top2_idx[0].item())

                if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                    states[best_cls] = self._update_state(
                        states[best_cls], patches_norm, feat_norm, match_thresh, max_K,
                    )

                # ── Periodic logging ────────────────────────────────────────
                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- Exp4FullFusion's test accuracy: {running:.2f}. ----")

        # ── Final results ─────────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- Exp4FullFusion's test accuracy: {final_acc:.2f}. ----\n")

        with open("outputs/result.txt", "a") as f:
            f.write(
                f"Exp4FullFusion's performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> Exp4FullFusionAdapter:
    """Factory function for runner.py dynamic import."""
    return Exp4FullFusionAdapter(cfg)
