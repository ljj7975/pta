"""
Experiment 3: Gaussian Prototypes (Center + Variance)
=====================================================

Represent each prototype as a Gaussian (center + per-dimension variance) and
replace cosine-similarity matching with a Mahalanobis-like distance:

    gaussian_score = exp(-0.5 * ((patch - center)² / variance).sum(dim=-1))

Key change vs multi_proto_pta.py:
  - prototypes store both `center` and `variance` (per-dimension)
  - matching uses Gaussian probability instead of cosine similarity
  - variance is EMA-updated during incremental K-means
  - variance clamped for numerical stability

This naturally downweights uncertain prototypes (high variance) and
upweights tight, high-confidence clusters.
"""
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.multi_proto_pta_base import (
    _safe_normalize,
    _incremental_kmeans_step,
    _extract_patch_embeddings,
)
from utils import get_clip_logits, cls_acc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_from_evidence(n_images: float, n_half: float = 15.0) -> float:
    if n_images <= 0:
        return 0.0
    return min(1.0, math.log(1.0 + n_images) / math.log(1.0 + n_half))


def _gaussian_score_for_class(
    patches_norm: torch.Tensor,
    centers_norm: torch.Tensor,
    variance: torch.Tensor,
    appearance: torch.Tensor,
    update_samples: int,
    top_m: int,
    patch_group_threshold: float = 0.9,
    variance_min: float = 0.001,
) -> torch.Tensor:
    """
    Score one class using Gaussian prototype similarity.

    For each patch group and prototype, compute Gaussian score:
        score = exp(-0.5 * Σ_d (patch_d - center_d)² / variance_d)

    Then apply one-to-one assignment (same as MPTA) and top-M aggregation.

    Returns a scalar score for this class.
    """
    num_input_patches = patches_norm.shape[0]
    if num_input_patches == 0 or centers_norm.shape[0] == 0:
        return torch.tensor(0.0, device=centers_norm.device)

    K = centers_norm.shape[0]

    # ── Patch grouping (same as MPTA) ──────────────────────────────────────
    patch_sims = patches_norm @ patches_norm.t()  # [P, P]
    unassigned = torch.ones(num_input_patches, dtype=torch.bool, device=patches_norm.device)
    rep_centers = []
    group_members = []

    while unassigned.any():
        cand_idx = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        sub_sims = patch_sims[cand_idx][:, cand_idx]
        anchor_local = int(sub_sims.mean(dim=1).argmax().item())
        anchor_idx = int(cand_idx[anchor_local].item())

        group_mask = unassigned & (patch_sims[anchor_idx] >= patch_group_threshold)
        member_idx = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
        if member_idx.numel() == 0:
            member_idx = torch.tensor([anchor_idx], device=patches_norm.device, dtype=torch.long)
            group_mask = torch.zeros_like(unassigned)
            group_mask[anchor_idx] = True

        rep = _safe_normalize(patches_norm[member_idx].mean(dim=0), dim=-1)
        rep_centers.append(rep)
        group_members.append(member_idx)
        unassigned[group_mask] = False

    rep_patches = torch.stack(rep_centers, dim=0)  # [G, D]

    # ── Gaussian score between each patch group and each prototype ─────────
    # diff[g, k, d] = rep_patches[g, d] - centers_norm[k, d]
    # gaussian_score[g, k] = exp(-0.5 * Σ_d diff² / variance_k_d)
    var_clamped = variance.clamp(min=variance_min)  # [K, D]
    diff = rep_patches[:, None, :] - centers_norm[None, :, :]  # [G, K, D]
    scaled_maha = (diff.pow(2) / var_clamped[None, :, :]).sum(dim=-1)  # [G, K]
    gaussian_scores = torch.exp(-0.5 * scaled_maha)  # [G, K]

    # ── One-to-one assignment (same as MPTA) ───────────────────────────────
    num_groups, K = gaussian_scores.shape
    proto_best_vals = gaussian_scores.max(dim=0).values
    proto_order = torch.argsort(proto_best_vals, descending=True)
    used_groups = torch.zeros(num_groups, dtype=torch.bool, device=gaussian_scores.device)

    best_per_proto = torch.zeros(K, device=gaussian_scores.device)
    for proto_idx in proto_order.tolist():
        scores = gaussian_scores[:, proto_idx].clone()
        scores[used_groups] = float("-inf")
        best_val, group_idx = scores.max(dim=0)
        if torch.isneginf(best_val):
            continue
        best_per_proto[proto_idx] = best_val
        used_groups[group_idx] = True

    # ── Appearance weighting and top-M aggregation ─────────────────────────
    denom = max(float(update_samples), 1e-6)
    app_w = appearance / denom
    weighted = best_per_proto * app_w  # [K]

    k = min(top_m, weighted.numel())
    if k <= 0:
        return torch.tensor(0.0, device=gaussian_scores.device)
    return weighted.topk(k).values.mean()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Exp3GaussianPrototypesAdapter(BaseAdapter):

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
        Update prototype centers, appearance counts AND variance.

        Variance update uses EMA:
            new_var = (1 - gaussian_ema) * old_var + gaussian_ema * batch_var

        where batch_var is the per-dimension variance of patches that matched
        this prototype in the current image.
        """
        centers    = state["centers"]        # [K, D]
        apps       = state["appearance"]     # [K]
        variances  = state["variance"]       # [K, D]
        old_K      = centers.shape[0]

        gaussian_ema   = float(self.cfg.get("gaussian_ema", 0.1))
        variance_min   = float(self.cfg.get("variance_min", 0.001))
        variance_max   = float(self.cfg.get("variance_max", 1.0))
        default_new_var = float(variance_min * 10)  # moderate default for new protos

        grow_cap = max(max_K + int(patches_norm.shape[0]), max_K)

        # ── Run incremental K-means ──────────────────────────────────────────
        if old_K == 0:
            # Seed with global feature when no prototypes exist yet
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

        # ── Build variance tensor (one entry per prototype in updated_centers) ──
        # Strategy: start with old prototypes (EMA-update or create if seed),
        # then append new prototypes from unmatched patch groups.
        all_vars = []
        all_apps = []

        # ── Process "old" prototypes ────────────────────────────────────────
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

        # ── Process new prototypes (from unmatched patch groups) ────────────
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

        # ── Prune to max_K by appearance weight ─────────────────────────────
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

    def run(self, loader, clip_model, clip_weights, dataset_name: str) -> float:
        # ── Config ──────────────────────────────────────────────────────────
        max_K              = int(self.cfg.get("max_K", 100))
        match_thresh       = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh        = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half             = float(self.cfg.get("n_half", 15.0))
        alpha_max          = float(self.cfg.get("alpha_max", 0.2))
        top_m              = int(self.cfg.get("soft_nn_top_m", 4))
        exclude_pos        = bool(self.cfg.get("exclude_pos", False))
        tau_proto          = float(self.cfg.get("tau_proto", 20.0))
        quality_eps        = float(self.cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self.cfg.get("patch_group_threshold", 0.9))
        # ── Exp 3 specific ──────────────────────────────────────────────────
        variance_min       = float(self.cfg.get("variance_min", 0.001))
        variance_max       = float(self.cfg.get("variance_max", 1.0))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(clip_weights.t().float())   # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        states     = [self._make_class_state(D, device) for _ in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[Exp3] {dataset_name}")):
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # 1) CLIP text branch
                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
                feat       = image_features.squeeze(0).float()
                feat_norm  = _safe_normalize(feat)

                patch_embs   = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)
                patches_norm = _safe_normalize(patch_embs)

                text_logits = 100.0 * (text_proto @ feat_norm)  # [C]

                # 2) Gaussian prototype branch
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

                # 3) Fuse text + prototype (same formula as MPTA)
                alpha = torch.tensor(
                    [min(alpha_max, _alpha_from_evidence(states[c]["n_images"], n_half))
                     for c in range(C)],
                    device=device,
                )
                proto_var    = raw_proto.var()
                quality_gate = proto_var / (proto_var + quality_eps)

                final_logits = (text_logits + tau_proto * alpha * quality_gate * raw_proto).unsqueeze(0)

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # 4) Online memory update (same gate as MPTA)
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)
                top2_vals, top2_idx = pred_conf.topk(min(2, C))
                best_conf   = float(top2_vals[0].item())
                second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                conf_margin = best_conf - second_conf
                best_cls    = int(top2_idx[0].item())

                if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                    states[best_cls] = self._update_state(
                        states[best_cls], patches_norm, feat_norm, match_thresh, max_K
                    )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- Exp3 {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- Exp3 FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(f"Exp3GaussianPrototypes's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> Exp3GaussianPrototypesAdapter:
    return Exp3GaussianPrototypesAdapter(cfg)
