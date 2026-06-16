"""
Experiment 2: Adaptive Tau Proto (50/50 Score Blending)
=======================================================

Replace fixed `tau_proto=20` exponential weighting with a principled 50/50
blend of CLIP text logits and prototype similarity scores.

Key change vs multi_proto_pta.py:
  - tau_proto / alpha / quality_gate removed from fusion
  - text and proto scores are softmax-normalised and blended 50/50
  - classes with < `min_protos_for_full_agg` prototypes fall back to mean
    of all prototype scores (instead of top-M)

This is model-agnostic because both scores come from the same CLIP framework
and are normalised to the same [0,1] probability scale before blending.
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
# Helpers (adapted from multi_proto_pta.py)
# ---------------------------------------------------------------------------

def _alpha_from_evidence(n_images: float, n_half: float = 15.0) -> float:
    if n_images <= 0:
        return 0.0
    return min(1.0, math.log(1.0 + n_images) / math.log(1.0 + n_half))


def _proto_score_for_class(
    patches_norm: torch.Tensor,
    centers_norm: torch.Tensor,
    appearance: torch.Tensor,
    update_samples: int,
    top_m: int,
    min_protos_full_agg: int = 5,
    patch_group_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Score one class by prototype similarity.

    For classes with enough prototypes (>= min_protos_full_agg):
      Use top-M mean aggregation (current MPTA method with patch grouping + one-to-one assignment).

    For classes with too few prototypes (< min_protos_full_agg):
      Use mean of ALL prototype scores (more robust against low-sample noise).

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

    # ── Similarity between patch groups and prototypes ─────────────────────
    sims = rep_patches @ centers_norm.t()  # [G, K]

    # ── One-to-one assignment (same as MPTA) ───────────────────────────────
    num_groups = sims.shape[0]
    proto_best_vals = sims.max(dim=0).values
    proto_order = torch.argsort(proto_best_vals, descending=True)
    used_groups = torch.zeros(num_groups, dtype=torch.bool, device=sims.device)

    best_per_proto = torch.zeros(K, device=sims.device)
    for proto_idx in proto_order.tolist():
        scores = sims[:, proto_idx].clone()
        scores[used_groups] = float("-inf")
        best_val, group_idx = scores.max(dim=0)
        if torch.isneginf(best_val):
            continue
        best_per_proto[proto_idx] = best_val
        used_groups[group_idx] = True

    # ── Appearance weighting ───────────────────────────────────────────────
    denom = max(float(update_samples), 1e-6)
    app_w = appearance / denom
    weighted = best_per_proto * app_w  # [K]

    # ── Aggregation depends on prototype count ────────────────────────────
    if K >= min_protos_full_agg:
        # Top-M mean (standard MPTA)
        k = min(top_m, weighted.numel())
        if k <= 0:
            return torch.tensor(0.0, device=sims.device)
        return weighted.topk(k).values.mean()
    else:
        # Low-count fallback: mean of ALL prototypes
        return weighted.mean()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Exp2AdaptiveTauProtoAdapter(BaseAdapter):

    def _make_class_state(self, D: int, device: torch.device) -> Dict:
        return {
            "centers":    torch.empty(0, D, device=device),
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
        """Same update logic as MPTA (multi_proto_pta.py)."""
        centers = state["centers"]
        apps    = state["appearance"]
        old_K   = centers.shape[0]
        grow_cap = max(max_K + int(patches_norm.shape[0]), max_K)

        if old_K == 0:
            init = global_feat.unsqueeze(0)
            updated_centers, appeared, _, _, _ = _incremental_kmeans_step(
                init, patches_norm, match_threshold, grow_cap
            )
            n_new = updated_centers.shape[0] - 1
            updated_apps = torch.ones(1 + n_new, device=apps.device)
        else:
            updated_centers, appeared, _, _, _ = _incremental_kmeans_step(
                centers, patches_norm, match_threshold, grow_cap
            )
            updated_apps = apps.clone()
            updated_apps[appeared] += 1
            n_new = updated_centers.shape[0] - old_K
            if n_new > 0:
                updated_apps = torch.cat([
                    updated_apps,
                    torch.ones(n_new, device=apps.device),
                ])

        n_images_next = max(int(state.get("n_images", 0)) + 1, 1)
        if updated_centers.shape[0] > max_K:
            app_w = updated_apps / float(n_images_next)
            keep  = torch.argsort(app_w, descending=True)[:max_K]
            keep  = torch.sort(keep).values
            updated_centers = updated_centers[keep]
            updated_apps    = updated_apps[keep]

        state = dict(state)
        state["centers"]    = updated_centers
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
        top_m              = int(self.cfg.get("soft_nn_top_m", 4))
        exclude_pos        = bool(self.cfg.get("exclude_pos", False))
        quality_eps        = float(self.cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self.cfg.get("patch_group_threshold", 0.9))
        # ── Exp 2 specific ──────────────────────────────────────────────────
        blend_text_weight  = float(self.cfg.get("blend_text_weight", 0.5))
        blend_proto_weight = float(self.cfg.get("blend_proto_weight", 0.5))
        min_protos_full_agg = int(self.cfg.get("min_protos_for_full_agg", 5))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(clip_weights.t().float())   # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        states     = [self._make_class_state(D, device) for _ in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[Exp2] {dataset_name}")):
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

                # 2) Prototype branch
                raw_proto = torch.zeros(C, device=device)
                for c in range(C):
                    if states[c]["centers"].shape[0] > 0:
                        centers_norm = _safe_normalize(states[c]["centers"])
                        raw_proto[c] = _proto_score_for_class(
                            patches_norm,
                            centers_norm,
                            states[c]["appearance"],
                            update_samples=int(states[c]["n_images"]),
                            top_m=top_m,
                            min_protos_full_agg=min_protos_full_agg,
                            patch_group_threshold=patch_group_threshold,
                        )

                # 3) 50/50 blend: normalise both to probabilities then blend
                text_probs  = F.softmax(text_logits, dim=0)      # [C], sum=1
                proto_probs = F.softmax(raw_proto, dim=0)        # [C], sum=1

                final_probs = (
                    blend_text_weight * text_probs + blend_proto_weight * proto_probs
                )  # [C], sum=1

                final_logits = final_probs.unsqueeze(0)          # [1, C]

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
                    print(f"---- Exp2 {running:.2f}% ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- Exp2 FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(f"Exp2AdaptiveTauProto's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> Exp2AdaptiveTauProtoAdapter:
    return Exp2AdaptiveTauProtoAdapter(cfg)
