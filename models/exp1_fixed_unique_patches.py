"""
Experiment 1: Fixed Unique Patches Per Sample
==============================================
Instead of a threshold-based patch grouping, keep a fixed number of
maximally-diverse patch representatives per sample (farthest-first selection).

Key change vs multi_proto_pta.py:
  - `patch_group_threshold` removed
  - `n_unique_patches_per_sample` (default 15) controls how many patch
     representatives are selected per image before prototype matching
  - Diversity selection = greedy farthest-first from the patch set

This stabilises the patch count regardless of positional encoding and avoids
the aggressive merging that occurs when exclude_pos=True.
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


def _select_diverse_patches(
    patches_norm: torch.Tensor,   # [P, D] L2-normalised
    n: int,                        # how many to keep
) -> torch.Tensor:
    """
    Greedy farthest-first selection of n patch representatives.

    Starts from the patch closest to the set centroid (most "central") and
    iteratively adds the patch that is farthest from all already-selected
    patches (maximises minimum distance to the current set).

    Returns indices of the selected patches (length = min(n, P)).
    """
    P = patches_norm.shape[0]
    if P <= n:
        return torch.arange(P, device=patches_norm.device)

    # Seed: patch closest to centroid
    centroid = _safe_normalize(patches_norm.mean(dim=0))          # [D]
    sims_to_centroid = patches_norm @ centroid                     # [P]
    selected = [int(sims_to_centroid.argmax().item())]

    # Cosine distances (1 - cosine_sim) tracked per patch to nearest selected
    # We work in similarity space and negate: farther = lower similarity
    selected_tensor = patches_norm[selected[0]].unsqueeze(0)       # [1, D]
    min_sims = (patches_norm @ selected_tensor.t()).squeeze(1)     # [P]  sim to nearest

    for _ in range(n - 1):
        # The least-similar patch to all already-selected patches is the farthest
        next_idx = int(min_sims.argmin().item())
        selected.append(next_idx)
        new_sims = patches_norm @ patches_norm[next_idx]           # [P]
        min_sims = torch.max(min_sims, new_sims)                   # update nearest-selected sims

    return torch.tensor(selected, device=patches_norm.device, dtype=torch.long)


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
) -> torch.Tensor:
    """Top-M mean prototype score (same aggregation as MPTA B2C1)."""
    if patches_norm.shape[0] == 0 or centers_norm.shape[0] == 0:
        return torch.tensor(0.0, device=centers_norm.device)

    sims = patches_norm @ centers_norm.t()          # [P, K]
    best_per_proto = sims.max(dim=0).values         # [K]

    denom = max(float(update_samples), 1e-6)
    app_w = appearance / denom
    weighted = best_per_proto * app_w

    k = min(top_m, weighted.numel())
    if k <= 0:
        return torch.tensor(0.0, device=sims.device)
    return weighted.topk(k).values.mean()


class Exp1FixedUniquePatchesAdapter(BaseAdapter):

    def _make_class_state(self, D: int, device: torch.device) -> Dict:
        return {
            "centers":    torch.empty(0, D, device=device),
            "appearance": torch.empty(0,    device=device),
            "n_images":   0,
        }

    def _update_state(
        self,
        state: Dict,
        patches_norm: torch.Tensor,   # already-diverse subset, normalised
        global_feat: torch.Tensor,
        match_threshold: float,
        max_K: int,
    ) -> Dict:
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
        max_K            = int(self.cfg.get("max_K", 100))
        match_thresh     = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh      = float(self.cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half           = float(self.cfg.get("n_half", 15.0))
        alpha_max        = float(self.cfg.get("alpha_max", 0.2))
        top_m            = int(self.cfg.get("soft_nn_top_m", 4))
        exclude_pos      = bool(self.cfg.get("exclude_pos", False))
        tau_proto        = float(self.cfg.get("tau_proto", 20.0))
        quality_eps      = float(self.cfg.get("quality_eps", 1e-3))
        # ── Exp 1 specific ──────────────────────────────────────────────────
        n_unique = int(self.cfg.get("n_unique_patches_per_sample", 15))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(clip_weights.t().float())     # [C, D]
        C, D       = text_proto.shape
        device     = text_proto.device

        states     = [self._make_class_state(D, device) for _ in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[Exp1] {dataset_name}")):
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
                feat       = image_features.squeeze(0).float()
                feat_norm  = _safe_normalize(feat)

                # Extract patches then select diverse subset
                patch_embs   = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)
                patches_norm_all = _safe_normalize(patch_embs)
                div_idx      = _select_diverse_patches(patches_norm_all, n_unique)
                patches_norm = patches_norm_all[div_idx]           # [n_unique, D]

                text_logits = 100.0 * (text_proto @ feat_norm)     # [C]

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
                        )

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

                # Update memory
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
                    print(f"---- Exp1 {running:.2f}%  n_unique={n_unique} ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- Exp1 FINAL {final_acc:.2f}% ----")

        with open("outputs/result.txt", "a") as f:
            f.write(f"Exp1FixedUniquePatches's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> Exp1FixedUniquePatchesAdapter:
    return Exp1FixedUniquePatchesAdapter(cfg)
