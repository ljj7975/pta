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
# OVERVIEW (non-technical)
# ---------------------------------------------------------------------------
# This variant keeps several patch-level prototypes per class and combines them
# with CLIP's text score at test time.
#
# Intuition:
# 1) CLIP text score gives a strong general prior.
# 2) Prototype score adds class-specific visual memory from confident test images.
# 3) The prototype branch is gated by evidence and quality so it ramps up
#    gradually instead of dominating too early.
# ---------------------------------------------------------------------------


def _proto_score_for_class(
    patches_norm: torch.Tensor,
    centers_norm: torch.Tensor,
    appearance: torch.Tensor,
    update_samples: int,
    top_m: int,
    patch_group_threshold: float = 0.9,
    return_details: bool = False,
) -> torch.Tensor:
    """
    Score one class by asking: does any patch match each prototype strongly?

    For each prototype we keep the best patch match, weight by appearance
    frequency, then average the top-M weighted scores.
    """
    # Reduce near-duplicate local regions first:
    # greedily group highly similar patches and use one centroid per group.
    # This avoids neighboring patches (e.g., both eye patches) dominating matches.
    num_input_patches = patches_norm.shape[0]
    if num_input_patches == 0:
        out = torch.tensor(0.0, device=centers_norm.device)
        if return_details:
            empty = torch.empty(0, device=centers_norm.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=centers_norm.device)
            return out, empty, empty, empty_idx, empty, empty_idx
        return out

    patch_sims = patches_norm @ patches_norm.t()  # [P, P]
    unassigned = torch.ones(num_input_patches, dtype=torch.bool, device=patches_norm.device)
    rep_centers = []
    group_members = []

    while unassigned.any():
        cand_idx = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        # Pick a dense anchor among remaining patches.
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

    # Compute cosine similarity between grouped patch representatives and prototypes.
    sims = rep_patches @ centers_norm.t()  # [num_groups, num_prototypes]

    num_groups, num_prototypes = sims.shape
    if num_prototypes == 0:
        out = torch.tensor(0.0, device=sims.device)
        if return_details:
            empty = torch.empty(0, device=sims.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=sims.device)
            return out, empty, empty, empty_idx, empty, empty_idx
        return out

    # Enforce one-to-one patch usage for prototype best matches:
    # once a patch is assigned to one prototype, it cannot be reused.
    proto_best_vals = sims.max(dim=0).values
    proto_order = torch.argsort(proto_best_vals, descending=True)
    used_groups = torch.zeros(num_groups, dtype=torch.bool, device=sims.device)

    best_per_proto = torch.zeros(num_prototypes, device=sims.device)
    best_patch_idx = torch.full((num_prototypes,), -1, dtype=torch.long, device=sims.device)

    for proto_idx in proto_order.tolist():
        scores = sims[:, proto_idx].clone()
        scores[used_groups] = float("-inf")
        best_val, group_idx = scores.max(dim=0)
        if torch.isneginf(best_val):
            continue
        best_per_proto[proto_idx] = best_val

        # For GUI/debug, map the selected representative group back to one
        # original patch: choose the group member most similar to this prototype.
        members = group_members[int(group_idx.item())]
        member_sims = patches_norm[members] @ centers_norm[proto_idx]
        local_best = int(member_sims.argmax().item())
        best_patch_idx[proto_idx] = members[local_best]

        used_groups[group_idx] = True

    # Appearance weight is normalized by number of update samples for the class,
    # not by total appearance sum across prototypes.
    denom = max(float(update_samples), 1e-6)
    app_w = appearance / denom

    weighted = best_per_proto * app_w

    k = min(top_m, weighted.numel())
    if k <= 0:
        out = torch.tensor(0.0, device=sims.device)
        if return_details:
            empty = torch.empty(0, device=sims.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=sims.device)
            return out, best_per_proto, app_w, best_patch_idx, weighted, empty_idx
        return out

    top_vals, top_idx = weighted.topk(k)
    score = top_vals.mean()

    if return_details:
        return score, best_per_proto, app_w, best_patch_idx, weighted, top_idx
    return score


def _alpha_from_evidence(n_images: float, n_half: float = 15.0) -> float:
    """
    Turn class evidence count into a smooth [0, 1] weight.

    More confident updates for a class -> larger alpha for that class.
    """
    if n_images <= 0:
        return 0.0
    return min(1.0, math.log(1.0 + n_images) / math.log(1.0 + n_half))


class MultiProtoPTAAdapter(BaseAdapter):

    # -----------------------------------------------------------------------
    # Class memory state
    # -----------------------------------------------------------------------

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
        """
        Update a single class memory using the current image patches.

        Special case: when the class has no prototypes yet, seed one prototype
        from the global image feature, then let incremental clustering refine it.

        We allow temporary growth past max_K during clustering, then prune the
        lowest-app_w prototypes so the bank stays within max_K.
        """
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
            keep = torch.argsort(app_w, descending=True)[:max_K]
            keep = torch.sort(keep).values
            updated_centers = updated_centers[keep]
            updated_apps = updated_apps[keep]

        state = dict(state)
        state["centers"]    = updated_centers
        state["appearance"] = updated_apps
        state["n_images"]   = state["n_images"] + 1
        return state

    def run(
        self,
        loader,
        clip_model,
        clip_weights,
        dataset_name: str,
    ) -> float:
        """
        Main inference loop.

        Per sample:
          1) get CLIP text logits and patch embeddings
          2) compute class prototype deltas
          3) gate prototype influence (evidence + quality)
          4) fuse text and prototype logits
          5) update class memories using confident pseudo-labels
        """
        # -------------------------------------------------------------------
        # Config
        # -------------------------------------------------------------------
        max_K        = int(self.cfg.get("max_K", 100))
        match_thresh = float(self.cfg.get("match_threshold", 0.60))
        conf_thresh  = float(self.cfg.get("conf_threshold", 0.5))
        # Require a clear CLIP winner before updating memory to reduce label drift.
        # Margin is measured on softmax(clip_logits): top1_prob - top2_prob.
        conf_margin_thresh = float(self.cfg.get("conf_margin_threshold", 0.05))
        n_half       = float(self.cfg.get("n_half", 15.0))
        alpha_max    = float(self.cfg.get("alpha_max", 0.2))
        top_m        = int(self.cfg.get("soft_nn_top_m", 4))
        patch_group_threshold = float(self.cfg.get("patch_group_threshold", 0.9))
        exclude_pos  = bool(self.cfg.get("exclude_pos", False))
        tau_proto    = float(self.cfg.get("tau_proto", 20.0))
        quality_eps  = float(self.cfg.get("quality_eps", 1e-3))

        os.makedirs("outputs", exist_ok=True)

        # -------------------------------------------------------------------
        # Setup
        # -------------------------------------------------------------------
        text_proto = _safe_normalize(clip_weights.t().float())
        C, D       = text_proto.shape
        device     = text_proto.device

        states     = [self._make_class_state(D, device) for _ in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[MPTA] {dataset_name}")):
                # Some datasets provide a list of augmented tensors.
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # 1) CLIP text branch + patch features
                # Get CLIP's global image embedding and text-based predictions
                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
                feat         = image_features.squeeze(0).float()
                feat_norm    = _safe_normalize(feat)  # Normalize global image embedding
                
                # Extract patch-level embeddings from Vision Transformer
                patch_embs   = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)  # shape: [num_patches, D]
                patches_norm = _safe_normalize(patch_embs)  # Normalize each patch embedding

                # Compute logits from text embeddings × global image embedding
                text_logits = 100.0 * (text_proto @ feat_norm)  # shape: [C] where C = num_classes

                # 2) Prototype branch per class
                # Compute raw prototype scores for all classes
                raw_proto = torch.zeros(C, device=device)

                for c in range(C):
                    # Only compute prototype score if we have learned prototypes for this class
                    if states[c]["centers"].shape[0] > 0:
                        centers_norm = _safe_normalize(states[c]["centers"])
                        raw_proto[c] = _proto_score_for_class(
                            patches_norm,
                            centers_norm,
                            states[c]["appearance"],
                            update_samples=int(states[c]["n_images"]),
                            top_m=top_m,
                            patch_group_threshold=patch_group_threshold,
                        )

                delta_proto = raw_proto

                # 3) Evidence gate (per class) and quality gate (per sample)
                # Per-class alpha: how much to trust this class's prototypes
                # Alpha increases smoothly as we accumulate more confident updates (n_images)
                alpha = torch.tensor(
                    [
                        # Per-class trust: 0 (new class) → alpha_max (well-explored class)
                        min(alpha_max, _alpha_from_evidence(states[c]["n_images"], n_half))
                        for c in range(C)
                    ],
                    device=device,
                )

                # Global quality gate: measures if prototypes give well-separated predictions
                # High variance → classes have distinct prototype scores → prototypes are informative
                proto_var    = delta_proto.var()  # Spread of prototype logits across classes
                quality_gate = proto_var / (proto_var + quality_eps)  # Smooth ratio: ∈ [0, 1]

                # 4) Fuse text + prototype contributions
                # final_logits = text_logits + (scaled prototype delta)
                # Scaling includes:
                #   - tau_proto: temperature controlling overall prototype influence strength
                #   - alpha: per-class confidence in prototypes (0 for new classes, up to alpha_max)
                #   - quality_gate: sample-level quality of prototype predictions
                #   - delta_proto: re-centered prototype scores for each class
                final_logits = (text_logits + tau_proto * alpha * quality_gate * delta_proto).unsqueeze(0)

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # 5) Online memory update via strict top-1 pseudo-label gate.
                # Update only the best CLIP class when confidence is high AND
                # clearly separated from the runner-up.
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)  # CLIP's class probabilities
                top2_vals, top2_idx = pred_conf.topk(min(2, C))

                best_conf = float(top2_vals[0].item())
                second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                conf_margin = best_conf - second_conf
                best_cls = int(top2_idx[0].item())

                # Gate update by both absolute confidence and top1-top2 separation.
                if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                    # Update even when the bank is full; _update_state prunes by app_w.
                    states[best_cls] = self._update_state(
                        states[best_cls], patches_norm, feat_norm, match_thresh, max_K
                    )

                if i % 500 == 0:
                    running  = sum(accuracies) / len(accuracies)
                    k_vals   = [s["centers"].shape[0] for s in states]
                    k_active = sum(1 for k in k_vals if k > 0)
                    k_mean   = sum(k_vals) / C
                    print(f"---- MPTA {running:.2f}%  active={k_active}/{C}  mean_K={k_mean:.2f} ----")

        final_acc = sum(accuracies) / len(accuracies)
        k_vals    = [s["centers"].shape[0] for s in states]
        k_active  = sum(1 for k in k_vals if k > 0)
        k_mean    = sum(k_vals) / C
        a_mean    = sum(_alpha_from_evidence(s["n_images"], n_half) for s in states) / C

        print(f"---- MPTA FINAL {final_acc:.2f}% ----")
        print(f"     active={k_active}/{C}  mean_K={k_mean:.2f}  mean_alpha={a_mean:.3f}")
        print(f"     K dist: min={min(k_vals)}  med={sorted(k_vals)[C//2]}  max={max(k_vals)}")

        with open("outputs/result.txt", "a") as f:
            f.write(f"MultiProtoPTA's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> MultiProtoPTAAdapter:
    return MultiProtoPTAAdapter(cfg)