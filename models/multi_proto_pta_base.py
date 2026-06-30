import os
from abc import ABC
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from utils import get_clip_logits, cls_acc

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# This file implements MultiProtoPTA: a test-time adaptation method that builds
# a "visual memory" for each class by watching test images as they arrive.
#
# The core idea is simple:
#   - An image is made up of many small patches (e.g., 196 patches for a ViT)
#   - Patches from the same visual concept (e.g., "dog's snout") tend to look
#     similar to each other across different images
#   - We group similar patches together into "prototypes" (representative embeddings)
#   - When classifying a new image, we check whether those prototype patterns
#     are present anywhere in the image
#   - Prototypes that appear consistently across many images are likely foreground
#     (the thing we care about), while prototypes that appear rarely are likely
#     background noise
# ─────────────────────────────────────────────────────────────────────────────


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # Shrink every vector to length 1 so that dot-product = cosine similarity.
    # The clamp(min=eps) prevents dividing by zero for all-zero vectors.
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def _extract_patch_embeddings(image: torch.Tensor, clip_model, exclude_pos: bool = False) -> torch.Tensor:
    """
    Ask CLIP's vision encoder for per-patch embeddings instead of one global summary.

    A ViT (Vision Transformer) processes images by splitting them into a grid of
    small patches (e.g., 14x14 = 196 patches for ViT-B/16). Each patch produces
    its own embedding vector. We need these individual patch vectors so we can
    later cluster them into prototypes.

    Args:
        exclude_pos: If True, run the transformer without positional embeddings so
            patch tokens encode only visual content (no spatial location bias).
            Useful for prototype matching where a "nose" patch should match the nose
            prototype regardless of where it appears in the image.

    If the model does not support patch-level output, this raises an error
    (we explicitly do NOT fall back to the global image feature, because the
    whole point of this method is patch-level analysis).
    """
    with torch.no_grad():
        model_dtype = clip_model.visual.conv1.weight.dtype
        if exclude_pos:
            embeds = clip_model.visual(image.to(model_dtype), return_patches_no_pos=True)
        else:
            embeds = clip_model.visual(image.to(model_dtype), return_patches=True)  # ask for patch-level output

    if embeds is None:
        raise RuntimeError(
            "Patch embeddings are unavailable from clip_model.visual(..., return_patches=True)."
        )

    if embeds.dim() == 3:   # shape is [batch, num_patches, feature_dim]
        if embeds.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 for patch extraction, got {tuple(embeds.shape)}.")
        embeds = embeds.squeeze(0)  # drop batch dim → [num_patches, feature_dim]

    if embeds.dim() != 2:
        raise RuntimeError(f"Unexpected patch embedding shape: {tuple(embeds.shape)}")

    if embeds.shape[0] <= 1:
        raise RuntimeError(
            "Patch extraction produced <= 1 embedding. This implementation requires true patch-level embeddings."
        )

    return embeds.float()  # [P, D]  P = num patches, D = feature dimension


def _incremental_kmeans_step(
    cluster_centers: torch.Tensor,   # [K, D]  existing prototypes
    new_patches: torch.Tensor,       # [P, D]  patches from the incoming image
    match_threshold: float = 0.8,    # how similar a patch must be to "belong" to a prototype
    max_clusters: int = 30,          # hard cap: never create more than this many prototypes
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Update prototypes with one new image's patches. Deterministic (no randomness).

    Think of prototypes as "template" patches. For each incoming patch we ask:
      "Is this patch similar enough to any existing template?"
      YES → that template absorbs the patch slightly (moves toward it a little)
      NO  → the patch is unmatched; we may create a new template from it

    For unmatched patches we use a greedy grouping:
      1. Pick the first unmatched patch as an anchor
      2. Any other unmatched patch within threshold of it joins the same new cluster
      3. Their average becomes the new prototype
      4. Repeat with the remaining unmatched patches
    This is fully deterministic because processing order is fixed.

    Returns:
        updated_centers: [K', D]  all prototypes after update (K' >= K, up to max_clusters)
        appeared:        [K] bool  which OLD prototypes were matched by at least 1 patch
                         (used for per-sample appearance counting, not per-patch)
        matched:         [P] bool  which input patches matched an old prototype
        best_clusters:   [P] int   which old prototype each patch matched (meaningful only where matched=True)
        new_groups:      list of index tensors, one per new prototype created
    """
    K = cluster_centers.shape[0]
    ema_alpha = 0.1  # how much each new patch nudges the prototype (10% per update)

    patches_norm = _safe_normalize(new_patches, dim=-1)      # [P, D]

    # ── Empty-bank fast path ──────────────────────────────────────────────────
    # When K=0 there are no existing prototypes to compare against, so every
    # patch is unmatched and the similarity / EMA steps are skipped entirely.
    if K == 0:
        appeared      = torch.zeros(0, dtype=torch.bool, device=cluster_centers.device)
        matched       = torch.zeros(new_patches.shape[0], dtype=torch.bool, device=cluster_centers.device)
        best_clusters = torch.zeros(new_patches.shape[0], dtype=torch.long, device=cluster_centers.device)
        updated_centers = cluster_centers.clone()
        new_groups: List[torch.Tensor] = []
        remaining_idx = torch.arange(new_patches.shape[0], device=cluster_centers.device)
        while remaining_idx.numel() > 0 and updated_centers.shape[0] < max_clusters:
            anchor_idx = remaining_idx[0]
            anchor = patches_norm[anchor_idx]
            sims_to_anchor = patches_norm[remaining_idx] @ anchor
            members = sims_to_anchor >= match_threshold
            member_idx = remaining_idx[members]
            new_center = _safe_normalize(patches_norm[member_idx].mean(dim=0), dim=-1)
            updated_centers = torch.cat([updated_centers, new_center.unsqueeze(0)], dim=0)
            new_groups.append(member_idx)
            remaining_idx = remaining_idx[~members]
        return updated_centers, appeared, matched, best_clusters, new_groups

    # Normalize so that dot-product equals cosine similarity
    centers_norm = _safe_normalize(cluster_centers, dim=-1)  # [K, D]

    # For every patch, compute similarity to every existing prototype
    similarities = patches_norm @ centers_norm.t()           # [P, K]
    best_sims, best_clusters = similarities.max(dim=1)       # [P], [P]  best match per patch
    matched = best_sims >= match_threshold                   # [P] bool: does this patch fit any prototype?

    # Track which prototypes were visited by at least one patch this image
    appeared = torch.zeros(K, dtype=torch.bool, device=cluster_centers.device)  # [K]
    updated_centers = centers_norm.clone()                   # start from current prototypes

    # Update matched prototypes: nudge center slightly toward the incoming patches
    for k in range(K):
        mask = matched & (best_clusters == k)  # which patches matched prototype k
        if mask.any():
            appeared[k] = True
            mean_patch = patches_norm[mask].mean(dim=0)  # average of all matching patches
            # EMA: new_prototype = 90% old + 10% new_mean  (small step toward new data)
            updated_centers[k] = _safe_normalize(
                (1 - ema_alpha) * centers_norm[k] + ema_alpha * mean_patch,
                dim=-1,
            )

    # Handle unmatched patches: greedily form new prototypes
    remaining_idx = torch.nonzero(~matched, as_tuple=False).squeeze(1)  # indices of unmatched patches
    new_groups: List[torch.Tensor] = []

    while remaining_idx.numel() > 0 and updated_centers.shape[0] < max_clusters:
        # Take the first unmatched patch as the anchor for a new prototype
        anchor_idx = remaining_idx[0]
        anchor = patches_norm[anchor_idx]                    # [D]

        # Group any other unmatched patch that is similar to this anchor
        sims_to_anchor = patches_norm[remaining_idx] @ anchor  # [remaining]
        members = sims_to_anchor >= match_threshold            # which ones are similar
        member_idx = remaining_idx[members]

        # New prototype = mean of the group, normalized
        new_center = _safe_normalize(patches_norm[member_idx].mean(dim=0), dim=-1)
        updated_centers = torch.cat([updated_centers, new_center.unsqueeze(0)], dim=0)
        new_groups.append(member_idx)

        remaining_idx = remaining_idx[~members]              # continue with the rest

    return updated_centers, appeared, matched, best_clusters, new_groups


class MultiProtoPTABase(BaseAdapter, ABC):
    """
    Shared base class for all MultiProtoPTA variants.

    Subclasses can override _score_prototype_bank() to change how prototype
    similarity scores are computed. Everything else (online clustering, appearance
    tracking, cross-class penalty, main loop) lives here.
    """

    def _make_class_state(self, prototype_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        D      = prototype_vec.shape[0]
        device = prototype_vec.device
        return {
            "centers":    torch.empty(0, D, device=device),
            "appearance": torch.empty(0,    device=device),
        }

    def _update_class_state(
        self,
        state: Dict[str, torch.Tensor],
        patch_embeddings: torch.Tensor,  # [P, D]  patches from one image
        match_threshold: float,
        max_K: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Update memory for one class after seeing a new (high-confidence) image.

        Steps:
          1. Compare each patch to existing prototypes
          2. Matched patches nudge their prototype slightly (EMA update)
          3. Unmatched patches may create new prototypes (up to max_K total)
          4. Increment the appearance count for each prototype that was matched
             (once per IMAGE, not once per patch — background patches that
             flood an image shouldn't look more important than they are)
        """
        centers = state["centers"]   # [K, D]
        apps    = state["appearance"] # [K]
        old_K   = centers.shape[0]

        updated_centers, appeared, _, _, _ = _incremental_kmeans_step(
            centers, patch_embeddings, match_threshold, max_K
        )
        # appeared[k] = True means prototype k was seen in this image

        # Add 1 to appearance count for every prototype that matched this sample
        updated_apps = apps.clone()
        updated_apps[appeared] += 1

        # New prototypes start with appearance = 1 (they came from this image)
        num_new = updated_centers.shape[0] - old_K
        if num_new > 0:
            new_apps = torch.ones(num_new, device=apps.device)
            updated_apps = torch.cat([updated_apps, new_apps], dim=0)

        state = dict(state)  # shallow copy so we don't mutate the original
        state["centers"]    = updated_centers
        state["appearance"] = updated_apps
        return state

    def _score_prototype_bank(
        self,
        patch_embeddings: torch.Tensor,
        feat: torch.Tensor,
        state: Dict[str, torch.Tensor],
        top_n: int = 5,
        global_gate: bool = False,
        softmax_vote: bool = False,
        softmax_vote_temp: float = 0.05,
    ) -> torch.Tensor:
        """
        Convert one class's prototype bank into a single score for this image.

        Flow:
          1. For each prototype, find the best-matching patch in the image.
          2. Weight that prototype by how often it has appeared historically.
          3. Aggregate the strongest prototypes (mean top-N or softmax vote).
        """
        centers    = state["centers"]     # [K, D]
        appearance = state["appearance"]  # [K]

        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=centers.device)

        patches_norm = _safe_normalize(patch_embeddings, dim=-1)  # [P, D]

        sims = patches_norm @ _safe_normalize(centers, dim=-1).t()  # [P, K]
        max_sims = sims.max(dim=0)[0]  # [K]

        app_weights = appearance / appearance.sum().clamp_min(1e-6)  # [K]
        weighted_scores = max_sims * app_weights  # [K]

        if global_gate:
            gate = (feat @ _safe_normalize(centers[0], dim=-1)).clamp(min=0.0)
            weighted_scores = weighted_scores * (0.5 + 0.5 * gate)

        k = min(top_n, weighted_scores.numel())
        if k == 1:
            return weighted_scores.max()

        if softmax_vote:
            top_vals = weighted_scores.topk(k).values
            w = F.softmax(top_vals / softmax_vote_temp, dim=0)
            return (top_vals * w).sum()

        return weighted_scores.topk(k).values.mean()

    def _compute_common_penalty(
        self,
        class_id: int,
        states: List[Dict[str, torch.Tensor]],
        common_threshold: float,
    ) -> float:
        """
        Compute how "generic" (non-distinctive) a class's prototypes are.

        If a prototype of class A looks very similar to a prototype of class B,
        that prototype doesn't help us tell A and B apart — it might be something
        both classes share (e.g., a grass background).

        For each prototype in the target class, we check the maximum similarity
        to any prototype in any other class. If that similarity exceeds
        common_threshold, the prototype is flagged as "common".

        The penalty is the fraction of prototypes that are common (0 = none, 1 = all).
        This is then applied as a multiplicative reduction to score_proto.
        """
        C = len(states)
        protos_c = states[class_id]["centers"]  # [K_c, D]

        penalty = 0.0
        for k_c in range(protos_c.shape[0]):
            proto = protos_c[k_c]  # [D]  one prototype from this class
            max_cross_sim = 0.0

            for c_other in range(C):
                if c_other == class_id:
                    continue
                protos_other = states[c_other]["centers"]   # [K_other, D]
                if protos_other.shape[0] == 0:
                    continue
                cross_sims   = protos_other @ proto          # [K_other]
                max_cross_sim = max(max_cross_sim, cross_sims.max().item())

            # This prototype is too similar to something in another class
            if max_cross_sim > common_threshold:
                penalty += 1.0 / protos_c.shape[0]  # each common proto contributes equally

        return min(penalty, 1.0)

    def run(
        self,
        loader,
        clip_model,
        clip_weights,
        dataset_name: str,
    ) -> float:
        """
        Main evaluation loop. Processes test images one by one.

        For each image:
          1. Get CLIP zero-shot prediction (text branch)
          2. Extract patch embeddings
          3. Score each class using both text similarity and prototype memory
          4. Record accuracy
          5. If the prediction was confident, update the prototype memory
        """
        # ──────────────────────────────────────────────────────────────────────
        # CONFIG — all settings loaded from per-dataset YAML
        # ──────────────────────────────────────────────────────────────────────
        max_K             = int(self.cfg.get("max_K", 30))
        match_thresh      = float(self.cfg.get("match_threshold", 0.8))
        conf_thresh       = float(self.cfg.get("conf_threshold", 0.1))
        tau_proto         = float(self.cfg.get("tau_proto", self.cfg.get("T", 20.0)))
        penalize_common   = self.cfg.get("penalize_common", True)
        common_thresh     = float(self.cfg.get("common_threshold", 0.7))
        proto_top_n       = int(self.cfg.get("proto_top_n", 5))
        disable_centering = bool(self.cfg.get("disable_centering", False))
        topk_update       = int(self.cfg.get("topk_update", 1))
        adaptive_tau      = bool(self.cfg.get("adaptive_tau", False))
        use_global_gate   = bool(self.cfg.get("global_gate", False))
        use_softmax_vote  = bool(self.cfg.get("softmax_vote", False))
        softmax_vote_temp = float(self.cfg.get("softmax_vote_temp", 0.05))

        os.makedirs("outputs", exist_ok=True)

        # ──────────────────────────────────────────────────────────────────────
        # SETUP
        # ──────────────────────────────────────────────────────────────────────
        # clip_weights: [D, C]  — transpose to get [C, D] (one row per class)
        text_proto = _safe_normalize(clip_weights.t().float())  # [C, D]
        C, _ = text_proto.shape  # C = number of classes
        device = text_proto.device

        states = [self._make_class_state(text_proto[c]) for c in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[MultiProtoPTA] {dataset_name}")):
                # OOD datasets (R/S/A) return a list of tensors; ImageNet-I returns a tensor.
                # Normalise to a single tensor on device before any model call.
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # ── CLIP zero-shot branch ───────────────────────────────────────────
                # Get the global image feature and zero-shot logits from CLIP
                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
                feat = image_features.squeeze(0).float()  # [D]  global image summary

                # ── Extract patch embeddings ───────────────────────────────────────
                # Get per-patch embeddings for prototype-level scoring and update
                patch_embs = _extract_patch_embeddings(images, clip_model)  # [P, D]

                # ── Compute per-class logits ──────────────────────────────────────
                # Step 1: gather raw proto scores for all classes
                raw_proto_scores = torch.zeros(C, device=device)
                for c in range(C):
                    raw_score = self._score_prototype_bank(
                        patch_embeddings=patch_embs,
                        feat=feat,
                        state=states[c],
                        top_n=proto_top_n,
                        global_gate=use_global_gate,
                        softmax_vote=use_softmax_vote,
                        softmax_vote_temp=softmax_vote_temp,
                    )
                    K_c = states[c]["centers"].shape[0]
                    evidence_weight = min(1.0, 0.5 + 0.5 * (K_c - 1) / 4.0) if K_c >= 1 else 0.0
                    raw_proto_scores[c] = evidence_weight * raw_score

                # Step 2: optionally center proto scores across classes
                centered_proto = raw_proto_scores if disable_centering else raw_proto_scores - raw_proto_scores.mean()

                # Step 3: adaptive tau — scale proto weight by how discriminative it is this sample
                eff_tau_proto = tau_proto
                if adaptive_tau:
                    proto_var = centered_proto.var().item()
                    eff_tau_proto = tau_proto * (1.0 + proto_var / (proto_var + 0.01))

                # Step 4: fuse text + centered proto into final logits
                final_logits = torch.zeros(1, C, device=device)
                for c in range(C):
                    score_text = 100.0 * (text_proto[c] @ feat)
                    if penalize_common:
                        common_penalty = self._compute_common_penalty(c, states, common_thresh)
                        centered_proto[c] = centered_proto[c] * (1 - common_penalty)
                    final_logits[0, c] = score_text + eff_tau_proto * centered_proto[c]

                # ── Accuracy ─────────────────────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # ── Online update (after prediction) ──────────────────────────────
                # Only update memory if CLIP was confident about its prediction.
                # Using a confident prediction as a pseudo-label reduces noise from
                # uncertain or ambiguous images.
                pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)  # [C]
                if topk_update == 1:
                    pred_class = pred_conf.argmax()
                    if pred_conf[pred_class] > conf_thresh:
                        states[pred_class] = self._update_class_state(
                            states[pred_class], patch_embs, match_thresh, max_K,
                        )
                else:
                    topk_vals, topk_idx = pred_conf.topk(min(topk_update, C))
                    for rank, (conf_val, cls_idx) in enumerate(zip(topk_vals, topk_idx)):
                        threshold = conf_thresh * (0.6 ** rank)
                        if conf_val > threshold:
                            states[cls_idx] = self._update_class_state(
                                states[cls_idx], patch_embs, match_thresh, max_K,
                            )

                if i % 1000 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- MultiProtoPTA test accuracy: {running:.2f}. ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- MultiProtoPTA test accuracy: {final_acc:.2f}. ----\n")

        with open("outputs/result.txt", "a") as f:
            f.write(f"MultiProtoPTA's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc