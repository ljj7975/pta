"""Incremental k-means and Gaussian scoring helpers for prototype-based TTA.

These functions are used exclusively by GaussianPatchLevel and the proto_viz
tool — they are not general-purpose utilities.
"""
from typing import List, Optional, Tuple

import torch

from utils.clip_inference import _safe_normalize
from utils.proto_stats import zscore as _zscore


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
        appeared:        [K] long  number of patches that matched each OLD prototype
                         (used for per-sample appearance counting with patch-count threshold)
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
        appeared      = torch.zeros(0, dtype=torch.long, device=cluster_centers.device)
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
    appeared = torch.zeros(K, dtype=torch.long, device=cluster_centers.device)  # [K]
    updated_centers = centers_norm.clone()                   # start from current prototypes

    # Update matched prototypes: nudge center slightly toward the incoming patches
    for k in range(K):
        mask = matched & (best_clusters == k)  # which patches matched prototype k
        if mask.any():
            appeared[k] += 1
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


def _gaussian_score_for_class(
    patches_norm: torch.Tensor,
    centers_norm: torch.Tensor,
    variance: torch.Tensor,
    appearance: torch.Tensor,
    update_samples: int,
    top_m: int,
    patch_group_threshold: float = 0.9,
    variance_min: float = 0.001,
    aggregation: str = "top_m_mean",
    return_details: bool = False,
    proto_mu: Optional[torch.Tensor] = None,
    proto_sigma: Optional[torch.Tensor] = None,
    proto_valid: Optional[torch.Tensor] = None,
    sigma_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Score one class using Gaussian prototype similarity.

    For each patch group and prototype, compute Gaussian score:
        score = exp(-0.5 * Σ_d (patch_d - center_d)² / variance_d)

    Then apply one-to-one assignment (same as MPTA) and top-M aggregation.

    The ``zscore_*`` aggregations first normalize each prototype's raw score
    against that prototype's own reference distribution (``proto_mu`` /
    ``proto_sigma``, maintained online by utils.proto_stats), so that scores from
    prototypes with different natural score ranges become comparable. They
    require ``proto_mu``/``proto_sigma``/``proto_valid``; without them the call
    falls back to the raw-score path.

    Returns a scalar score for this class.
    """
    num_input_patches = patches_norm.shape[0]
    if num_input_patches == 0 or centers_norm.shape[0] == 0:
        empty = torch.tensor(0.0, device=centers_norm.device)
        if return_details:
            K0 = centers_norm.shape[0]
            zeros = torch.zeros(K0, device=centers_norm.device)
            return empty, {
                "best_per_proto": zeros,
                "weighted": zeros.clone(),
                "app_w": zeros.clone(),
            }
        return empty

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
    #
    # The Mahalanobis distance sums over all D dimensions. With D=512 (ViT-B/16),
    # a per-dimension variance floor of 0.001 produces total distances of 200–1000,
    # causing exp(-0.5 * maha) to underflow to exactly 0.0 in float32.
    #
    # Fix: scale the effective variance floor by D so that the total Mahalanobis
    # distance stays in a numerically stable range (~1–20 instead of ~200–1000).
    # The stored variance values are unchanged; this only affects the scoring floor.
    D = variance.shape[-1]
    effective_var_min = variance_min * D  # e.g. 0.001 * 512 ≈ 0.512
    var_clamped = variance.clamp(min=effective_var_min)  # [K, D]
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
        empty = torch.tensor(0.0, device=gaussian_scores.device)
        if return_details:
            return empty, {
                "best_per_proto": best_per_proto,
                "weighted": weighted,
                "app_w": app_w,
            }
        return empty

    # ── Z-score normalization (optional) ───────────────────────────────────
    # Convert each prototype's raw score into a prototype-specific z-score, then
    # aggregate with appearance weights normalized to sum to 1 within the class.
    # This makes class scores comparable even when classes hold different
    # numbers of prototypes with different natural score ranges.
    z = None
    w_norm = None
    if aggregation.startswith("zscore") and proto_mu is not None:
        z = _zscore(best_per_proto, proto_mu, proto_sigma, proto_valid, sigma_eps)
        w_norm = app_w / app_w.sum().clamp(min=1e-6)
        z_weighted = w_norm * z  # [K]

        if aggregation == "zscore_top_m_mean":
            score = z_weighted.topk(k).values.mean()
        else:  # "zscore_weighted_mean" — the weighted mean z-score
            score = z_weighted.sum()

        if return_details:
            return score, {
                "best_per_proto": best_per_proto,  # [K] raw score, kept for comparison
                "weighted": z_weighted,            # [K] normalized-weight × z
                "app_w": app_w,                    # [K] raw appearance weights
                "mu": proto_mu,                    # [K] reference mean
                "sigma": proto_sigma,              # [K] reference std
                "z": z,                            # [K] prototype-specific z-score
                "w_norm": w_norm,                  # [K] appearance weights, sum to 1
            }
        return score

    if aggregation == "top_m_mean":
        score = weighted.topk(k).values.mean()
    elif aggregation == "max":
        score = weighted.max()
    elif aggregation == "sum":
        score = weighted.sum()
    elif aggregation == "mean":
        score = weighted.mean()
    elif aggregation == "top_m_mean_plus_mean":
        top_m_val = weighted.topk(k).values.mean()
        all_mean = weighted.mean()
        score = (top_m_val + all_mean) / 2.0
    elif aggregation == "weighted_mean":
        # Appearance-weighted average: sum(best_per_proto * app_w) / sum(app_w)
        app_w_sum = app_w.sum()
        score = weighted.sum() / app_w_sum.clamp(min=1e-6)
    else:
        score = weighted.topk(k).values.mean()

    if return_details:
        return score, {
            "best_per_proto": best_per_proto,   # [K] Gaussian score per prototype (before weighting)
            "weighted": weighted,               # [K] after appearance weighting
            "app_w": app_w,                     # [K] appearance weights
        }
    return score
