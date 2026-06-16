import torch

from models.multi_proto_pta_base import (
    MultiProtoPTABase,
    _safe_normalize,
    _incremental_kmeans_step,
)


class MultiProtoPTAGaussianAdapter(MultiProtoPTABase):
    """
    Gaussian-style extension of MultiProtoPTA.

    The base class (MultiProtoPTABase) scores prototypes by raw cosine similarity:
    "how closely does the best patch match this prototype?"

    This class adds a concept of SPREAD (variance) to each prototype:
    "how tightly packed are the patches that belong to this prototype?"

    Think of it like this:
      - Base class prototype = a single point in feature space
      - Gaussian prototype   = a fuzzy "cloud" centred on that point

    A patch that falls close to the prototype's center scores HIGH.
    A patch that falls far away scores LOW.
    A patch that falls exactly at the centre of a WIDE cloud scores LOWER than
    the same patch at the centre of a TIGHT cloud (because a tight cloud means
    the prototype is very specific — a better discriminator).

    This uses a simplified version of a Mahalanobis distance:
      score = -0.5 * Σ_d (patch_d - center_d)² / variance_d
    Higher score = patch is closer to the prototype relative to its spread.

    Config keys specific to this class:
        gaussian_ema       (float, default 0.1)   — how quickly variance updates (0=never, 1=instant)
        gaussian_min_var   (float, default 1e-4)  — minimum allowed spread (prevents instability)
        gaussian_new_var   (float, default 0.05)  — spread used when a new prototype
                                                    is born from too little data (e.g., 1 patch)
    """

    def _make_class_state(self, prototype_vec: torch.Tensor):
        state = super()._make_class_state(prototype_vec)
        # Starts empty; each prototype receives a variance when it is created.
        state["variance"] = torch.empty_like(state["centers"])
        return state

    def _update_class_state(
        self,
        state,
        patch_embeddings: torch.Tensor,  # [P, D]  patches from the current image
        match_threshold: float,
        max_K: int,
    ):
        """
        Update prototype centers, appearances AND variances after seeing a new image.

        Variance update intuition:
          After patches are assigned to a prototype, we measure how far each patch
          strayed from the prototype center (these distances are called residuals).
          The average squared residual = how spread out this prototype's patches are.
          We blend that with the old variance via EMA (exponential moving average):
            new_variance = (1 - beta) * old_variance + beta * batch_variance
          Small beta → slow to change, large beta → quickly adapts to new data.

        For brand-new prototypes, we compute variance from the patches that formed
        them (or fall back to a default if only one patch was in the group).
        """
        centers   = state["centers"]    # [K, D]  current prototype centers
        apps      = state["appearance"] # [K]     per-sample appearance counts
        variances = state["variance"]   # [K, D]  current spread per prototype

        old_K = centers.shape[0]

        # Run the base clustering step: update centers, find which appeared, which patches matched
        updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
            centers, patch_embeddings, match_threshold, max_K
        )

        # Normalise for consistent distance computation
        patches_norm  = _safe_normalize(patch_embeddings, dim=-1)  # [P, D]
        centers_old   = _safe_normalize(centers, dim=-1)            # [K, D]

        beta          = float(self.cfg.get("gaussian_ema",     0.1))    # variance learning rate
        min_var       = float(self.cfg.get("gaussian_min_var", 1e-4))   # floor to avoid /0
        default_new_var = float(self.cfg.get("gaussian_new_var", 0.05)) # fallback for new protos

        updated_vars = variances.clone()  # start from current variances

        # Update variance for existing prototypes
        for k in range(old_K):
            mask = matched & (best_clusters == k)  # which patches matched prototype k
            if mask.any():
                # Residual = how far each matching patch is from the prototype center
                residuals = patches_norm[mask] - centers_old[k]  # [num_matched, D]
                # Average squared residual per dimension = batch variance
                batch_var = residuals.pow(2).mean(dim=0).clamp_min(min_var)  # [D]
                # EMA blend: nudge old variance toward this new batch estimate
                updated_vars[k] = (1 - beta) * variances[k] + beta * batch_var

        # Compute variance for newly created prototypes
        num_new = updated_centers.shape[0] - old_K
        if num_new > 0:
            new_vars = []
            for idx, group_idx in enumerate(new_groups):
                group_patches = patches_norm[group_idx]         # patches that formed this new cluster
                group_center  = updated_centers[old_K + idx]   # the new cluster's center
                # If the cluster was formed from very few patches, use a stable default spread.
                if group_patches.shape[0] <= 1:
                    group_var = torch.full(
                        (updated_centers.shape[1],),
                        default_new_var,
                        device=updated_centers.device,
                    )
                else:
                    group_residuals = group_patches - group_center  # deviation from center
                    # Variance = average squared deviation (how spread the founding patches were)
                    group_var = group_residuals.pow(2).mean(dim=0).clamp_min(min_var)
                new_vars.append(group_var)

            if len(new_vars) > 0:
                new_vars = torch.stack(new_vars, dim=0)  # [num_new, D]
            else:
                # Fallback: only happens if new_groups is empty despite num_new > 0
                new_vars = torch.full(
                    (num_new, updated_centers.shape[1]),
                    default_new_var,
                    device=updated_centers.device,
                )

            updated_vars = torch.cat([updated_vars, new_vars], dim=0)  # [K', D]

        # Update appearance counts (same logic as base class)
        updated_apps = apps.clone()
        updated_apps[appeared] += 1  # +1 per prototype per sample (not per patch)
        if num_new > 0:
            new_apps = torch.ones(num_new, device=apps.device)
            updated_apps = torch.cat([updated_apps, new_apps], dim=0)

        state = dict(state)  # shallow copy so we don't mutate the original
        state["centers"]    = updated_centers
        state["appearance"] = updated_apps
        state["variance"]   = updated_vars
        return state

    def _score_prototype_bank(
        self,
        patch_embeddings: torch.Tensor,
        feat: torch.Tensor,
        state,
        top_n: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score how well the image matches this class's prototype bank.

        Instead of raw cosine similarity (base class), we use a Gaussian score:
          score(patch, prototype) = -0.5 * Σ_d (patch_d - center_d)² / variance_d

        Intuition:
          - A patch that is very close to the prototype center \u2192 small squared distance \u2192 high score
          - A prototype with small variance (tight cluster) \u2192 divides by small number \u2192 penalises
            patches that aren't very close (it's a strict prototype)
          - A prototype with large variance (loose cluster) \u2192 more forgiving of distant patches

        For each prototype we take the HIGHEST score across all patches:
          "Does ANY patch in the image strongly resemble this prototype?"

        Then we take the appearance-weighted top-N and average them.
        """
        patches_norm = _safe_normalize(patch_embeddings, dim=-1)  # [P, D]
        centers      = state["centers"]     # [K, D]
        appearance   = state["appearance"]  # [K]
        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=patch_embeddings.device)

        variance     = state["variance"].clamp_min(  # [K, D]  clamp to avoid dividing by ~0
            float(self.cfg.get("gaussian_min_var", 1e-4))
        )

        # Compute (patch - center) for every patch-prototype pair
        # diff[p, k, d] = patch p's feature d minus prototype k's center feature d
        diff = patches_norm[:, None, :] - centers[None, :, :]     # [P, K, D]

        # Mahalanobis-like score: squared diff normalised by variance, averaged across D
        # Using .mean (not .sum) keeps scores scale-independent of feature dimension D
        maha = (diff.pow(2) / variance[None, :, :]).mean(dim=-1)   # [P, K]

        # Convert distance to score: closer = higher (negate and scale by -0.5)
        # Then take the BEST patch per prototype
        proto_scores = (-0.5 * maha).max(dim=0)[0]                # [K]

        # Weight by appearance: frequent prototypes matter more
        app_weights     = appearance / appearance.sum().clamp_min(1e-6)  # [K] sums to 1
        weighted_scores = proto_scores * app_weights               # [K]

        # Average the top-N most activated prototypes (ignore weak/background ones)
        k = min(top_n, weighted_scores.numel())
        if k == 1:
            return weighted_scores.max()
        return weighted_scores.topk(k).values.mean()


def build(cfg: dict) -> MultiProtoPTAGaussianAdapter:
    """
    Factory function called by runner.py to instantiate this adapter.
    Usage: python runner.py --method multi_proto_pta_gaussian ...
    """
    return MultiProtoPTAGaussianAdapter(cfg)