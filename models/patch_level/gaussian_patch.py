"""
Gaussian Patch-Level Prototypes

Represent each prototype as a Gaussian (center + per-dimension variance) and
replace cosine-similarity matching with a Mahalanobis-like distance.

This is the patch-level component that implements BasePatchLevel, intended to
be composed into a full adapter (e.g., via PTA's multi-component setup).
"""
import torch
import torchvision.transforms.functional as TF
import clip as _clip  # local vendored CLIP — for tokenize()

from typing import Optional

from models.patch_level.base import BasePatchLevel
from utils.kmeans import (
    _incremental_kmeans_step,
    _gaussian_score_for_class,
)
from utils.clip_inference import (
    _safe_normalize,
    compute_surgery_scores,
    filter_patches_by_text_alignment,
)
from utils import proto_stats


# Fixed augmentation ranges — not exposed as config
_AUG_ROTATION_DEG   = 15.0   # ±degrees
_AUG_TRANSLATE_FRAC = 0.10   # ±fraction of image size
_AUG_SCALE_MIN      = 0.85
_AUG_SCALE_MAX      = 1.15
_AUG_BRIGHTNESS_MAG = 0.20   # additive offset ±this value
_AUG_CONTRAST_MIN   = 0.80
_AUG_CONTRAST_MAX   = 1.20
_AUG_BLUR_KERNEL_MIN = 3
_AUG_BLUR_KERNEL_MAX = 11
_AUG_BLUR_SIGMA_MIN  = 0.1
_AUG_BLUR_SIGMA_MAX  = 2.0


def _augment_image(image: torch.Tensor) -> torch.Tensor:
    """
    Apply a randomly-parameterized composite augmentation to a CLIP-preprocessed image tensor.

    All five transforms are applied jointly each call:
      - Rotation: uniform ±_AUG_ROTATION_DEG degrees
      - Affine (translation + scale): uniform translation ±_AUG_TRANSLATE_FRAC
        of each spatial dimension, scale uniform in [_AUG_SCALE_MIN, _AUG_SCALE_MAX]
      - Brightness: additive offset uniform in [-_AUG_BRIGHTNESS_MAG, +_AUG_BRIGHTNESS_MAG]
      - Contrast: linear rescaling around per-channel mean, factor in
        [_AUG_CONTRAST_MIN, _AUG_CONTRAST_MAX]
      - Gaussian blur: kernel size uniform in [_AUG_BLUR_KERNEL_MIN, _AUG_BLUR_KERNEL_MAX],
        sigma uniform in [_AUG_BLUR_SIGMA_MIN, _AUG_BLUR_SIGMA_MAX]

    Brightness and contrast are implemented as raw tensor ops (no [0,1] clamping)
    so they remain valid for CLIP's zero-centred normalised pixel values.

    Args:
        image: ``[1, C, H, W]`` float tensor (CLIP-preprocessed, on any device).

    Returns:
        Augmented copy with the same shape and dtype as *image*.
    """
    img = image.squeeze(0)                             # [C, H, W]
    _, H, W = img.shape

    # ── Rotation ─────────────────────────────────────────────────────
    angle = (torch.rand(1).item() * 2 - 1) * _AUG_ROTATION_DEG
    img = TF.rotate(img, angle=angle)

    # ── Affine: translation + scale ──────────────────────────────────
    tx = int((torch.rand(1).item() * 2 - 1) * _AUG_TRANSLATE_FRAC * W)
    ty = int((torch.rand(1).item() * 2 - 1) * _AUG_TRANSLATE_FRAC * H)
    scale = _AUG_SCALE_MIN + torch.rand(1).item() * (_AUG_SCALE_MAX - _AUG_SCALE_MIN)
    img = TF.affine(img, angle=0, translate=[tx, ty], scale=scale, shear=0)

    # ── Brightness: additive offset (tensor-safe) ─────────────────────
    brightness_delta = (torch.rand(1, device=image.device).item() * 2 - 1) * _AUG_BRIGHTNESS_MAG
    img = img + brightness_delta

    # ── Contrast: rescale around per-channel spatial mean ─────────────
    contrast_factor = _AUG_CONTRAST_MIN + torch.rand(1, device=image.device).item() * (
        _AUG_CONTRAST_MAX - _AUG_CONTRAST_MIN
    )
    channel_mean = img.mean(dim=[-2, -1], keepdim=True)  # [C, 1, 1]
    img = contrast_factor * img + (1 - contrast_factor) * channel_mean

    # ── Gaussian blur ──────────────────────────────────────────────────
    kernel_size = _AUG_BLUR_KERNEL_MIN + int(
        torch.rand(1).item() * (_AUG_BLUR_KERNEL_MAX - _AUG_BLUR_KERNEL_MIN)
    )
    # Ensure kernel size is odd (required by TF.gaussian_blur)
    kernel_size += kernel_size % 2 == 0
    sigma = _AUG_BLUR_SIGMA_MIN + torch.rand(1).item() * (
        _AUG_BLUR_SIGMA_MAX - _AUG_BLUR_SIGMA_MIN
    )
    img = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    return img.unsqueeze(0).to(dtype=image.dtype)       # [1, C, H, W]


class GaussianPatchLevel(BasePatchLevel):

    def init_state(self, text_features):
        C, D = text_features.shape
        device = text_features.device
        return [
            {
                "centers":    torch.empty(0, D, device=device),
                "variance":   torch.empty(0, D, device=device),
                "appearance": torch.empty(0,    device=device),
                "n_images":   0,
                "top_rep_patches": [],  # list of lists: top-3 (patch_idx, image_idx, sim) per cluster
                # Running reference-score statistics, one entry per prototype.
                # Grown and permuted in lockstep with centers/variance/appearance.
                **proto_stats.init_stats(0, device),
            }
            for _ in range(C)
        ]

    def set_text_context(self, clip_weights, clip_model, device):
        """Cache text features and empty-text baseline for patch filtering.
        
        Must be called once before the TTA loop when patch_filter_mode != "none".
        
        Args:
            clip_weights: [D, C] CLIP text weight matrix
            clip_model: CLIP model (for encoding empty text)
            device: torch device
        """
        self._text_features = clip_weights.t().float()  # [C, D], L2-normalised per class
        self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Empty-text baseline: raw tokenization, no templates.
        # Bypass Encoder wrappers (which add prompt templates) by calling the
        # underlying raw CLIP model's encode_text directly with pre-tokenized tokens.
        tokens = _clip.tokenize([""]).to(device)
        raw_model = clip_model.model if hasattr(clip_model, "model") else clip_model
        with torch.no_grad():
            empty_feat = raw_model.encode_text(tokens).float()  # [1, D]
            empty_feat = empty_feat / empty_feat.norm(dim=-1, keepdim=True)
        self._empty_text_feat = empty_feat  # [1, D]
        
        self._filter_mode = self._cfg.get("patch_filter_mode", "none")
        self._filter_threshold = float(self._cfg.get("patch_filter_threshold", 0.5))

    def precompute_filter_scores(
        self,
        images: torch.Tensor,
        encoder,
    ) -> Optional[torch.Tensor]:
        """Compute surgery relevance scores for all patches and classes.

        Intended to be called **once per image** in the adapter loop before
        ``update_state``, so the surgery forward pass is shared across all
        class updates for that image.

        Args:
            images:  ``[1, C_img, H, W]`` input image tensor.
            encoder: encoder wrapper (provides encode_image for surgery pass).

        Returns:
            ``[P, C]`` tensor of scores, or ``None`` if the configured
            ``patch_filter_mode`` is not a surgery mode or if text context
            has not yet been set via :meth:`set_text_context`.
        """
        if not hasattr(self, "_text_features"):
            return None
        return compute_surgery_scores(
            images, encoder,
            self._text_features,
            self._empty_text_feat,
            self._filter_mode,
        )

    def compute_patch_logits(self, images, encoder, states,
                              return_details=False,
                              *, filter_scores=None, target_class_idx=None):
        """Score every class's prototype bank against one image.

        Note: when prototype score statistics are enabled (a ``zscore_*``
        aggregation, or ``proto_stats_track``) this method **mutates** the
        per-class dicts in *states*, folding this image's raw prototype scores
        into each bank's running reference distribution. The write happens only
        after every class has been scored, so the current image is never part of
        the statistics used to score it.
        """
        # ── Config ────────────────────────────────────────────────────────────
        exclude_pos = bool(self._cfg.get("exclude_pos", False))
        top_m = int(self._cfg.get("soft_nn_top_m", 4))
        quality_eps = float(self._cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self._cfg.get("patch_group_threshold", 0.9))
        variance_min = float(self._cfg.get("variance_min", 0.001))
        aggregation = str(self._cfg.get("aggregation", "top_m_mean"))

        # ── Prototype score statistics ────────────────────────────────────────
        stats_min_count = int(self._cfg.get("proto_stats_min_count", 10))
        stats_sigma_eps = float(self._cfg.get("proto_stats_sigma_eps", 1e-6))
        stats_sigma_warn = float(self._cfg.get("proto_stats_sigma_warn", 1e-4))
        stats_log_every = int(self._cfg.get("proto_stats_log_every", 0))
        stats_enabled = (
            aggregation.startswith("zscore")
            or bool(self._cfg.get("proto_stats_track", False))
        )
        # The z-score path needs per-prototype raw scores, which only come back
        # via the details dict.
        want_details = bool(return_details) or stats_enabled

        num_classes = len(states)
        device = states[0]["centers"].device if num_classes > 0 else images.device

        # ── Extract patch embeddings ─────────────────────────────────────────
        patch_embs = encoder.get_patch_embeddings(images, exclude_pos=exclude_pos)  # extract patch embeddings
        patches_norm = _safe_normalize(patch_embs)  # [P, D]

        # ── Patch relevance filtering (on full patch pool) ────────────────────
        filter_mode = self._cfg.get("patch_filter_mode", "none")
        keep_mask = None
        filter_scores_per_class = None
        if (filter_mode != "none"
                and hasattr(self, "_text_features")
                and target_class_idx is not None):
            if filter_scores is None and filter_mode in ("surgery_with_labels", "surgery_no_labels"):
                filter_scores = compute_surgery_scores(
                    images, encoder, self._text_features, self._empty_text_feat, filter_mode
                )  # [P, C]
            keep_mask, filter_scores_per_class = filter_patches_by_text_alignment(
                patches_norm, target_class_idx,
                text_features=self._text_features,
                empty_text_feat=self._empty_text_feat,
                filter_mode=filter_mode,
                filter_threshold=self._filter_threshold,
                aug_copies=0,
                precomputed_scores=filter_scores[:, target_class_idx] if filter_scores is not None else None,
                encoder=encoder,
                return_scores=True,
            )
            patches_norm = patches_norm[keep_mask]  # [P_filtered, D]

        # ── Compute per-class Gaussian score ──────────────────────────────────
        raw_proto = torch.zeros(num_classes, device=device)
        per_class_details = {}  # c → {best_per_proto, weighted, app_w, ...}
        for c in range(num_classes):
            if states[c]["centers"].shape[0] > 0:
                centers_norm = _safe_normalize(states[c]["centers"])

                # Reference distribution for this class's prototypes, built from
                # every image seen so far — never from the current one.
                proto_mu = proto_sigma = proto_valid = None
                if stats_enabled and proto_stats.has_stats(states[c]):
                    proto_mu, proto_sigma, proto_valid = proto_stats.compute_mu_sigma(
                        states[c], stats_min_count, stats_sigma_eps
                    )

                result = _gaussian_score_for_class(
                    patches_norm,
                    centers_norm,
                    states[c]["variance"],
                    states[c]["appearance"],
                    update_samples=int(states[c]["n_images"]),
                    top_m=top_m,
                    patch_group_threshold=patch_group_threshold,
                    variance_min=variance_min,
                    aggregation=aggregation,
                    return_details=want_details,
                    proto_mu=proto_mu,
                    proto_sigma=proto_sigma,
                    proto_valid=proto_valid,
                    sigma_eps=stats_sigma_eps,
                )
                if want_details:
                    raw_proto[c], per_class_details[c] = result
                else:
                    raw_proto[c] = result

        # ── Fold this image into every bank's reference distribution ──────────
        # Strictly after all scoring above, so no bank's statistics can contain
        # the image they were just used to score.
        if stats_enabled:
            self._accumulate_proto_stats(
                states, per_class_details,
                sigma_warn=stats_sigma_warn,
                min_count=stats_min_count,
                sigma_eps=stats_sigma_eps,
                log_every=stats_log_every,
            )

        # ── Quality gate: how discriminative are the raw prototype scores? ───
        proto_var = raw_proto.var()
        quality_gate = proto_var / (proto_var + quality_eps)

        if return_details:
            details = dict(per_class_details)
            if keep_mask is not None:
                details["keep_mask"] = keep_mask
                details["filter_scores"] = filter_scores_per_class
            return raw_proto, quality_gate, details
        return raw_proto, quality_gate

    def _accumulate_proto_stats(self, states, per_class_details, *,
                                sigma_warn, min_count, sigma_eps, log_every):
        """Fold one image's raw prototype scores into every bank's statistics.

        Mutates ``states`` in place. Also refreshes ``self.last_stats_diag`` with
        a health summary — a large ``n_low_var`` means prototypes whose reference
        distribution is near-degenerate, so their z-scores are dominated by the
        sigma floor.
        """
        n_protos = 0
        n_valid = 0
        n_low_var = 0
        sigma_total = 0.0

        for c, details in per_class_details.items():
            best_per_proto = details.get("best_per_proto")
            if best_per_proto is None or best_per_proto.numel() == 0:
                continue
            states[c].update(proto_stats.accumulate(states[c], best_per_proto))

            _, sigma, valid = proto_stats.compute_mu_sigma(
                states[c], min_count, sigma_eps
            )
            n_protos += sigma.numel()
            n_valid += int(valid.sum().item())
            n_low_var += proto_stats.low_variance_count(sigma, valid, sigma_warn)
            sigma_total += float(sigma.sum().item())

        self.last_stats_diag = {
            "n_protos": n_protos,
            "n_valid": n_valid,
            "n_low_var": n_low_var,
            "mean_sigma": sigma_total / max(n_protos, 1),
        }

        self._stats_calls = getattr(self, "_stats_calls", 0) + 1
        if log_every > 0 and self._stats_calls % log_every == 0:
            d = self.last_stats_diag
            print(
                f"[proto_stats] call={self._stats_calls} "
                f"protos={d['n_protos']} valid={d['n_valid']} "
                f"low_var={d['n_low_var']} mean_sigma={d['mean_sigma']:.5f}"
            )

    def update_state(self, state, images, encoder, global_feat,
                     *, filter_scores=None, target_class_idx=None):
        # ── Config ────────────────────────────────────────────────────────────
        match_threshold = float(self._cfg.get("match_threshold", 0.60))
        max_K = int(self._cfg.get("max_K", 100))
        exclude_pos = bool(self._cfg.get("exclude_pos", False))
        gaussian_ema = float(self._cfg.get("gaussian_ema", 0.1))
        variance_min = float(self._cfg.get("variance_min", 0.001))
        variance_max = float(self._cfg.get("variance_max", 1.0))

        aug_copies = int(self._cfg.get("aug_copies", 0))

        # ── Build all views (original + augmented), then batch extract ────────
        all_views = [images]
        for _ in range(aug_copies):
            all_views.append(_augment_image(images))
        batched_images = torch.cat(all_views, dim=0)  # [(1+aug_copies), C, H, W]
        patch_embs = encoder.get_patch_embeddings(batched_images, exclude_pos=exclude_pos)  # [B, P, D]
        patch_embs = patch_embs.reshape(-1, patch_embs.shape[-1])
        patches_norm = _safe_normalize(patch_embs)  # [(1+aug_copies)*P, D]

        default_new_var = patches_norm.var(dim=0).mean().clamp(variance_min, variance_max).item()

        # ── Patch relevance filtering (on full concatenated pool) ──────────────
        filter_mode = self._cfg.get("patch_filter_mode", "none")
        keep_mask = None
        if (filter_mode != "none"
                and hasattr(self, "_text_features")
                and target_class_idx is not None):
            # Surgery modes: compute scores on every view (original + augmented)
            # so the filter reflects foreground evidence from the full patch pool.
            if filter_scores is None and filter_mode in ("surgery_with_labels", "surgery_no_labels"):
                filter_scores = torch.cat([
                    compute_surgery_scores(
                        img, encoder, self._text_features, self._empty_text_feat, filter_mode
                    )[:, target_class_idx]
                    for img in all_views
                ])  # [(1+aug_copies)*P]
            keep_mask = filter_patches_by_text_alignment(
                patches_norm, target_class_idx,
                text_features=self._text_features,
                empty_text_feat=self._empty_text_feat,
                filter_mode=filter_mode,
                filter_threshold=self._filter_threshold,
                aug_copies=aug_copies,
                precomputed_scores=filter_scores,
                encoder=encoder,
            )
            patches_norm = patches_norm[keep_mask]  # [P_filtered, D]

        # Map filtered indices back to original grid positions (0..P-1)
        # When filter_mode="none", keep_mask is None and this is identity.
        P = 14 * 14
        if keep_mask is not None:
            _concat_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
            _filtered_to_grid = (_concat_indices % P).long()
        else:
            _filtered_to_grid = torch.arange(patches_norm.shape[0], device=patches_norm.device).long()

        centers = state["centers"]     # [K, D]
        apps = state["appearance"]     # [K]
        variances = state["variance"]  # [K, D]
        old_K = centers.shape[0]

        grow_cap = max(max_K + int(patches_norm.shape[0]), max_K)

        # ── Incremental K-means ───────────────────────────────────────────────
        if old_K == 0:
            init = _safe_normalize(global_feat, dim=-1).unsqueeze(0)
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                init, patches_norm, match_threshold, grow_cap
            )
        else:
            init = None
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                centers, patches_norm, match_threshold, grow_cap
            )

        total_K = updated_centers.shape[0]
        n_new = total_K - old_K if old_K > 0 else total_K - 1

        # ── Build variance and appearance tensors ─────────────────────────────
        all_vars = []
        all_apps = []

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
            # Seed prototype (index 0 in updated_centers from init)
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

        # Grow the score accumulators to match the new bank size. Freshly created
        # prototypes start with zero observations and abstain from z-scoring
        # until they have accumulated proto_stats_min_count reference scores.
        updated_stats = proto_stats.grow(
            state, updated_apps.shape[0], updated_centers.device
        )

        # ── Track top-3 representative patches per cluster ─────────────────────
        _TOP3 = 5
        current_image_idx = int(state.get("n_images", 0))
        old_top_rep = list(state.get("top_rep_patches", []))

        centers_new_norm = _safe_normalize(updated_centers, dim=-1)
        sims_to_centers = patches_norm @ centers_new_norm.T  # [P, total_K]

        new_top_rep = []

        def _merge_top3(existing, new_candidates):
            merged = list(existing)
            for pidx, iidx, sim in new_candidates:
                merged.append((pidx, iidx, sim))
            merged.sort(key=lambda x: x[2], reverse=True)
            return merged[:_TOP3]

        if old_K > 0:
            for k in range(old_K):
                old_entries = old_top_rep[k] if k < len(old_top_rep) else []
                mask = matched & (best_clusters == k)
                if mask.any():
                    candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    candidate_sims = sims_to_centers[candidate_indices, k]
                    top_vals, top_local_pos = candidate_sims.topk(min(_TOP3, candidate_sims.numel()))
                    new_cands = [
                        (int(_filtered_to_grid[candidate_indices[pos]].item()), current_image_idx, float(sim))
                        for pos, sim in zip(top_local_pos.tolist(), top_vals.tolist())
                    ]
                    new_top_rep.append(_merge_top3(old_entries, new_cands))
                else:
                    new_top_rep.append(old_entries)
        else:
            mask = matched & (best_clusters == 0)
            if mask.any():
                candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                candidate_sims = sims_to_centers[candidate_indices, 0]
                top_vals, top_local_pos = candidate_sims.topk(min(_TOP3, candidate_sims.numel()))
                new_top_rep.append([
                    (int(_filtered_to_grid[candidate_indices[pos]].item()), current_image_idx, float(sim))
                    for pos, sim in zip(top_local_pos.tolist(), top_vals.tolist())
                ])
            else:
                new_top_rep.append([])

        for i, group_idx in enumerate(new_groups):
            new_k = (old_K if old_K > 0 else 1) + i
            if new_k < updated_centers.shape[0] and group_idx.numel() > 0:
                sims_new = sims_to_centers[group_idx, new_k]
                top_vals, top_local_pos = sims_new.topk(min(_TOP3, sims_new.numel()))
                new_top_rep.append([
                    (int(_filtered_to_grid[group_idx[pos]].item()), current_image_idx, float(sim))
                    for pos, sim in zip(top_local_pos.tolist(), top_vals.tolist())
                ])
            else:
                new_top_rep.append([])

        while len(new_top_rep) < updated_centers.shape[0]:
            new_top_rep.append([])

        # ── Prune to max_K by appearance weight ───────────────────────────────
        n_images_next = max(int(state.get("n_images", 0)) + 1, 1)
        if updated_centers.shape[0] > max_K:
            app_w = updated_apps / float(n_images_next)
            keep = torch.argsort(app_w, descending=True)[:max_K]
            keep = torch.sort(keep).values
            updated_centers = updated_centers[keep]
            updated_vars = updated_vars[keep]
            updated_apps = updated_apps[keep]
            # Score statistics must follow the same reindexing, or every
            # surviving prototype inherits a different prototype's history.
            updated_stats = proto_stats.permute(updated_stats, keep)
            keep_list = keep.tolist()
            new_top_rep = [new_top_rep[k] for k in keep_list]

        state = dict(state)
        state["centers"] = updated_centers
        state["variance"] = updated_vars
        state["appearance"] = updated_apps
        state["n_images"] = state["n_images"] + 1
        state["top_rep_patches"] = new_top_rep
        state.update(updated_stats)
        if keep_mask is not None:
            state["keep_mask"] = keep_mask[:P]
        else:
            state["keep_mask"] = torch.ones(P, dtype=torch.bool, device=updated_centers.device)

        return state


def build(cfg: dict) -> GaussianPatchLevel:
    return GaussianPatchLevel(cfg)
