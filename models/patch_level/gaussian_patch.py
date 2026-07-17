"""
Gaussian Patch-Level Prototypes

Represent each prototype as a Gaussian (center + per-dimension variance) and
replace cosine-similarity matching with a Mahalanobis-like distance.

This is the patch-level component that implements BasePatchLevel, intended to
be composed into a full adapter (e.g., via PTA's multi-component setup).
"""
import torch
import torchvision.transforms.functional as TF

from models.patch_level.base import (
    BasePatchLevel,
    _safe_normalize,
    _extract_patch_embeddings,
    _incremental_kmeans_step,
    _gaussian_score_for_class,
)


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
            }
            for _ in range(C)
        ]

    def compute_patch_logits(self, images, clip_model, states):
        # ── Config ────────────────────────────────────────────────────────────
        exclude_pos = bool(self._cfg.get("exclude_pos", False))
        top_m = int(self._cfg.get("soft_nn_top_m", 4))
        quality_eps = float(self._cfg.get("quality_eps", 1e-3))
        patch_group_threshold = float(self._cfg.get("patch_group_threshold", 0.9))
        variance_min = float(self._cfg.get("variance_min", 0.001))
        aggregation = str(self._cfg.get("aggregation", "top_m_mean"))

        num_classes = len(states)
        device = states[0]["centers"].device if num_classes > 0 else images.device

        # ── Extract patch embeddings ─────────────────────────────────────────
        patch_embs = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)
        patches_norm = _safe_normalize(patch_embs)  # [P, D]

        # ── Compute per-class Gaussian score ──────────────────────────────────
        raw_proto = torch.zeros(num_classes, device=device)
        for c in range(num_classes):
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
                    aggregation=aggregation,
                )

        # ── Quality gate: how discriminative are the raw prototype scores? ───
        proto_var = raw_proto.var()
        quality_gate = proto_var / (proto_var + quality_eps)

        return raw_proto, quality_gate

    def update_state(self, state, images, clip_model, global_feat, is_high_confidence):
        if not is_high_confidence:
            return state

        # ── Config ────────────────────────────────────────────────────────────
        match_threshold = float(self._cfg.get("match_threshold", 0.60))
        max_K = int(self._cfg.get("max_K", 100))
        exclude_pos = bool(self._cfg.get("exclude_pos", False))
        gaussian_ema = float(self._cfg.get("gaussian_ema", 0.1))
        variance_min = float(self._cfg.get("variance_min", 0.001))
        variance_max = float(self._cfg.get("variance_max", 1.0))
        default_new_var = variance_min * 10.0

        aug_copies = int(self._cfg.get("aug_copies", 0))

        # ── Extract patches ──────────────────────────────────────────────────
        patch_embs = _extract_patch_embeddings(images, clip_model, exclude_pos=exclude_pos)
        patches_norm = _safe_normalize(patch_embs)  # [P, D]

        # ── Augmented views (Option A: concatenate into single k-means pass) ─
        if aug_copies > 0:
            aug_patch_list = [patches_norm]
            for _ in range(aug_copies):
                aug_img = _augment_image(images)
                aug_embs = _extract_patch_embeddings(aug_img, clip_model, exclude_pos=exclude_pos)
                aug_patch_list.append(_safe_normalize(aug_embs))
            patches_norm = torch.cat(aug_patch_list, dim=0)  # [(1+aug_copies)*P, D]

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

        # ── Prune to max_K by appearance weight ───────────────────────────────
        n_images_next = max(int(state.get("n_images", 0)) + 1, 1)
        if updated_centers.shape[0] > max_K:
            app_w = updated_apps / float(n_images_next)
            keep = torch.argsort(app_w, descending=True)[:max_K]
            keep = torch.sort(keep).values
            updated_centers = updated_centers[keep]
            updated_vars = updated_vars[keep]
            updated_apps = updated_apps[keep]

        state = dict(state)
        state["centers"] = updated_centers
        state["variance"] = updated_vars
        state["appearance"] = updated_apps
        state["n_images"] = state["n_images"] + 1
        return state


def build(cfg: dict) -> GaussianPatchLevel:
    return GaussianPatchLevel(cfg)
