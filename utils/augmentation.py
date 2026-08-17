"""Shared augmentation helpers for patch-level prototype updates and scoring.

Moved from models/patch_level/gaussian_patch.py so both the patch-level module
and any future consumers can reuse the same augmentation logic.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# Fixed augmentation ranges — not exposed as config
_AUG_ROTATION_DEG       = 15.0   # ±degrees
_AUG_TRANSLATE_FRAC     = 0.10   # ±fraction of image size
_AUG_SCALE_MIN          = 0.85
_AUG_SCALE_MAX          = 1.15
_AUG_BRIGHTNESS_MAG     = 0.20   # additive offset ±this value
_AUG_CONTRAST_MIN       = 0.80
_AUG_CONTRAST_MAX       = 1.20
_AUG_BLUR_KERNEL_MIN    = 3
_AUG_BLUR_KERNEL_MAX    = 11
_AUG_BLUR_SIGMA_MIN     = 0.1
_AUG_BLUR_SIGMA_MAX     = 2.0
_AUG_BLUR_P             = 0.30

# Weighted extra-op pool (sample exactly one op per augmented copy).
_AUG_EXTRA_GRAYSCALE_P  = 0.30
_AUG_EXTRA_EDGE_BLEND_P = 0.30
_AUG_EXTRA_NONE_P       = 0.40   # not explicitly used

# Extra-op magnitudes.
_AUG_EDGE_BLEND_MIN = 0.25
_AUG_EDGE_BLEND_MAX = 0.55


def _sample_uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1).item()


def _to_grayscale_like(img: torch.Tensor) -> torch.Tensor:
    """Convert to grayscale while preserving [C, H, W] shape."""
    c = img.shape[0]
    if c == 3:
        gray = TF.rgb_to_grayscale(img, num_output_channels=1)
        return gray.repeat(3, 1, 1)
    gray = img.mean(dim=0, keepdim=True)
    return gray.repeat(c, 1, 1)


def _edge_map_like(img: torch.Tensor) -> torch.Tensor:
    """Sobel edge magnitude map with per-channel shape preservation."""
    x = img.unsqueeze(0)  # [1, C, H, W]
    c = x.shape[1]
    kx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=x.device, dtype=x.dtype)
    kx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    mag = mag / mag.mean(dim=[2, 3], keepdim=True).clamp(min=1e-8)
    return mag.squeeze(0)


def _apply_weighted_extra_op(img: torch.Tensor) -> torch.Tensor:
    """Sample and apply one extra geometry-focused op from a weighted pool."""
    r = torch.rand(1).item()
    p_gray = _AUG_EXTRA_GRAYSCALE_P
    p_edge = p_gray + _AUG_EXTRA_EDGE_BLEND_P

    if r < p_gray:
        return _to_grayscale_like(img)

    if r < p_edge:
        lam = _sample_uniform(_AUG_EDGE_BLEND_MIN, _AUG_EDGE_BLEND_MAX)
        edges = _edge_map_like(img)
        return (1.0 - lam) * img + lam * edges

    # "none" path keeps the base chain unchanged.
    return img


def _augment_image(image: torch.Tensor) -> torch.Tensor:
    """
    Apply a randomly-parameterized composite augmentation to a CLIP-preprocessed image tensor.

    Base transforms are always applied each call:
      - Rotation: uniform ±_AUG_ROTATION_DEG degrees
      - Affine (translation + scale): uniform translation ±_AUG_TRANSLATE_FRAC
        of each spatial dimension, scale uniform in [_AUG_SCALE_MIN, _AUG_SCALE_MAX]
      - Brightness: additive offset uniform in [-_AUG_BRIGHTNESS_MAG, +_AUG_BRIGHTNESS_MAG]
      - Contrast: linear rescaling around per-channel mean, factor in
        [_AUG_CONTRAST_MIN, _AUG_CONTRAST_MAX]
      - Gaussian blur (with probability _AUG_BLUR_P): kernel size uniform in
        [_AUG_BLUR_KERNEL_MIN, _AUG_BLUR_KERNEL_MAX], sigma uniform in
        [_AUG_BLUR_SIGMA_MIN, _AUG_BLUR_SIGMA_MAX]

    Then exactly one extra operation is sampled from a weighted pool:
      - grayscale, edge-blend, or none.

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

    # ── Gaussian blur (stochastic) ──────────────────────────────────────
    if torch.rand(1).item() < _AUG_BLUR_P:
        kernel_size = _AUG_BLUR_KERNEL_MIN + int(
            torch.rand(1).item() * (_AUG_BLUR_KERNEL_MAX - _AUG_BLUR_KERNEL_MIN)
        )
        # Ensure kernel size is odd (required by TF.gaussian_blur)
        kernel_size += kernel_size % 2 == 0
        sigma = _AUG_BLUR_SIGMA_MIN + torch.rand(1).item() * (
            _AUG_BLUR_SIGMA_MAX - _AUG_BLUR_SIGMA_MIN
        )
        img = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    # ── Weighted extra op: grayscale/edge/none ────────────────────────────
    img = _apply_weighted_extra_op(img)

    return img.unsqueeze(0).to(dtype=image.dtype)       # [1, C, H, W]
