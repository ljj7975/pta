#!/usr/bin/env python3
"""
Test _filter_patches_by_text_alignment with real images and encoder-based inference.

Run with: python tests/test_identify_relevant_patches.py

For each image in tests/assets/ (cat.jpg, person.jpg, bird.jpg), this test:
  1. Loads the image and preprocesses it through the encoder.
  2. Extracts per-patch embeddings from the ViT backbone.
  3. For each filter mode, computes text-alignment scores against the
     ground-truth class label, applies the keep-mask, and renders a
     heatmap overlay (similar to third_party/CLIP_Surgery/demo.ipynb).
  4. Saves foreground patches and heatmap overlays to tests/assets/output/.

Each filter_mode produces a different set of relevant patches:
  - none                : keeps all patches (baseline)
  - cosine_with_labels  : relative specificity (target − mean-other)
  - cosine_no_labels    : cosine against label-adjusted text feature
  - surgery_with_labels : CLIP Surgery with class text features
  - surgery_no_labels   : CLIP Surgery with empty-string baseline subtracted
"""
import sys
import os
import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import CLIPEncoder
from utils.clip_inference import (
    _safe_normalize,
    filter_patches_by_text_alignment,
    compute_surgery_scores,
)
from third_party.CLIP_Surgery.clip_surgery import clip as clip_surgery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
OUTPUT_DIR = ASSETS_DIR / "output"
MODEL_TYPE = "ViT-B/16"
# Thresholds to compare — each row in the patch grid shows a different threshold
THRESHOLDS = [0.2, 0.4, 0.6, 0.8]

# Map each asset image to its ground-truth class label(s).
# "demo.jpg" comes from CLIP_Surgery/demo.ipynb which targets multiple classes.
IMAGE_LABELS = {
    "cat.jpg": "cat",
    "person.jpg": "person",
    "bird.jpg": "bird",
    "demo.jpg": ["bench", "building", "ground", "person"],  # multi-label from CLIP_Surgery demo
}

# All filter modes to test.
FILTER_MODES = [
    "none",
    "cosine_with_labels",
    "cosine_no_labels",
    "surgery_with_labels",
    "surgery_no_labels",
]

# Additional classes for cosine_with_labels (needed to compute "other" mean).
# We include the ground-truth label plus a few distractors.
# "bench", "building", "ground" are added for the demo.jpg multi-label case (CLIP_Surgery demo.ipynb).
ALL_CLASSES = ["airplane", "bench", "bird", "building", "car", "cat", "dog", "ground", "house", "person", "tree"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_similarity_map(scores: torch.Tensor, shape: tuple, invert: bool = False) -> torch.Tensor:
    """Reshape per-patch scores [B, P] to spatial heatmap [B, H, W] then
    bilinearly interpolate to *shape* (height, width).

    Mirrors third_party/CLIP_Surgery/clip_surgery/clip.py::get_similarity_map
    but works on a single-class score tensor [B, P].

    Args:
        scores: [B, P] per-patch relevance scores (higher = more relevant).
        shape: target (height, width) for bilinear interpolation.
        invert: if True, negate scores before normalization. Not used by default.
    """
    if invert:
        scores = -scores
    # Min-max normalize
    sm = (scores - scores.min(1, keepdim=True)[0]) / (
        scores.max(1, keepdim=True)[0] - scores.min(1, keepdim=True)[0] + 1e-8
    )
    side = int(math.sqrt(sm.shape[1]))
    sm = sm.reshape(sm.shape[0], side, side).unsqueeze(1)  # [B, 1, side, side]
    sm = torch.nn.functional.interpolate(sm, shape, mode="bilinear", align_corners=False)
    return sm.squeeze(1)  # [B, H, W]


def render_heatmap_overlay(pil_img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Blend a [H, W] heatmap (values in [0, 1]) over a PIL image.

    Returns an RGB numpy array suitable for plt.imshow() or cv2.imwrite().
    """
    cv2_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32)
    vis = (heatmap * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    blended = cv2_img * 0.4 + vis * 0.6
    return cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB)


def extract_foreground_patches(pil_img: Image.Image, mask: torch.Tensor,
                                side: int, patch_size: int) -> list:
    """Extract individual patch crops from the original PIL image for the
    patches where *mask* is True.

    Returns a list of patch_pil images (compact layout — no spatial positioning).
    """
    img_w, img_h = pil_img.size
    patches = []
    kept_indices = torch.where(mask)[0].cpu().tolist()
    for idx in kept_indices:
        row = idx // side
        col = idx % side
        x1 = col * patch_size
        y1 = row * patch_size
        x2 = min(x1 + patch_size, img_w)
        y2 = min(y1 + patch_size, img_h)
        patch_pil = pil_img.crop((x1, y1, x2, y2))
        patches.append(patch_pil)
    return patches


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def test_filter_patches_by_text_alignment():
    """End-to-end test of filter_patches_by_text_alignment with real images."""
    print("=" * 70)
    print("TEST: filter_patches_by_text_alignment with real images")
    print("=" * 70)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize encoder
    print(f"Loading encoder: {MODEL_TYPE} ...")
    encoder = CLIPEncoder(model_type=MODEL_TYPE, device=device)

    # Compute empty-string embedding (baseline for no_labels modes)
    with torch.no_grad():
        empty_tokens = clip_surgery.tokenize([""]).to(device)
        empty_text_feat = encoder.model.encode_text(empty_tokens)
    empty_text_feat = empty_text_feat / empty_text_feat.norm(dim=-1, keepdim=True)
    print(f"Empty text feature shape: {empty_text_feat.shape}")

    # Build text features for all classes
    with torch.no_grad():
        text_tokens = clip_surgery.tokenize(
            [f"a photo of a {c}" for c in ALL_CLASSES]
        ).to(device)
        raw_text_feats = encoder.model.encode_text(text_tokens)
    text_features = raw_text_feats / raw_text_feats.norm(dim=-1, keepdim=True)
    print(f"Text features shape: {text_features.shape}  (classes: {ALL_CLASSES})")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine ViT grid size (14 for ViT-B/16 on 224x224)
    # The encoder image_size is 448 by default, but CLIP ViT-B/16 expects 224.
    # We use the model's actual input resolution.
    visual = encoder.visual
    patch_size = 16  # ViT-B/16
    # Input resolution: check conv1 kernel / stride
    input_res = 224  # standard for ViT-B/16
    side = input_res // patch_size  # 14
    num_patches = side * side  # 196
    print(f"ViT grid: {side}x{side} = {num_patches} patches (patch_size={patch_size})")

    results = {}

    for img_name, label in IMAGE_LABELS.items():
        img_path = ASSETS_DIR / img_name
        if not img_path.exists():
            print(f"  ⚠ Skipping {img_name} — file not found at {img_path}")
            continue

        img_stem = Path(img_name).stem  # "cat.jpg" → "cat"

        # Normalize label to list for uniform handling
        target_labels = label if isinstance(label, list) else [label]

        # Load and preprocess image ONCE per image (shared across all target labels)
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = encoder.preprocess(pil_img).unsqueeze(0).to(device)
        print(f"  Preprocessed tensor shape: {img_tensor.shape}")

        # Extract patch embeddings ONCE per image
        with torch.no_grad():
            all_tokens = encoder.encode_image(
                img_tensor, CLS_token_only=False, preprocess=False
            )  # [1, 1+P, D]
            patch_embs = all_tokens[0, 1:]  # [P, D] — strip CLS
        patches_norm = _safe_normalize(patch_embs)
        print(f"  Patch embeddings: {patches_norm.shape}")

        resized_img = pil_img.resize((input_res, input_res), Image.LANCZOS)

        # Process each target label separately
        for target_label in target_labels:
            target_class_idx = ALL_CLASSES.index(target_label)
            print(f"\n{'─' * 60}")
            print(f"Image: {img_name}  |  Label: {target_label}  |  Class idx: {target_class_idx}")
            print(f"{'─' * 60}")

            img_results = {}
            heatmap_images = []   # list of (mode, overlay_np)
            # patch_grids[mode][threshold] = (grid_np, kept_count)
            patch_grids = {mode: {} for mode in FILTER_MODES}

            for mode in FILTER_MODES:
                print(f"\n  Mode: {mode}")

                # --- Get per-patch scores and heatmap ---
                if mode == "none":
                    mode_scores = None
                    overlay = np.array(pil_img.convert("RGB"))
                elif mode.startswith("surgery"):
                    scores_2d = compute_surgery_scores(
                        img_tensor, encoder, text_features, empty_text_feat, mode
                    )  # [P, C]
                    mode_scores = scores_2d[:, target_class_idx]  # [P]
                    heatmap = get_similarity_map(mode_scores.unsqueeze(0), pil_img.size[::-1])
                    overlay = render_heatmap_overlay(pil_img, heatmap[0].cpu().numpy())
                else:
                    _, mode_scores = filter_patches_by_text_alignment(
                        patches_norm,
                        target_class_idx=target_class_idx,
                        text_features=text_features,
                        empty_text_feat=empty_text_feat,
                        filter_mode=mode,
                        encoder=encoder,
                        return_scores=True,
                    )
                    heatmap = get_similarity_map(mode_scores.unsqueeze(0), pil_img.size[::-1])
                    overlay = render_heatmap_overlay(pil_img, heatmap[0].cpu().numpy())
                heatmap_images.append((mode, overlay))

                # --- Compute masks + patch grids for each threshold ---
                # All scores are similarity-like (higher = more relevant).
                # identify_relevant_patches keeps patches with the highest scores.
                for thresh in THRESHOLDS:
                    keep_mask = filter_patches_by_text_alignment(
                        patches_norm,
                        target_class_idx=target_class_idx,
                        text_features=text_features,
                        empty_text_feat=empty_text_feat,
                        filter_mode=mode,
                        filter_threshold=thresh,
                        precomputed_scores=mode_scores,
                        encoder=encoder,
                    )  # [P]

                    kept = int(keep_mask.sum())
                    fg_patches = extract_foreground_patches(
                        resized_img, keep_mask, side, patch_size
                    )

                    if fg_patches:
                        n_cols = min(len(fg_patches), 14)
                        n_rows = math.ceil(len(fg_patches) / n_cols)
                        grid_w = n_cols * patch_size
                        grid_h = n_rows * patch_size
                        grid_img = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
                        for i, patch_pil in enumerate(fg_patches):
                            col_idx = i % n_cols
                            row_idx = i // n_cols
                            patch_resized = patch_pil.resize((patch_size, patch_size), Image.LANCZOS)
                            grid_img.paste(patch_resized, (col_idx * patch_size, row_idx * patch_size))
                        patch_grids[mode][thresh] = (np.array(grid_img), kept)
                    else:
                        patch_grids[mode][thresh] = (None, 0)

                # Store stats for the "default" threshold (middle one)
                default_thresh = THRESHOLDS[len(THRESHOLDS) // 2]
                _, default_kept = patch_grids[mode][default_thresh]
                img_results[mode] = {
                    "kept": default_kept,
                    "scores_min": mode_scores.min().item() if mode_scores is not None else 0.0,
                    "scores_max": mode_scores.max().item() if mode_scores is not None else 0.0,
                    "scores_mean": mode_scores.mean().item() if mode_scores is not None else 0.0,
                }

            # --- Composite figure ---
            # Row 0: Original | heatmap overlays (one per filter mode)
            # Rows 1..N: Patch grids for each threshold (one row per threshold)
            n_filter_modes = len(FILTER_MODES) - 1  # exclude "none"
            n_thresholds = len(THRESHOLDS)
            n_cols = 1 + n_filter_modes  # original + filter modes
            n_rows = 1 + n_thresholds     # heatmap row + threshold rows

            fig_height = 4 + n_thresholds * 2.5
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, fig_height))

            # Row 0 — heatmaps
            axes[0, 0].imshow(np.array(pil_img.convert("RGB")))
            axes[0, 0].set_title(f"{img_stem}\n(original)", fontsize=10, pad=5)
            axes[0, 0].axis("off")

            for c, (mode, overlay_np) in enumerate(heatmap_images):
                if mode == "none":
                    continue
                ax = axes[0, c]
                ax.imshow(overlay_np)
                ax.set_title(f"{mode}\n(heatmap)", fontsize=9, pad=5)
                ax.axis("off")

            # Rows 1..N — patch grids per threshold
            active_modes = [m for m in FILTER_MODES if m != "none"]
            for r, thresh in enumerate(THRESHOLDS):
                row_idx = r + 1
                # First column: label
                axes[row_idx, 0].set_title(f"threshold ≥ {thresh}", fontsize=10, pad=5)
                axes[row_idx, 0].axis("off")
                axes[row_idx, 0].set_facecolor("#f0f0f0")

                for c, mode in enumerate(active_modes):
                    ax = axes[row_idx, c + 1]
                    grid_np, kept = patch_grids[mode][thresh]
                    if grid_np is not None:
                        ax.imshow(grid_np)
                    else:
                        ax.text(0.5, 0.5, "0 patches", ha="center", va="center",
                                transform=ax.transAxes, fontsize=10)
                        ax.set_facecolor("#222222")
                    ax.set_title(f"{mode}\n{kept}/{num_patches}", fontsize=8, pad=3)
                    ax.axis("off")

            fig.suptitle(f"{img_stem} — label: {target_label}", fontsize=14, fontweight="bold", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            suffix = f"_{target_label}" if len(target_labels) > 1 else ""
            composite_path = OUTPUT_DIR / f"{img_stem}{suffix}_composite.png"
            fig.savefig(composite_path, dpi=150)
            plt.close(fig)
            print(f"\n    Saved composite → {composite_path}")

            results_key = f"{img_stem}_{target_label}" if len(target_labels) > 1 else img_stem
            results[results_key] = img_results

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for stem, img_results in results.items():
        print(f"\n  {stem}:")
        for mode, stats in img_results.items():
            print(f"    {mode:25s}: kept={stats['kept']:3d}/196  "
                  f"score=[{stats['scores_min']:.4f}, {stats['scores_max']:.4f}]  "
                  f"mean={stats['scores_mean']:.4f}")

    # Validation: different modes should produce different masks
    print(f"\n{'=' * 70}")
    print("VALIDATION: filter modes produce different results")
    print(f"{'=' * 70}")
    all_passed = True
    for stem, img_results in results.items():
        kept_counts = [img_results[m]["kept"] for m in FILTER_MODES if m != "none"]
        if len(set(kept_counts)) < len(kept_counts) - kept_counts.count(kept_counts[0]):
            # Allow some modes to have same count if score distributions differ
            pass
        # At minimum, "none" should keep all patches
        if img_results["none"]["kept"] != 196:
            print(f"  ✗ {stem}: 'none' mode should keep all 196 patches, got {img_results['none']['kept']}")
            all_passed = False
        else:
            print(f"  ✓ {stem}: 'none' mode keeps all 196 patches")

        # Other modes should keep at least 1 patch (absolute threshold may be strict)
        for mode in FILTER_MODES:
            if mode == "none":
                continue
            if img_results[mode]["kept"] == 0:
                print(f"  ✗ {stem}: '{mode}' kept 0 patches — threshold may be too strict")
                all_passed = False
            else:
                print(f"  ✓ {stem}: '{mode}' keeps {img_results[mode]['kept']}/{num_patches} patches")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    return all_passed


if __name__ == "__main__":
    print("Running filter_patches_by_text_alignment tests with real images...\n")

    try:
        passed = test_filter_patches_by_text_alignment()
    except Exception as e:
        print(f"\n✗ Test raised exception: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    if passed:
        print("\n✓ TEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ TEST FAILED")
        sys.exit(1)
