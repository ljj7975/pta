#!/usr/bin/env python3
"""
debug_gaussian_patch.py — Unified step-by-step visualisation for GaussianPatchLevel.

Links update_state and compute_patch_logits into one workflow.  For each step i:

  ① update_state(image_i)
        → cluster assignment heatmap on image_i
        → patch crops grouped by cluster (showing *which* patches formed each prototype)

  ② compute_patch_logits(image_{i+1})  [uses state built from images 0..i]
        → cluster assignment heatmap on image_{i+1}  (how new patches map to existing prototypes)
        → patch_proto_logits bar chart
        → quality_gate scalar

Each step is saved as one combined figure so you can page through them and see
how clusters accumulate over time and how they score new images.

Typical usage
-------------
  # Augment a single image (smoke-test, no dataset needed):
  python tests/debug_gaussian_patch.py \\
      --image tests/assets/cat.jpg --class-name cat \\
      --n-images 6 --filter-modes none cosine_with_labels

  # Directory of same-class images:
  python tests/debug_gaussian_patch.py \\
      --images-dir /data/caltech101/cat/ --class-name cat \\
      --n-images 10 --filter-modes none cosine_with_labels cosine_no_labels

  # Compare filter modes on a summary grid (adds one extra figure):
  ... --compare-modes
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image as PilImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import CLIPEncoder
from models.patch_level.gaussian_patch import GaussianPatchLevel, _augment_image
from utils.clip_inference import _safe_normalize

# ── ViT-B/16 patch-grid constants ────────────────────────────────────────────
GRID  = 14        # 14 × 14 = 196 patches
PSIZE = 16        # 224 / 14 = 16 pixels per patch (exact)
IMPX  = 224       # CLIP input resolution

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# Pre-build color table once so colors are consistent across steps / modes.
_CMAP = [plt.cm.tab20(i) for i in range(20)]


# ── Low-level image helpers ──────────────────────────────────────────────────

def _denorm(t: torch.Tensor) -> np.ndarray:
    """CLIP-denormalize [C, H, W] → [H, W, 3] uint8."""
    t = t.cpu().float() * CLIP_STD.view(3, 1, 1) + CLIP_MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _rgb(k: int) -> np.ndarray:
    """Cluster-k color as [3] float array in [0, 1]."""
    return np.array(_CMAP[k % 20][:3])


def _assignment_overlay(image_np: np.ndarray, assignments, K: int,
                         alpha: float = 0.55) -> np.ndarray:
    """
    Blend a 14×14 cluster-color grid over a 224×224 image.

    assignments : [196] int array (patch_idx → cluster_idx), or None.
    K           : number of clusters (for cmap range; not used if assignments is None).
    alpha       : blending weight for the color grid (0 = original image only).

    Returns [224, 224, 3] uint8.
    """
    if assignments is None or K == 0:
        return image_np.copy()

    grid = np.zeros((GRID, GRID, 3), dtype=np.float32)
    covered = np.zeros((GRID, GRID), dtype=bool)
    for pidx, k in enumerate(assignments):
        r, c = int(pidx) // GRID, int(pidx) % GRID
        grid[r, c] = _rgb(int(k))
        covered[r, c] = True

    # Gray-out uncovered patches (filtered out or absent)
    for r in range(GRID):
        for c in range(GRID):
            if not covered[r, c]:
                grid[r, c] = (0.65, 0.65, 0.65)

    grid_pil = PilImage.fromarray((grid * 255).astype(np.uint8))
    grid_pil = grid_pil.resize((IMPX, IMPX), PilImage.NEAREST)
    grid_np  = np.array(grid_pil).astype(np.float32)

    out = image_np.astype(np.float32) * (1 - alpha) + grid_np * alpha
    return out.clip(0, 255).astype(np.uint8)


def _crops_strip(image_np: np.ndarray, assignments, K: int,
                 sims: np.ndarray = None,
                 max_per_cluster: int = 12, cell: int = 32,
                 state_after_i: dict = None, tensors: list = None,
                 keep_mask: np.ndarray = None) -> np.ndarray:
    """
    Build a [K × (cell+6), W, 3] uint8 canvas.

    Each row = one cluster:
      [4px color stripe] [top-3 rep patches with border] [gap] [patch crops from current image]

    keep_mask: [196] bool array — when provided, only show patches where True.
    """
    if assignments is None or K == 0:
        return np.full((cell, cell, 3), 200, dtype=np.uint8)

    row_h = cell + 6
    col_w = cell + 2
    n_rep = 5
    rep_block_w = n_rep * (cell + 2) + 4
    canvas = np.full((K * row_h, rep_block_w + 4 + max_per_cluster * col_w, 3), 240, dtype=np.uint8)

    top_rep = state_after_i.get("top_rep_patches", []) if state_after_i else []

    for k in range(K):
        ry = k * row_h
        canvas[ry:ry + row_h, :4] = (_rgb(k) * 255).astype(np.uint8)

        entries = top_rep[k] if k < len(top_rep) else []
        border_color = (_rgb(k) * 255).astype(np.uint8)

        for j, (rpidx, rimgidx, _) in enumerate(entries[:n_rep]):
            if rpidx >= 0 and rimgidx >= 0 and tensors is not None and rimgidx < len(tensors):
                rep_img = _denorm(tensors[rimgidx].squeeze(0))
                r, c = int(rpidx) // GRID, int(rpidx) % GRID
                raw = rep_img[r * PSIZE:(r + 1) * PSIZE, c * PSIZE:(c + 1) * PSIZE]
                thumb = np.array(PilImage.fromarray(raw).resize((cell, cell), PilImage.NEAREST))
                thumb[:2, :] = border_color
                thumb[-2:, :] = border_color
                thumb[:, :2] = border_color
                thumb[:, -2:] = border_color
                y = ry + 3
                x = 4 + j * (cell + 2) + 1
                canvas[y:y + cell, x:x + cell] = thumb

        patch_indices = np.where(assignments == k)[0]
        if keep_mask is not None:
            patch_indices = patch_indices[keep_mask[patch_indices]]
        if sims is not None and k < sims.shape[1]:
            patch_sims = sims[patch_indices, k]
            sorted_order = np.argsort(-patch_sims)
            patch_indices = patch_indices[sorted_order]
        patch_indices = patch_indices[:max_per_cluster]
        for j, pidx in enumerate(patch_indices):
            r, c = int(pidx) // GRID, int(pidx) % GRID
            raw = image_np[r * PSIZE:(r + 1) * PSIZE, c * PSIZE:(c + 1) * PSIZE]
            crop = np.array(PilImage.fromarray(raw).resize((cell, cell), PilImage.NEAREST))
            y = ry + 3
            x = rep_block_w + 4 + j * col_w + 1
            canvas[y:y + cell, x:x + cell] = crop

    return canvas


# ── Patch assignment (for visualization, computed after update_state) ────────

def _assign_all_patches(tensor: torch.Tensor, state: dict, encoder) -> tuple:
    """
    Assign every patch from *tensor* to its nearest cluster center.

    Returns (assignments, sims) where assignments is [196] int array
    and sims is [196, K] similarity matrix, or (None, None) if no clusters.
    """
    if state["centers"].shape[0] == 0:
        return None, None
    with torch.no_grad():
        patches = encoder.get_patch_embeddings(tensor)           # [196, D]
        p_norm  = _safe_normalize(patches)                       # [196, D]
        c_norm  = _safe_normalize(state["centers"])              # [K, D]
        sims    = p_norm @ c_norm.T                              # [196, K]
    return sims.argmax(dim=1).cpu().numpy(), sims.cpu().numpy()  # [196], [196, K]


# ── Per-step figure ───────────────────────────────────────────────────────────

def _filter_heatmap_overlay(image_np: np.ndarray, keep_mask: torch.Tensor,
                             scores: torch.Tensor = None,
                             alpha: float = 0.6) -> np.ndarray:
    """
    Blend a 14×14 filter mask over a 224×224 image.

    keep_mask : [196] bool tensor — True = patch kept by filter.
    scores    : [196] float tensor — raw relevance scores (for color intensity).
    alpha     : blending weight for the color overlay.

    Kept patches get a green overlay; filtered-out patches get a red overlay.
    When scores are provided, the green intensity scales with score magnitude.
    """
    grid = np.zeros((GRID, GRID, 3), dtype=np.float32)
    mask_np = keep_mask.cpu().numpy()

    for pidx in range(GRID * GRID):
        r, c = pidx // GRID, pidx % GRID
        if mask_np[pidx]:
            # Kept patch — green overlay
            if scores is not None:
                # Scale intensity by score (normalized to [0.3, 1.0])
                intensity = 0.3 + 0.7 * float(scores[pidx])
                grid[r, c] = (0.0, min(intensity, 1.0), 0.0)
            else:
                grid[r, c] = (0.0, 0.6, 0.0)
        else:
            # Filtered out — red overlay
            grid[r, c] = (0.7, 0.0, 0.0)

    grid_pil = PilImage.fromarray((grid * 255).astype(np.uint8))
    grid_pil = grid_pil.resize((IMPX, IMPX), PilImage.NEAREST)
    grid_np  = np.array(grid_pil).astype(np.float32)

    out = image_np.astype(np.float32) * (1 - alpha) + grid_np * alpha
    return out.clip(0, 255).astype(np.uint8)


def _save_step_figure(
    step: int,
    filter_mode: str,
    tensor_i: torch.Tensor,
    tensor_next: torch.Tensor,
    state_after_i: dict,
    raw_proto: torch.Tensor,
    quality_gate: torch.Tensor,
    class_names: list,
    encoder,
    out_dir: Path,
    proto_details: dict = None,
    tensors: list = None,
):
    """
    Produce one combined figure for step i.

    Layout (2 rows × 3 cols):
      [0,0] image_i with cluster overlay      (what update_state absorbed)
      [0,1] image_{i+1} with cluster overlay  (how next image maps to current state)
      [0,2] image_{i+1} with filter heatmap   (which patches the filter keeps for scoring)
      [1, :] patch crops from image_i, one row per cluster
    """
    K = state_after_i["centers"].shape[0]

    img_i    = _denorm(tensor_i.squeeze(0))
    img_next = _denorm(tensor_next.squeeze(0))

    assign_i, sims_i    = _assign_all_patches(tensor_i,    state_after_i, encoder)
    assign_next, _      = _assign_all_patches(tensor_next, state_after_i, encoder)

    overlay_i    = _assignment_overlay(img_i,    assign_i,    K)
    overlay_next = _assignment_overlay(img_next, assign_next, K)

    km = state_after_i.get("keep_mask", None)
    km_np = km.cpu().numpy() if km is not None else None
    strip        = _crops_strip(img_i, assign_i, K, sims=sims_i,
                                state_after_i=state_after_i, tensors=tensors,
                                keep_mask=km_np)

    # Filter heatmap for test image (image_{i+1})
    filter_overlay_next = img_next.copy()
    if proto_details is not None and "keep_mask" in proto_details:
        keep_mask = proto_details["keep_mask"]
        scores = proto_details.get("filter_scores", None)
        filter_overlay_next = _filter_heatmap_overlay(img_next, keep_mask, scores=scores)

    fig = plt.figure(figsize=(22, 10))
    gs  = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.1, 1.4],
        width_ratios=[1, 1, 1, 1.3],
        hspace=0.50, wspace=0.30,
    )

    # ── [0,0] image_i with cluster overlay ───────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(overlay_i)
    ax00.set_title(
        f"image_{step}  →  update_state\n"
        f"K = {K} clusters after this step",
        fontsize=9,
    )
    ax00.axis("off")
    if K > 0:
        handles = [
            mpatches.Patch(color=_CMAP[k % 20][:3], label=f"C{k}")
            for k in range(min(K, 10))
        ]
        ax00.legend(handles=handles, loc="lower right", fontsize=6,
                    ncol=min(K, 5), framealpha=0.75)

    # ── [0,1] image_{i+1} with cluster overlay ───────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(overlay_next)
    ax01.set_title(
        f"image_{step + 1}  →  compute_patch_logits\n"
        f"(scored against state from step {step})",
        fontsize=9,
    )
    ax01.axis("off")

    # ── [0,2] image_{i+1} with filter heatmap ────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(filter_overlay_next)
    kept_count = None
    threshold_str = ""
    if proto_details is not None and "keep_mask" in proto_details:
        keep_mask = proto_details["keep_mask"]
        scores = proto_details.get("filter_scores", None)
        kept_count = int(keep_mask.sum().item())
        # Compute the actual threshold from normalized scores
        if scores is not None:
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores_norm = (scores - s_min) / (s_max - s_min)
                threshold_val = float(scores_norm[keep_mask].min())
                threshold_str = f"\nthreshold={threshold_val:.3f}"
            else:
                threshold_str = "\nthreshold=N/A (flat scores)"
    ax02.set_title(
        f"image_{step + 1}  →  filter heatmap\n"
        f"(green=kept, red=filtered)\n"
        f"kept={kept_count}/196{threshold_str}" if kept_count is not None else f"[filter_mode={filter_mode!r}]",
        fontsize=9,
    )
    ax02.axis("off")
    # Legend for filter colors
    green_patch = mpatches.Patch(color=(0.0, 0.6, 0.0), label="kept")
    red_patch = mpatches.Patch(color=(0.7, 0.0, 0.0), label="filtered")
    ax02.legend(handles=[green_patch, red_patch], loc="lower right",
                fontsize=7, ncol=2, framealpha=0.75)

    # ── [0,3] logit bar chart + quality_gate ─────────────────────────────────
    ax03 = fig.add_subplot(gs[0, 3])
    vals = raw_proto.cpu().float().numpy()
    bar_colors = ["tab:blue" if i == 0 else "tab:gray" for i in range(len(class_names))]
    bars = ax03.bar(range(len(class_names)), vals, color=bar_colors)
    ax03.set_xticks(range(len(class_names)))
    ax03.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax03.set_ylabel("patch_proto_logit", fontsize=8)
    ax03.axhline(0, color="black", linewidth=0.5, linestyle="--")
    qg = float(quality_gate)
    ax03.set_title(
        f"patch_proto_logits  (step {step})\n"
        f"quality_gate = {qg:.4f}",
        fontsize=9,
    )
    # Annotate each bar with its value
    for bar, v in zip(bars, vals):
        if abs(v) > 1e-6:
            ax03.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(abs(vals)) * 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    # ── [1, :] crops strip ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    ax1.imshow(strip)
    ax1.set_title(
        f"Patch crops from image_{step} grouped by cluster  "
        f"[filter_mode={filter_mode!r}]",
        fontsize=9,
    )
    ax1.axis("off")
    # Cluster labels on left (with appearance count / n_images and weight)
    # Per-prototype scores on right (from compute_patch_logits details)
    if K > 0 and assign_i is not None:
        row_h = 32 + 6
        n_images = int(state_after_i["n_images"])
        appearances = state_after_i["appearance"].cpu().numpy()
        km = km_np  # keep_mask from outer scope (already numpy)
        for k in range(K):
            y_frac = ((k * row_h + row_h / 2) / strip.shape[0])
            if km is not None:
                n_patches = int(((assign_i == k) & km).sum())
            else:
                n_patches = int((assign_i == k).sum())
            app_raw = int(round(appearances[k]))
            app_w = appearances[k] / max(n_images, 1)
            ax1.text(
                -0.005, 1 - y_frac,
                f"C{k} (n={n_patches}, app={app_raw}/{n_images}, w={app_w:.2f})",
                transform=ax1.transAxes,
                fontsize=7, va="center", ha="right",
                color=_CMAP[k % 20][:3],
            )
            # Per-prototype scores on the right side
            if proto_details is not None and 0 in proto_details:
                details = proto_details[0]
                best_per_proto = details["best_per_proto"].cpu().numpy()  # [K]
                weighted = details["weighted"].cpu().numpy()              # [K]
                if "z" in details:
                    mu_np = details["mu"].cpu().numpy()
                    sigma_np = details["sigma"].cpu().numpy()
                    z_np = details["z"].cpu().numpy()
                    score_str = (
                        f"s={best_per_proto[k]:.3f} μ={mu_np[k]:.3f} "
                        f"σ={sigma_np[k]:.3f} z={z_np[k]:+.2f} → w={weighted[k]:.4f}"
                    )
                else:
                    score_str = f"s={best_per_proto[k]:.3f} → w={weighted[k]:.4f}"
                ax1.text(
                    1.005, 1 - y_frac,
                    score_str,
                    transform=ax1.transAxes,
                    fontsize=6, va="center", ha="left",
                    color="dimgray",
                )

    fig.suptitle(
        f"Step {step}  |  filter_mode={filter_mode!r}  |  "
        f"n_images_in_state={int(state_after_i['n_images'])}",
        fontsize=11, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:02d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Filter-mode comparison figure ────────────────────────────────────────────

def _save_compare_figure(
    step: int,
    filter_modes: list,
    tensors_by_mode: dict,   # mode → (tensor_i, tensor_next)
    overlays_by_mode: dict,  # mode → (overlay_i, overlay_next)
    logits_by_mode: dict,    # mode → (raw_proto, quality_gate, K)
    class_names: list,
    out_dir: Path,
):
    """
    One figure with one row per filter mode, showing overlay_i side-by-side.
    Useful for seeing at a glance how different filters produce different clusters.
    """
    n_modes = len(filter_modes)
    fig, axes = plt.subplots(
        n_modes, 3,
        figsize=(15, 4.5 * n_modes),
        squeeze=False,
    )

    for row, mode in enumerate(filter_modes):
        overlay_i, overlay_next = overlays_by_mode[mode]
        raw_proto, quality_gate, K = logits_by_mode[mode]

        axes[row, 0].imshow(overlay_i)
        axes[row, 0].set_title(f"[{mode}] image_{step} (K={K})", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_next)
        axes[row, 1].set_title(f"[{mode}] image_{step+1}", fontsize=8)
        axes[row, 1].axis("off")

        vals = raw_proto.cpu().float().numpy()
        bars = axes[row, 2].bar(range(len(class_names)), vals,
                                color=["tab:blue" if i == 0 else "tab:gray"
                                       for i in range(len(class_names))])
        axes[row, 2].set_xticks(range(len(class_names)))
        axes[row, 2].set_xticklabels(class_names, rotation=30, ha="right", fontsize=7)
        axes[row, 2].set_title(
            f"[{mode}] logits  qg={float(quality_gate):.3f}", fontsize=8)
        axes[row, 2].axhline(0, color="black", linewidth=0.5, linestyle="--")
        for bar, v in zip(bars, vals):
            if abs(v) > 1e-6:
                axes[row, 2].text(
                    bar.get_x() + bar.get_width() / 2,
                    v + max(abs(vals)) * 0.02 if max(abs(vals)) > 0 else v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6,
                )

    fig.suptitle(f"Filter-mode comparison — step {step}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = out_dir / f"compare_step_{step:02d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Config builder ────────────────────────────────────────────────────────────

def _build_cfg(args, filter_mode: str) -> dict:
    return {
        "patch_level": {
            "match_threshold":              args.match_threshold,
            "max_K":                        args.max_k,
            "gaussian_ema":                 0.1,
            "variance_min":                 0.001,
            "variance_max":                 1.0,
            "aug_copies":                   0,   # handle augmentation in outer loop
            "exclude_pos":                  False,
            "soft_nn_top_m":                args.top_m,
            "quality_eps":                  1e-3,
            "patch_group_threshold":        0.9,
            "aggregation":                  args.aggregation,
            "patch_filter_mode":            filter_mode,
            "patch_filter_threshold":       args.filter_threshold,
            # Reference-distribution statistics. min_count defaults far below the
            # production value because this script only walks a handful of
            # images — at the production default every prototype would still be
            # in warm-up and every z-score would be 0.
            "proto_stats_min_count":        args.proto_stats_min_count,
            "proto_stats_sigma_eps":        1e-6,
            "proto_stats_sigma_warn":       1e-4,
            "proto_stats_log_every":        0,
        }
    }


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_tensors(args, encoder, device) -> list:
    """Return list of [1, 3, 224, 224] preprocessed tensors."""
    if args.images_dir:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        paths = []
        for ext in exts:
            paths.extend(sorted(Path(args.images_dir).glob(ext)))
        paths = paths[:args.n_images]
        if not paths:
            raise ValueError(f"No images found in {args.images_dir}")
        tensors = [
            encoder.preprocess(PilImage.open(p).convert("RGB")).unsqueeze(0).to(device)
            for p in paths
        ]
    else:
        base = encoder.preprocess(
            PilImage.open(args.image).convert("RGB")
        ).unsqueeze(0).to(device)
        torch.manual_seed(args.seed)
        tensors = [base] + [_augment_image(base) for _ in range(args.n_images - 1)]

    print(f"Loaded {len(tensors)} image(s).")
    return tensors


# ── Main run loop for one filter mode ────────────────────────────────────────

def _print_proto_table(proto_details: dict, raw_proto, class_names: list):
    """Print the per-prototype breakdown behind each class-level score.

    For every class with a non-empty bank, one row per prototype showing the raw
    score, its reference statistics, the resulting z-score, and both the raw and
    normalised appearance weights — then the class score and the prediction.
    Lets raw-score aggregation and z-score aggregation be compared side by side.
    """
    # compute_patch_logits also stuffs "keep_mask" / "filter_scores" into the
    # details dict when a patch filter is active, so select the integer class
    # keys before sorting.
    class_details = {c: d for c, d in proto_details.items()
                     if isinstance(c, int) and isinstance(d, dict)}
    has_z = any("z" in d for d in class_details.values())

    for c, det in sorted(class_details.items()):
        bpp = det["best_per_proto"].cpu().numpy()
        if bpp.size == 0:
            continue
        apw = det["app_w"].cpu().numpy()
        wgt = det["weighted"].cpu().numpy()
        mu = det["mu"].cpu().numpy() if "mu" in det else None
        sigma = det["sigma"].cpu().numpy() if "sigma" in det else None
        z = det["z"].cpu().numpy() if "z" in det else None
        w_norm = det["w_norm"].cpu().numpy() if "w_norm" in det else None

        name = class_names[c] if c < len(class_names) else f"class_{c}"
        print(f"    class {c} ({name}):")
        if has_z and z is not None:
            print("      proto |  raw_score |      mu |   sigma |  z_score |  app_w | norm_w")
            for k in range(bpp.shape[0]):
                print(
                    f"      {k:5d} | {bpp[k]:10.6f} | {mu[k]:7.4f} | {sigma[k]:7.4f} | "
                    f"{z[k]:8.3f} | {apw[k]:6.3f} | {w_norm[k]:6.3f}"
                )
        else:
            print("      proto |  raw_score |  app_w | weighted")
            for k in range(bpp.shape[0]):
                print(
                    f"      {k:5d} | {bpp[k]:10.6f} | {apw[k]:6.3f} | {wgt[k]:10.6f}"
                )
        print(f"      final_class_score = {float(raw_proto[c]):.6f}")

    pred = int(raw_proto.argmax())
    pred_name = class_names[pred] if pred < len(class_names) else f"class_{pred}"
    print(f"    predicted class = {pred} ({pred_name})")


def _run_mode(
    filter_mode: str,
    tensors: list,
    class_names: list,
    encoder,
    args,
    out_dir: Path,
):
    """
    Step-by-step loop for a single filter mode.

    Returns a dict of per-step data for the comparison figure.
    """
    device = tensors[0].device
    N = len(tensors)

    cfg = _build_cfg(args, filter_mode)
    patch_level = GaussianPatchLevel(cfg)

    # Text features: encode_text returns [C, D] (L2-normalized)
    text_feats = encoder.encode_text(class_names).float()  # [C, D]

    if filter_mode != "none":
        # set_text_context expects clip_weights as [D, C]
        patch_level.set_text_context(text_feats.T.contiguous(), encoder, device)

    # Initialize per-class states (list of C state dicts)
    states = patch_level.init_state(text_feats)   # uses text_feats [C, D]

    mode_dir = out_dir / filter_mode.replace("/", "_")

    per_step_compare_data = {}

    for i in range(N - 1):
        tensor_i    = tensors[i]
        tensor_next = tensors[i + 1]

        # ── Step A: update_state on image_i ───────────────────────────────────
        with torch.no_grad():
            # CLS-token global feature for seeding the first prototype
            global_feat = encoder.encode_image(tensor_i).squeeze(0).float()
            # encode_image already normalizes; _safe_normalize is a no-op here
            # but keep it explicit for robustness
            global_feat = _safe_normalize(global_feat)

        states[0] = patch_level.update_state(
            states[0], tensor_i, encoder, global_feat,
            target_class_idx=0,
        )

        K = states[0]["centers"].shape[0]

        # ── Step B: compute_patch_logits on image_{i+1} ───────────────────────
        raw_proto, quality_gate, proto_details = patch_level.compute_patch_logits(
            tensor_next, encoder, states, return_details=True,
            target_class_idx=0,
        )

        qg  = float(quality_gate)
        best_logit = float(raw_proto.max())
        print(
            f"  [{filter_mode}] step={i:02d}  K={K:3d}  "
            f"quality_gate={qg:.4f}  best_proto_logit={best_logit:.4f}"
        )
        _print_proto_table(proto_details, raw_proto, class_names)

        # ── Save per-step figure ──────────────────────────────────────────────
        path = _save_step_figure(
            step=i,
            filter_mode=filter_mode,
            tensor_i=tensor_i,
            tensor_next=tensor_next,
            state_after_i=states[0],
            raw_proto=raw_proto,
            quality_gate=quality_gate,
            class_names=class_names,
            encoder=encoder,
            out_dir=mode_dir,
            proto_details=proto_details,
            tensors=tensors,
        )
        print(f"    → {path}")

        # Collect data for cross-mode comparison figure
        if args.compare_modes:
            assign_i, _    = _assign_all_patches(tensor_i,    states[0], encoder)
            assign_next, _ = _assign_all_patches(tensor_next, states[0], encoder)
            img_i    = _denorm(tensor_i.squeeze(0))
            img_next = _denorm(tensor_next.squeeze(0))
            per_step_compare_data[i] = {
                "overlay_i":    _assignment_overlay(img_i,    assign_i,    K),
                "overlay_next": _assignment_overlay(img_next, assign_next, K),
                "raw_proto":    raw_proto,
                "quality_gate": quality_gate,
                "K":            K,
            }

    # Final update (no "next" image to score)
    if N > 0:
        with torch.no_grad():
            global_feat = encoder.encode_image(tensors[-1]).squeeze(0).float()
            global_feat = _safe_normalize(global_feat)
        states[0] = patch_level.update_state(
            states[0], tensors[-1], encoder, global_feat, target_class_idx=0
        )
        print(f"  [{filter_mode}] final update done  K={states[0]['centers'].shape[0]}")

    return per_step_compare_data


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Debug GaussianPatchLevel: update_state → compute_patch_logits per step."
    )

    # ── Image source ──────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",      metavar="PATH",
                     help="Single image path; will be augmented --n-images times.")
    src.add_argument("--images-dir", metavar="DIR",
                     help="Directory of same-class images (first --n-images used).")

    # ── Class setup ───────────────────────────────────────────────────────────
    p.add_argument("--class-name", required=True,
                   help="Target class name (index 0 in states).")
    p.add_argument("--distractor-classes", nargs="*",
                   default=["dog", "car"],
                   help="Extra class names to make quality_gate non-trivial "
                        "(default: dog car).")

    # ── Run config ────────────────────────────────────────────────────────────
    p.add_argument("--n-images",     type=int, default=6,
                   help="Number of images / update steps (default: 6).")
    p.add_argument("--filter-modes", nargs="+",
                   default=["none"],
                   choices=["none", "cosine_with_labels", "cosine_no_labels",
                            "surgery_with_labels", "surgery_no_labels"],
                   help="Filter mode(s) to visualise (default: none).")
    p.add_argument("--compare-modes", action="store_true",
                   help="Also save a cross-mode comparison figure per step.")
    p.add_argument("--backbone", default="ViT-B/16",
                   help="CLIP backbone (default: ViT-B/16).")
    p.add_argument("--match-threshold",     type=float, default=0.60)
    p.add_argument("--max-k",               type=int,   default=20,
                   help="Max clusters per class (default: 20).")
    p.add_argument("--aggregation",         default="weighted_mean",
                   choices=["top_m_mean", "max", "sum", "mean",
                            "top_m_mean_plus_mean", "weighted_mean",
                            "zscore_weighted_mean", "zscore_top_m_mean"],
                   help="Prototype-score aggregation (default: weighted_mean). "
                        "The zscore_* variants normalise each prototype against "
                        "its own reference distribution before aggregating.")
    p.add_argument("--top-m",               type=int,   default=4,
                   help="M for the top_m_* aggregations (default: 4).")
    p.add_argument("--proto-stats-min-count", type=int, default=2,
                   help="Reference observations before a prototype's z-score is "
                        "trusted (default: 2, vs 10 in production configs — this "
                        "script only walks a few images).")
    p.add_argument("--filter-threshold",    type=float, default=0.5,
                   help="Threshold on normalised [0,1] score for patch filtering (default: 0.5).")
    p.add_argument("--seed",                type=int,   default=42,
                   help="RNG seed for image augmentation.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="tests/assets/gaussian_debug",
                   help="Root directory for output figures.")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Encoder ──────────────────────────────────────────────────────────────
    print(f"Loading encoder ({args.backbone}) ...")
    encoder = CLIPEncoder(model_type=args.backbone, device=device)

    # ── Images ───────────────────────────────────────────────────────────────
    tensors = _load_tensors(args, encoder, device)

    # ── Class names: target first, then distractors ───────────────────────────
    class_names = [args.class_name] + (args.distractor_classes or [])
    print(f"Classes: {class_names}  (index 0 = target)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run each filter mode ──────────────────────────────────────────────────
    all_mode_data = {}   # mode → {step → compare_data}
    for mode in args.filter_modes:
        print(f"\n{'='*60}")
        print(f"Filter mode: {mode!r}")
        print(f"{'='*60}")
        all_mode_data[mode] = _run_mode(
            filter_mode=mode,
            tensors=tensors,
            class_names=class_names,
            encoder=encoder,
            args=args,
            out_dir=out_dir,
        )

    # ── Cross-mode comparison figures ─────────────────────────────────────────
    if args.compare_modes and len(args.filter_modes) > 1:
        N = len(tensors)
        print(f"\nSaving comparison figures ...")
        for i in range(N - 1):
            overlays_by_mode = {}
            logits_by_mode   = {}
            valid = True
            for mode in args.filter_modes:
                step_data = all_mode_data[mode].get(i)
                if step_data is None:
                    valid = False
                    break
                overlays_by_mode[mode] = (step_data["overlay_i"], step_data["overlay_next"])
                logits_by_mode[mode]   = (step_data["raw_proto"],
                                          step_data["quality_gate"],
                                          step_data["K"])
            if not valid:
                continue
            path = _save_compare_figure(
                step=i,
                filter_modes=args.filter_modes,
                tensors_by_mode=None,
                overlays_by_mode=overlays_by_mode,
                logits_by_mode=logits_by_mode,
                class_names=class_names,
                out_dir=out_dir,
            )
            print(f"  → {path}")

    print(f"\nDone.  Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
