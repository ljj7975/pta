"""Layer-alignment diagnostic for CS-ViT-B/16 on DTD.

Question
--------
Do patch tokens at intermediate transformer depths (blocks 8 / 10) keep
text-aligned semantics comparable to the final layer?  This informs whether
an "early-exit" patch layer (Option B) is feasible for the prototype bank.

Approach
--------
For each image, run the CS vision transformer manually, capturing the token
state after blocks 8, 10 and the final block (11).  Each captured state is
projected with the model's own tail (CLS-swap + ln_post + proj) and compared
against class text embeddings (DTD template ``'{} texture.'``).  Standard
(non-surgery) CLIP final-layer patches are included as a contrast row, with
the direction correction the codebase applies elsewhere (negation).

Usage
-----
    python scripts/layer_alignment_diagnostic.py [--num-images 30]
        [--data-root data] [--seed 0] [--json-out outputs/layer_alignment_diag.json]

Prints one table row per depth.  Optionally dumps raw JSON for the findings
document.
"""
import argparse
import json
import os

import torch

from third_party.CLIP_Surgery.clip_surgery.clip import clip_feature_surgery
from utils import build_test_data_loader
from utils.clip_inference import _identify_relevant_patches, _safe_normalize

# Depths (0-indexed transformer blocks) to capture for the early-exit sweep.
CAPTURE_BLOCKS = (8, 10, 11)
# Surgery replaces attention in the last 6 blocks (6..11), which then carry
# the dual-path [x, x_ori] state; blocks below this are plain CLIP blocks.
SURGERY_START = 6


def build_text_features(classnames, template, encoder):
    """Class text embeddings [C, D], mirroring utils.clip_inference.clip_classifier.

    clip_classifier is hard-wired to .cuda(); this CPU variant formats the
    same templates and averages per-class embeddings the same way.
    """
    with torch.no_grad():
        embeds = []
        for name in classnames:
            name = name.replace("_", " ")
            texts = [t.format(name) for t in template]
            class_embeddings = encoder.encode_text(texts)  # [n_templates, D]
            class_embeddings = _safe_normalize(class_embeddings)
            embeds.append(_safe_normalize(class_embeddings.mean(dim=0)))
        return torch.stack(embeds, dim=0)  # [C, D]


def forward_with_capture(visual, img, capture_blocks=(8, 10, 11)):
    """Replicate VisionTransformer.forward, capturing states after target blocks.

    Returns a dict {block_index: projected_tokens} where projected_tokens is
    ``[1, 1+P, D]`` — the CLS-swapped state at that depth pushed through the
    model's own ln_post + proj tail.  The final entry must equal the model's
    ``encode_image`` output (verified by the caller).

    The surgery install (replacing attn in blocks 6-11) happens lazily inside
    the model's first forward, so the caller must trigger one forward before
    calling this.
    """
    x = visual.conv1(img)  # [1, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, width, grid**2]
    x = x.permute(0, 2, 1)  # [1, grid**2, width]
    x = torch.cat([
        visual.class_embedding.to(x.dtype)
        + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x,
    ], dim=1)  # [1, 1 + grid**2, width]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND (seq-first, matches transformer internals)
    captured = {}
    for i, block in enumerate(visual.transformer.resblocks):
        x = block(x)
        if i in capture_blocks:
            captured[i] = x

    # Apply the model's tail to each captured state: blocks >= SURGERY_START
    # hold a dual-path list [x, x_ori]; the CLS token is taken from the
    # original path, exactly like the final forward does.
    out = {}
    for block_idx, state in captured.items():
        if isinstance(state, (list, tuple)):
            x_d, x_ori_d = state
            x_d = x_d.clone()
            x_d[0, :, :] = x_ori_d[0, :, :]
            state = x_d
        state = state.permute(1, 0, 2)  # LND -> NLD
        state = visual.ln_post(state)
        state = state @ visual.proj
        out[block_idx] = state  # [1, 1+P, D]
    return out


def alignment_metrics(patches_norm, text_feats, true_idx, surgery_col=None):
    """Alignment proxies for one image's patch embeddings.

    Args:
        patches_norm: [P, D] L2-normalised patch embeddings.
        text_feats:   [C, D] L2-normalised class text embeddings.
        true_idx:     int, ground-truth class index.
        surgery_col:  [P] per-patch CLIP Surgery relevance scores for the true
                      class (optional).  When given, keep_frac/kept_acc mirror
                      the real write filter (filter_patches_by_text_alignment
                      thresholds the surgery score column at 0.5).

    Returns a dict of scalar metrics:
        top1_sim   — mean over patches of max-over-classes cosine similarity.
        true_sim   — mean over patches of cosine similarity to the true class.
        patch_acc  — fraction of patches whose best-matching class is the true one.
        keep_frac  — fraction of patches kept by _identify_relevant_patches
                     (threshold 0.5) on the true-class surgery score column.
        kept_acc   — fraction of kept patches whose best-matching class is true.
    """
    sims = patches_norm @ text_feats.T  # [P, C]
    true_col = sims[:, true_idx]
    best_class = sims.argmax(dim=1)

    if surgery_col is not None:
        keep_mask = _identify_relevant_patches(surgery_col, threshold=0.5, min_patches=1)
    else:
        keep_mask = torch.ones(sims.shape[0], dtype=torch.bool, device=sims.device)
    kept = sims[keep_mask]
    kept_acc = (kept.argmax(dim=1) == true_idx).float().mean().item() if kept.shape[0] else 0.0

    return {
        "top1_sim": float(sims.max(dim=1).values.mean().item()),
        "true_sim": float(true_col.mean().item()),
        "patch_acc": float((best_class == true_idx).float().mean().item()),
        "keep_frac": float(keep_mask.float().mean().item()),
        "kept_acc": kept_acc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=30,
                        help="Number of test images to evaluate (default: 30)")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Dataset root directory (default: data)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Loader shuffle seed (default: 0)")
    parser.add_argument("--json-out", type=str, default="",
                        help="Optional JSON output path for the findings doc")
    parser.add_argument("--no-standard-clip", action="store_true",
                        help="Skip the standard CLIP contrast row")
    args = parser.parse_args()

    device = "cpu"
    torch.manual_seed(args.seed)

    from encoder.clip_encoder import CLIPSurgeryEncoder

    encoder = CLIPSurgeryEncoder(model_type="ViT-B/16", device=device)
    model = encoder.model
    visual = model.visual

    # Surgery installs lazily on the first forward; run a dummy to trigger it
    # before the manual block loop, so blocks 6-11 use the surgery attention.
    with torch.no_grad():
        _ = visual(torch.zeros(1, 3, 224, 224, device=device))

    # ------------------------------------------------------------------ loader
    test_loader, classnames, template = build_test_data_loader(
        "dtd", args.data_root, encoder.preprocess, shuffle=True, seed=args.seed,
    )
    print(f"[layer-alignment] DTD test set, {args.num_images} images, "
          f"{len(classnames)} classes, template={template}")

    # ------------------------------------------------------------------ text
    text_feats = build_text_features(classnames, template, encoder)  # [C, D]

    # ------------------------------------------------------- standard CLIP
    std_model = None
    std_visual = None
    if not args.no_standard_clip:
        import clip as standard_clip
        std_model, _ = standard_clip.load("ViT-B/16", device=device)
        std_visual = std_model.visual

    # --------------------------------------------------------------- sweep
    accum = {d: [] for d in list(CAPTURE_BLOCKS) + (["std_clip"] if std_visual else [])}
    max_diff_final = 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.num_images:
                break
            images, target = batch
            if isinstance(images, list):
                images = torch.cat(images, dim=0)
            images = images.to(device)
            true_idx = int(target.item())

            # CS depths (single pass captures all three)
            captures = forward_with_capture(visual, images)
            for depth, tokens in captures.items():
                tokens_norm = _safe_normalize(tokens)  # [1, 1+P, D]
                # Sanity: the final capture must match the model's own output.
                if depth == 11:
                    ref = _safe_normalize(model.encode_image(images))
                    max_diff_final = max(
                        max_diff_final,
                        float((tokens_norm - ref).abs().max().item()),
                    )
                # Surgery relevance scores for the keep-fraction proxy.
                scores = clip_feature_surgery(
                    tokens_norm.float(), text_feats.float()
                )  # [1, 1+P, C]
                patches_norm = tokens_norm[0, 1:]  # [P, D]
                # True-class score column drives _identify_relevant_patches.
                accum[depth].append(
                    alignment_metrics(
                        patches_norm, text_feats, true_idx,
                        surgery_col=scores[0, 1:, true_idx],
                    )
                )

            if std_model is not None:
                # clip_feature_surgery weights scores by the CLS row, so build
                # CLS+patches tokens (same as compute_surgery_scores via
                # encode_image(CLS_token_only=False)).
                std_cls = std_model.encode_image(images)  # [1, D]
                std_patches = std_visual(images, return_patches=True)  # [1, P, D]
                std_tokens = torch.cat([std_cls.unsqueeze(1), std_patches], dim=1)  # [1, 1+P, D]
                std_tokens_norm = _safe_normalize(std_tokens)
                std_norm = std_tokens_norm[0, 1:]  # [P, D]
                # Standard CLIP patch cosine similarity is inverted relative to
                # CS features; negate for the corrected direction (same
                # convention as filter_patches_by_text_alignment).
                # Surgery scores are likewise negated (compute_surgery_scores).
                std_scores = clip_feature_surgery(
                    std_tokens_norm.float(), text_feats.float()
                )  # [1, 1+P, C]
                accum["std_clip"].append(
                    alignment_metrics(
                        -std_norm, text_feats, true_idx,
                        surgery_col=-std_scores[0, 1:, true_idx],
                    )
                )

    # -------------------------------------------------------------- report
    print(f"[layer-alignment] final capture == encode_image: max_abs_diff={max_diff_final:.2e}")
    print()
    header = ("depth", "top1_sim", "true_sim", "patch_acc", "keep_frac", "kept_acc")
    print(f"{header[0]:<10} {header[1]:>9} {header[2]:>9} {header[3]:>9} "
          f"{header[4]:>9} {header[5]:>9}")
    print("-" * 60)
    results = {}
    for depth, rows in accum.items():
        n = len(rows)
        mean = {k: sum(r[k] for r in rows) / n for k in rows[0]}
        results[str(depth)] = mean
        label = {
            8: "block8 (early-exit)",
            10: "block10 (early-exit)",
            11: "final (block11)",
            "std_clip": "std CLIP final (fixed)",
        }[depth]
        print(f"{label:<10} {mean['top1_sim']:>9.4f} {mean['true_sim']:>9.4f} "
              f"{mean['patch_acc']:>9.4f} {mean['keep_frac']:>9.4f} "
              f"{mean['kept_acc']:>9.4f}")

    results["final_verify_max_abs_diff"] = max_diff_final
    results["num_images"] = args.num_images
    results["classnames"] = classnames
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
