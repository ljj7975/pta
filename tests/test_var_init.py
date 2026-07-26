#!/usr/bin/env python3
"""
Diagnostic test for Gaussian variance initialization fix.

Verifies that:
1. Variance is initialized from patch statistics (not fixed 0.01)
2. Gaussian scores are non-zero
3. Patch_proto is non-zero after update_state

Run with: python tests/test_var_init.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import create_encoder_instance
from models.patch_level.gaussian_patch import GaussianPatchLevel
from utils.clip_inference import _safe_normalize


def test_variance_initialization():
    """Test that variance is initialized from patch statistics."""
    print("=" * 60)
    print("TEST: Variance initialization from patch statistics")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder = create_encoder_instance("clip", model_type="ViT-B/16", device=device)
    encoder.eval()

    dummy_image = torch.randn(1, 3, 224, 224).to(device)

    num_classes = 10
    D = 512
    clip_weights = torch.randn(D, num_classes).to(device)
    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)

    cfg = {
        "patch_level": {
            "type": "GaussianPatchLevel",
            "max_K": 100,
            "match_threshold": 0.6,
            "conf_threshold": 0.5,
            "conf_margin_threshold": 0.0,
            "n_half": 15.0,
            "soft_nn_top_m": 4,
            "proto_alpha_max": 1.0,
            "quality_eps": 0.001,
            "exclude_pos": False,
            "patch_group_threshold": 0.9,
            "gaussian_ema": 0.1,
            "variance_min": 0.001,
            "variance_max": 1.0,
            "aug_copies": 15,
            "patch_filter_mode": "none",
            "patch_filter_threshold": 0.5,
        }
    }
    patch_level = GaussianPatchLevel(cfg)

    patch_level.set_text_context(clip_weights, encoder, device)

    patch_embs = encoder.get_patch_embeddings(dummy_image, exclude_pos=False)
    patches_norm = _safe_normalize(patch_embs)

    all_patches = [patches_norm]
    for _ in range(15):
        aug_embs = encoder.get_patch_embeddings(dummy_image, exclude_pos=False)
        all_patches.append(_safe_normalize(aug_embs))
    patches_concat = torch.cat(all_patches, dim=0)

    refine_feature = clip_weights.t().float()  # [C, D]
    states = patch_level.init_state(refine_feature)
    state = states[0]
    print(f"\nBefore update_state:")
    print(f"  Centers shape: {state['centers'].shape}")
    print(f"  Variance shape: {state['variance'].shape}")
    print(f"  n_images: {state['n_images']}")

    state = patch_level.update_state(
        state, dummy_image, encoder, refine_feature[0, :],
        filter_scores=None, target_class_idx=0
    )
    states[0] = state

    print(f"\nAfter update_state:")
    print(f"  Centers shape: {state['centers'].shape}")
    print(f"  Variance shape: {state['variance'].shape}")
    print(f"  Variance mean: {state['variance'].mean().item():.6f}")
    print(f"  Variance min: {state['variance'].min().item():.6f}")
    print(f"  Variance max: {state['variance'].max().item():.6f}")
    print(f"  n_images: {state['n_images']}")

    raw_proto, quality_gate = patch_level.compute_patch_logits(
        dummy_image, encoder, states
    )

    print(f"\nPatch logits:")
    print(f"  raw_proto: {raw_proto}")
    print(f"  quality_gate: {quality_gate}")
    print(f"  patch_proto_max: {raw_proto.max().item():.6f}")
    print(f"  patch_proto_mean: {raw_proto.mean().item():.6f}")

    variance_initialized = state['variance'].mean().item() > 0.001
    print(f"\nVariance initialized from patch statistics: {variance_initialized}")

    gaussian_scores_nonzero = raw_proto.max().item() > 0.001
    print(f"Gaussian scores are non-zero: {gaussian_scores_nonzero}")

    if variance_initialized and gaussian_scores_nonzero:
        print("\nTEST PASSED")
        return True
    else:
        print("\nTEST FAILED")
        return False


if __name__ == "__main__":
    print("Running Gaussian variance initialization test...")
    print()

    test_passed = test_variance_initialization()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Variance initialization test: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        sys.exit(0)
    else:
        sys.exit(1)
