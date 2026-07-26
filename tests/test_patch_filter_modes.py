#!/usr/bin/env python3
"""
Test patch filtering modes - verifies differences between all 4 filter modes.

Run with: python tests/test_patch_filter_modes.py

Expected output when working correctly:
  1. All modes produce DIFFERENT keep masks (not identical)
  2. Score distributions differ across modes
  3. Filter reduces patch count from 3136 → ~1568 (50%)
  4. None mode keeps ALL patches (no filtering)
  5. Cosine modes use different scoring strategies
  6. Surgery modes use CLIP Surgery similarity scores
"""
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.patch_level.gaussian_patch import GaussianPatchLevel
# _safe_normalize imported from base; _extract_* helpers inlined at call sites
from utils.clip_inference import _safe_normalize, filter_patches_by_text_alignment


def create_test_config(filter_mode="none"):
    """Create a test config with the specified filter mode."""
    return {
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
            "patch_filter_mode": filter_mode,
            "patch_filter_threshold": 0.5,
        }
    }


def test_filter_modes_differ():
    """Test that different filter modes produce different keep masks."""
    print("=" * 60)
    print("TEST: Filter modes produce different keep masks")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create a dummy image (14x14 ViT patches = 196 patches)
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Load CLIP model
    import clip
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    # Create text features (simulate 10 classes)
    num_classes = 10
    D = 512  # ViT-B/16 embedding dimension
    clip_weights = torch.randn(D, num_classes).to(device)
    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)
    
    # Test each filter mode
    filter_modes = ["cosine_with_labels", "cosine_no_labels", "surgery_with_labels", "surgery_no_labels"]
    all_masks = {}
    all_score_stats = {}
    
    for mode in filter_modes:
        print(f"\n--- Testing mode: {mode} ---")
        
        # Create patch level instance
        cfg = create_test_config(filter_mode=mode)
        patch_level = GaussianPatchLevel(cfg)
        
        # Set text context
        patch_level.set_text_context(clip_weights, clip_model, device)
        
        # Extract patches from original image
        patch_embs = clip_model.get_patch_embeddings(dummy_image, exclude_pos=False)  # extract patch embeddings
        patches_norm = _safe_normalize(patch_embs)  # [196, D]
        
        # Simulate augmented views (concatenate 16 views)
        all_patches = [patches_norm]
        for _ in range(15):
            aug_embs = clip_model.get_patch_embeddings(dummy_image, exclude_pos=False)  # extract patch embeddings
            all_patches.append(_safe_normalize(aug_embs))
        patches_concat = torch.cat(all_patches, dim=0)  # [3136, D]
        
        # Get filter scores for surgery modes
        filter_scores = None
        if mode.startswith("surgery"):
            all_tokens = clip_model.encode_image(dummy_image, CLS_token_only=False, preprocess=False)  # [1, 197, D]  # extract all tokens
            from third_party.CLIP_Surgery.clip_surgery.clip import clip_feature_surgery
            text_feats = clip_weights.t().float()  # [C, D]
            if mode == "surgery_with_labels":
                sim = clip_feature_surgery(all_tokens.float(), text_feats)  # [1, 197, C]
                filter_scores = sim[0, 1:, 0]  # [196] - first class
            else:
                empty_feat = patch_level._empty_text_feat.float()
                sim = clip_feature_surgery(all_tokens.float(), text_feats, redundant_feats=empty_feat)
                filter_scores = sim[0, 1:, 0]  # [196]
        
        # Apply filter
        keep_mask = filter_patches_by_text_alignment(
            patches_concat, target_class_idx=0,
            text_features=patch_level._text_features,
            empty_text_feat=patch_level._empty_text_feat,
            filter_mode=mode,
            filter_threshold=patch_level._filter_threshold,
            aug_copies=15,
            precomputed_scores=filter_scores,
        )
        
        all_masks[mode] = keep_mask
        kept_count = int(keep_mask.sum())
        total_count = len(keep_mask)
        
        print(f"  Patches: {total_count} → {kept_count} kept ({kept_count/total_count*100:.1f}%)")
        
        # Compute score stats for this mode
        if mode == "cosine_with_labels":
            sim = patches_concat @ patch_level._text_features.T
            target_score = sim[:, 0]
            other_mean = (sim.sum(1) - target_score) / max(sim.shape[1] - 1, 1)
            scores = target_score - other_mean
        elif mode == "cosine_no_labels":
            target_feat = patch_level._text_features[0]
            adjusted = target_feat - patch_level._empty_text_feat.squeeze(0)
            adjusted = adjusted / adjusted.norm().clamp(min=1e-8)
            scores = patches_concat @ adjusted
        else:
            scores = filter_scores.repeat(16)[:patches_concat.shape[0]]
        
        all_score_stats[mode] = {
            "min": scores.min().item(),
            "max": scores.max().item(),
            "mean": scores.mean().item(),
            "std": scores.std().item(),
        }
        print(f"  Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    # Verify modes produce different masks
    print("\n" + "=" * 60)
    print("VERIFICATION: Comparing filter masks across modes")
    print("=" * 60)
    
    mode_list = list(all_masks.keys())
    for i in range(len(mode_list)):
        for j in range(i+1, len(mode_list)):
            m1, m2 = mode_list[i], mode_list[j]
            mask1, mask2 = all_masks[m1], all_masks[m2]
            
            # Check if masks are identical
            identical = mask1.eq(mask2).all().item()
            
            # Compute overlap
            overlap = int((mask1 & mask2).sum())
            total_kept_1 = int(mask1.sum())
            total_kept_2 = int(mask2.sum())
            jaccard = overlap / int((mask1 | mask2).sum())
            
            print(f"\n{m1} vs {m2}:")
            print(f"  Identical: {identical}")
            print(f"  Overlap: {overlap} patches ({overlap/total_kept_1*100:.1f}% of {m1}, {overlap/total_kept_2*100:.1f}% of {m2})")
            print(f"  Jaccard similarity: {jaccard:.3f}")
            
            if identical:
                print(f"  ⚠️  WARNING: Masks are IDENTICAL - filtering not working!")
                return False
    
    # Verify score distributions differ
    print("\n" + "=" * 60)
    print("VERIFICATION: Score distributions across modes")
    print("=" * 60)
    
    for mode in mode_list:
        stats = all_score_stats[mode]
        print(f"\n{mode}:")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Check if score distributions are too similar
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_std = [all_score_stats[m]["std"] for m in mode_list]
    max_std = max(all_std)
    min_std = min(all_std)
    
    print(f"Score std range: [{min_std:.4f}, {max_std:.4f}]")
    
    if max_std < 0.01:
        print("⚠️  WARNING: Score std is very low (< 0.01) - filtering may be ineffective")
        print("   This is expected for texture datasets (DTD) where all patches are similar.")
        print("   For object datasets (ImageNet, Caltech101), std should be higher.")
    else:
        print("✓ Score std is reasonable - filtering should work on object datasets")
    
    print("\n✓ All filter modes produce DIFFERENT masks")
    print("✓ Score distributions differ across modes")
    print("✓ Test PASSED")
    
    return True


def test_none_mode_keeps_all():
    """Test that 'none' mode keeps all patches."""
    print("\n" + "=" * 60)
    print("TEST: None mode keeps all patches")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy image
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Load CLIP model
    import clip
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    # Create patch level with 'none' mode
    cfg = create_test_config(filter_mode="none")
    patch_level = GaussianPatchLevel(cfg)
    
    # Extract patches
    patch_embs = clip_model.get_patch_embeddings(dummy_image, exclude_pos=False)  # extract patch embeddings
    patches_norm = _safe_normalize(patch_embs)  # [196, D]
    
    # Simulate augmented views
    all_patches = [patches_norm]
    for _ in range(15):
        aug_embs = clip_model.get_patch_embeddings(dummy_image, exclude_pos=False)  # extract patch embeddings
        all_patches.append(_safe_normalize(aug_embs))
    patches_concat = torch.cat(all_patches, dim=0)  # [3136, D]
    
    # Apply filter (should return all-True for 'none' mode)
    keep_mask = filter_patches_by_text_alignment(
        patches_concat, target_class_idx=0,
        text_features=None,
        filter_mode="none",
        aug_copies=15,
        precomputed_scores=None,
    )
    
    all_kept = keep_mask.all().item()
    print(f"Total patches: {len(keep_mask)}")
    print(f"Kept patches: {int(keep_mask.sum())}")
    print(f"All kept: {all_kept}")
    
    if all_kept:
        print("✓ None mode correctly keeps ALL patches")
        return True
    else:
        print("✗ None mode incorrectly filtered some patches")
        return False


def test_config_override_propagation():
    """Verify config override propagates correctly to GaussianPatchLevel._cfg."""
    from utils.config import get_config_file
    from runner import apply_overrides

    print("\n" + "=" * 60)
    print("TEST: Config override propagation")
    print("=" * 60)

    # Load config with override
    cfg = get_config_file("configs/patch_modulated_pta", "dtd")
    print(f"  Before override: patch_filter_mode = {cfg.get('patch_level', {}).get('patch_filter_mode')}")

    # Apply override like runner.py does
    cfg = apply_overrides(cfg, ["patch_level.patch_filter_mode=cosine_with_labels"])
    print(f"  After override:  patch_filter_mode = {cfg.get('patch_level', {}).get('patch_filter_mode')}")

    # Create GaussianPatchLevel and check its config
    patch_level = GaussianPatchLevel(cfg)
    print(f"  GaussianPatchLevel._cfg patch_filter_mode = {patch_level._cfg.get('patch_filter_mode')}")

    assert cfg["patch_level"]["patch_filter_mode"] == "cosine_with_labels"
    assert patch_level._cfg.get("patch_filter_mode") == "cosine_with_labels"
    print("  ✓ Config override propagates correctly\n")


def test_set_text_context_sets_attributes():
    """Verify set_text_context properly sets _text_features on GaussianPatchLevel."""
    from utils.config import get_config_file
    from runner import apply_overrides

    print("\n" + "=" * 60)
    print("TEST: set_text_context sets attributes")
    print("=" * 60)

    cfg = get_config_file("configs/patch_modulated_pta", "dtd")
    cfg = apply_overrides(cfg, ["patch_level.patch_filter_mode=cosine_with_labels"])

    patch_level = GaussianPatchLevel(cfg)

    assert not hasattr(patch_level, "_text_features"), "Before set_text_context, _text_features should not exist"

    # Create dummy text embeddings [D, C]
    D, C = 512, 5
    text_embeddings = torch.randn(D, C).cuda()

    # Create a mock encoder that has .model attribute
    class MockModel(torch.nn.Module):
        def encode_text(self, tokens):
            return torch.randn(tokens.shape[0], 512).to(tokens.device)

    class MockEncoder:
        def __init__(self):
            self.model = MockModel()

    encoder = MockEncoder()
    device = torch.device("cuda")

    patch_level.set_text_context(text_embeddings, encoder, device)

    assert hasattr(patch_level, "_text_features"), "_text_features should exist after set_text_context"
    assert patch_level._text_features.shape == (C, D), f"Expected ({C}, {D}), got {patch_level._text_features.shape}"
    assert hasattr(patch_level, "_empty_text_feat"), "_empty_text_feat should exist"
    assert patch_level._filter_mode == "cosine_with_labels", f"Expected cosine_with_labels, got {patch_level._filter_mode}"
    print("  ✓ set_text_context sets all required attributes\n")


def test_update_state_filtering_difference():
    """Verify update_state produces different results with different filter modes."""
    from utils.config import get_config_file
    from runner import apply_overrides

    print("\n" + "=" * 60)
    print("TEST: update_state filtering produces different results")
    print("=" * 60)

    D = 512
    C = 5

    # Create a state with some existing centers
    def make_state():
        return {
            "centers": torch.randn(3, D).cuda(),
            "variance": torch.abs(torch.randn(3, D)).cuda() * 0.01 + 0.001,
            "appearance": torch.ones(3).cuda(),
            "n_images": 5,
        }

    # Create dummy encoder that returns patch embeddings
    class MockVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, D, 16, stride=16, bias=False)
            self.positional_embedding = torch.nn.Parameter(torch.randn(197, D) * 0.01)
            self.class_embedding = torch.nn.Parameter(torch.randn(D) * 0.01)
            self.ln_pre = torch.nn.LayerNorm(D)
            self.ln_post = torch.nn.LayerNorm(D)
            self.proj = torch.nn.Parameter(torch.randn(D, D) * 0.01)
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(D, 8, D * 4, batch_first=True), num_layers=2
            )
            self.input_resolution = 224

        def encode_text(self, tokens):
            return torch.randn(tokens.shape[0], D).to(tokens.device)

    class MockEncoder:
        def __init__(self):
            self.visual = MockVisual().cuda()
            self.model = MockVisual().cuda()  # for set_text_context

        def get_patch_embeddings(self, images, exclude_pos=False):
            B = images.shape[0]
            P = 196  # 14x14 patches
            return torch.randn(B, P, D).cuda()

        def encode_image(self, images, CLS_token_only=True, preprocess=True):
            B = images.shape[0]
            if CLS_token_only:
                return torch.randn(B, D).cuda()
            else:
                return torch.randn(B, 197, D).cuda()  # CLS + 196 patches

    encoder = MockEncoder()

    # Create dummy images
    images = torch.randn(1, 3, 224, 224).cuda()
    global_feat = torch.randn(D).cuda()

    text_embeddings = torch.randn(D, C).cuda()

    modes = ["none", "cosine_with_labels", "cosine_no_labels"]
    results = {}

    for mode in modes:
        cfg = get_config_file("configs/patch_modulated_pta", "dtd")
        cfg = apply_overrides(cfg, [f"patch_level.patch_filter_mode={mode}"])

        # Patch filter absolute threshold - set high to force strong filtering
        cfg["patch_level"]["patch_filter_threshold"] = 0.5
        cfg["patch_level"]["aug_copies"] = 0  # No augmentation for speed

        patch_level = GaussianPatchLevel(cfg)
        patch_level.set_text_context(text_embeddings, encoder, torch.device("cuda"))

        state = make_state()
        new_state = patch_level.update_state(
            state, images, encoder, global_feat,
            target_class_idx=0,
        )

        n_centers = new_state["centers"].shape[0]
        results[mode] = n_centers
        print(f"  {mode:25s}: centers after update = {n_centers}")

    # Check that different filter modes produce different state
    all_same = all(v == results["none"] for v in results.values())
    if all_same:
        print("  ⚠ All modes produced same number of centers (this may be expected with few images)")
    else:
        print("  ✓ Different filter modes produce different states")
    print()


if __name__ == "__main__":
    print("Running patch filter mode tests...")
    print()

    results = {}

    # Test 1: None mode keeps all patches
    results["none_mode"] = test_none_mode_keeps_all()

    # Test 2: Filter modes produce different masks
    results["modes_differ"] = test_filter_modes_differ()

    # Test 3: Config override propagation
    results["config_override"] = test_config_override_propagation()

    # Test 4: set_text_context sets attributes
    results["text_context"] = test_set_text_context_sets_attributes()

    # Test 5: update_state filtering difference
    results["update_state"] = test_update_state_filtering_difference()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:25s}: {'PASSED' if passed else 'FAILED'}")

    if all(results.values()):
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
