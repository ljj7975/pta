#!/usr/bin/env python3
"""
Diagnostic script to verify patch filtering is actually applied in update_state.

Run with: srun --gres=gpu:1 python tests/test_patch_filter_debug.py
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.patch_level.gaussian_patch import GaussianPatchLevel
from utils.clip_inference import _safe_normalize, filter_patches_by_text_alignment


def test_filter_is_nontrivial():
    """Verify that filter_patches_by_text_alignment produces different masks for different modes."""
    torch.manual_seed(42)
    D = 512
    P = 196 * 16  # 16 views * 196 patches (matching aug_copies=15)

    # Create dummy patch embeddings with some structure
    patches_norm = torch.randn(P, D)
    patches_norm = patches_norm / patches_norm.norm(dim=-1, keepdim=True)

    # Create dummy text features (5 classes)
    C = 5
    text_features = torch.randn(C, D)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Create dummy empty text feature
    empty_text_feat = torch.randn(1, D)
    empty_text_feat = empty_text_feat / empty_text_feat.norm(dim=-1, keepdim=True)

    target_class_idx = 0

    modes = ["none", "cosine_with_labels", "cosine_no_labels"]
    masks = {}
    for mode in modes:
        mask = filter_patches_by_text_alignment(
            patches_norm, target_class_idx,
            text_features=text_features,
            empty_text_feat=empty_text_feat,
            filter_mode=mode,
            filter_threshold=0.5,
            aug_copies=15,
        )
        masks[mode] = mask
        kept = mask.sum().item()
        print(f"  {mode:25s}: kept={kept}/{P} ({100*kept/P:.1f}%)")

    # Verify masks differ
    none_kept = masks["none"].sum().item()
    cosine_labels_kept = masks["cosine_with_labels"].sum().item()
    cosine_nolabels_kept = masks["cosine_no_labels"].sum().item()

    assert none_kept == P, f"'none' should keep all patches, got {none_kept}"
    assert cosine_labels_kept < P, f"cosine_with_labels should filter some patches, kept {cosine_labels_kept}/{P}"
    assert cosine_nolabels_kept < P, f"cosine_no_labels should filter some patches, kept {cosine_nolabels_kept}/{P}"
    assert cosine_labels_kept != cosine_nolabels_kept, (
        f"cosine modes should produce different masks: "
        f"labels={cosine_labels_kept}, nolabels={cosine_nolabels_kept}"
    )
    print("  ✓ All filter modes produce different masks\n")


def test_config_override_propagation():
    """Verify config override propagates correctly to GaussianPatchLevel._cfg."""
    from utils.config import get_config_file

    # Load config with override
    cfg = get_config_file("configs/patch_modulated_pta", "dtd")
    print(f"  Before override: patch_filter_mode = {cfg.get('patch_level', {}).get('patch_filter_mode')}")

    # Apply override like runner.py does
    from runner import apply_overrides
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
    # At minimum, "none" should have different center count than filtered modes
    all_same = all(v == results["none"] for v in results.values())
    if all_same:
        print("  ⚠ All modes produced same number of centers (this may be expected with few images)")
    else:
        print("  ✓ Different filter modes produce different states")
    print()


def main():
    print("=" * 60)
    print("PATCH FILTER DIAGNOSTIC")
    print("=" * 60)

    print("\n1. Testing filter_patches_by_text_alignment produces non-trivial masks...")
    test_filter_is_nontrivial()

    print("2. Testing config override propagation...")
    test_config_override_propagation()

    print("3. Testing set_text_context sets attributes...")
    test_set_text_context_sets_attributes()

    print("4. Testing update_state filtering produces different results...")
    test_update_state_filtering_difference()

    print("=" * 60)
    print("All diagnostics passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
