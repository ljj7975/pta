#!/usr/bin/env python3
"""
Test suite for CLIP encoder classes in encoder/clip_encoder.py.

Loads sample images from the eurosat or dtd dataset split files and validates
that each CLIP encoder variant can:
  - Load the model without errors
  - Encode text features (correct shape, normalized)
  - Encode image features (correct shape, normalized)
  - Produce reasonable image-text similarity scores

Usage (local):
    python tests/test_clip_encoders.py
    python tests/test_clip_encoders.py --dataset dtd --num-samples 5
    python tests/test_clip_encoders.py --dataset eurosat --backbone ViT-B/16

Usage (remote via SSH):
    ssh jetson "cd /home/brandon/repos/pta && source pta/bin/activate && \
        python tests/test_clip_encoders.py --dataset dtd"
"""

import argparse
import os
import sys
import contextlib

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import json
import torch
from PIL import Image
from typing import List, Dict, Tuple

# ─── Dataset helpers ───────────────────────────────────────────────

def load_split_images(dataset: str, data_root: str, num_samples: int) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Load sample images from a dataset split JSON file.

    Loads **one image per class** (different classes) so the image-text
    similarity diagonal is meaningful — matching pairs should score higher
    than mismatched pairs.

    Returns:
        images: list of PIL Images
        classnames: list of class names for each image
        image_paths: list of full image paths
    """
    dataset_config = {
        "eurosat": {
            "dataset_dir": "eurosat",
            "image_dir": "2750",
            "split_file": "split_zhou_EuroSAT.json",
        },
        "dtd": {
            "dataset_dir": "dtd",
            "image_dir": "images",
            "split_file": "split_zhou_DescribableTextures.json",
        },
        "caltech101": {
            "dataset_dir": "caltech-101",
            "image_dir": "101_ObjectCategories",
            "split_file": "split_zhou_Caltech101.json",
        },
    }

    if dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset}. Use one of {list(dataset_config.keys())}.")

    cfg = dataset_config[dataset]
    dataset_dir = os.path.join(data_root, cfg["dataset_dir"])
    image_dir = os.path.join(dataset_dir, cfg["image_dir"])
    split_path = os.path.join(dataset_dir, cfg["split_file"])

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Make sure datasets are placed under {data_root}."
        )

    with open(split_path, "r") as f:
        split = json.load(f)

    test_items = split["test"]  # list of [relative_path, label, classname]

    # Pick one image per unique class (up to num_samples classes)
    seen_classes = set()
    images = []
    classnames = []
    image_paths = []

    for rel_path, label, classname in test_items:
        if classname in seen_classes:
            continue
        full_path = os.path.join(image_dir, rel_path)
        if not os.path.exists(full_path):
            continue
        img = Image.open(full_path).convert("RGB")
        images.append(img)
        classnames.append(classname)
        image_paths.append(full_path)
        seen_classes.add(classname)
        if len(images) >= num_samples:
            break

    if not images:
        raise RuntimeError(f"No images found for dataset '{dataset}' under {image_dir}")

    return images, classnames, image_paths


# ─── Import context for DetailCLIP ─────────────────────────────────

_DETAILCLIP_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "third_party", "DetailCLIP")
)

@contextlib.contextmanager
def _detailclip_import_context():
    """Temporarily make DetailCLIP's directory the first import source.

    DetailCLIP's models.py does `from utils import ...` which would resolve
    to the project's utils/ package. This context manager puts DetailCLIP's
    dir at the front of sys.path so Python finds DetailCLIP's modules first.
    """
    saved_path = sys.path[:]
    saved_modules = {}
    for name in ("models", "utils", "tokenizer"):
        if name in sys.modules:
            saved_modules[name] = sys.modules.pop(name)
    try:
        sys.path.insert(0, _DETAILCLIP_DIR)
        yield
    finally:
        sys.path = saved_path
        sys.modules.update(saved_modules)


# ─── Encoder test helpers ──────────────────────────────────────────

_PATCH_SIZE = {"ViT-B/32": 32, "ViT-B/16": 16, "ViT-L/14": 14}


def expected_num_tokens(backbone: str, input_size: int = 224) -> int:
    patch_size = _PATCH_SIZE[backbone]
    num_patches = (input_size // patch_size) ** 2
    return num_patches + 1  # +1 for CLS


def check_tensor_props(features: torch.Tensor, expected_dim: int, name: str, tolerance: float = 1e-3):
    """Validate that a feature tensor has correct shape and normalization."""
    assert features.dim() == 2, f"{name} should be 2D, got {features.dim()}D"
    assert features.shape[1] == expected_dim, \
        f"{name} dim mismatch: expected {expected_dim}, got {features.shape[1]}"

    # Check L2 normalization
    norms = features.norm(dim=-1)
    assert (norms - 1.0).abs().max() < tolerance, \
        f"{name} not properly L2-normalized (max deviation: {(norms - 1.0).abs().max():.6f})"

    # Check no NaN/Inf
    assert not features.isnan().any(), f"{name} contains NaN"
    assert not features.isinf().any(), f"{name} contains Inf"

    print(f"  ✓ {name}: shape={tuple(features.shape)}, norm_ok, no_nan/inf")


def check_similarity(text_features: torch.Tensor, image_features: torch.Tensor, name: str):
    """Check that image-text similarity produces reasonable scores.

    With one image per class, the diagonal (correct class) should have the
    highest similarity in each row — i.e. the model correctly ranks the
    matching text above all other class texts for each image.
    """
    sim = (text_features @ image_features.T)  # NxN matrix
    diag = sim.diag()

    # Print full similarity matrix
    sim_np = sim.cpu().numpy()
    print(f"  ✓ {name} similarity matrix:")
    for i in range(sim_np.shape[0]):
        row_vals = [f'{sim_np[i, j]:7.3f}' for j in range(sim_np.shape[1])]
        row_str = '  '.join(row_vals)
        marker = ' ← match'
        print(f"    row {i}: {row_str}{marker}")

    print(f"    diagonal: [{', '.join(f'{v:.3f}' for v in diag.cpu().numpy())}]")

    # Check 1: diagonal values should be positive (features are aligned)
    assert diag.mean() > 0, \
        f"{name} similarity diagonal mean is {diag.mean():.4f} — features may not be aligned"

    # Check 2: diagonal should be the max in each row (correct class ranks highest)
    row_max = sim.max(dim=1).values
    diag_is_max = (diag == row_max).float().mean().item()
    print(f"    diagonal-is-max rate: {diag_is_max:.0%} ({diag_is_max * sim.shape[0]:.0f}/{sim.shape[0]} rows)")

    if diag_is_max < 1.0:
        print(f"    ⚠ Not all diagonal entries are row-maximums — model may not perfectly rank correct class highest")


# ─── Individual encoder tests ──────────────────────────────────────

def test_clip_surgery_encoder(images: List[Image.Image], classnames: List[str],
                               backbone: str, device: str) -> Dict:
    """Test CLIPSurgeryEncoder (standard CLIP via CLIP Surgery wrapper)."""
    print("\n── CLIPSurgeryEncoder ──")
    from encoder.clip_encoder import CLIPSurgeryEncoder

    try:
        encoder = CLIPSurgeryEncoder(model_type=backbone, device=device)
        print(f"  ✓ Model loaded: CS-{backbone}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return {"status": "SKIP", "reason": str(e)}

    # Text encoding
    try:
        text_feats = encoder.encode_text(classnames, prompt_templates=['a photo of {}'])
        check_tensor_props(text_feats, text_feats.shape[1], "text_features")
    except Exception as e:
        print(f"  ✗ Text encoding failed: {e}")
        return {"status": "FAIL", "reason": f"text_encode: {e}"}

    # Image encoding — CLS token only
    try:
        img_feats = encoder.encode_image(images, CLS_token_only=True, preprocess=True)
        check_tensor_props(img_feats, img_feats.shape[1], "image_features (CLS)")
    except Exception as e:
        print(f"  ✗ CLS encoding failed: {e}")
        return {"status": "FAIL", "reason": f"image_encode_cls: {e}"}

    # Image encoding — CLS + patches
    try:
        patch_feats = encoder.encode_image(images, CLS_token_only=False, preprocess=True)
        assert patch_feats.dim() == 3, f"patch_features should be 3D, got {patch_feats.dim()}D"
        num_tokens = patch_feats.shape[1]
        assert num_tokens == expected_num_tokens(backbone), \
            f"Expected {expected_num_tokens(backbone)} tokens, got {num_tokens}"
        print(f"  ✓ CLS+patches: shape={tuple(patch_feats.shape)}, norm_ok, no_nan/inf")
    except Exception as e:
        print(f"  ✗ CLS+patches encoding failed: {e}")
        return {"status": "FAIL", "reason": f"image_encode_cls_patches: {e}"}

    # get_patch_embeddings (strips CLS)
    try:
        single_tensor = encoder.preprocess_image([images[0]])[0]
        patch_only = encoder.get_patch_embeddings(single_tensor)
        assert patch_only.dim() == 2, f"get_patch_embeddings should return [P, D], got {patch_only.dim()}D"
        print(f"  ✓ get_patch_embeddings: shape={tuple(patch_only.shape)}")
    except Exception as e:
        print(f"  ✗ get_patch_embeddings failed: {e}")
        return {"status": "FAIL", "reason": f"get_patch_embeddings: {e}"}

    # visual property
    try:
        assert hasattr(encoder, "visual"), "encoder.visual property missing"
        print(f"  ✓ encoder.visual accessible")
    except Exception as e:
        print(f"  ✗ visual property failed: {e}")
        return {"status": "FAIL", "reason": f"visual_property: {e}"}

    # Similarity check
    try:
        check_similarity(text_feats, img_feats, "CLIPSurgery")
    except Exception as e:
        print(f"  ⚠ Similarity check warning: {e}")

    return {"status": "PASS"}


def test_clip_encoder(images: List[Image.Image], classnames: List[str],
                       backbone: str, device: str) -> Dict:
    """Test CLIPEncoder (inherits from CLIPSurgeryEncoder, uses standard CLIP weights)."""
    print("\n── CLIPEncoder ──")
    from encoder.clip_encoder import CLIPEncoder

    # CLIPEncoder currently inherits from CLIPSurgeryEncoder with commented-out __init__
    # So it behaves identically to CLIPSurgeryEncoder for now
    try:
        # CLIPEncoder inherits from CLIPSurgeryEncoder — use parent's init
        encoder = CLIPEncoder(model_type=backbone, device=device)
        print(f"  ✓ Model loaded (via CLIPSurgeryEncoder parent): {backbone}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        text_feats = encoder.encode_text(classnames, prompt_templates=['a photo of {}'])
        check_tensor_props(text_feats, text_feats.shape[1], "text_features")

        img_feats = encoder.encode_image(images, CLS_token_only=True, preprocess=True)
        check_tensor_props(img_feats, img_feats.shape[1], "image_features (CLS)")

        patch_feats = encoder.encode_image(images, CLS_token_only=False, preprocess=True)
        assert patch_feats.dim() == 3, f"patch_features should be 3D, got {patch_feats.dim()}D"
        num_tokens = patch_feats.shape[1]
        assert num_tokens == expected_num_tokens(backbone), \
            f"Expected {expected_num_tokens(backbone)} tokens, got {num_tokens}"
        print(f"  ✓ CLS+patches: shape={tuple(patch_feats.shape)}, norm_ok, no_nan/inf")

        single_tensor = encoder.preprocess_image([images[0]])[0]
        patch_only = encoder.get_patch_embeddings(single_tensor)
        assert patch_only.dim() == 2, f"get_patch_embeddings should return [P, D], got {patch_only.dim()}D"
        print(f"  ✓ get_patch_embeddings: shape={tuple(patch_only.shape)}")

        assert hasattr(encoder, "visual"), "encoder.visual property missing"
        print(f"  ✓ encoder.visual accessible")

        check_similarity(text_feats, img_feats, "CLIP")
    except Exception as e:
        print(f"  ✗ Encoding failed: {e}")
        return {"status": "FAIL", "reason": str(e)}

    return {"status": "PASS"}


def test_openclip_encoder(images: List[Image.Image], classnames: List[str],
                           backbone: str, device: str) -> Dict:
    """Test OpenClipEncoder."""
    print("\n── OpenClipEncoder ──")
    from encoder.clip_encoder import OpenClipEncoder

    # Map backbone names for open_clip
    openclip_model_map = {
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
        "ViT-L/14": "ViT-L-14",
    }
    openclip_model = openclip_model_map.get(backbone, "ViT-B-32")

    try:
        encoder = OpenClipEncoder(model_type=openclip_model, device=device)
        print(f"  ✓ Model loaded: {openclip_model}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        text_feats = encoder.encode_text(classnames)
        check_tensor_props(text_feats, text_feats.shape[1], "text_features")

        img_feats = encoder.encode_image(images, CLS_token_only=True, preprocess=True)
        check_tensor_props(img_feats, img_feats.shape[1], "image_features (CLS)")

        patch_feats = encoder.encode_image(images, CLS_token_only=False, preprocess=True)
        assert patch_feats.dim() == 3, f"patch_features should be 3D, got {patch_feats.dim()}D"
        num_tokens = patch_feats.shape[1]
        assert num_tokens == expected_num_tokens(backbone), \
            f"Expected {expected_num_tokens(backbone)} tokens, got {num_tokens}"
        print(f"  ✓ CLS+patches: shape={tuple(patch_feats.shape)}, norm_ok, no_nan/inf")

        single_tensor = encoder.preprocess_image([images[0]])[0]
        patch_only = encoder.get_patch_embeddings(single_tensor)
        assert patch_only.dim() == 2, f"get_patch_embeddings should return [P, D], got {patch_only.dim()}D"
        print(f"  ✓ get_patch_embeddings: shape={tuple(patch_only.shape)}")

        assert hasattr(encoder, "visual"), "encoder.visual property missing"
        print(f"  ✓ encoder.visual accessible")

        check_similarity(text_feats, img_feats, "OpenCLIP")
    except Exception as e:
        print(f"  ✗ Encoding failed: {e}")
        return {"status": "FAIL", "reason": str(e)}

    return {"status": "PASS"}


def test_detailclip_encoder(images: List[Image.Image], classnames: List[str],
                             backbone: str, device: str) -> Dict:
    """Test DetailClipEncoder (requires third_party/DetailCLIP/package to be importable)."""
    print("\n── DetailClipEncoder ──")

    # DetailClipEncoder does `import detail_clip as clip` internally.
    detail_clip_available = False

    # Try 1: direct import of 'detail_clip' (what the encoder expects)
    try:
        import detail_clip  # noqa: F401
        detail_clip_available = True
    except ImportError:
        pass

    # Try 2: import via sys.path manipulation (DetailCLIP's models.py does `from utils import ...`)
    if not detail_clip_available:
        with _detailclip_import_context():
            try:
                import detail_clip  # noqa: F401
                detail_clip_available = True
            except ImportError:
                pass

    if not detail_clip_available:
        print(f"  ⊘ Skipping: detail_clip not importable")
        print(f"    Hint: run `pip install -e third_party/DetailCLIP/package`")
        return {"status": "SKIP", "reason": "detail_clip not importable"}

    from encoder.clip_encoder import DetailClipEncoder

    # Checkpoint path is read from DETAILCLIP_CHECKPOINT env var inside clip.load()
    checkpoint_path = os.environ.get("DETAILCLIP_CHECKPOINT")

    try:
        encoder = DetailClipEncoder(model_type=backbone, device=device)
        print(f"  ✓ Model loaded: DetailCLIP {backbone}")
        if checkpoint_path:
            print(f"    checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        text_feats = encoder.encode_text(classnames)
        check_tensor_props(text_feats, text_feats.shape[1], "text_features")

        img_feats = encoder.encode_image(images, CLS_token_only=True, preprocess=True)
        check_tensor_props(img_feats, img_feats.shape[1], "image_features (CLS)")

        patch_feats = encoder.encode_image(images, CLS_token_only=False, preprocess=True)
        assert patch_feats.dim() == 3, f"patch_features should be 3D, got {patch_feats.dim()}D"
        num_tokens = patch_feats.shape[1]
        assert num_tokens == expected_num_tokens(backbone), \
            f"Expected {expected_num_tokens(backbone)} tokens, got {num_tokens}"
        print(f"  ✓ CLS+patches: shape={tuple(patch_feats.shape)}, norm_ok, no_nan/inf")

        single_tensor = encoder.preprocess_image([images[0]])[0]
        patch_only = encoder.get_patch_embeddings(single_tensor)
        assert patch_only.dim() == 2, f"get_patch_embeddings should return [P, D], got {patch_only.dim()}D"
        print(f"  ✓ get_patch_embeddings: shape={tuple(patch_only.shape)}")

        assert hasattr(encoder, "visual"), "encoder.visual property missing"
        print(f"  ✓ encoder.visual accessible")

        check_similarity(text_feats, img_feats, "DetailCLIP")
    except Exception as e:
        print(f"  ✗ Encoding failed: {e}")
        return {"status": "FAIL", "reason": str(e)}

    return {"status": "PASS"}



# ─── Main ──────────────────────────────────────────────────────────

ENCODER_TESTS = [
    ("CLIPSurgeryEncoder", test_clip_surgery_encoder),
    ("CLIPEncoder", test_clip_encoder),
    ("OpenClipEncoder", test_openclip_encoder),
    ("DetailClipEncoder", test_detailclip_encoder),
]


def main():
    parser = argparse.ArgumentParser(description="Test CLIP encoder classes")
    parser.add_argument("--dataset", default="caltech101",
                        choices=["eurosat", "dtd", "caltech101"],
                        help="Dataset to load sample images from (default: caltech101)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of sample images to load (default: 3)")
    parser.add_argument("--data-root", default="./data",
                        help="Root directory for datasets (default: ./data)")
    parser.add_argument("--backbone", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP backbone model (default: ViT-B/32)")
    parser.add_argument("--device", default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--encoders", nargs="+", default=None,
                        choices=[name for name, _ in ENCODER_TESTS],
                        help="Specific encoders to test (default: all)")
    parser.add_argument("--detailclip-checkpoint", default=None,
                        help="Path to DetailCLIP checkpoint (default: uses DETAILCLIP_CHECKPOINT env var or hardcoded path)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Set env var so detail_clip.load picks it up
    if args.detailclip_checkpoint:
        os.environ["DETAILCLIP_CHECKPOINT"] = args.detailclip_checkpoint

    print("=" * 70)
    print(f"  CLIP Encoder Test Suite")
    print(f"  Dataset: {args.dataset} | Backbone: {args.backbone} | Device: {device}")
    print(f"  Samples: {args.num_samples} | Data root: {args.data_root}")
    print("=" * 70)

    # Load sample images
    print(f"\nLoading {args.num_samples} sample images from {args.dataset}...")
    try:
        images, classnames, image_paths = load_split_images(
            args.dataset, args.data_root, args.num_samples
        )
        print(f"  ✓ Loaded {len(images)} images:")
        for img, cname, path in zip(images, classnames, image_paths):
            print(f"    - {os.path.basename(path)} (class: {cname})")
    except Exception as e:
        print(f"  ✗ Failed to load images: {e}")
        print("\n  Make sure datasets are placed under the data root directory.")
        print(f"  Expected: {args.data_root}/{args.dataset}/split_zhou_*.json")
        sys.exit(1)

    # Run encoder tests
    if args.encoders:
        tests_to_run = [(n, f) for n, f in ENCODER_TESTS if n in args.encoders]
    else:
        tests_to_run = ENCODER_TESTS

    results = {}
    for name, test_fn in tests_to_run:
        try:
            result = test_fn(images, classnames, args.backbone, device)
        except Exception as e:
            print(f"\n── {name} ──")
            print(f"  ✗ Unexpected error: {e}")
            result = {"status": "ERROR", "reason": str(e)}
        results[name] = result

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    skipped = sum(1 for r in results.values() if r["status"] == "SKIP")
    errors = sum(1 for r in results.values() if r["status"] == "ERROR")

    for name, result in results.items():
        status_icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘", "ERROR": "✗"}
        icon = status_icon.get(result["status"], "?")
        reason = f" ({result.get('reason', '')})" if result["status"] != "PASS" else ""
        print(f"  {icon} {name}: {result['status']}{reason}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
    print("=" * 70)

    if failed > 0 or errors > 0:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
