#!/usr/bin/env python3
"""Debug script: check whether data order and image features are deterministic."""

import os
import sys
import random
import numpy as np

# Must be set BEFORE torch import for deterministic CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import clip
from utils import get_config_file, build_test_data_loader, clip_classifier

SEED = 42
DATASET = "dtd"
BACKBONE = "ViT-B/16"
CONFIG_DIR = "configs"


def set_full_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def check_data_order():
    """Check if data loader returns same order across two instantiations."""
    print("=== DATA ORDER CHECK ===")
    set_full_seed(SEED)

    clip_model, preprocess = clip.load(BACKBONE)
    clip_model.eval()

    # First loader
    loader1, classnames, template = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=True
    )
    labels1 = []
    for images, target in loader1:
        labels1.append(target.item())
        if len(labels1) >= 20:
            break

    # Second loader
    set_full_seed(SEED)
    loader2, _, _ = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=True
    )
    labels2 = []
    for images, target in loader2:
        labels2.append(target.item())
        if len(labels2) >= 20:
            break

    print(f"  Run 1 first 20 labels: {labels1}")
    print(f"  Run 2 first 20 labels: {labels2}")
    print(f"  Labels match: {labels1 == labels2}")
    return labels1 == labels2


def check_image_features():
    """Check if CLIP image features are deterministic."""
    print("\n=== IMAGE FEATURE CHECK ===")
    set_full_seed(SEED)

    clip_model, preprocess = clip.load(BACKBONE)
    clip_model.eval()

    loader, classnames, template = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=True
    )
    clip_weights = clip_classifier(classnames, template, clip_model)

    # Get first 5 image features
    features1 = []
    for images, target in loader:
        images = images.cuda()
        with torch.no_grad():
            feat = clip_model.encode_image(images)
            feat /= feat.norm(dim=-1, keepdim=True)
        features1.append(feat.cpu().numpy())
        if len(features1) >= 5:
            break

    # Reset and repeat
    set_full_seed(SEED)
    clip_model2, preprocess2 = clip.load(BACKBONE)
    clip_model2.eval()

    loader2, _, _ = build_test_data_loader(
        DATASET, "./data", preprocess2, shuffle=True
    )

    features2 = []
    for images, target in loader2:
        images = images.cuda()
        with torch.no_grad():
            feat = clip_model2.encode_image(images)
            feat /= feat.norm(dim=-1, keepdim=True)
        features2.append(feat.cpu().numpy())
        if len(features2) >= 5:
            break

    all_match = True
    for i in range(5):
        match = np.allclose(features1[i], features2[i], atol=1e-6)
        if not match:
            diff = np.abs(features1[i] - features2[i]).max()
            print(f"  Feature {i}: MISMATCH (max diff={diff:.2e})")
            all_match = False

    if all_match:
        print("  All 5 image features match perfectly")
    return all_match


def check_logits():
    """Check if CLIP logits are deterministic."""
    print("\n=== LOGITS CHECK ===")
    set_full_seed(SEED)

    clip_model, preprocess = clip.load(BACKBONE)
    clip_model.eval()

    loader, classnames, template = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=True
    )
    clip_weights = clip_classifier(classnames, template, clip_model)

    # Get first 5 logits
    logits1 = []
    for images, target in loader:
        images = images.cuda()
        with torch.no_grad():
            feat = clip_model.encode_image(images)
            feat /= feat.norm(dim=-1, keepdim=True)
        logit = 100.0 * feat @ clip_weights
        logits1.append(logit.cpu().numpy())
        if len(logits1) >= 5:
            break

    # Reset and repeat
    set_full_seed(SEED)
    clip_model2, preprocess2 = clip.load(BACKBONE)
    clip_model2.eval()

    loader2, _, _ = build_test_data_loader(
        DATASET, "./data", preprocess2, shuffle=True
    )
    clip_weights2 = clip_classifier(classnames, template, clip_model2)

    logits2 = []
    for images, target in loader2:
        images = images.cuda()
        with torch.no_grad():
            feat = clip_model2.encode_image(images)
            feat /= feat.norm(dim=-1, keepdim=True)
        logit = 100.0 * feat @ clip_weights2
        logits2.append(logit.cpu().numpy())
        if len(logits2) >= 5:
            break

    all_match = True
    for i in range(5):
        match = np.allclose(logits1[i], logits2[i], atol=1e-4)
        if not match:
            diff = np.abs(logits1[i] - logits2[i]).max()
            print(f"  Logits {i}: MISMATCH (max diff={diff:.2e})")
            all_match = False

    if all_match:
        print("  All 5 logits match perfectly")
    return all_match


if __name__ == "__main__":
    print(f"Debug determinism for {DATASET} ({BACKBONE}), seed={SEED}")
    print("=" * 60)

    data_ok = check_data_order()
    feat_ok = check_image_features()
    logit_ok = check_logits()

    print("\n" + "=" * 60)
    print(f"  Data order:     {'PASS' if data_ok else 'FAIL'}")
    print(f"  Image features: {'PASS' if feat_ok else 'FAIL'}")
    print(f"  Logits:         {'PASS' if logit_ok else 'FAIL'}")
    print("=" * 60)
