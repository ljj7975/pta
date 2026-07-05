#!/usr/bin/env python3
"""Fidelity snapshot capture script for PTA.

Captures per-sample accuracy from a TTA method for post-refactoring
bit-exact verification. Monkey-patches utils.cls_acc before calling
the adapter's run().
"""

import argparse
import importlib
import json
import os
import random
import sys

# Ensure the repo root is on sys.path so that 'utils' and 'models' are
# importable regardless of how this script is invoked.
try:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _REPO_ROOT = os.getcwd()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import get_config_file


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Capture fidelity snapshot for bit-exact verification.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Adapter method name (e.g. pta, exp12_patch_quality_modulation).",
    )
    parser.add_argument(
        "--config",
        default="configs",
        help="Path to config directory (default: configs).",
    )
    parser.add_argument(
        "--dataset",
        default="oxford_pets",
        help="Dataset name (default: oxford_pets).",
    )
    parser.add_argument(
        "--backbone",
        default="ViT-B/16",
        choices=["RN50", "ViT-B/16"],
        help="CLIP backbone (default: ViT-B/16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Max batches to process (default: 50).",
    )
    parser.add_argument(
        "--output",
        default="tests/fidelity_snapshots/",
        help="Output directory for snapshots (default: tests/fidelity_snapshots/).",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root directory containing datasets (default: ./data).",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    import numpy as np
    import torch
    import clip

    from utils import build_test_data_loader, clip_classifier

    # ------------------------------------------------------------------ seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # ------------------------------------------------------------------ env
    os.environ["MAX_BATCHES"] = str(args.max_batches)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    dataset_name = args.dataset

    # ------------------------------------------------------------------ config
    cfg = get_config_file(args.config, dataset_name)
    if not isinstance(cfg, dict):
        cfg = {}

    # ------------------------------------------------------------------ CLIP
    print(f"Loading CLIP backbone: {args.backbone}")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # ------------------------------------------------------------------ adapter
    module_path = f"models.{args.method}"
    adapter_module = importlib.import_module(module_path)
    adapter = adapter_module.build(cfg)

    # ------------------------------------------------------------------ data
    test_loader, classnames, template = build_test_data_loader(
        dataset_name, args.data_root, preprocess, shuffle=True,
    )
    clip_weights = clip_classifier(classnames, template, clip_model)

    # ------------------------------------------------------------------ monkey-patch cls_acc
    from utils import cls_acc as _original_cls_acc
    import utils as _utils_mod

    # Also need to patch every adapter module that imports cls_acc via
    # "from utils import cls_acc", since those modules have their own
    # reference independent from _utils_mod.cls_acc.
    _adapter_mod = adapter_module

    per_sample_accuracies = []

    def patched_cls_acc(output, target, topk=1):
        result = _original_cls_acc(output, target, topk=topk)
        per_sample_accuracies.append(float(result))
        return result

    _utils_mod.cls_acc = patched_cls_acc
    if hasattr(_adapter_mod, "cls_acc"):
        _adapter_mod.cls_acc = patched_cls_acc

    # Also patch the function directly in the adapter's globals if possible
    if hasattr(adapter, "run"):
        run_globals = adapter.run.__globals__
        if "cls_acc" in run_globals:
            run_globals["cls_acc"] = patched_cls_acc

    # Limit the test loader to max_batches batches
    def limited_loader(loader, max_batches):
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            yield batch

    limited_test_loader = limited_loader(test_loader, args.max_batches)

    # ------------------------------------------------------------------ run
    try:
        adapter.run(limited_test_loader, clip_model, clip_weights, dataset_name)
    finally:
        _utils_mod.cls_acc = _original_cls_acc
        if hasattr(_adapter_mod, "cls_acc"):
            _adapter_mod.cls_acc = _original_cls_acc
        if hasattr(adapter, "run") and "cls_acc" in adapter.run.__globals__:
            adapter.run.__globals__["cls_acc"] = _original_cls_acc

    # ------------------------------------------------------------------ save
    os.makedirs(args.output, exist_ok=True)
    fname = f"{args.method}_{dataset_name}.json"
    out_path = os.path.join(args.output, fname)
    snapshot = {
        "method": args.method,
        "dataset": dataset_name,
        "backbone": args.backbone,
        "seed": args.seed,
        "max_batches": args.max_batches,
        "per_sample_accuracies": per_sample_accuracies,
    }
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Snapshot saved to {out_path}")


if __name__ == "__main__":
    main()
