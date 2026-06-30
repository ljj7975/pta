"""
Generic TTA evaluation runner.

Usage:
    python runner.py --method pta --config configs --datasets caltech101/dtd --backbone ViT-B/16

Adding a new method:
    1. Create models/<method>.py implementing BaseAdapter + a module-level build(cfg) factory.
    2. Pass --method <method> — no changes to this file required.
"""

import importlib
import os
import random
import argparse

import torch
import clip

from utils import get_config_file, build_test_data_loader, clip_classifier, get_imagenet_subset_remap


def get_arguments():
    parser = argparse.ArgumentParser(description="Generic TTA evaluation runner.")

    parser.add_argument(
        "--method",
        dest="method",
        required=True,
        help=(
            "Name of the adapter module inside models/. "
            "E.g. 'pta' loads models/pta.py and calls models.pta.build(cfg)."
        ),
    )
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="Path to the directory containing per-dataset YAML configs.",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        type=str,
        required=True,
        help=(
            "Dataset(s) to evaluate, separated by '/'. "
            "Examples: 'caltech101/dtd/eurosat'  or  'I/V/R/S/A'"
        ),
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        type=str,
        default="./data",
        help="Root directory that contains all dataset folders.",
    )
    parser.add_argument(
        "--backbone",
        dest="backbone",
        type=str,
        choices=["RN50", "ViT-B/16"],
        required=True,
        help="CLIP backbone: RN50 or ViT-B/16.",
    )
    parser.add_argument(
        "--wandb-log",
        dest="wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1,
        help="Random seed (default: 1).",
    )

    return parser.parse_args()


def load_adapter_module(method: str):
    """
    Dynamically import models.<method> and return the module.
    Raises ImportError with a helpful message if the module doesn't exist.
    """
    module_path = f"models.{method}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import adapter '{module_path}'. "
            f"Make sure models/{method}.py exists and implements BaseAdapter. "
            f"Original error: {e}"
        )


def main():
    args = get_arguments()

    # ------------------------------------------------------------------ setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------ CLIP
    print(f"Loading CLIP backbone: {args.backbone}")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # ------------------------------------------------------------------ adapter module
    adapter_module = load_adapter_module(args.method)

    # ------------------------------------------------------------------ per-dataset loop
    datasets = args.datasets.split("/")

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset : {dataset_name}")
        print(f"  Method  : {args.method}")
        print(f"{'='*60}")

        cfg = get_config_file(args.config, dataset_name)
        print("Config:", cfg)

        # Build a fresh adapter instance per dataset so running state
        # (prototypes, caches, etc.) never leaks across datasets.
        adapter = adapter_module.build(cfg)

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc = adapter.run(test_loader, clip_model, clip_weights, dataset_name)

        print(f"\n  >> [{args.method.upper()}] {dataset_name}: {acc:.2f}%\n")


if __name__ == "__main__":
    main()
