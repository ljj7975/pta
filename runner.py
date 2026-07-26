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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from utils import get_config_file, build_test_data_loader, clip_classifier, get_imagenet_subset_remap
from encoder import create_encoder_instance


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
        "--clip-model",
        dest="clip_model",
        type=str,
        choices=["clip", "clip_surgery", "detail-clip"],
        default="clip_surgery",
        help="Encoder type (default: clip_surgery).",
    )
    parser.add_argument(
        "--clip-checkpoint",
        dest="clip_checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required for detail-clip).",
    )
    parser.add_argument(
        "--backbone",
        dest="backbone",
        type=str,
        default="ViT-B/16",
        help="Vision backbone name passed to the encoder (default: ViT-B/16).",
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
    parser.add_argument(
        "--override",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one or more config values after loading the YAML. "
            "Supports dotted paths for nested keys. "
            "Values are auto-cast to float/int when possible. "
            "Example: --override conf_threshold=0.3 conf_margin_threshold=0.1"
        ),
    )

    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply key=value override strings to a config dict.

    Supports dotted paths (e.g. ``patch_level.conf_threshold=0.3``).
    Values are auto-cast to float or int when possible; otherwise kept as str.
    """
    for item in overrides:
        key, _, raw = item.partition("=")
        # Auto-cast value
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        # Walk dotted path
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------ Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder: {args.clip_model} / {args.backbone}")

    encoder_kwargs = {"model_type": args.backbone, "device": device}
    if args.clip_checkpoint and args.clip_model == "detail-clip":
        os.environ["DETAILCLIP_CHECKPOINT"] = args.clip_checkpoint

    encoder = create_encoder_instance(args.clip_model, **encoder_kwargs)
    preprocess = encoder.preprocess

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
        if args.overrides:
            cfg = apply_overrides(cfg, args.overrides)
        print("Config:", cfg)

        # Build a fresh adapter instance per dataset so running state
        # (prototypes, caches, etc.) never leaks across datasets.
        adapter = adapter_module.build(cfg)

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, shuffle=True
        )
        text_embeddings = clip_classifier(classnames, template, encoder)

        acc = adapter.run(test_loader, encoder, text_embeddings, dataset_name)

        print(f"\n  >> [{args.method.upper()}] {dataset_name}: {acc:.2f}%\n")


if __name__ == "__main__":
    main()
