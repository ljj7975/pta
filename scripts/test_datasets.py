#!/usr/bin/env python3
"""
Smoke-test: verify that the 7 CD benchmark datasets are correctly laid out and
that images can be read from disk.  Runs entirely on CPU, no GPU required.

Modes
-----
Data-only (default, fast ~10 s):
    python scripts/test_datasets.py

Full pipeline (loads CLIP on CPU, slow ~10-20 min for ViT-B/16):
    python scripts/test_datasets.py --backbone ViT-B/16

Options
-------
--data-root DIR     Root that contains caltech-101/, dtd/, etc.  (default: ./data)
--n-samples N       Images to load per dataset before declaring PASS. (default: 5)
--backbone NAME     If given, load CLIP and run encode_image on each sample.
--datasets A/B/C    Slash-separated subset to test. Default: all 7.
"""
import argparse
import os
import sys
import traceback

# Allow imports from the project root regardless of where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T

# Patch DataLoader to use num_workers=0 so this script works on machines where
# torch worker processes can't import numpy (e.g. numpy 2.x / torch 1.x mismatch).
_orig_dataloader_init = torch.utils.data.DataLoader.__init__
def _patched_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["pin_memory"] = False
    _orig_dataloader_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _patched_init
from torchvision.transforms.functional import InterpolationMode

from utils import build_test_data_loader

ALL_DATASETS = [
    "caltech101", "dtd", "eurosat", "fgvc",
    "oxford_flowers", "oxford_pets", "ucf101",
]

# CLIP ViT-B/16 preprocessing — hardcoded so we can run data-only checks
# without loading the full model.
_CLIP_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data",
                   help="Root directory that contains all dataset folders.")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of samples to successfully load before marking a dataset PASS.")
    p.add_argument("--backbone", choices=["ViT-B/16", "RN50"], default=None,
                   help="Load CLIP and run the image encoder on each sample (slow on CPU).")
    p.add_argument("--datasets", default=None,
                   help="Slash-separated list of datasets to test. Default: all 7.")
    return p.parse_args()


def main():
    args = parse_args()

    datasets = args.datasets.split("/") if args.datasets else ALL_DATASETS
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}", file=sys.stderr)
        sys.exit(1)

    device = "cpu"
    clip_model = None

    if args.backbone:
        import clip as clip_lib
        print(f"Loading CLIP {args.backbone} on CPU (may take ~30 s and download weights)…")
        clip_model, preprocess = clip_lib.load(args.backbone, device=device)
        clip_model.eval()
        print("CLIP loaded.\n")
    else:
        preprocess = _CLIP_TRANSFORM

    mode = f"CLIP {args.backbone} on CPU" if clip_model else "data-only (no CLIP)"
    print(f"Mode     : {mode}")
    print(f"Data root: {args.data_root}")
    print(f"Datasets : {', '.join(datasets)}")
    print(f"Samples  : {args.n_samples} per dataset")
    print()

    results = {}

    for ds in datasets:
        print(f"  {ds} … ", end="", flush=True)
        try:
            loader, classnames, _ = build_test_data_loader(ds, args.data_root, preprocess)

            n_loaded = 0
            for images, targets in loader:
                if clip_model is not None:
                    with torch.no_grad():
                        feats = clip_model.encode_image(images.to(device))
                    assert feats.ndim == 2 and feats.shape[0] == images.shape[0], \
                        f"unexpected feature shape {feats.shape}"
                else:
                    # Sanity-check tensor shape: (B, C, H, W)
                    assert images.ndim == 4 and images.shape[1] == 3, \
                        f"unexpected image shape {images.shape}"
                n_loaded += images.shape[0]
                if n_loaded >= args.n_samples:
                    break

            results[ds] = ("PASS", f"{n_loaded} samples OK, {len(classnames)} classes")

        except Exception:
            tb = traceback.format_exc().strip().splitlines()
            short = tb[-1]  # last line is the actual error
            results[ds] = ("FAIL", short)

    # Summary
    print()
    print("=" * 62)
    print(f"  {'Dataset':<22} {'Status':<6}  Detail")
    print("=" * 62)
    for ds in datasets:
        status, detail = results[ds]
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {ds:<21} {status:<6}  {detail}")
    print("=" * 62)

    failed = [ds for ds, (s, _) in results.items() if s == "FAIL"]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll datasets passed.")


if __name__ == "__main__":
    main()
