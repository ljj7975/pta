"""One-off feature extraction for the representation-upgrade pre-validation gates.

Encodes each dataset's test split once with the SAME encoder/preprocess pipeline
as runner.py (clip_surgery / ViT-B/16 by default) and dumps L2-normalized CLS
image features + labels to <out>/<ds>.npz.

Usage:
    python scripts/extract_features.py --datasets dtd/oxford_flowers/oxford_pets
    python scripts/extract_features.py --datasets dtd --max-images 10   # smoke test

Notes:
    - Features are seed-independent (deterministic encoding); the loader seed
      only affects ordering, which is irrelevant for the pooled gate analyses.
    - Output is fp32 numpy; features are L2-normalized (encode_image default).
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils import build_test_data_loader
from encoder import create_encoder_instance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="outputs_repr_upgrade/features")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--clip-model", default="clip_surgery")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Stop after N images per dataset (smoke testing).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip datasets whose npz already exists with the "
                         "full test-set size.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder {args.clip_model} / {args.backbone} on {device}")
    encoder = create_encoder_instance(args.clip_model, model_type=args.backbone, device=device)
    preprocess = encoder.preprocess

    for ds in args.datasets.split("/"):
        out = os.path.join(args.out, f"{ds}.npz")
        if args.skip_existing and os.path.exists(out):
            n_existing = np.load(out)["features"].shape[0]
            print(f"[{ds}] skipping (existing npz with {n_existing} features)",
                  flush=True)
            continue
        loader, classnames, _ = build_test_data_loader(
            ds, args.data_root, preprocess, shuffle=True, seed=1
        )
        feats, labels = [], []
        for j, (images, target) in enumerate(loader):
            # images: [1,3,H,W] already preprocessed by the loader;
            # encode_image moves to device + L2-normalizes (same as get_clip_logits).
            f = encoder.encode_image(images)
            feats.append(f.float().cpu().numpy())
            labels.append(int(target.item()))
            if j % 200 == 0:
                print(f"[{ds}] {j} images encoded", flush=True)
            if args.max_images is not None and len(labels) >= args.max_images:
                break
        feats = np.concatenate(feats, axis=0)
        labels = np.array(labels)
        np.savez_compressed(out, features=feats, labels=labels,
                            classnames=np.array(classnames))
        print(f"[{ds}] saved {feats.shape} (fp32, L2-norm) -> {out}", flush=True)


if __name__ == "__main__":
    main()
