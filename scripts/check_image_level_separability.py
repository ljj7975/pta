#!/usr/bin/env python3
"""Phase 3a — cheap separability pre-check (no adapter; diagnostic-only,
uses ground-truth labels retrospectively exactly as
experimental_results/PTA_Limitations_and_Patch_Signal_Analysis.md Part 2.2
did for the patch-level bank).

Patch-level separability was proven negative everywhere (margins -0.11 to
-0.24). This checks whether the *same* clustering idea is viable at the
image (CLS) level, where zero-shot CLIP is already known to separate
classes reasonably well.

Method: one forward pass per image (feature dump, no sequential TTA state)
to get CLS embeddings + ground-truth labels for the whole test set. Per
class, run a simple deterministic batch k-means (cosine distance) for
k in {1, 2, 3}. For k >= 2, compute:
  - intra-class coherence: mean pairwise cosine similarity among a class's
    own k centers.
  - nearest cross-class confusability: for each class, the max center-to-
    center similarity against its single most similar other class.
  - separability margin = intra - nearest cross-class confusability.
(k=1 has only one center per class, so "intra-class coherence between
centers" is undefined there; we instead report the k=1 cross-class
centroid confusability as a reference baseline.)

Go/no-go (see plan): proceed to Phase 3b only if k=2 or k=3 margins are
POSITIVE (or far less negative than the patch-level -0.11 to -0.24) on
>=2 of 3 dev datasets.

Usage:
    python scripts/check_image_level_separability.py \
        --datasets dtd/oxford_flowers/oxford_pets --seed 1 \
        --out outputs/image_level_separability_report.md
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import create_encoder_instance
from utils.clip_inference import clip_classifier, _safe_normalize
from utils.data import build_test_data_loader

REGRESSION_REF = -0.24  # worst patch-level margin (Part 2.2); "far less negative" reference


def dump_embeddings(dataset_name, encoder, seed, data_root, max_batches=None):
    preprocess = encoder.preprocess
    test_loader, classnames, _template = build_test_data_loader(
        dataset_name, data_root, preprocess, shuffle=True, seed=seed
    )
    feats, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(
            tqdm(test_loader, desc=f"[embed] {dataset_name}")
        ):
            if max_batches is not None and i >= max_batches:
                break
            images = images.cuda()
            f = encoder.encode_image(images)
            f = _safe_normalize(f.float())
            feats.append(f.squeeze(0).cpu())
            labels.append(int(target.item()))
    return torch.stack(feats), torch.tensor(labels), len(classnames)


def kmeans_cosine(X, k, iters=25, seed=0):
    """Deterministic batch k-means under cosine similarity (X L2-normalized).

    Returns [k', D] centers (k' <= k if X has fewer points than k).
    """
    n = X.shape[0]
    if n == 0:
        return X.new_zeros(0, X.shape[-1] if X.dim() > 1 else 1)
    if n <= k:
        return X.clone()

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    centers = X[idx].clone()

    for _ in range(iters):
        sims = X @ centers.t()          # [n, k]
        assign = sims.argmax(dim=1)     # [n]
        new_centers = []
        for j in range(k):
            members = X[assign == j]
            if members.shape[0] == 0:
                new_centers.append(centers[j])
            else:
                c = members.mean(dim=0)
                c = _safe_normalize(c)
                new_centers.append(c)
        centers = torch.stack(new_centers)
    return centers


def compute_margins(feats, labels, num_classes, k, seed=0):
    """Per-k separability analysis. Returns dict of aggregate stats."""
    centers_by_class = {}
    for c in range(num_classes):
        mask = labels == c
        Xc = feats[mask]
        if Xc.shape[0] == 0:
            continue
        centers_by_class[c] = kmeans_cosine(Xc, k, seed=seed)

    intras = []
    crosses = []
    margins = []
    classes = sorted(centers_by_class.keys())
    for c in classes:
        own = centers_by_class[c]
        if own.shape[0] >= 2:
            sim = own @ own.t()
            mask = ~torch.eye(own.shape[0], dtype=torch.bool)
            intra = sim[mask].mean().item()
        else:
            intra = None

        best_cross = None
        for c2 in classes:
            if c2 == c:
                continue
            other = centers_by_class[c2]
            if other.shape[0] == 0:
                continue
            s = (own @ other.t()).max().item()
            if best_cross is None or s > best_cross:
                best_cross = s
        crosses.append(best_cross)
        if intra is not None and best_cross is not None:
            margins.append(intra - best_cross)
            intras.append(intra)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    return {
        "mean_intra": _mean(intras),
        "mean_nearest_cross": _mean(crosses),
        "mean_margin": _mean(margins),
        "n_classes": len(classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--clip-model", default="clip_surgery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--out", default="outputs/image_level_separability_report.md")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = create_encoder_instance(args.clip_model, model_type=args.backbone, device=device)
    ks = [int(x) for x in args.ks.split(",")]

    lines = ["# Phase 3a: Image-Level Multi-Prototype — Separability Pre-Check", ""]
    lines.append(f"Backbone: {args.backbone} | clip-model: {args.clip_model} | seed: {args.seed}")
    lines.append("")
    lines.append(
        "Reference: patch-level separability margins (Part 2.2) were "
        f"-0.11 to {REGRESSION_REF} on every dataset — the failure mode this "
        "checks whether the SAME clustering idea avoids at the image level."
    )
    lines.append("")

    pass_count = {}
    dataset_names = args.datasets.split("/")
    for dataset_name in dataset_names:
        feats, labels, num_classes = dump_embeddings(
            dataset_name, encoder, args.seed, args.data_root, max_batches=args.max_batches
        )
        lines.append(f"## {dataset_name} (N={feats.shape[0]}, C={num_classes})")
        lines.append("")
        lines.append("| k | classes | mean intra-class coherence | mean nearest cross-class | mean margin |")
        lines.append("|---|---|---|---|---|")
        for k in ks:
            r = compute_margins(feats, labels, num_classes, k, seed=args.seed)
            intra_s = f"{r['mean_intra']:.3f}" if r["mean_intra"] is not None else "N/A (k=1)"
            cross_s = f"{r['mean_nearest_cross']:.3f}" if r["mean_nearest_cross"] is not None else "N/A"
            margin_s = f"{r['mean_margin']:+.3f}" if r["mean_margin"] is not None else "N/A"
            lines.append(f"| {k} | {r['n_classes']} | {intra_s} | {cross_s} | {margin_s} |")
            if k >= 2 and r["mean_margin"] is not None:
                pass_count.setdefault(dataset_name, False)
                if r["mean_margin"] > 0:  # primary go/no-go bar: positive margin
                    pass_count[dataset_name] = True
        lines.append("")

    n_pass = sum(1 for v in pass_count.values() if v)
    n_total = len(dataset_names)
    required = min(2, n_total)
    verdict = "PROCEED to Phase 3b" if n_pass >= required else "STOP — do not build the multi-prototype adapter"
    lines.append(f"Go/no-go bar: k>=2 margin is POSITIVE on >={required}/{n_total} datasets.")
    lines.append(f"**Verdict: {verdict}** ({n_pass}/{n_total} datasets cleared the bar)")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
