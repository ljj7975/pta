#!/usr/bin/env python3
"""Phase 3a -- cheap diagnostic for the multi-view consistency trust signal
(see /home/brandon/.claude/plans/toasty-pondering-eagle.md).

For each test image (no sequential TTA state -- this is a one-pass
diagnostic, mirroring the methodology that validated the patch-vote signal
in the prior report's Part 3.1), build N views: the original plus (N-1)
copies from utils/augmentation.py:_augment_image (the same perturbation
pipeline already used to build the patch-level Gaussian bank). Get each
view's independent zero-shot top-1 prediction and compute:

    agreement = fraction of the (N-1) augmented views whose top-1 prediction
                matches the ORIGINAL (unaugmented) view's top-1 prediction

Then report the purity gap: accuracy of the original view's own zero-shot
prediction when agreement == 1.0 (all augmented views agree) vs. when
agreement < 1.0 (at least one disagrees). This is compared against the
already-validated patch-vote purity gap (+9 to +35pp, prior report Part 3.1).

Also records wall-clock throughput (samples/sec) to gauge whether a full
sequential-TTA sweep at this N is feasible on a single 16GB GPU.

Usage:
    python scripts/check_view_consistency_signal.py \
        --datasets dtd/oxford_flowers/oxford_pets --seed 1 --n-views 4 \
        --out outputs/view_consistency_diagnostic.md
"""
import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import create_encoder_instance
from utils.clip_inference import clip_classifier, _safe_normalize
from utils.data import build_test_data_loader
from utils.augmentation import _augment_image

REFERENCE_PATCH_VOTE_GAP_LOW = 9.0    # prior report Part 3.1, easy-set low end
REFERENCE_PATCH_VOTE_GAP_HIGH = 35.0  # prior report Part 3.1, hard-set high end


def run_one_dataset(dataset_name, encoder, seed, data_root, n_views, max_batches=None):
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset_name, data_root, preprocess, shuffle=True, seed=seed
    )
    text_embeddings = clip_classifier(classnames, template, encoder)  # [D, C]

    n_agree_correct = 0
    n_agree_total = 0
    n_disagree_correct = 0
    n_disagree_total = 0

    t0 = time.time()
    n_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(test_loader, desc=f"[view-consistency] {dataset_name}")):
            if max_batches is not None and i >= max_batches:
                break
            target = target.cuda()

            views = [images]
            for _ in range(n_views - 1):
                views.append(_augment_image(images))
            batched = torch.cat(views, dim=0).cuda()

            feats = encoder.encode_image(batched)
            feats = _safe_normalize(feats.float())
            logits = 100.0 * feats @ text_embeddings.float()  # [n_views, C]
            preds = logits.argmax(dim=-1)  # [n_views]

            pred0 = int(preds[0].item())
            agreement = (preds[1:] == preds[0]).float().mean().item() if n_views > 1 else 1.0
            correct0 = int(pred0 == int(target.item()))

            if agreement >= 1.0:
                n_agree_total += 1
                n_agree_correct += correct0
            else:
                n_disagree_total += 1
                n_disagree_correct += correct0
            n_samples += 1
    elapsed = time.time() - t0
    samples_per_sec = n_samples / elapsed if elapsed > 0 else float("nan")

    agree_acc = 100.0 * n_agree_correct / n_agree_total if n_agree_total else None
    disagree_acc = 100.0 * n_disagree_correct / n_disagree_total if n_disagree_total else None
    gap = (agree_acc - disagree_acc) if (agree_acc is not None and disagree_acc is not None) else None

    return {
        "n_agree_total": n_agree_total,
        "n_disagree_total": n_disagree_total,
        "agree_acc": agree_acc,
        "disagree_acc": disagree_acc,
        "gap": gap,
        "samples_per_sec": samples_per_sec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--clip-model", default="clip_surgery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="outputs/view_consistency_diagnostic.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = create_encoder_instance(args.clip_model, model_type=args.backbone, device=device)

    lines = ["# Phase 3a: Multi-View Consistency Trust Signal -- Diagnostic", ""]
    lines.append(f"Backbone: {args.backbone} | clip-model: {args.clip_model} | seed: {args.seed} | n_views: {args.n_views}")
    lines.append("")
    lines.append(
        "Reference: the already-validated confidence x patch-agreement signal has a purity gap of "
        f"+{REFERENCE_PATCH_VOTE_GAP_LOW:.0f} to +{REFERENCE_PATCH_VOTE_GAP_HIGH:.0f}pp (prior report Part 3.1). "
        "This checks whether cross-view agreement under perturbation is comparably informative."
    )
    lines.append("")
    lines.append("| Dataset | N agree | Agree Acc | N disagree | Disagree Acc | Gap | Samples/sec |")
    lines.append("|---|---|---|---|---|---|---|")

    dataset_names = args.datasets.split("/")
    n_pass = 0
    for dataset_name in dataset_names:
        r = run_one_dataset(dataset_name, encoder, args.seed, args.data_root, args.n_views, args.max_batches)
        agree_s = f"{r['agree_acc']:.2f}" if r["agree_acc"] is not None else "N/A"
        disagree_s = f"{r['disagree_acc']:.2f}" if r["disagree_acc"] is not None else "N/A"
        gap_s = f"{r['gap']:+.2f}" if r["gap"] is not None else "N/A"
        lines.append(
            f"| {dataset_name} | {r['n_agree_total']} | {agree_s} | {r['n_disagree_total']} | "
            f"{disagree_s} | {gap_s} | {r['samples_per_sec']:.1f} |"
        )
        if r["gap"] is not None and r["gap"] >= REFERENCE_PATCH_VOTE_GAP_LOW:
            n_pass += 1

    n_total = len(dataset_names)
    required = min(2, n_total)
    verdict = "PROCEED to Phase 3b" if n_pass >= required else "STOP -- do not build the view-consistency adapter"
    lines.append("")
    lines.append(
        f"Go/no-go bar: purity gap >= {REFERENCE_PATCH_VOTE_GAP_LOW:.0f}pp (the low end of the "
        f"already-validated patch-vote gap) on >= {required}/{n_total} datasets."
    )
    lines.append(f"**Verdict: {verdict}** ({n_pass}/{n_total} datasets cleared the bar)")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
