#!/usr/bin/env python3
"""Phase 2a — cheap diagnostic (no adapter, no TTA loop, no ground-truth use
beyond final accuracy reporting).

Checks whether a CLIP-Surgery relevance-weighted patch pooling is even
competitive with the CLS token in isolation, before committing to building
the full Phase 2b adapter (models/surgery_embed_pta.py). Mirrors the
"standalone accuracy" diagnostic methodology already used for the patch vote
(experimental_results/PTA_Limitations_and_Patch_Signal_Analysis.md Part 3.2).

For each image (single forward pass, no sequential TTA state):
  1. CLS embedding + zero-shot CLIP prediction (standard).
  2. Per-patch, per-class CLIP-Surgery relevance via
     utils.clip_inference.compute_surgery_scores (causal: uses only the
     CLIP-predicted top-1 class's relevance column, no label leakage).
  3. fg_embedding = normalize(sum_i relu(relevance_i[top1]) * patch_i
                               / sum_i relu(relevance_i[top1]))
  4. Standalone zero-shot accuracy of fg_embedding alone, vs. CLS zero-shot
     accuracy, plus mean cosine(fg_embedding, cls_embedding).

Go/no-go (see plan): proceed to Phase 2b only if fg_embedding standalone
accuracy is within ~10pp of CLS zero-shot on >=2 of 3 dev datasets.

Usage:
    python scripts/check_surgery_embedding.py --datasets dtd/oxford_flowers/oxford_pets \
        --backbone ViT-B/16 --seed 1 --out outputs/surgery_embedding_diagnostic.md
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import create_encoder_instance
from utils.clip_inference import clip_classifier, get_clip_logits, compute_surgery_scores, _safe_normalize
from utils.data import build_test_data_loader
from utils.metrics import cls_acc

GAIN_BAR_PP = 10.0  # go/no-go: fg standalone acc must be within this of CLS zero-shot


def run_one_dataset(dataset_name, encoder, backbone, seed, data_root, max_batches=None):
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset_name, data_root, preprocess, shuffle=True, seed=seed
    )
    text_embeddings = clip_classifier(classnames, template, encoder)          # [D, C]
    text_features_cd = _safe_normalize(text_embeddings.t().float())            # [C, D]

    tokens_empty = None
    empty_text_feat = encoder.encode_text([""])
    empty_text_feat = _safe_normalize(empty_text_feat.float())

    n = 0
    cls_acc_sum = 0.0
    fg_acc_sum = 0.0
    cos_sum = 0.0

    with torch.no_grad():
        for i, (images, target) in enumerate(
            tqdm(test_loader, desc=f"[diag] {dataset_name}")
        ):
            if max_batches is not None and i >= max_batches:
                break
            target = target.cuda()

            image_features, clip_logits, _, _, _ = get_clip_logits(
                images, encoder, text_embeddings
            )
            top1 = int(clip_logits.argmax(dim=-1).item())

            scores = compute_surgery_scores(
                images if images.dim() == 4 else images.unsqueeze(0),
                encoder, text_features_cd, empty_text_feat,
                filter_mode="surgery_with_labels",
            )  # [P, C] or None

            patches = _safe_normalize(
                encoder.get_patch_embeddings(images).float()
            )  # [P, D]

            if scores is None:
                fg_embedding = image_features.float()
            else:
                rel = scores[:, top1].clamp(min=0.0)  # relu
                denom = rel.sum()
                if float(denom.item()) < 1e-8:
                    fg_embedding = image_features.float()
                else:
                    fg_embedding = (rel.unsqueeze(-1) * patches).sum(dim=0, keepdim=True) / denom
                    fg_embedding = _safe_normalize(fg_embedding)

            fg_logits = 100.0 * fg_embedding.float() @ text_embeddings.float()

            n += 1
            cls_acc_sum += cls_acc(clip_logits, target)
            fg_acc_sum += cls_acc(fg_logits, target)
            cos_sum += float(
                (fg_embedding.float() @ image_features.float().t()).item()
            )

    return {
        "n": n,
        "cls_acc": cls_acc_sum / n if n else None,
        "fg_acc": fg_acc_sum / n if n else None,
        "mean_cos_fg_cls": cos_sum / n if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--clip-model", default="clip_surgery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="outputs/surgery_embedding_diagnostic.md")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = create_encoder_instance(args.clip_model, model_type=args.backbone, device=device)

    results = {}
    for dataset_name in args.datasets.split("/"):
        results[dataset_name] = run_one_dataset(
            dataset_name, encoder, args.backbone, args.seed, args.data_root,
            max_batches=args.max_batches,
        )

    lines = ["# Phase 2a: Foreground-Weighted Embedding — Cheap Diagnostic", ""]
    lines.append(f"Backbone: {args.backbone} | clip-model: {args.clip_model} | seed: {args.seed}")
    lines.append("")
    lines.append("| Dataset | N | CLS zero-shot acc | fg_embedding standalone acc | Δ (fg - cls) | mean cos(fg, cls) |")
    lines.append("|---|---|---|---|---|---|")
    n_pass = 0
    n_total = 0
    for ds, r in results.items():
        if r["cls_acc"] is None:
            continue
        n_total += 1
        delta = r["fg_acc"] - r["cls_acc"]
        passed = delta >= -GAIN_BAR_PP
        n_pass += int(passed)
        lines.append(
            f"| {ds} | {r['n']} | {r['cls_acc']:.2f}% | {r['fg_acc']:.2f}% "
            f"| {delta:+.2f}pp | {r['mean_cos_fg_cls']:.3f} |"
        )
    lines.append("")
    required = min(2, n_total)
    verdict = "PROCEED to Phase 2b" if n_pass >= required else "STOP — do not build Phase 2b adapter"
    lines.append(f"Go/no-go bar: fg standalone acc within {GAIN_BAR_PP}pp of CLS zero-shot on >={required}/{n_total} datasets.")
    lines.append(f"**Verdict: {verdict}** ({n_pass}/{n_total} datasets cleared the bar)")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
