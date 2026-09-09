#!/usr/bin/env python3
"""Phase 2a -- cheap diagnostic for the prototype confusability monitor
(see /home/brandon/.claude/plans/toasty-pondering-eagle.md).

Runs the UNMODIFIED base-PTA sequential adaptation loop (same update rule as
models/pta.py + models/image_level/pta_image.py, same fusion as base PTA)
on dtd/oxford_flowers/oxford_pets. The only addition is bookkeeping: at the
end of the run, compute the [C, C] prototype self-similarity matrix from the
final prototype bank and, for each class, its nearest-other-class
similarity ("confusability"). Correlates that against final per-class
accuracy (already tracked the same way models/pta.py tracks it).

No CLIP confidence, no patch content -- purely a function of the bank's own
final geometry.

Usage:
    python scripts/check_prototype_confusability.py \
        --datasets dtd/oxford_flowers/oxford_pets --seed 1 \
        --out outputs/prototype_confusability_report.md
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import create_encoder_instance
from utils.clip_inference import clip_classifier, get_clip_logits, _safe_normalize
from utils.data import build_test_data_loader
from utils import cls_acc
from models.image_level.pta_image import PTAImageLevel
from models.fusion import WeightedFusion


def run_one_dataset(dataset_name, encoder, seed, data_root, max_batches=None):
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset_name, data_root, preprocess, shuffle=True, seed=seed
    )
    text_embeddings = clip_classifier(classnames, template, encoder)  # [D, C]
    C = text_embeddings.shape[1]

    image_level_cfg = {"image_level": {"alpha": 0.01, "T": 20.0}}
    image_level = PTAImageLevel(image_level_cfg["image_level"])
    fusion = WeightedFusion({"fusion": {"tau_text": 1.0, "tau_image_proto": 100.0, "tau_patch_proto": 0.0}})

    refine_feature = text_embeddings.t()  # [C, D]
    prototype_state = image_level.init_state(refine_feature)

    cls_total = [0] * C
    cls_correct = [0] * C

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(test_loader, desc=f"[confusability] {dataset_name}")):
            if max_batches is not None and i >= max_batches:
                break
            image_features, clip_logits, _, _, _ = get_clip_logits(images, encoder, text_embeddings)
            target = target.cuda()

            refine_feature, prototype_state = image_level.update_prototypes(
                image_features, clip_logits, refine_feature, prototype_state
            )
            image_proto_logits = image_level.compute_logits(image_features, refine_feature)
            final_logits = fusion.forward(clip_logits.clone(), image_proto_logits, None)

            acc = cls_acc(final_logits, target)
            t = int(target.item())
            cls_total[t] += 1
            if acc:
                cls_correct[t] += 1

    # ── Final-state confusability (no CLIP calls, just the bank itself) ────
    proto_norm = _safe_normalize(prototype_state, dim=-1)  # [C, D]
    sims = proto_norm @ proto_norm.t()  # [C, C]
    sims.fill_diagonal_(-1.0)
    nearest_other, _ = sims.max(dim=1)  # [C]

    per_class = []
    for c in range(C):
        acc_c = (100.0 * cls_correct[c] / cls_total[c]) if cls_total[c] else None
        per_class.append({
            "class": c,
            "confusability": nearest_other[c].item(),
            "acc": acc_c,
            "n": cls_total[c],
        })
    return per_class


def tercile_gap(per_class):
    valid = [p for p in per_class if p["acc"] is not None and p["n"] > 0]
    if len(valid) < 6:
        return None
    ordered = sorted(valid, key=lambda p: p["confusability"])
    k = len(ordered) // 3
    low_conf = ordered[:k]         # least confusable (safest) classes
    high_conf = ordered[-k:]       # most confusable classes
    low_acc = sum(p["acc"] for p in low_conf) / len(low_conf)
    high_acc = sum(p["acc"] for p in high_conf) / len(high_conf)
    return low_acc, high_acc, low_acc - high_acc


def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--clip-model", default="clip_surgery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="outputs/prototype_confusability_report.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = create_encoder_instance(args.clip_model, model_type=args.backbone, device=device)

    lines = ["# Phase 2a: Prototype Confusability Monitor -- Diagnostic", ""]
    lines.append(f"Backbone: {args.backbone} | clip-model: {args.clip_model} | seed: {args.seed}")
    lines.append("")
    lines.append(
        "Runs the UNMODIFIED base-PTA update loop; only addition is a final "
        "[C,C] prototype self-similarity matrix computed from the finished bank. "
        "Checks whether classes whose final prototype is close to another class's "
        "prototype (\"confusable\") actually have lower accuracy."
    )
    lines.append("")

    dataset_names = args.datasets.split("/")
    pass_count = 0
    for dataset_name in dataset_names:
        per_class = run_one_dataset(dataset_name, encoder, args.seed, args.data_root, args.max_batches)
        valid = [p for p in per_class if p["n"] > 0]
        xs = [p["confusability"] for p in valid]
        ys = [p["acc"] for p in valid]
        corr = pearson_corr(xs, ys)
        gap = tercile_gap(per_class)

        lines.append(f"## {dataset_name} (C={len(per_class)})")
        lines.append("")
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        pctl = lambda p: xs_sorted[min(n - 1, int(p * n))]
        lines.append(
            f"Confusability distribution: min={xs_sorted[0]:.3f} p25={pctl(0.25):.3f} "
            f"median={pctl(0.5):.3f} p75={pctl(0.75):.3f} max={xs_sorted[-1]:.3f}"
        )
        lines.append("")
        lines.append(f"Pearson correlation(confusability, per-class accuracy): {corr:+.3f}" if corr is not None else "Pearson correlation: N/A")
        lines.append("")
        if gap is not None:
            low_acc, high_acc, delta = gap
            lines.append(
                f"Low-confusability tercile acc: {low_acc:.2f} | "
                f"High-confusability tercile acc: {high_acc:.2f} | "
                f"Gap (low - high): {delta:+.2f}pp"
            )
            if delta >= 5.0:
                pass_count += 1
        else:
            lines.append("Tercile gap: N/A (too few classes)")
        lines.append("")

    n_total = len(dataset_names)
    required = min(2, n_total)
    verdict = "PROCEED to Phase 2b" if pass_count >= required else "STOP -- do not build the confusability-gated adapter"
    lines.append(f"Go/no-go bar: low-vs-high-confusability tercile accuracy gap >= 5pp on >= {required}/{n_total} datasets.")
    lines.append(f"**Verdict: {verdict}** ({pass_count}/{n_total} datasets cleared the bar)")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
