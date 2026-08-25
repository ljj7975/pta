#!/usr/bin/env python3
"""Experiment I -- compare patch-level aggregation signals for the H-style
corroboration vote.

See outputs/patch_fix_experiments_plan.md ("Experiment I"). Experiment H
(scripts/frozen_bank_eval_v2.py) used a single pooling choice -- mean cosine
similarity across all patches -- to build a second, bank-free "patch vote"
that corroborates CLIP's own (CLS-pooled) prediction. Mean pooling can be
diluted by background patches, especially when the object of interest is
small relative to the image; a max-style pooling answers a conceptually
different question ("does *any* patch look like class c" instead of "does
the *average* patch look like class c"). This compares several different
pooling choices -- including mean, for direct comparison -- against the same
three diagnostics H used (agreement rate, CLS-accuracy-by-agreement,
write-purity-by-agreement), plus a new standalone-accuracy check and
rescue/corrupt rates.

All variants come from ONE CLIP Surgery forward pass per image (patch
embeddings already extracted for the patch-level branch); the two
surgery-relevance variants call compute_surgery_scores() once more on the
same original, unaugmented image -- still one model, no augmentation,
consistent with the "single model, original image, any number of calls"
constraint used throughout this investigation.

Usage::

    python scripts/analyze_patch_vote_aggregation.py --dataset dtd
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import clip as _clip  # noqa: E402 -- local vendored CLIP, for tokenize()
from encoder import create_encoder_instance  # noqa: E402
from utils import build_test_data_loader  # noqa: E402
from utils.clip_inference import (  # noqa: E402
    _safe_normalize,
    clip_classifier,
    compute_surgery_scores,
    _identify_relevant_patches,
)
from utils.patch_vote import topk_pool, _otsu_threshold_1d, otsu_mean  # noqa: E402,F401
from frozen_bank_eval import (  # noqa: E402
    MATCH_THRESHOLD,
    WRITE_CONF_THRESHOLD,
    BACKBONE,
    CLIP_MODEL,
    SEED,
    load_bank,
    load_stored_records,
)

FILTER_THRESHOLD = 0.5  # matches GaussianPatchLevel's default patch_filter_threshold
# Raw cosine-similarity pooled scores cluster tightly (~0.15-0.35), so a
# softmax without temperature scaling is near-uniform and the resulting
# margin is uninformative. Match CLIP's own convention (get_clip_logits
# multiplies by 100 before softmax) purely for the margin metric -- doesn't
# change any variant's argmax/pred, which is scale-invariant.
MARGIN_LOGIT_SCALE = 100.0


# ---------------------------------------------------------------------------
# Pooling variants -- each is importable standalone (Experiment J reuses
# these directly) and operates on sims [P, C] (raw patch-to-text cosine) and,
# for the two surgery variants, surgery_scores [P, C] (CLIP Surgery's
# per-patch, per-class relevance map from the same forward pass).
# ---------------------------------------------------------------------------

def mean_pool(sims):
    return sims.mean(dim=0)


def max_pool(sims):
    return sims.max(dim=0).values


def power_mean_pool(sims, p):
    """Generalized mean: p=1 -> mean, p->inf -> max. Similarities clamped to
    >=0 before the power (a negative cosine isn't "anti-evidence" in a way a
    fractional power handles sensibly)."""
    pos = sims.clamp(min=0)
    return (pos.pow(p).mean(dim=0) + 1e-12).pow(1.0 / p)


def surgery_masked_mean(sims, surgery_scores):
    """For each class, keep only the patches CLIP Surgery's own relevance
    map calls foreground for that class (same _identify_relevant_patches
    threshold logic gaussian_patch.py already uses at write time), then
    mean-pool raw cosine similarity over just those patches."""
    C = sims.shape[1]
    out = torch.zeros(C, device=sims.device)
    for c in range(C):
        mask = _identify_relevant_patches(surgery_scores[:, c], threshold=FILTER_THRESHOLD)
        out[c] = sims[mask, c].mean()
    return out


def surgery_score_mean(surgery_scores):
    return surgery_scores.mean(dim=0)


VARIANTS = [
    ("mean", lambda sims, ss: mean_pool(sims)),
    ("max", lambda sims, ss: max_pool(sims)),
    ("topk3", lambda sims, ss: topk_pool(sims, 3)),
    ("topk5", lambda sims, ss: topk_pool(sims, 5)),
    ("topk10", lambda sims, ss: topk_pool(sims, 10)),
    ("topk20", lambda sims, ss: topk_pool(sims, 20)),
    ("pmean2", lambda sims, ss: power_mean_pool(sims, 2)),
    ("pmean4", lambda sims, ss: power_mean_pool(sims, 4)),
    ("pmean8", lambda sims, ss: power_mean_pool(sims, 8)),
    ("pmean16", lambda sims, ss: power_mean_pool(sims, 16)),
    ("surgery_masked_mean", lambda sims, ss: surgery_masked_mean(sims, ss)),
    ("surgery_score_mean", lambda sims, ss: surgery_score_mean(ss)),
    ("otsu_mean", lambda sims, ss: otsu_mean(sims)),
]


def _new_bucket():
    return {"n": 0, "correct": 0}


def _rate(bucket):
    return 100.0 * bucket["correct"] / bucket["n"] if bucket["n"] else None


def run_dataset(dataset, dumps_dir, device, limit=0):
    bank = load_bank(Path(dumps_dir) / f"{dataset}-s1.pt")
    stored = load_stored_records(dataset)
    C = len(bank["classes"])

    per_class_centers = []
    for cls in bank["classes"]:
        centers = cls["centers"]
        if centers.shape[0] == 0:
            per_class_centers.append(None)
            continue
        per_class_centers.append(_safe_normalize(centers).to(device))

    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device=device)
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset, "./data", preprocess, shuffle=True, seed=SEED,
    )
    text_embeddings = clip_classifier(classnames, template, encoder).to(device)  # [D, C]
    text_features = _safe_normalize(text_embeddings.t().float())  # [C, D], for surgery scoring

    tokens = _clip.tokenize([""]).to(device)
    raw_model = encoder.model if hasattr(encoder, "model") else encoder
    with torch.no_grad():
        empty_feat = raw_model.encode_text(tokens).float()
        empty_feat = _safe_normalize(empty_feat)

    stats = {
        name: {
            "n_correct": 0, "n_total": 0,
            "agree_total": 0, "agree_correct": 0,
            "disagree_total": 0, "disagree_correct": 0,
            "write_agree": _new_bucket(), "write_disagree": _new_bucket(),
            "n_cls_wrong": 0, "rescue": 0,
            "n_cls_right": 0, "corrupt": 0,
            "margin_sum": 0.0, "margin_agree_sum": 0.0, "margin_disagree_sum": 0.0,
        }
        for name, _ in VARIANTS
    }

    n_total = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            if limit and i >= limit:
                break
            if i not in stored:
                continue
            rec = stored[i]
            target_i = int(rec["target"])
            clip_logits = rec["clip"].to(device)
            cls_pred = int(clip_logits.argmax().item())

            images = images.to(device) if not isinstance(images, list) else torch.cat(images, dim=0).to(device)
            patch_embs = encoder.get_patch_embeddings(images, exclude_pos=False)  # [P, D]
            patches_norm = _safe_normalize(patch_embs.float())
            sims = patches_norm @ text_embeddings  # [P, C]

            surgery_scores = compute_surgery_scores(
                images, encoder, text_features, empty_feat, "surgery_no_labels"
            )  # [P, C]

            # --- Variant-independent: which bank clusters this image's
            # patches touch for each confidently-written class (replays the
            # same multi_gate write rule and match_threshold as Experiment
            # B/H). One (class, n_touched, correct_write) event per class.
            write_probs = F.softmax(clip_logits, dim=-1)
            written_classes = (write_probs > WRITE_CONF_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
            write_events = []
            for c in written_classes:
                centers_c = per_class_centers[c]
                if centers_c is None:
                    continue
                csims = patches_norm @ centers_c.T
                best_val, best_idx = csims.max(dim=1)
                touched = torch.unique(best_idx[best_val >= MATCH_THRESHOLD])
                n_touched = touched.numel()
                if n_touched == 0:
                    continue
                write_events.append((c, n_touched, target_i == c))

            # --- Per-variant predictions + margins from the same sims/surgery_scores ---
            for name, pool_fn in VARIANTS:
                scores = pool_fn(sims, surgery_scores)
                probs = F.softmax(scores * MARGIN_LOGIT_SCALE, dim=-1)
                top2 = probs.topk(min(2, probs.shape[0]))
                margin = float((top2.values[0] - top2.values[-1]).item())
                pred = int(scores.argmax().item())

                s = stats[name]
                s["n_total"] += 1
                s["n_correct"] += int(pred == target_i)
                s["margin_sum"] += margin

                if pred == cls_pred:
                    s["agree_total"] += 1
                    s["agree_correct"] += int(cls_pred == target_i)
                    s["margin_agree_sum"] += margin
                else:
                    s["disagree_total"] += 1
                    s["disagree_correct"] += int(cls_pred == target_i)
                    s["margin_disagree_sum"] += margin

                if cls_pred != target_i:
                    s["n_cls_wrong"] += 1
                    s["rescue"] += int(pred == target_i)
                else:
                    s["n_cls_right"] += 1
                    s["corrupt"] += int(pred != target_i)

                for c, n_touched, correct_write in write_events:
                    bucket = s["write_agree"] if (c == pred) else s["write_disagree"]
                    bucket["n"] += n_touched
                    bucket["correct"] += n_touched * int(correct_write)

            n_total += 1
            if n_total % 200 == 0:
                mean_acc = 100.0 * stats["mean"]["n_correct"] / n_total
                max_acc = 100.0 * stats["max"]["n_correct"] / n_total
                print(
                    f"[{dataset}] {n_total} images — mean standalone {mean_acc:.2f}% "
                    f"max standalone {max_acc:.2f}%",
                    file=sys.stderr,
                )

    result = {"dataset": dataset, "n_total": n_total, "variants": {}}
    for name, _ in VARIANTS:
        s = stats[name]
        result["variants"][name] = {
            "standalone_acc": 100.0 * s["n_correct"] / max(s["n_total"], 1),
            "agreement_rate": 100.0 * s["agree_total"] / max(s["n_total"], 1),
            "cls_acc_when_agree": 100.0 * s["agree_correct"] / max(s["agree_total"], 1) if s["agree_total"] else None,
            "cls_acc_when_disagree": 100.0 * s["disagree_correct"] / max(s["disagree_total"], 1) if s["disagree_total"] else None,
            "write_purity_agree": _rate(s["write_agree"]),
            "write_purity_agree_n": s["write_agree"]["n"],
            "write_purity_disagree": _rate(s["write_disagree"]),
            "write_purity_disagree_n": s["write_disagree"]["n"],
            "rescue_rate_when_cls_wrong": 100.0 * s["rescue"] / max(s["n_cls_wrong"], 1) if s["n_cls_wrong"] else None,
            "corrupt_rate_when_cls_right": 100.0 * s["corrupt"] / max(s["n_cls_right"], 1) if s["n_cls_right"] else None,
            "mean_margin": s["margin_sum"] / max(s["n_total"], 1),
            "mean_margin_agree": s["margin_agree_sum"] / max(s["agree_total"], 1) if s["agree_total"] else None,
            "mean_margin_disagree": s["margin_disagree_sum"] / max(s["disagree_total"], 1) if s["disagree_total"] else None,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "oxford_flowers", "oxford_pets"])
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out-dir", default="outputs/patch_vote_aggregation")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_dataset(args.dataset, args.dumps, device, limit=args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_v = result["variants"]["mean"]
    max_v = result["variants"]["max"]
    print(
        f"[OK] {args.dataset}: n={result['n_total']} "
        f"mean: standalone={mean_v['standalone_acc']:.2f}% agree_rate={mean_v['agreement_rate']:.1f}% | "
        f"max: standalone={max_v['standalone_acc']:.2f}% agree_rate={max_v['agreement_rate']:.1f}% "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
