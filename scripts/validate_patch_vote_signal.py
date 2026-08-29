#!/usr/bin/env python3
"""Experiment J -- deep validation of Experiment I's shortlisted aggregation
variants (mean, topk20, pmean16, surgery_masked_mean).

See outputs/patch_fix_experiments_plan.md ("Experiment J"). Experiment I's
metrics are coarse, single-seed, whole-dataset averages. Before committing
to a live write-time change (Experiment K) built on one of these variants,
this checks:

  1. Multi-seed replication -- does the write-purity-by-agreement gap hold
     up against *different* final bank states (seeds 2-4), not just the
     seed-1 bank Experiments H/I used? (Standalone accuracy / agreement
     rate / CLS-accuracy-by-agreement are provably seed-invariant -- they
     only depend on each image's own zero-shot CLIP logits and patch
     embeddings, neither of which depends on test-set processing order --
     so only the bank-dependent write-purity metric needs new seeds; see
     the plan doc for why.)
  2. Margin-graded analysis -- does each variant's own top1-top2 margin
     carry graded (not just binary agree/disagree) information?
  3. Per-class breakdown -- does the corroboration benefit concentrate in
     classes where CLIP itself is weak, or classes with certain properties,
     or is it spread evenly?
  4. Cross-signal correlation -- is agreement just re-deriving (a) the
     confident-vs-ambiguous CLIP write-margin split, or (b) Experiment F's
     per-prototype confusability margin?
  5. Qualitative spot-check -- concrete rescue/corrupt examples.

Purely diagnostic -- no bank mutation, no live prediction change.

Usage::

    python scripts/validate_patch_vote_signal.py --dataset dtd
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

from utils import build_test_data_loader  # noqa: E402
from utils.clip_inference import _safe_normalize, clip_classifier, compute_surgery_scores  # noqa: E402
from encoder import create_encoder_instance  # noqa: E402
from frozen_bank_eval import (  # noqa: E402
    MATCH_THRESHOLD,
    WRITE_CONF_THRESHOLD,
    BACKBONE,
    CLIP_MODEL,
    load_bank,
)
from frozen_bank_eval_v2 import compute_per_prototype_margin  # noqa: E402
from analyze_patch_vote_aggregation import (  # noqa: E402
    mean_pool,
    topk_pool,
    power_mean_pool,
    surgery_masked_mean,
    otsu_mean,
)

CONFIDENT_WRITE_MARGIN = 0.2  # matches the earlier confident-vs-ambiguous split


def load_stored_records_for_seed(dataset, seed):
    """Like frozen_bank_eval.load_stored_records, but for an arbitrary seed
    (that function hardcodes -s1). Records are keyed by batch_idx, i.e. the
    *position* in that seed's own shuffled test loader -- so a seed-N loader
    MUST be paired with seed-N's own records file, never seed-1's (shuffle
    order differs per seed, so position i is a different image under each
    seed; pairing mismatched seeds silently scrambles target/prediction
    pairing down to near-chance accuracy)."""
    path = REPO_ROOT / "outputs" / "records" / f"PatchModPTA-CS-{dataset}-s{seed}" / "records.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            out[rec["batch_idx"]] = {
                "clip": torch.tensor(rec["logits"]["clip"], dtype=torch.float32),
                "target": rec["target"],
            }
    return out

SHORTLIST = [
    ("mean", lambda sims, ss: mean_pool(sims)),
    ("topk20", lambda sims, ss: topk_pool(sims, 20)),
    ("pmean16", lambda sims, ss: power_mean_pool(sims, 16)),
    ("surgery_masked_mean", lambda sims, ss: surgery_masked_mean(sims, ss)),
    ("otsu_mean", lambda sims, ss: otsu_mean(sims)),
]
MARGIN_LOGIT_SCALE = 100.0


def clip_margin(clip_logits):
    probs = F.softmax(clip_logits, dim=-1)
    top2 = probs.topk(2)
    return float((top2.values[0] - top2.values[1]).item())


def collect_samples(dataset, seed, dumps_dir, device, need_bank_geometry, limit=0):
    """One pass over the test set. Returns per-sample records plus (if
    need_bank_geometry) the flattened bank + per-prototype confusability
    margins for the confusability cross-check."""
    stored = load_stored_records_for_seed(dataset, seed)

    bank = load_bank(Path(dumps_dir) / f"{dataset}-s{seed}.pt")
    C = len(bank["classes"])

    per_class_centers = []
    global_offset = []
    offset = 0
    for cls in bank["classes"]:
        centers = cls["centers"]
        if centers.shape[0] == 0:
            per_class_centers.append(None)
            global_offset.append(None)
            continue
        per_class_centers.append(_safe_normalize(centers).to(device))
        global_offset.append(offset)
        offset += centers.shape[0]

    proto_margin = None
    if need_bank_geometry:
        centers_list, var_list, class_ids_list = [], [], []
        for c, cls in enumerate(bank["classes"]):
            centers = cls["centers"]
            if centers.shape[0] == 0:
                continue
            centers_list.append(_safe_normalize(centers))
            var_list.append(cls["variance"])
            class_ids_list.append(torch.full((centers.shape[0],), c, dtype=torch.long))
        centers_all = torch.cat(centers_list, dim=0)
        var_all = torch.cat(var_list, dim=0)
        class_ids_all = torch.cat(class_ids_list, dim=0)
        proto_margin, _, _ = compute_per_prototype_margin(centers_all, var_all, class_ids_all)

    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device=device)
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset, "./data", preprocess, shuffle=True, seed=seed,
    )
    text_embeddings = clip_classifier(classnames, template, encoder).to(device)

    text_features = surgery_empty = None
    if need_bank_geometry:
        import clip as _clip
        text_features = _safe_normalize(text_embeddings.t().float())
        tokens = _clip.tokenize([""]).to(device)
        raw_model = encoder.model if hasattr(encoder, "model") else encoder
        with torch.no_grad():
            surgery_empty = _safe_normalize(raw_model.encode_text(tokens).float())

    samples = []
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
            patch_embs = encoder.get_patch_embeddings(images, exclude_pos=False)
            patches_norm = _safe_normalize(patch_embs.float())
            sims = patches_norm @ text_embeddings

            surgery_scores = None
            if need_bank_geometry:
                surgery_scores = compute_surgery_scores(
                    images, encoder, text_features, surgery_empty, "surgery_no_labels"
                )

            variants = {}
            for name, pool_fn in SHORTLIST:
                scores = pool_fn(sims, surgery_scores)
                probs = F.softmax(scores * MARGIN_LOGIT_SCALE, dim=-1)
                top2 = probs.topk(min(2, probs.shape[0]))
                margin = float((top2.values[0] - top2.values[-1]).item())
                pred = int(scores.argmax().item())
                variants[name] = {"pred": pred, "margin": margin}

            write_events = []
            if need_bank_geometry:
                write_probs = F.softmax(clip_logits, dim=-1)
                written_classes = (write_probs > WRITE_CONF_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
                w_margin = clip_margin(clip_logits)
                for c in written_classes:
                    centers_c = per_class_centers[c]
                    if centers_c is None:
                        continue
                    csims = patches_norm @ centers_c.T
                    best_val, best_idx = csims.max(dim=1)
                    touched = torch.unique(best_idx[best_val >= MATCH_THRESHOLD])
                    if touched.numel() == 0:
                        continue
                    write_events.append({
                        "class": c,
                        "n_touched": int(touched.numel()),
                        "correct": bool(target_i == c),
                        "clip_write_margin": w_margin,
                        "global_proto_idxs": [global_offset[c] + int(k) for k in touched.tolist()],
                    })

            samples.append({
                "idx": i, "target": target_i, "cls_pred": cls_pred,
                "cls_correct": cls_pred == target_i,
                "variants": variants,
                "write_events": write_events,
            })

    return {
        "dataset": dataset, "seed": seed, "n_total": len(samples),
        "samples": samples, "classnames": classnames,
        "proto_margin": proto_margin,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_multiseed(per_seed_data):
    """variant -> seed -> {standalone_acc, agreement_rate, write_purity_agree, write_purity_disagree}"""
    out = {}
    for name, _ in SHORTLIST:
        out[name] = {}
        for seed, data in per_seed_data.items():
            n_correct = n_total = agree_n = agree_correct = 0
            wa_n = wa_c = wd_n = wd_c = 0
            for s in data["samples"]:
                n_total += 1
                pred = s["variants"][name]["pred"]
                n_correct += int(pred == s["target"])
                if pred == s["cls_pred"]:
                    agree_n += 1
                    agree_correct += int(s["cls_correct"])
                for ev in s["write_events"]:
                    if ev["class"] == pred:
                        wa_n += ev["n_touched"]
                        wa_c += ev["n_touched"] * int(ev["correct"])
                    else:
                        wd_n += ev["n_touched"]
                        wd_c += ev["n_touched"] * int(ev["correct"])
            out[name][seed] = {
                "n_total": n_total,
                "standalone_acc": 100.0 * n_correct / max(n_total, 1),
                "agreement_rate": 100.0 * agree_n / max(n_total, 1),
                "cls_acc_when_agree": 100.0 * agree_correct / max(agree_n, 1) if agree_n else None,
                "write_purity_agree": 100.0 * wa_c / max(wa_n, 1) if wa_n else None,
                "write_purity_agree_n": wa_n,
                "write_purity_disagree": 100.0 * wd_c / max(wd_n, 1) if wd_n else None,
                "write_purity_disagree_n": wd_n,
            }
    return out


def section_margin_deciles(data, n_bins=10):
    """variant -> list of {bin, margin_range, cls_acc, write_purity, n}.
    Rank-based binning (equal sample count per bin, not equal margin width)
    -- avoids degenerate bins when many samples share a tied margin."""
    out = {}
    for name, _ in SHORTLIST:
        if not data["samples"]:
            out[name] = []
            continue
        order = sorted(range(len(data["samples"])), key=lambda idx: data["samples"][idx]["variants"][name]["margin"])
        bin_size = max(len(order) // n_bins, 1)
        bins = []
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(order)
            idxs = order[start:end]
            if not idxs:
                continue
            samples_in_bin = [data["samples"][i] for i in idxs]
            margin_lo = samples_in_bin[0]["variants"][name]["margin"]
            margin_hi = samples_in_bin[-1]["variants"][name]["margin"]
            cls_correct = sum(int(s["cls_correct"]) for s in samples_in_bin)
            wa_n = wa_c = 0
            for s in samples_in_bin:
                pred = s["variants"][name]["pred"]
                for ev in s["write_events"]:
                    if ev["class"] == pred:
                        wa_n += ev["n_touched"]
                        wa_c += ev["n_touched"] * int(ev["correct"])
            bins.append({
                "bin": b, "margin_range": [margin_lo, margin_hi], "n": len(idxs),
                "cls_acc": 100.0 * cls_correct / len(idxs),
                "write_purity": 100.0 * wa_c / max(wa_n, 1) if wa_n else None,
                "write_purity_n": wa_n,
            })
        out[name] = bins
    return out


def section_per_class(data, min_class_n=5):
    """variant -> list of {class, classname, cls_baseline_acc, agreement_rate, gap}"""
    classnames = data["classnames"]
    by_class = {}
    for s in data["samples"]:
        by_class.setdefault(s["target"], []).append(s)

    baseline_acc = {}
    for c, samples_c in by_class.items():
        baseline_acc[c] = 100.0 * sum(int(s["cls_correct"]) for s in samples_c) / len(samples_c)

    out = {}
    for name, _ in SHORTLIST:
        rows = []
        for c, samples_c in by_class.items():
            if len(samples_c) < min_class_n:
                continue
            agree = [s for s in samples_c if s["variants"][name]["pred"] == s["cls_pred"]]
            disagree = [s for s in samples_c if s["variants"][name]["pred"] != s["cls_pred"]]
            acc_agree = 100.0 * sum(int(s["cls_correct"]) for s in agree) / len(agree) if agree else None
            acc_disagree = 100.0 * sum(int(s["cls_correct"]) for s in disagree) / len(disagree) if disagree else None
            gap = (acc_agree - acc_disagree) if (acc_agree is not None and acc_disagree is not None) else None
            rows.append({
                "class": c, "classname": classnames[c] if c < len(classnames) else str(c),
                "n": len(samples_c),
                "cls_baseline_acc": baseline_acc[c],
                "agreement_rate": 100.0 * len(agree) / len(samples_c),
                "acc_when_agree": acc_agree, "acc_when_disagree": acc_disagree, "gap": gap,
            })
        rows.sort(key=lambda r: r["cls_baseline_acc"])
        out[name] = rows
    return out


def section_cross_signal(data):
    """variant -> {confident_agree, confident_disagree, ambiguous_agree,
    ambiguous_disagree} write-purity quadrants, plus mean confusability
    margin of touched prototypes split by agreement."""
    out = {}
    proto_margin = data["proto_margin"]
    for name, _ in SHORTLIST:
        quad = {k: {"n": 0, "correct": 0} for k in
                ["confident_agree", "confident_disagree", "ambiguous_agree", "ambiguous_disagree"]}
        conf_margin_agree, conf_margin_disagree = [], []
        for s in data["samples"]:
            pred = s["variants"][name]["pred"]
            for ev in s["write_events"]:
                agree = ev["class"] == pred
                confident = ev["clip_write_margin"] >= CONFIDENT_WRITE_MARGIN
                key = f"{'confident' if confident else 'ambiguous'}_{'agree' if agree else 'disagree'}"
                quad[key]["n"] += ev["n_touched"]
                quad[key]["correct"] += ev["n_touched"] * int(ev["correct"])
                if proto_margin is not None:
                    m = float(proto_margin[ev["global_proto_idxs"]].mean().item())
                    (conf_margin_agree if agree else conf_margin_disagree).append(m)
        out[name] = {
            "write_purity_by_quadrant": {
                k: (100.0 * v["correct"] / v["n"] if v["n"] else None) for k, v in quad.items()
            },
            "write_purity_n_by_quadrant": {k: v["n"] for k, v in quad.items()},
            "mean_confusability_margin_agree": (
                sum(conf_margin_agree) / len(conf_margin_agree) if conf_margin_agree else None
            ),
            "mean_confusability_margin_disagree": (
                sum(conf_margin_disagree) / len(conf_margin_disagree) if conf_margin_disagree else None
            ),
        }
    return out


def section_qualitative(data, n_examples=5):
    classnames = data["classnames"]
    out = {}
    for name, _ in SHORTLIST:
        rescue, corrupt = [], []
        for s in data["samples"]:
            pred = s["variants"][name]["pred"]
            if not s["cls_correct"] and pred == s["target"]:
                rescue.append(s)
            elif s["cls_correct"] and pred != s["target"]:
                corrupt.append(s)
        def _fmt(s):
            return {
                "idx": s["idx"],
                "target": classnames[s["target"]] if s["target"] < len(classnames) else s["target"],
                "cls_pred": classnames[s["cls_pred"]] if s["cls_pred"] < len(classnames) else s["cls_pred"],
                "variant_pred": classnames[s["variants"][name]["pred"]]
                    if s["variants"][name]["pred"] < len(classnames) else s["variants"][name]["pred"],
            }
        out[name] = {
            "rescue_examples": [_fmt(s) for s in rescue[:n_examples]],
            "corrupt_examples": [_fmt(s) for s in corrupt[:n_examples]],
            "n_rescue_total": len(rescue), "n_corrupt_total": len(corrupt),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "oxford_flowers", "oxford_pets"])
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out-dir", default="outputs/patch_vote_validation")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed_data = {}
    for seed in seeds:
        dump_path = Path(args.dumps) / f"{args.dataset}-s{seed}.pt"
        if not dump_path.exists():
            print(f"[skip] {dump_path} not found", file=sys.stderr)
            continue
        print(f"[{args.dataset}-s{seed}] collecting samples ...", file=sys.stderr)
        per_seed_data[seed] = collect_samples(
            args.dataset, seed, args.dumps, device,
            need_bank_geometry=True, limit=args.limit,
        )

    if not per_seed_data:
        raise SystemExit(f"No bank dumps found for {args.dataset} under {args.dumps}")

    primary_seed = min(per_seed_data.keys())
    primary = per_seed_data[primary_seed]

    result = {
        "dataset": args.dataset,
        "seeds_used": sorted(per_seed_data.keys()),
        "primary_seed": primary_seed,
        "multiseed": section_multiseed(per_seed_data),
        "margin_deciles": section_margin_deciles(primary),
        "per_class": section_per_class(primary),
        "cross_signal": section_cross_signal(primary),
        "qualitative": section_qualitative(primary),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {args.dataset}: seeds={sorted(per_seed_data.keys())} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
