#!/usr/bin/env python3
"""Experiments F (confusability-weighted appearance) + H (CLS-vs-patch-vote
corroboration, diagnostic only) -- shared pass, extends frozen_bank_eval.py.

See outputs/patch_fix_experiments_plan.md for both. Reuses
frozen_bank_eval.py's bank loading, patch grouping, and Gaussian scoring
(imported, not duplicated) and adds:

  F) A third scoring condition alongside control/TCR: appearance weight
     multiplied by each prototype's own geometric confusability margin
     (intra-class match minus nearest-cross-class match, same variance-aware
     formula as Experiment A / analyze_prototype_separability_v2.py, but
     computed per-prototype instead of aggregated), normalized within each
     class (so a prototype is judged against its *own classmates*, not
     against an absolute scale -- Experiment A already showed every class is
     confusable on some absolute scale, so what matters is which prototype
     within a class is relatively more/less confusable than its siblings).

  H) Diagnostic only (no fusion change): patch embeddings, already
     extracted for the patch-level branch, dotted directly against text
     embeddings (mean-pooled across patches) gives a second, bank-free,
     state-free classification vote for the same image from the same single
     model. Checks whether this vote agreeing/disagreeing with CLIP's own
     (CLS-pooled) vote predicts write purity and prediction correctness --
     i.e. whether this is an informative signal at all before building
     anything on top of it.

Usage::

    python scripts/frozen_bank_eval_v2.py --dataset dtd
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

from encoder import create_encoder_instance  # noqa: E402
from utils import build_test_data_loader  # noqa: E402
from utils.clip_inference import _safe_normalize, clip_classifier  # noqa: E402
from frozen_bank_eval import (  # noqa: E402
    MATCH_THRESHOLD,
    PATCH_GROUP_THRESHOLD,
    VARIANCE_MIN,
    WRITE_CONF_THRESHOLD,
    BACKBONE,
    CLIP_MODEL,
    SEED,
    load_bank,
    load_stored_records,
    group_patches,
    score_all_classes,
    fuse,
)

MARGIN_CHUNK = 200


# ---------------------------------------------------------------------------
# F: per-prototype confusability margin
# ---------------------------------------------------------------------------

def compute_per_prototype_margin(centers_all, var_all, class_ids_all, variance_min=VARIANCE_MIN):
    """margin[i] = intra-class match (mean, own class, excl. self) minus
    nearest-cross-class match (max, any other class), same Gaussian-match
    formula as utils/kmeans.py::_gaussian_score_for_class. NaN intra where a
    class has only one prototype (no sibling to compare against)."""
    K = centers_all.shape[0]
    D = var_all.shape[-1]
    eff_min = variance_min * D
    var_clamped = var_all.clamp(min=eff_min)

    intra = torch.zeros(K)
    nearest_cross = torch.zeros(K)
    for start in range(0, K, MARGIN_CHUNK):
        end = min(start + MARGIN_CHUNK, K)
        c_i = centers_all[start:end]                              # [b, D]
        diff = c_i[:, None, :] - centers_all[None, :, :]           # [b, K, D]
        maha = (diff.pow(2) / var_clamped[None, :, :]).sum(-1)     # [b, K]
        scores = torch.exp(-0.5 * maha)                            # [b, K]
        for bi in range(end - start):
            i = start + bi
            same_mask = class_ids_all == class_ids_all[i]
            same_mask = same_mask.clone()
            same_mask[i] = False
            other_mask = ~(class_ids_all == class_ids_all[i])
            intra[i] = scores[bi][same_mask].mean() if same_mask.any() else float("nan")
            nearest_cross[i] = scores[bi][other_mask].max() if other_mask.any() else 0.0

    margin = intra - nearest_cross
    return margin, intra, nearest_cross


def normalize_margin_within_class(margin, class_ids_all, C, eps=1e-6):
    """rel[i] in [0, 1]: this prototype's margin relative to its own class's
    min/max. Singleton classes (nan margin) -> rel=1.0 (no penalty possible,
    nothing to compare against)."""
    K = margin.shape[0]
    rel = torch.ones(K)
    for c in range(C):
        idx = (class_ids_all == c).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue  # singleton or empty -> rel stays 1.0
        vals = margin[idx]
        valid = ~torch.isnan(vals)
        if valid.sum() <= 1:
            continue
        vmin, vmax = vals[valid].min(), vals[valid].max()
        span = (vmax - vmin).clamp(min=eps)
        rel_vals = (vals - vmin) / span
        rel_vals[~valid] = 1.0
        rel[idx] = rel_vals
    return rel


# ---------------------------------------------------------------------------
# H: patch-to-text zero-shot vote (bank-free, stateless)
# ---------------------------------------------------------------------------

def patch_text_vote(patches_norm, text_embeddings):
    """Mean- and max-pooled patch-to-text cosine similarity -> [C] logits
    each. text_embeddings: [D, C] (clip_classifier's convention)."""
    sims = patches_norm @ text_embeddings  # [P, C]
    mean_vote = sims.mean(dim=0)
    max_vote = sims.max(dim=0).values
    return mean_vote, max_vote


# ---------------------------------------------------------------------------
# Main per-dataset pass
# ---------------------------------------------------------------------------

def run_dataset(dataset, dumps_dir, device, limit=0):
    bank = load_bank(Path(dumps_dir) / f"{dataset}-s1.pt")
    stored = load_stored_records(dataset)
    C = len(bank["classes"])

    centers_list, var_list, app_list, class_ids_list = [], [], [], []
    per_class_centers = []
    for c, cls in enumerate(bank["classes"]):
        centers = cls["centers"]
        if centers.shape[0] == 0:
            per_class_centers.append(None)
            continue
        centers_n = _safe_normalize(centers).to(device)
        per_class_centers.append(centers_n)
        centers_list.append(centers_n)
        var_list.append(cls["variance"].to(device))
        app_list.append(cls["appearance"].to(device))
        class_ids_list.append(torch.full((centers.shape[0],), c, dtype=torch.long, device=device))

    centers_all = torch.cat(centers_list, dim=0)
    var_all = torch.cat(var_list, dim=0)
    app_all = torch.cat(app_list, dim=0)
    class_ids_all = torch.cat(class_ids_list, dim=0)

    print(f"[{dataset}] computing per-prototype confusability margin (F) ...", file=sys.stderr)
    margin, intra, nearest_cross = compute_per_prototype_margin(centers_all.cpu(), var_all.cpu(), class_ids_all.cpu())
    rel_margin = normalize_margin_within_class(margin, class_ids_all.cpu(), C)
    f_weight_all = (app_all.cpu() * rel_margin).to(device)

    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device=device)
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset, "./data", preprocess, shuffle=True, seed=SEED,
    )
    text_embeddings = clip_classifier(classnames, template, encoder).to(device)  # [D, C]

    purity_count = {c: torch.zeros(bank["classes"][c]["centers"].shape[0]) for c in range(C) if bank["classes"][c]["centers"].shape[0] > 0}
    purity_correct = {c: torch.zeros(bank["classes"][c]["centers"].shape[0]) for c in range(C) if bank["classes"][c]["centers"].shape[0] > 0}

    n_correct_control = n_correct_f = 0
    n_total = 0

    # H bookkeeping
    write_agree = {"n": 0, "correct": 0}
    write_disagree = {"n": 0, "correct": 0}
    img_agree_correct = img_agree_total = 0
    img_disagree_correct = img_disagree_total = 0
    n_cls_wrong_patch_right = n_cls_wrong = 0
    n_cls_right_patch_wrong = n_cls_right = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            if limit and i >= limit:
                break
            if i not in stored:
                continue
            rec = stored[i]
            target_i = int(rec["target"])
            clip_logits = rec["clip"].to(device).unsqueeze(0)
            image_proto_logits = rec["image_proto"].to(device).unsqueeze(0)
            cls_pred = int(rec["clip"].argmax().item())

            images = images.to(device) if not isinstance(images, list) else torch.cat(images, dim=0).to(device)
            patch_embs = encoder.get_patch_embeddings(images, exclude_pos=False)  # [P, D]
            patches_norm = _safe_normalize(patch_embs.float())

            # --- H: patch-to-text vote (bank-free) ---
            mean_vote, _max_vote = patch_text_vote(patches_norm, text_embeddings)
            patch_vote_pred = int(mean_vote.argmax().item())
            img_agree = patch_vote_pred == cls_pred

            if img_agree:
                img_agree_total += 1
                img_agree_correct += int(cls_pred == target_i)
            else:
                img_disagree_total += 1
                img_disagree_correct += int(cls_pred == target_i)

            if cls_pred != target_i:
                n_cls_wrong += 1
                n_cls_wrong_patch_right += int(patch_vote_pred == target_i)
            else:
                n_cls_right += 1
                n_cls_right_patch_wrong += int(patch_vote_pred != target_i)

            # --- Experiment B-style purity, split by H agreement ---
            write_probs = F.softmax(clip_logits.squeeze(0), dim=-1)
            written_classes = (write_probs > WRITE_CONF_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
            for c in written_classes:
                centers_c = per_class_centers[c]
                if centers_c is None:
                    continue
                sims = patches_norm @ centers_c.T
                best_val, best_idx = sims.max(dim=1)
                touched = torch.unique(best_idx[best_val >= MATCH_THRESHOLD])
                for k in touched.tolist():
                    purity_count[c][k] += 1
                    correct_write = target_i == c
                    if correct_write:
                        purity_correct[c][k] += 1
                    bucket = write_agree if (c == patch_vote_pred) else write_disagree
                    bucket["n"] += 1
                    bucket["correct"] += int(correct_write)

            # --- F: control vs. confusability-weighted scoring ---
            rep_patches = group_patches(patches_norm)
            control_scores = score_all_classes(rep_patches, centers_all, var_all, app_all, class_ids_all, C)
            f_scores = score_all_classes(rep_patches, centers_all, var_all, f_weight_all, class_ids_all, C)

            final_control = fuse(clip_logits, image_proto_logits, control_scores.unsqueeze(0))
            final_f = fuse(clip_logits, image_proto_logits, f_scores.unsqueeze(0))

            n_correct_control += int(final_control.argmax(dim=-1).item() == target_i)
            n_correct_f += int(final_f.argmax(dim=-1).item() == target_i)
            n_total += 1

            if n_total % 200 == 0:
                print(
                    f"[{dataset}] {n_total} images — control {100*n_correct_control/n_total:.2f}% "
                    f"f_weighted {100*n_correct_f/n_total:.2f}% "
                    f"img_agree_rate {100*img_agree_total/n_total:.1f}%",
                    file=sys.stderr,
                )

    acc_control = 100.0 * n_correct_control / max(n_total, 1)
    acc_f = 100.0 * n_correct_f / max(n_total, 1)

    purity_table = []
    for c in range(C):
        if c not in purity_count:
            continue
        counts, corrects = purity_count[c], purity_correct[c]
        appearance = bank["classes"][c]["appearance"]
        for k in range(counts.shape[0]):
            if counts[k].item() == 0:
                continue
            purity_table.append({
                "class": c, "cluster": k,
                "appearance": float(appearance[k].item()),
                "touched": float(counts[k].item()),
                "correct": float(corrects[k].item()),
                "purity": float(corrects[k].item() / counts[k].item()),
            })

    def _rate(bucket):
        return 100.0 * bucket["correct"] / bucket["n"] if bucket["n"] else None

    return {
        "dataset": dataset,
        "n_total": n_total,
        "acc_control": acc_control,
        "acc_f_weighted": acc_f,
        "purity_table": purity_table,
        "h_diagnostic": {
            "write_purity_agree": _rate(write_agree),
            "write_purity_agree_n": write_agree["n"],
            "write_purity_disagree": _rate(write_disagree),
            "write_purity_disagree_n": write_disagree["n"],
            "img_agree_rate": 100.0 * img_agree_total / max(n_total, 1),
            "cls_acc_when_agree": 100.0 * img_agree_correct / max(img_agree_total, 1) if img_agree_total else None,
            "cls_acc_when_disagree": 100.0 * img_disagree_correct / max(img_disagree_total, 1) if img_disagree_total else None,
            "n_cls_wrong": n_cls_wrong,
            "rescue_rate_when_cls_wrong": 100.0 * n_cls_wrong_patch_right / max(n_cls_wrong, 1) if n_cls_wrong else None,
            "n_cls_right": n_cls_right,
            "corrupt_rate_when_cls_right": 100.0 * n_cls_right_patch_wrong / max(n_cls_right, 1) if n_cls_right else None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "oxford_flowers", "oxford_pets"])
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out-dir", default="outputs/frozen_bank_eval_v2")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_dataset(args.dataset, args.dumps, device, limit=args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    h = result["h_diagnostic"]
    print(
        f"[OK] {args.dataset}: control={result['acc_control']:.2f}% f_weighted={result['acc_f_weighted']:.2f}% "
        f"(n={result['n_total']}) | H: agree_purity={h['write_purity_agree']} "
        f"disagree_purity={h['write_purity_disagree']} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
