#!/usr/bin/env python3
"""Experiments B (per-cluster purity) + C (TCR reweighting) — shared pass.

Both experiments need the same thing: for every test image, extract its
patch embeddings and compare them against the *frozen* final patch bank
(`outputs/patch_bank_dumps/`) — no live adaptation, no bank updates, no seed
sweep. This script does that once per dataset and computes both:

  B) Per-prototype purity: for the classes an image's patches get written to
     (replaying the multi_gate write rule from `analyze_prototype_purity.py`,
     using the already-stored zero-shot CLIP logits), which specific
     prototype(s) in that class's bank the image's patches actually match
     (cosine >= match_threshold, same rule `update_state` uses live). Pooled
     over all images: purity per prototype = how often the images that
     matched it were genuinely that class.

  C) Control vs. TCR accuracy: score every image against the frozen bank
     twice — once weighted by the bank's real appearance counts (control),
     once by a TCR-style weight computed from the frozen bank alone (no
     labels) — using the same simplified "weighted_mean" aggregation for
     both (see outputs/offline_experiments_plan.md for why the live
     z-score aggregation isn't reproduced exactly). Fuse with the
     already-stored, already-validated clip/image-proto logits from
     `outputs/records_patch_benefit/PatchModPTA-CS-<dataset>-s1/` using the
     same ProtoAlphaFusion formula the live method uses, and compare
     resulting accuracy.

Usage::

    python scripts/frozen_bank_eval.py --dataset dtd
    python scripts/frozen_bank_eval.py --dataset oxford_flowers
    python scripts/frozen_bank_eval.py --dataset oxford_pets
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder import create_encoder_instance  # noqa: E402
from utils import build_test_data_loader, clip_classifier  # noqa: E402
from utils.clip_inference import _safe_normalize  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — all taken from the resolved config used for these bank dumps
# (configs/patch_modulated_pta/patch_modulated_pta.yaml, identical across
# dtd/oxford_flowers/oxford_pets — no per-dataset overrides exist).
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 0.6          # live match_threshold (update_state)
PATCH_GROUP_THRESHOLD = 0.9    # live patch grouping threshold
VARIANCE_MIN = 0.001           # live variance floor
WRITE_CONF_THRESHOLD = 0.3     # live multi_gate write conf_threshold
TAU_IMAGE_PROTO = 80.0
TAU_PATCH_PROTO = 10.0
PATCH_SQUASH_SCALE = 1.0       # tanh(x / scale)
BACKBONE = "ViT-B/16"
CLIP_MODEL = "clip_surgery"
SEED = 1


def load_bank(dump_path):
    return torch.load(dump_path, map_location="cpu", weights_only=False)


def load_stored_records(dataset):
    """clip_logits / image_proto_logits / target per batch_idx, from the
    original study's PatchModPTA-CS seed-1 run (same deterministic run these
    bank dumps came from — verified identical accuracy in the earlier smoke
    check)."""
    path = REPO_ROOT / "outputs" / "records" / f"PatchModPTA-CS-{dataset}-s1" / "records.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            out[rec["batch_idx"]] = {
                "clip": torch.tensor(rec["logits"]["clip"], dtype=torch.float32),
                "image_proto": torch.tensor(rec["logits"]["image_proto"], dtype=torch.float32),
                "target": rec["target"],
            }
    return out


# ---------------------------------------------------------------------------
# TCR weight computation (Experiment C) — pure function of the frozen bank,
# no labels used. See outputs/patch_benefit_report.md Section 9.4 for the
# definition being implemented here.
# ---------------------------------------------------------------------------

def compute_tcr_weights(bank, device="cpu", eps=1e-8):
    C = len(bank["classes"])
    centers_list, density_list, class_ids_list = [], [], []
    for c, cls in enumerate(bank["classes"]):
        centers = cls["centers"]
        if centers.shape[0] == 0:
            continue
        appearance = cls["appearance"]
        n_images = max(int(cls["n_images"]), 1)
        centers_list.append(_safe_normalize(centers).to(device))
        density_list.append((appearance / n_images).to(device))
        class_ids_list.append(torch.full((centers.shape[0],), c, dtype=torch.long, device=device))

    centers_all = torch.cat(centers_list, dim=0)      # [K, D]
    density_all = torch.cat(density_list, dim=0)       # [K]
    class_ids_all = torch.cat(class_ids_list, dim=0)   # [K]
    K = centers_all.shape[0]

    # contrib[i, c'] = max_{j in c'} sim(i, j) * density(j), for c' != class(i);
    # contrib[i, class(i)] = density(i) (own contribution).
    contrib = torch.zeros(K, C, device=device)
    sim_full = centers_all @ centers_all.T  # [K, K]
    weighted_sim = sim_full * density_all[None, :]  # column j scaled by density(j)
    for c_prime in range(C):
        mask = class_ids_all == c_prime
        if not mask.any():
            continue
        contrib[:, c_prime] = weighted_sim[:, mask].max(dim=1).values
    contrib[torch.arange(K), class_ids_all] = density_all  # overwrite own-class entry

    total_occ = contrib.sum(dim=1).clamp(min=eps)          # [K]
    own_density = density_all
    ctw = own_density / total_occ                          # [K]

    p = contrib / total_occ[:, None].clamp(min=eps)         # [K, C]
    entropy = -(p * (p + eps).log()).sum(dim=1)             # [K]
    h_max = math.log(max(C, 2))
    entropy_factor = 1.0 - (entropy / h_max)

    tcr = own_density * ctw * entropy_factor                # [K]

    # Re-split back into a per-class list matching bank["classes"] indexing.
    tcr_per_class = [torch.zeros(0) for _ in range(C)]
    offset = 0
    for c, cls in enumerate(bank["classes"]):
        k = cls["centers"].shape[0]
        if k == 0:
            continue
        tcr_per_class[c] = tcr[offset:offset + k].cpu()
        offset += k
    return tcr_per_class


# ---------------------------------------------------------------------------
# Patch grouping (Experiment B + C) — identical algorithm to
# utils/kmeans.py's `_gaussian_score_for_class` grouping step.
# ---------------------------------------------------------------------------

def group_patches(patches_norm, threshold=PATCH_GROUP_THRESHOLD):
    patch_sims = patches_norm @ patches_norm.t()
    n = patches_norm.shape[0]
    unassigned = torch.ones(n, dtype=torch.bool, device=patches_norm.device)
    rep_centers = []
    while unassigned.any():
        cand_idx = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        sub_sims = patch_sims[cand_idx][:, cand_idx]
        anchor_local = int(sub_sims.mean(dim=1).argmax().item())
        anchor_idx = int(cand_idx[anchor_local].item())
        group_mask = unassigned & (patch_sims[anchor_idx] >= threshold)
        member_idx = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
        if member_idx.numel() == 0:
            member_idx = torch.tensor([anchor_idx], device=patches_norm.device, dtype=torch.long)
            group_mask = torch.zeros_like(unassigned)
            group_mask[anchor_idx] = True
        rep = _safe_normalize(patches_norm[member_idx].mean(dim=0), dim=-1)
        rep_centers.append(rep)
        unassigned[group_mask] = False
    return torch.stack(rep_centers, dim=0)  # [G, D]


def score_all_classes(rep_patches, centers_all, var_all, weight_all, class_ids_all, C, chunk=2000):
    """best_per_proto (max over groups, no one-to-one assignment — see
    outputs/offline_experiments_plan.md for why this simplification is used),
    then appearance/TCR-weighted mean per class."""
    device = rep_patches.device
    K_total = centers_all.shape[0]
    best_per_proto = torch.zeros(K_total, device=device)
    D = var_all.shape[-1]
    eff_min = VARIANCE_MIN * D
    for start in range(0, K_total, chunk):
        end = min(start + chunk, K_total)
        c_chunk = centers_all[start:end]
        v_chunk = var_all[start:end].clamp(min=eff_min)
        diff = rep_patches[:, None, :] - c_chunk[None, :, :]       # [G, k, D]
        maha = (diff.pow(2) / v_chunk[None, :, :]).sum(dim=-1)      # [G, k]
        best_per_proto[start:end] = torch.exp(-0.5 * maha).max(dim=0).values

    weighted = best_per_proto * weight_all
    num = torch.zeros(C, device=device).scatter_add_(0, class_ids_all, weighted)
    den = torch.zeros(C, device=device).scatter_add_(0, class_ids_all, weight_all)
    return num / den.clamp(min=1e-6)


def fuse(clip_logits, image_proto_logits, patch_scores):
    squashed = torch.tanh(patch_scores / PATCH_SQUASH_SCALE)
    return clip_logits + TAU_IMAGE_PROTO * image_proto_logits + TAU_PATCH_PROTO * squashed


# ---------------------------------------------------------------------------
# Main per-dataset pass
# ---------------------------------------------------------------------------

def run_dataset(dataset, dumps_dir, device, limit=0):
    bank = load_bank(Path(dumps_dir) / f"{dataset}-s1.pt")
    stored = load_stored_records(dataset)
    C = len(bank["classes"])

    print(f"[{dataset}] computing TCR weights ...", file=sys.stderr)
    tcr_per_class = compute_tcr_weights(bank, device=device)

    # Flatten frozen bank once: centers/variance/appearance/tcr concatenated,
    # with a class_ids index — reused for every image (bank never changes).
    centers_list, var_list, app_list, tcr_list, class_ids_list = [], [], [], [], []
    per_class_centers = []  # kept per-class (normalized) for Experiment B matching
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
        tcr_list.append(tcr_per_class[c].to(device))
        class_ids_list.append(torch.full((centers.shape[0],), c, dtype=torch.long, device=device))

    centers_all = torch.cat(centers_list, dim=0)
    var_all = torch.cat(var_list, dim=0)
    app_all = torch.cat(app_list, dim=0)
    tcr_all = torch.cat(tcr_list, dim=0)
    class_ids_all = torch.cat(class_ids_list, dim=0)

    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device=device)
    preprocess = encoder.preprocess
    test_loader, classnames, template = build_test_data_loader(
        dataset, "./data", preprocess, shuffle=True, seed=SEED,
    )

    # Purity bookkeeping: per (class, cluster_idx_within_class) -> [n_touched, n_correct]
    purity_count = {c: torch.zeros(bank["classes"][c]["centers"].shape[0]) for c in range(C) if bank["classes"][c]["centers"].shape[0] > 0}
    purity_correct = {c: torch.zeros(bank["classes"][c]["centers"].shape[0]) for c in range(C) if bank["classes"][c]["centers"].shape[0] > 0}

    n_correct_control = 0
    n_correct_tcr = 0
    n_total = 0

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

            images = images.to(device) if not isinstance(images, list) else torch.cat(images, dim=0).to(device)
            patch_embs = encoder.get_patch_embeddings(images, exclude_pos=False)  # [P, D]
            patches_norm = _safe_normalize(patch_embs.float())

            # --- Experiment B: purity of clusters this image's patches touch ---
            write_probs = F.softmax(clip_logits.squeeze(0), dim=-1)
            written_classes = (write_probs > WRITE_CONF_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
            for c in written_classes:
                centers_c = per_class_centers[c]
                if centers_c is None:
                    continue
                sims = patches_norm @ centers_c.T  # [P, K_c]
                best_val, best_idx = sims.max(dim=1)
                touched = torch.unique(best_idx[best_val >= MATCH_THRESHOLD])
                for k in touched.tolist():
                    purity_count[c][k] += 1
                    if target_i == c:
                        purity_correct[c][k] += 1

            # --- Experiment C: control vs. TCR frozen-bank scoring ---
            rep_patches = group_patches(patches_norm)
            control_scores = score_all_classes(rep_patches, centers_all, var_all, app_all, class_ids_all, C)
            tcr_scores = score_all_classes(rep_patches, centers_all, var_all, tcr_all, class_ids_all, C)

            final_control = fuse(clip_logits, image_proto_logits, control_scores.unsqueeze(0))
            final_tcr = fuse(clip_logits, image_proto_logits, tcr_scores.unsqueeze(0))

            pred_control = int(final_control.argmax(dim=-1).item())
            pred_tcr = int(final_tcr.argmax(dim=-1).item())
            n_correct_control += int(pred_control == target_i)
            n_correct_tcr += int(pred_tcr == target_i)
            n_total += 1

            if n_total % 200 == 0:
                print(
                    f"[{dataset}] {n_total} images — control {100*n_correct_control/n_total:.2f}% "
                    f"tcr {100*n_correct_tcr/n_total:.2f}%",
                    file=sys.stderr,
                )

    acc_control = 100.0 * n_correct_control / max(n_total, 1)
    acc_tcr = 100.0 * n_correct_tcr / max(n_total, 1)

    # Per-prototype purity/appearance/confusability-ready table.
    purity_table = []
    for c in range(C):
        if c not in purity_count:
            continue
        counts = purity_count[c]
        corrects = purity_correct[c]
        appearance = bank["classes"][c]["appearance"]
        for k in range(counts.shape[0]):
            if counts[k].item() == 0:
                continue
            purity_table.append({
                "class": c,
                "cluster": k,
                "appearance": float(appearance[k].item()),
                "touched": float(counts[k].item()),
                "correct": float(corrects[k].item()),
                "purity": float(corrects[k].item() / counts[k].item()),
            })

    return {
        "dataset": dataset,
        "n_total": n_total,
        "acc_control": acc_control,
        "acc_tcr": acc_tcr,
        "purity_table": purity_table,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "oxford_flowers", "oxford_pets"])
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out-dir", default="outputs/frozen_bank_eval")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_dataset(args.dataset, args.dumps, device, limit=args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {args.dataset}: control={result['acc_control']:.2f}% tcr={result['acc_tcr']:.2f}% "
          f"(n={result['n_total']}) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
