#!/usr/bin/env python3
"""Patch-level prototype bank — inter/intra-class separability diagnostic.

Follow-up to ``scripts/analyze_prototype_purity.py``. That script found write
purity is *highest* on oxford_pets (85.7%) and *lowest* on dtd (62.6%) — the
opposite of what "patch fusion fails on flowers/pets because of write
contamination" would predict, since patch fusion actually hurts on
flowers/pets and mildly helps on dtd. So contamination isn't the bottleneck.

This script tests the next candidate mechanism: even a *correctly* built
bank (right-class patches only) may not be locally discriminative for
fine-grained classes — a Bengal-cat fur patch and a Siamese-cat fur patch can
sit closer together in patch-embedding space than two Bengal-cat patches sit
to each other, so the bank ends up "pure but confusable."

It loads the raw final per-class Gaussian bank state (``centers``,
``variance``, ``appearance``, ``n_images``) dumped via the opt-in
``DUMP_PATCH_BANK`` env var added to ``models/patch_modulated_pta.py``
(behavior-neutral — no-op unless set), and computes, per dataset:

  - intra-class coherence: mean pairwise cosine similarity between centers
    belonging to the SAME class (how tight is a class's own bank?)
  - inter-class confusability: for each center, cosine similarity to its
    nearest center in a DIFFERENT class (worst-case neighbor)
  - background inter-class similarity: mean cosine similarity across ALL
    cross-class center pairs (generic embedding-space closeness, as a
    baseline the "nearest" confusability number should be compared against)
  - separability margin = intra-class coherence − inter-class confusability
    (a large positive margin means a class's own centers are tighter
    together than its nearest confusor; a small/negative margin means the
    bank cannot geometrically distinguish that class from a neighbor even
    though every patch in it came from the right class)

All metrics are also computed restricted to each class's HIGH-APPEARANCE
centers only (top half by appearance weight, per class) — the "reasonably
high appearance weight" subset the hypothesis is specifically about, since
those are the prototypes that actually dominate ``compute_patch_logits``'s
scoring at inference time.

Usage::

    python scripts/analyze_prototype_separability.py \\
        --dumps outputs/patch_bank_dumps \\
        --out outputs/prototype_separability_report.md
"""

import argparse
import json
import sys
from pathlib import Path

import torch

DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]


def load_bank(dump_path):
    return torch.load(dump_path, map_location="cpu", weights_only=False)


def _normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def build_center_pool(bank, high_appearance_only=False):
    """Flatten per-class centers into (centers[N,D], class_ids[N]) tensors.

    When ``high_appearance_only``, keep only each class's centers whose
    appearance weight is >= that class's own median appearance (top half).
    Classes with <2 centers keep everything (no meaningful median split).
    """
    all_centers = []
    all_class_ids = []
    for c, cls in enumerate(bank["classes"]):
        centers = cls["centers"]
        appearance = cls["appearance"]
        if centers.shape[0] == 0:
            continue
        if high_appearance_only and centers.shape[0] >= 2:
            median = appearance.median()
            keep = appearance >= median
            centers = centers[keep]
        if centers.shape[0] == 0:
            continue
        all_centers.append(centers)
        all_class_ids.extend([c] * centers.shape[0])
    if not all_centers:
        return None, None
    centers = torch.cat(all_centers, dim=0).float()
    centers = _normalize(centers)
    class_ids = torch.tensor(all_class_ids, dtype=torch.long)
    return centers, class_ids


def compute_separability(centers, class_ids, device="cpu"):
    """Vectorized intra/inter-class cosine-similarity stats.

    Chunked by class to avoid an O(N^2) all-pairs matrix in memory at once
    (N can be ~10k for oxford_flowers at max_K=100).
    """
    centers = centers.to(device)
    class_ids = class_ids.to(device)
    n = centers.shape[0]

    intra_sims = []
    nearest_cross_sims = []
    background_cross_sims = []

    unique_classes = class_ids.unique().tolist()
    for c in unique_classes:
        own_mask = class_ids == c
        own = centers[own_mask]
        other = centers[~own_mask]
        if other.shape[0] == 0:
            continue

        # Intra-class: mean pairwise cosine sim among this class's own centers
        # (excluding self-similarity), only meaningful with >=2 centers.
        if own.shape[0] >= 2:
            sim_own = own @ own.T
            k = own.shape[0]
            off_diag = sim_own[~torch.eye(k, dtype=torch.bool, device=device)]
            intra_sims.append(off_diag.mean().item())

        # Inter-class: for each of this class's centers, nearest similarity
        # among centers of every OTHER class.
        sim_cross = own @ other.T  # [k, n_other]
        nearest = sim_cross.max(dim=1).values
        nearest_cross_sims.extend(nearest.tolist())

        # Background: mean cosine similarity across all cross-class pairs
        # for this class (not just the nearest) — the "generic closeness"
        # baseline.
        background_cross_sims.append(sim_cross.mean().item())

    def _stats(vals):
        if not vals:
            return {"mean": None, "median": None, "n": 0}
        t = torch.tensor(vals)
        return {
            "mean": t.mean().item(),
            "median": t.median().item(),
            "p90": t.kthvalue(max(1, int(0.9 * len(vals)))).values.item(),
            "n": len(vals),
        }

    intra = _stats(intra_sims)
    nearest_cross = _stats(nearest_cross_sims)
    background_cross = _stats(background_cross_sims)

    margin = None
    if intra["mean"] is not None and nearest_cross["mean"] is not None:
        margin = intra["mean"] - nearest_cross["mean"]

    return {
        "n_centers": n,
        "n_classes": len(unique_classes),
        "intra_class_coherence": intra,
        "nearest_cross_class_confusability": nearest_cross,
        "background_cross_class_similarity": background_cross,
        "separability_margin": margin,
    }


def analyze_dataset(dump_path, device="cpu"):
    bank = load_bank(dump_path)
    result = {}
    for label, high_only in (("all_centers", False), ("high_appearance_only", True)):
        centers, class_ids = build_center_pool(bank, high_appearance_only=high_only)
        if centers is None:
            result[label] = None
            continue
        result[label] = compute_separability(centers, class_ids, device=device)
    result["dataset"] = bank["dataset"]
    result["seed"] = bank["seed"]
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v):
    return "N/A" if v is None else f"{v:.3f}"


def render_report(results):
    lines = ["# Patch-Level Prototype Bank — Inter/Intra-Class Separability Diagnostic", ""]
    lines.append(
        "Loads the final per-class Gaussian bank (`centers`/`variance`/"
        "`appearance`/`n_images`) dumped via the opt-in `DUMP_PATCH_BANK` env "
        "var added to `models/patch_modulated_pta.py` (behavior-neutral, "
        "no-op unless set), one seed-1 run per dataset, default "
        "`PatchModPTA-CS` (ProtoAlphaFusion) config."
    )
    lines.append("")
    lines.append(
        "Tests whether a **pure** bank (right-class patches only, per "
        "`outputs/prototype_purity_report.md`) is still geometrically "
        "**confusable** with neighboring classes — the leading candidate "
        "explanation for why patch fusion hurts most on oxford_pets (85.7% "
        "write purity, worst accuracy hit) and helps on dtd (62.6% write "
        "purity, only dataset with a positive effect)."
    )
    lines.append("")

    for subset_label, title in (
        ("all_centers", "## All centers"),
        ("high_appearance_only", "## High-appearance-only centers (top half by appearance weight, per class)"),
    ):
        lines.append(title)
        lines.append("")
        lines.append(
            "| Dataset | Centers | Classes | Intra-class coherence (mean) "
            "| Nearest cross-class confusability (mean/p90) "
            "| Background cross-class sim (mean) | Separability margin |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            r = results[ds][subset_label]
            if r is None:
                lines.append(f"| {ds} | - | - | N/A | N/A | N/A | N/A |")
                continue
            intra = r["intra_class_coherence"]
            near = r["nearest_cross_class_confusability"]
            bg = r["background_cross_class_similarity"]
            lines.append(
                f"| {ds} | {r['n_centers']} | {r['n_classes']} "
                f"| {_fmt(intra['mean'])} "
                f"| {_fmt(near['mean'])} / {_fmt(near['p90'])} "
                f"| {_fmt(bg['mean'])} | {_fmt(r['separability_margin'])} |"
            )
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append(
        "- **Separability margin** = intra-class coherence − nearest cross-class "
        "confusability. Large positive → a class's own centers cluster tighter "
        "than its nearest confusor (geometrically separable). Small/negative → "
        "even a pure bank cannot distinguish this class from a neighbor at the "
        "patch level."
    )
    lines.append(
        "- **Nearest vs. background cross-class similarity**: if nearest ≈ "
        "background, there's no *specific* confusable neighbor — similarity is "
        "just generic high-dimensional closeness. If nearest ≫ background, "
        "specific near-duplicate confusor classes exist (e.g. two visually "
        "similar pet breeds)."
    )
    lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out", default="outputs/prototype_separability_report.md")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    dumps_dir = Path(args.dumps)
    results = {}
    for ds in DATASETS:
        dump_path = dumps_dir / f"{ds}-s1.pt"
        if not dump_path.is_file():
            print(f"[ERROR] missing dump: {dump_path}", file=sys.stderr)
            return 1
        print(f"[analyze_prototype_separability] {ds} ...", file=sys.stderr)
        results[ds] = analyze_dataset(dump_path, device=args.device)

    lines = render_report(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] separability report written to {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] raw numbers written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
