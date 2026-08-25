#!/usr/bin/env python3
"""Variance-aware separability re-check (Experiment A).

`scripts/analyze_prototype_separability.py` compared prototype centers by
plain cosine similarity. The live scoring code (`utils/kmeans.py`,
`_gaussian_score_for_class`) never does that — it scores a patch against a
prototype using a scaled-Mahalanobis distance (center AND per-dimension
variance), then exponentiates. Two prototypes can look identical by cosine
but be much easier (or harder) to tell apart once the target class's
variance is taken into account.

This script recomputes Diagnostic 2 (separability) using that same formula:
for a prototype in class A, "confusability with class B" = how well A's
centroid would score if treated as a query patch against each of B's
prototypes, using B's own variance. This directly reuses
`_gaussian_score_for_class`'s scoring math (imported, not reimplemented).

Reuses `outputs/patch_bank_dumps/` — no new GPU runs.

Usage::

    python scripts/analyze_prototype_separability_v2.py \\
        --dumps outputs/patch_bank_dumps \\
        --out outputs/prototype_separability_v2_report.md
"""

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.kmeans import _safe_normalize  # noqa: E402

DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]

# Matches configs/patch_modulated_pta/patch_modulated_pta.yaml /
# configs/base.yaml — the variance floor used by the live scoring formula.
VARIANCE_MIN = 0.001


def load_bank(dump_path):
    return torch.load(dump_path, map_location="cpu", weights_only=False)


def gaussian_match_score(query, center, variance, variance_min=VARIANCE_MIN):
    """Score a single query vector against one Gaussian prototype.

    Identical formula to `_gaussian_score_for_class`'s per-group scoring:
    ``exp(-0.5 * sum_d (query_d - center_d)^2 / var_d)``, with the same
    dimension-scaled variance floor used there (`variance_min * D`) to avoid
    float32 underflow.
    """
    D = variance.shape[-1]
    effective_var_min = variance_min * D
    var_clamped = variance.clamp(min=effective_var_min)
    diff = query - center
    scaled_maha = (diff.pow(2) / var_clamped).sum(dim=-1)
    return torch.exp(-0.5 * scaled_maha)


def build_pool(bank, high_appearance_only=False):
    """Flatten per-class (center, variance) pairs, optionally filtered."""
    pool = []  # list of (class_id, center[D], variance[D])
    for c, cls in enumerate(bank["classes"]):
        centers = cls["centers"]
        variances = cls["variance"]
        appearance = cls["appearance"]
        if centers.shape[0] == 0:
            continue
        keep = torch.ones(centers.shape[0], dtype=torch.bool)
        if high_appearance_only and centers.shape[0] >= 2:
            keep = appearance >= appearance.median()
        centers = _normalize(centers[keep])
        variances = variances[keep]
        for k in range(centers.shape[0]):
            pool.append((c, centers[k], variances[k]))
    return pool


def _normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def compute_separability_v2(pool):
    """Variance-aware intra/inter-class match-score stats.

    For each prototype P in class A: intra-class match = how well P's
    center scores against each OTHER prototype in A, using that other
    prototype's own variance. Cross-class confusability = the single best
    (highest) score P's center gets against any prototype in any OTHER
    class, using that other prototype's variance — i.e. "if this exact
    centroid showed up as a query at inference time, which other class
    would be most convinced it belongs to them."
    """
    by_class = {}
    for c, center, var in pool:
        by_class.setdefault(c, []).append((center, var))

    intra_scores = []
    nearest_cross_scores = []
    background_cross_scores = []

    classes = list(by_class.keys())
    for c in classes:
        own = by_class[c]
        other = [(cc, center, var) for cc in classes if cc != c for center, var in by_class[cc]]
        if not other:
            continue

        # Intra-class: score each prototype's center against every OTHER
        # prototype in the same class (using that prototype's variance).
        if len(own) >= 2:
            for i, (center_i, _) in enumerate(own):
                for j, (center_j, var_j) in enumerate(own):
                    if i == j:
                        continue
                    s = gaussian_match_score(center_i, center_j, var_j)
                    intra_scores.append(float(s.item()))

        # Cross-class: for each own prototype, its best (nearest) match
        # among every other class's prototypes, and the average match
        # across ALL other-class prototypes (background level).
        for center_i, _ in own:
            best = None
            total = 0.0
            n = 0
            for _, center_j, var_j in other:
                s = float(gaussian_match_score(center_i, center_j, var_j).item())
                total += s
                n += 1
                if best is None or s > best:
                    best = s
            nearest_cross_scores.append(best)
            background_cross_scores.append(total / max(n, 1))

    def _stats(vals):
        if not vals:
            return {"mean": None, "median": None, "n": 0}
        t = torch.tensor(vals)
        return {"mean": t.mean().item(), "median": t.median().item(), "n": len(vals)}

    intra = _stats(intra_scores)
    nearest = _stats(nearest_cross_scores)
    background = _stats(background_cross_scores)
    margin = (
        intra["mean"] - nearest["mean"]
        if intra["mean"] is not None and nearest["mean"] is not None
        else None
    )
    return {
        "n_prototypes": len(pool),
        "n_classes": len(classes),
        "intra_class_match": intra,
        "nearest_cross_class_match": nearest,
        "background_cross_class_match": background,
        "separability_margin": margin,
    }


def analyze_dataset(dump_path):
    bank = load_bank(dump_path)
    result = {}
    for label, high_only in (("all_centers", False), ("high_appearance_only", True)):
        pool = build_pool(bank, high_appearance_only=high_only)
        result[label] = compute_separability_v2(pool) if pool else None
    result["dataset"] = bank["dataset"]
    return result


def _fmt(v):
    return "N/A" if v is None else f"{v:.3f}"


def render_report(results, v1_margins):
    lines = ["# Prototype Separability — Variance-Aware Re-check (Experiment A)", ""]
    lines.append(
        "Same question as `outputs/prototype_separability_report.md`, but "
        "using the live scoring formula (center *and* variance, scaled "
        "Mahalanobis + exponential from `utils/kmeans.py`) instead of plain "
        "cosine similarity between centers. Reuses `outputs/patch_bank_dumps/` "
        "— no new GPU runs."
    )
    lines.append("")
    lines.append(
        "Match score is in the same [0, 1] range the live method actually "
        "computes (1.0 = perfect match). Separability margin = intra-class "
        "match score minus nearest cross-class match score; negative means "
        "a class's own prototypes aren't a better match for each other than "
        "for their nearest neighbor in another class."
    )
    lines.append("")

    for subset_label, title in (
        ("all_centers", "## All prototypes"),
        ("high_appearance_only", "## High-appearance-only prototypes (top half per class)"),
    ):
        lines.append(title)
        lines.append("")
        lines.append(
            "| Dataset | Prototypes | Classes | Intra-class match (mean) "
            "| Nearest cross-class match (mean) | Background cross-class match (mean) "
            "| Separability margin | v1 (cosine-only) margin |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            r = results[ds][subset_label]
            v1 = v1_margins.get(ds, {}).get(subset_label)
            if r is None:
                lines.append(f"| {ds} | - | - | N/A | N/A | N/A | N/A | {_fmt(v1)} |")
                continue
            intra = r["intra_class_match"]
            near = r["nearest_cross_class_match"]
            bg = r["background_cross_class_match"]
            lines.append(
                f"| {ds} | {r['n_prototypes']} | {r['n_classes']} "
                f"| {_fmt(intra['mean'])} | {_fmt(near['mean'])} | {_fmt(bg['mean'])} "
                f"| {_fmt(r['separability_margin'])} | {_fmt(v1)} |"
            )
        lines.append("")

    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="outputs/patch_bank_dumps")
    parser.add_argument("--out", default="outputs/prototype_separability_v2_report.md")
    args = parser.parse_args(argv)

    dumps_dir = Path(args.dumps)
    results = {}
    for ds in DATASETS:
        dump_path = dumps_dir / f"{ds}-s1.pt"
        if not dump_path.is_file():
            print(f"[ERROR] missing dump: {dump_path}", file=sys.stderr)
            return 1
        print(f"[analyze_prototype_separability_v2] {ds} ...", file=sys.stderr)
        results[ds] = analyze_dataset(dump_path)

    # v1 (cosine-only) margins, for side-by-side comparison, hardcoded from
    # outputs/prototype_separability_report.json (already computed).
    v1_json = Path("outputs/prototype_separability_report.json")
    v1_margins = {}
    if v1_json.is_file():
        v1_data = json.loads(v1_json.read_text())
        for ds in DATASETS:
            v1_margins[ds] = {
                "all_centers": v1_data[ds]["all_centers"]["separability_margin"],
                "high_appearance_only": v1_data[ds]["high_appearance_only"]["separability_margin"],
            }

    lines = render_report(results, v1_margins)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] report written to {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] raw numbers written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
