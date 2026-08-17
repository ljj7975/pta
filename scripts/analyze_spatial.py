#!/usr/bin/env python3
"""
analyze_spatial.py — Spatial sensitivity analysis (Exp 1.3).

Uses existing PatchModPTA records as proxy for spatial signal analysis.
Compares image_proto (global spatial) vs patch_proto (local spatial)
predictions and correlates with quality_gate values.

Sections:
  - Spatial sensitivity summary table
  - Local helps / Local hurts breakdown
  - quality_gate correlation statistics
  - Per-label detail tables

Usage:
    python scripts/analyze_spatial.py --records tests/fixtures/records_sample/ --out outputs/spatial.md
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reuse existing loading infrastructure from analyze_records.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_records import (  # noqa: E402
    list_labels,
    load_label_records,
    classify_label,
)

METHOD_PATCH = "PatchModPTA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def argmax_idx(vec):
    """Index of the maximum value in a list; None for empty/missing."""
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def _fmt(v, dp=2):
    """Format a float to dp decimal places, or '—' for None."""
    if v is None:
        return "—"
    return "{:.{}f}".format(v, dp)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def classify_spatial(sample):
    """Classify a single sample into a spatial role.

    Returns (role, quality_gate, img_correct, patch_correct) where role is one of:
      'local_helps'   — patch_proto correct, image_proto wrong
      'local_hurts'   — image_proto correct, patch_proto wrong
      'both_correct'  — both correct
      'both_wrong'    — both wrong
      'skip'          — patch_proto is null (not a PatchModPTA record)

    Also returns the sample's quality_gate value (or None).
    """
    lg = sample.get("logits") or {}
    image_proto = lg.get("image_proto")
    patch_proto = lg.get("patch_proto")

    if patch_proto is None:
        return "skip", None, None, None

    img_pred = argmax_idx(image_proto)
    patch_pred = argmax_idx(patch_proto)
    target = sample.get("target")

    if img_pred is None or patch_pred is None or target is None:
        return "skip", None, None, None

    img_correct = (img_pred == target)
    patch_correct = (patch_pred == target)
    quality_gate = sample.get("quality_gate")

    if patch_correct and not img_correct:
        return "local_helps", quality_gate, img_correct, patch_correct
    elif img_correct and not patch_correct:
        return "local_hurts", quality_gate, img_correct, patch_correct
    elif img_correct and patch_correct:
        return "both_correct", quality_gate, img_correct, patch_correct
    else:
        return "both_wrong", quality_gate, img_correct, patch_correct


def analyze_label(label, samples):
    """Compute spatial sensitivity stats for one label.

    Returns a dict with counts, quality_gate stats per role, etc.
    """
    counts = {"local_helps": 0, "local_hurts": 0, "both_correct": 0, "both_wrong": 0, "skip": 0}
    qg_by_role = {"local_helps": [], "local_hurts": [], "both_correct": [], "both_wrong": []}
    total_with_patch = 0

    for s in samples:
        role, qg, _, _ = classify_spatial(s)
        counts[role] += 1
        if role == "skip":
            continue
        total_with_patch += 1
        if qg is not None:
            qg_by_role[role].append(qg)

    # Aggregate quality_gate stats
    qg_stats = {}
    for role in ("local_helps", "local_hurts", "both_correct", "both_wrong"):
        vals = qg_by_role[role]
        if vals:
            qg_stats[role] = {
                "n": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        else:
            qg_stats[role] = {"n": 0, "mean": None, "min": None, "max": None}

    return {
        "label": label,
        "total": len(samples),
        "total_with_patch": total_with_patch,
        "total_skipped": counts["skip"],
        "counts": counts,
        "qg_stats": qg_stats,
        "qg_raw": qg_by_role,  # raw lists for aggregation
    }


def compute_global_correlation(all_qg_pairs):
    """Compute Pearson correlation between quality_gate and correctness.

    ``all_qg_pairs`` is a list of (quality_gate, correct) tuples.
    Returns (rho, n) where rho is the Pearson correlation (or 0.0 if
    insufficient data).
    """
    if len(all_qg_pairs) < 3:
        return 0.0, len(all_qg_pairs)
    vals = [(float(qg), 1.0 if correct else 0.0) for qg, correct in all_qg_pairs]
    n = len(vals)
    mx = sum(v[0] for v in vals) / n
    my = sum(v[1] for v in vals) / n
    num = sum((v[0] - mx) * (v[1] - my) for v in vals)
    dx = sum((v[0] - mx) ** 2 for v in vals) ** 0.5
    dy = sum((v[1] - my) ** 2 for v in vals) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0, n
    return num / (dx * dy), n


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(label_results, global_corr, global_n):
    """Build the full markdown report as a list of lines."""
    lines = []
    lines.append("# Spatial Sensitivity Analysis (Exp 1.3)")
    lines.append("")
    lines.append("Spatial signal analysis using PatchModPTA records as proxy.")
    lines.append("Compares image_proto (global spatial) vs patch_proto (local spatial) predictions.")
    lines.append("")

    # Aggregate totals
    total_all = sum(r["total"] for r in label_results)
    total_with_patch = sum(r["total_with_patch"] for r in label_results)
    agg_counts = {"local_helps": 0, "local_hurts": 0, "both_correct": 0, "both_wrong": 0}
    for r in label_results:
        for k in agg_counts:
            agg_counts[k] += r["counts"][k]

    lines.append("## Aggregate Spatial Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append("| Total samples (all labels) | {} |".format(total_all))
    lines.append("| Samples with patch_proto | {} |".format(total_with_patch))
    lines.append("| Samples skipped (null patch_proto) | {} |".format(total_all - total_with_patch))
    lines.append("| Local helps (patch correct, image wrong) | {} |".format(agg_counts["local_helps"]))
    lines.append("| Local hurts (image correct, patch wrong) | {} |".format(agg_counts["local_hurts"]))
    lines.append("| Both correct | {} |".format(agg_counts["both_correct"]))
    lines.append("| Both wrong | {} |".format(agg_counts["both_wrong"]))
    lines.append("")

    # Local Helps / Local Hurts section
    lines.append("## Local Helps")
    lines.append("")
    lines.append("Samples where patch_proto prediction is correct but image_proto is wrong.")
    lines.append("These are cases where local (patch-level) spatial signal corrects the global prediction.")
    lines.append("")
    n_helps = agg_counts["local_helps"]
    if n_helps > 0:
        helps_qg = []
        for r in label_results:
            helps_qg.extend(r.get("qg_raw", {}).get("local_helps", []))
        lines.append("| Stat | Value |")
        lines.append("|---|---|")
        lines.append("| Count | {} |".format(n_helps))
        lines.append("| Mean quality_gate | {} |".format(_fmt(sum(helps_qg) / len(helps_qg) if helps_qg else None)))
        lines.append("| Min quality_gate | {} |".format(_fmt(min(helps_qg) if helps_qg else None)))
        lines.append("| Max quality_gate | {} |".format(_fmt(max(helps_qg) if helps_qg else None)))
    else:
        lines.append("No local-helps cases found.")
    lines.append("")

    lines.append("## Local Hurts")
    lines.append("")
    lines.append("Samples where image_proto prediction is correct but patch_proto is wrong.")
    lines.append("These are cases where local (patch-level) spatial signal degrades the global prediction.")
    lines.append("")
    n_hurts = agg_counts["local_hurts"]
    if n_hurts > 0:
        hurts_qg = []
        for r in label_results:
            hurts_qg.extend(r.get("qg_raw", {}).get("local_hurts", []))
        lines.append("| Stat | Value |")
        lines.append("|---|---|")
        lines.append("| Count | {} |".format(n_hurts))
        lines.append("| Mean quality_gate | {} |".format(_fmt(sum(hurts_qg) / len(hurts_qg) if hurts_qg else None)))
        lines.append("| Min quality_gate | {} |".format(_fmt(min(hurts_qg) if hurts_qg else None)))
        lines.append("| Max quality_gate | {} |".format(_fmt(max(hurts_qg) if hurts_qg else None)))
    else:
        lines.append("No local-hurts cases found.")
    lines.append("")

    # quality_gate correlation statistics
    lines.append("## Quality Gate Correlation Statistics")
    lines.append("")
    lines.append("Pearson correlation between quality_gate value and sample correctness (final prediction).")
    lines.append("High positive correlation indicates quality_gate is discriminative: higher gate values")
    lines.append("correspond to correct predictions.")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("|---|---|")
    lines.append("| Pearson rho (quality_gate vs correct) | {} |".format(_fmt(global_corr)))
    lines.append("| N samples | {} |".format(global_n))
    lines.append("")

    # Interpretation
    if global_n >= 3:
        if global_corr > 0.3:
            verdict = "Strong positive — quality_gate is discriminative"
        elif global_corr > 0.1:
            verdict = "Moderate positive — quality_gate has some discriminative power"
        elif global_corr > -0.1:
            verdict = "Weak/no correlation — quality_gate not strongly discriminative"
        else:
            verdict = "Negative correlation — quality_gate may be anti-correlated with correctness"
        lines.append("**Interpretation:** {}".format(verdict))
    else:
        lines.append("**Interpretation:** Insufficient data for reliable correlation (N < 3).")
    lines.append("")

    # Per-label detail
    lines.append("## Per-Label Detail")
    lines.append("")
    for r in label_results:
        lines.append("### {}".format(r["label"]))
        lines.append("")
        lines.append("| Role | Count | Mean quality_gate | Min | Max |")
        lines.append("|---|---|---|---|---|")
        for role in ("local_helps", "local_hurts", "both_correct", "both_wrong"):
            qs = r["qg_stats"][role]
            lines.append("| {} | {} | {} | {} | {} |".format(
                role, qs["n"], _fmt(qs["mean"]), _fmt(qs["min"]), _fmt(qs["max"])))
        lines.append("")
        skipped = r["total_skipped"]
        if skipped > 0:
            lines.append("*{} samples skipped (null patch_proto — non-PatchModPTA method)*".format(skipped))
        lines.append("")

    return lines


# ---------------------------------------------------------------------------

def analyze_all(records_dir):
    """Run spatial analysis across all PatchModPTA labels.

    Returns (label_results, raw_qg_pairs) where raw_qg_pairs is a list of
    (quality_gate, correct) tuples for global correlation.
    """
    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return None, None

    label_results = []
    raw_qg_pairs = []  # (quality_gate, correct) for global correlation

    for label in labels:
        method = classify_label(label)
        if method != METHOD_PATCH:
            continue

        header, samples = load_label_records(records_dir, label)
        if not samples:
            continue

        # Per-label analysis
        result = analyze_label(label, samples)
        label_results.append(result)

        # Collect raw quality_gate + correctness pairs for global correlation
        for s in samples:
            lg = s.get("logits") or {}
            if lg.get("patch_proto") is None:
                continue  # skip non-patch records
            qg = s.get("quality_gate")
            if qg is None:
                continue
            correct = bool(s.get("correct", False))
            raw_qg_pairs.append((qg, correct))

    if not label_results:
        print("[ERROR] no PatchModPTA records with patch_proto found", file=sys.stderr)
        return None, None

    # Global correlation
    global_corr, global_n = compute_global_correlation(raw_qg_pairs)

    return label_results, (global_corr, global_n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spatial sensitivity analysis (Exp 1.3) using PatchModPTA records.")
    parser.add_argument("--records", required=True,
                        help="Record directory (DIR/<LABEL>/records.jsonl)")
    parser.add_argument("--out", required=True,
                        help="Markdown output file path")
    args = parser.parse_args(argv)

    records_dir = Path(args.records)
    if not records_dir.is_dir():
        print("[ERROR] records dir not found: {}".format(records_dir), file=sys.stderr)
        return 1

    result = analyze_all(records_dir)
    if result[0] is None:
        return 1
    label_results, (global_corr, global_n) = result

    lines = render_markdown(label_results, global_corr, global_n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] spatial analysis written to {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
