#!/usr/bin/env python3
"""Voting mechanism ablation — compare three fusion strategies offline.

Reads per-sample JSONL records produced by PatchModPTA runs with different
``fusion.type`` settings and compares their accuracy on the same datasets/seeds.

Fusion strategies compared:
  QualityGatedFusion  — baseline, always-on patch (no voting)
  MajorityVoteFusion  — 2-of-3 majority vote among clip/image/patch
  AgreementGateFusion — mute patch when its vote disagrees with CLIP

Output: Markdown report with per-dataset fusion strategy comparison tables
and per-class breakdown showing which classes benefit from voting.

Usage:
    python scripts/analyze_voting_ablation.py --records DIR --out FILE
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUSION_STRATEGIES = ["QualityGated", "MajorityVote", "AgreementGate"]

# Pattern: PatchModPTA-CS-{FusionType}-{dataset}-s{seed}
# Fusion type may contain uppercase letters (e.g., "QualityGated")
_LABEL_RE = re.compile(
    r"^PatchModPTA-CS-(?P<fusion>QualityGated|MajorityVote|AgreementGate)"
    r"-(?P<dataset>.+)-s(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Record loading (duplicated from analyze_records.py to stay self-contained)
# ---------------------------------------------------------------------------

def load_label_records(records_dir, label):
    """Return ``(header, samples)`` for ``DIR/label/records.jsonl``."""
    path = Path(records_dir) / label / "records.jsonl"
    if not path.is_file():
        return None, []
    header = None
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("__header__"):
                header = rec
            else:
                samples.append(rec)
    return header, samples


def list_labels(records_dir):
    """Sorted labels = subdirs containing records.jsonl."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    labels = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "records.jsonl").is_file():
            labels.append(sub.name)
    return labels


def parse_label(label):
    """Extract (fusion_type, dataset, seed) from a label string, or None."""
    m = _LABEL_RE.match(label)
    if not m:
        return None
    return m.group("fusion"), m.group("dataset"), int(m.group("seed"))


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def compute_accuracy(samples):
    """Overall accuracy (0-100) from correct flags."""
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.get("correct"))
    return 100.0 * correct / len(samples)


def per_class_accuracy(samples, num_classes=None):
    """Return {class_idx: (correct, total)} dict."""
    counts = defaultdict(lambda: [0, 0])
    for s in samples:
        target = s["target"]
        counts[target][1] += 1
        if s.get("correct"):
            counts[target][0] += 1
    if num_classes is not None:
        for c in range(num_classes):
            _ = counts[c]  # ensure key exists
    return dict(counts)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_records(records_dir):
    """Parse all fusion-variant records and return structured results.

    Returns:
        groups: dict keyed by (dataset, seed) -> {fusion_type: samples}
        all_classnames: dict keyed by (dataset, seed) -> classnames list
    """
    labels = list_labels(records_dir)
    groups = defaultdict(dict)
    all_classnames = {}

    for label in labels:
        parsed = parse_label(label)
        if parsed is None:
            continue
        fusion_type, dataset, seed = parsed
        key = (dataset, seed)
        header, samples = load_label_records(records_dir, label)
        if not samples:
            continue
        groups[key][fusion_type] = samples
        # Extract classnames from header
        cn = (header or {}).get("classnames") or []
        if cn and key not in all_classnames:
            all_classnames[key] = cn

    return dict(groups), all_classnames


def build_comparison_table(groups):
    """Build per-dataset comparison rows.

    Returns list of dicts, one per (dataset, seed):
        dataset, seed, n_samples,
        acc_QualityGated, acc_MajorityVote, acc_AgreementGate,
        delta_MajorityVote, delta_AgreementGate
    """
    rows = []
    for (dataset, seed) in sorted(groups.keys()):
        variants = groups[(dataset, seed)]
        n = min(len(s) for s in variants.values()) if variants else 0
        accs = {}
        for strat in FUSION_STRATEGIES:
            samples = variants.get(strat, [])
            accs[strat] = compute_accuracy(samples) if samples else None

        base = accs.get("QualityGated")
        row = {
            "dataset": dataset,
            "seed": seed,
            "n": n,
            "acc_QualityGated": base,
            "acc_MajorityVote": accs.get("MajorityVote"),
            "acc_AgreementGate": accs.get("AgreementGate"),
        }
        if base is not None:
            row["delta_MajorityVote"] = (accs.get("MajorityVote") or 0) - base
            row["delta_AgreementGate"] = (accs.get("AgreementGate") or 0) - base
        else:
            row["delta_MajorityVote"] = None
            row["delta_AgreementGate"] = None
        rows.append(row)
    return rows


def per_class_breakdown(groups):
    """Per-class comparison across fusion strategies.

    Returns: dict keyed by (dataset, seed) -> {class_idx: {strat: (correct, total)}}
    """
    breakdown = {}
    for (dataset, seed) in sorted(groups.keys()):
        variants = groups[(dataset, seed)]
        # Determine num_classes from any variant
        num_classes = 0
        for samples in variants.values():
            for s in samples:
                num_classes = max(num_classes, s["target"] + 1)
        class_data = {}
        for c in range(num_classes):
            class_data[c] = {}
            for strat in FUSION_STRATEGIES:
                samples = variants.get(strat, [])
                class_acc = per_class_accuracy(samples, num_classes)
                class_data[c][strat] = class_acc.get(c, [0, 0])
        breakdown[(dataset, seed)] = class_data
    return breakdown


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def format_markdown(groups, rows, breakdown, classnames_map):
    """Generate markdown report string."""
    lines = []
    lines.append("# Voting Mechanism Ablation — Fusion Strategy Comparison")
    lines.append("")
    lines.append("## Fusion Strategy Comparison")
    lines.append("")
    lines.append("Compares three PatchModPTA fusion policies on the same datasets and seeds.")
    lines.append("")

    # Summary table
    lines.append("### Overall Accuracy")
    lines.append("")
    lines.append(
        "| Dataset | Seed | N | QualityGated | MajorityVote | AgreementGate "
        "| Δ Vote | Δ Gate |"
    )
    lines.append(
        "|---------|------|---|-------------|-------------|--------------"
        "|--------|--------|"
    )
    for r in rows:
        qg = f"{r['acc_QualityGated']:.1f}%" if r["acc_QualityGated"] is not None else "N/A"
        mv = f"{r['acc_MajorityVote']:.1f}%" if r["acc_MajorityVote"] is not None else "N/A"
        ag = f"{r['acc_AgreementGate']:.1f}%" if r["acc_AgreementGate"] is not None else "N/A"
        dm = f"{r['delta_MajorityVote']:+.1f}" if r["delta_MajorityVote"] is not None else "N/A"
        da = f"{r['delta_AgreementGate']:+.1f}" if r["delta_AgreementGate"] is not None else "N/A"
        lines.append(
            f"| {r['dataset']} | {r['seed']} | {r['n']} "
            f"| {qg} | {mv} | {ag} | {dm} | {da} |"
        )
    lines.append("")

    # Per-class breakdown
    lines.append("### Per-Class Breakdown — Which Classes Benefit from Voting?")
    lines.append("")
    lines.append(
        "Positive Δ means the voting strategy improves over QualityGated baseline."
    )
    lines.append("")

    for (dataset, seed) in sorted(breakdown.keys()):
        class_data = breakdown[(dataset, seed)]
        cn = classnames_map.get((dataset, seed), [])

        lines.append(f"#### {dataset} (seed {seed})")
        lines.append("")
        lines.append("| Class | Name | QG Acc | MV Acc | MV Δ | AG Acc | AG Δ |")
        lines.append("|-------|------|--------|--------|------|--------|------|")

        for c in sorted(class_data.keys()):
            cdata = class_data[c]
            qg_c, qg_t = cdata.get("QualityGated", (0, 0))
            mv_c, mv_t = cdata.get("MajorityVote", (0, 0))
            ag_c, ag_t = cdata.get("AgreementGate", (0, 0))

            qg_acc = (100.0 * qg_c / qg_t) if qg_t > 0 else 0.0
            mv_acc = (100.0 * mv_c / mv_t) if mv_t > 0 else 0.0
            ag_acc = (100.0 * ag_c / ag_t) if ag_t > 0 else 0.0

            name = cn[c] if c < len(cn) else str(c)
            dm = mv_acc - qg_acc
            da = ag_acc - qg_acc

            lines.append(
                f"| {c} | {name} | {qg_acc:.0f}% "
                f"| {mv_acc:.0f}% | {dm:+.0f} | {ag_acc:.0f}% | {da:+.0f} |"
            )
        lines.append("")

    # Summary
    n_groups = len(rows)
    n_improved_mv = sum(1 for r in rows if (r.get("delta_MajorityVote") or 0) > 0)
    n_improved_ag = sum(1 for r in rows if (r.get("delta_AgreementGate") or 0) > 0)
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- Datasets evaluated: {n_groups}")
    lines.append(f"- MajorityVote improves over QualityGated: {n_improved_mv}/{n_groups}")
    lines.append(f"- AgreementGate improves over QualityGated: {n_improved_ag}/{n_groups}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voting mechanism ablation — compare fusion strategies offline."
    )
    parser.add_argument(
        "--records", required=True,
        help="Record directory containing PatchModPTA-CS-*-{fusion}-{dataset}-s{seed}/"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output markdown file path"
    )
    args = parser.parse_args()

    records_dir = args.records
    out_path = Path(args.out)

    groups, classnames_map = analyze_records(records_dir)
    if not groups:
        print(f"[ERROR] No fusion-variant records found in {records_dir}", file=sys.stderr)
        sys.exit(1)

    rows = build_comparison_table(groups)
    breakdown = per_class_breakdown(groups)
    report = format_markdown(groups, rows, breakdown, classnames_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] Fusion strategy comparison written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
