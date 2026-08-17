#!/usr/bin/env python3
"""Component ablation analysis (Exp 4.1): compare three method configurations.

Configurations compared:
  1. PTA                  — image-level only (no patch, no quality gate)
  2. PatchModPTA+QG       — patch + quality gate (QualityGatedFusion)
  3. PatchModPTA+Vote     — patch + quality gate + voting (MajorityVoteFusion)

For each dataset the script computes:
  - Accuracy per configuration
  - Accuracy deltas: +patch+QG, +voting, total
  - Cumulative contribution of each component

Usage:
    python scripts/analyze_component_ablation.py \\
        --records tests/fixtures/records_sample/ \\
        --out outputs/component_ablation.md
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Label matching patterns
# ---------------------------------------------------------------------------

# PTA baseline: PTA-CS-<dataset>-s<N>
_PTA_RE = re.compile(r"^PTA-CS-(?P<ds>.+)-s(?P<seed>\d+)$")

# PatchModPTA with QualityGatedFusion: PatchModPTA-CS-QualityGated-<dataset>-s<N>
_QG_RE = re.compile(r"^PatchModPTA-CS-QualityGated-(?P<ds>.+)-s(?P<seed>\d+)$")

# PatchModPTA with MajorityVoteFusion: PatchModPTA-CS-MajorityVote-<dataset>-s<N>
_VOTE_RE = re.compile(r"^PatchModPTA-CS-MajorityVote-(?P<ds>.+)-s(?P<seed>\d+)$")

CONFIGS = [
    ("PTA", _PTA_RE, "Image-level only (baseline)"),
    ("PatchModPTA+QG", _QG_RE, "Patch + quality gate"),
    ("PatchModPTA+Vote", _VOTE_RE, "Patch + quality gate + voting"),
]


# ---------------------------------------------------------------------------
# Record loading (reuses pattern from analyze_records.py)
# ---------------------------------------------------------------------------

def list_labels(records_dir):
    """Sorted labels = subdirs of records_dir containing records.jsonl."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "records.jsonl").is_file()
    )


def load_label_records(records_dir, label):
    """Return (header, samples) for DIR/label/records.jsonl."""
    path = Path(records_dir) / label / "records.jsonl"
    header = None
    samples = []
    try:
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
    except FileNotFoundError:
        print("[WARN] records file not found: {}".format(path), file=sys.stderr)
    except OSError as e:
        print("[WARN] error reading records file {}: {}".format(path, e),
              file=sys.stderr)
    return header, samples


def overall_acc(samples):
    """Overall accuracy as a percentage float (None if empty)."""
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


def per_class_stats(samples):
    """{target_idx: {"total": int, "correct": int}} recomputed from JSONL."""
    stats = {}
    for s in samples:
        t = int(s.get("target", -1))
        st = stats.setdefault(t, {"total": 0, "correct": 0})
        st["total"] += 1
        if s.get("correct"):
            st["correct"] += 1
    return stats


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_label(label):
    """Map label to (config_name, dataset, seed) or None."""
    for config_name, regex, _desc in CONFIGS:
        m = regex.match(label)
        if m:
            return config_name, m.group("ds"), int(m.group("seed"))
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(v):
    """Format a float value, or dash if None."""
    if v is None:
        return "—"
    return "{:.2f}".format(v)


def _fmt_delta(v):
    """Format a delta with sign, or dash if None."""
    if v is None:
        return "—"
    return "{:+.2f}".format(v)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records_dir):
    """Compute per-dataset, per-config accuracy and deltas.

    Returns (dataset_results, all_labels_by_config):
        dataset_results: {dataset: {config_name: {"acc": float, "n": int,
                           "seeds": {seed: acc}, "per_class": {...}}}}
        all_labels_by_config: {config_name: [label, ...]}
    """
    labels = list_labels(records_dir)
    if not labels:
        return {}, {}

    # Group labels by config
    by_config = defaultdict(list)  # config_name -> [(dataset, seed, label)]
    for label in labels:
        result = classify_label(label)
        if result is None:
            continue
        config_name, dataset, seed = result
        by_config[config_name].append((dataset, seed, label))

    # Per-dataset results
    dataset_results = defaultdict(dict)  # dataset -> config_name -> {acc, n, ...}
    all_labels_by_config = defaultdict(list)

    for config_name, entries in by_config.items():
        for dataset, seed, label in entries:
            header, samples = load_label_records(records_dir, label)
            if not samples:
                continue
            acc = overall_acc(samples)
            n = len(samples)
            pc = per_class_stats(samples)
            all_labels_by_config[config_name].append(label)

            ds_cfg = dataset_results[dataset].setdefault(config_name, {
                "accs": [],
                "total": 0,
                "per_class": defaultdict(lambda: {"total": 0, "correct": 0}),
            })
            ds_cfg["accs"].append(acc if acc is not None else 0.0)
            ds_cfg["total"] += n
            for cls_idx, st in pc.items():
                ds_cfg["per_class"][cls_idx]["total"] += st["total"]
                ds_cfg["per_class"][cls_idx]["correct"] += st["correct"]

    return dict(dataset_results), dict(all_labels_by_config)


def build_report(dataset_results, all_labels_by_config):
    """Build markdown report lines."""
    lines = []
    lines.append("# Component Ablation (Exp 4.1)")
    lines.append("")
    lines.append("Compares three method configurations:")
    lines.append("")
    for config_name, _regex, desc in CONFIGS:
        lines.append("- **{}** — {}".format(config_name, desc))
    lines.append("")
    lines.append("Cumulative component contribution:")
    lines.append("")
    lines.append("1. **+patch+QG**: PatchModPTA+QG minus PTA (adds patch-level prototypes "
                 "and quality gating)")
    lines.append("2. **+voting**: PatchModPTA+Vote minus PatchModPTA+QG "
                 "(adds majority vote fusion)")
    lines.append("")

    # ── Summary table ──────────────────────────────────────────────────
    lines.append("## Per-Dataset Accuracy Summary")
    lines.append("")

    # Header row
    header_cols = ["dataset"]
    for config_name, _r, _d in CONFIGS:
        header_cols.append("{} acc (%)".format(config_name))
    header_cols.extend(["+patch+QG (pp)", "+voting (pp)", "total (pp)"])
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    # Sort datasets alphabetically
    sorted_datasets = sorted(dataset_results.keys())
    config_names = [c[0] for c in CONFIGS]

    acc_by_config_all = defaultdict(list)  # config_name -> [acc per dataset]

    for ds in sorted_datasets:
        ds_data = dataset_results[ds]
        row = [ds]
        accs = {}
        for cn in config_names:
            info = ds_data.get(cn)
            if info and info["accs"]:
                avg_acc = sum(info["accs"]) / len(info["accs"])
            else:
                avg_acc = None
            accs[cn] = avg_acc
            row.append(_fmt(avg_acc))
            if avg_acc is not None:
                acc_by_config_all[cn].append(avg_acc)

        # Deltas
        pta_acc = accs.get("PTA")
        qg_acc = accs.get("PatchModPTA+QG")
        vote_acc = accs.get("PatchModPTA+Vote")

        delta_patch_qg = (qg_acc - pta_acc) if (qg_acc is not None and pta_acc is not None) else None
        delta_vote = (vote_acc - qg_acc) if (vote_acc is not None and qg_acc is not None) else None
        delta_total = (vote_acc - pta_acc) if (vote_acc is not None and pta_acc is not None) else None

        row.append(_fmt_delta(delta_patch_qg))
        row.append(_fmt_delta(delta_vote))
        row.append(_fmt_delta(delta_total))
        lines.append("| " + " | ".join(row) + " |")

    # ── Mean row ───────────────────────────────────────────────────────
    mean_row = ["**Mean**"]
    mean_accs = {}
    for cn in config_names:
        vals = acc_by_config_all.get(cn, [])
        mean_v = sum(vals) / len(vals) if vals else None
        mean_accs[cn] = mean_v
        mean_row.append("**{}**".format(_fmt(mean_v)))

    m_pta = mean_accs.get("PTA")
    m_qg = mean_accs.get("PatchModPTA+QG")
    m_vote = mean_accs.get("PatchModPTA+Vote")
    mean_row.append("**{}**".format(_fmt_delta(
        (m_qg - m_pta) if (m_qg is not None and m_pta is not None) else None)))
    mean_row.append("**{}**".format(_fmt_delta(
        (m_vote - m_qg) if (m_vote is not None and m_qg is not None) else None)))
    mean_row.append("**{}**".format(_fmt_delta(
        (m_vote - m_pta) if (m_vote is not None and m_pta is not None) else None)))
    lines.append("| " + " | ".join(mean_row) + " |")
    lines.append("")

    # ── Labels used ────────────────────────────────────────────────────
    lines.append("## Labels Used")
    lines.append("")
    for cn in config_names:
        lbls = all_labels_by_config.get(cn, [])
        lines.append("- **{}**: {}".format(cn, ", ".join(lbls) if lbls else "(none)"))
    lines.append("")

    # ── Per-dataset detail ─────────────────────────────────────────────
    lines.append("## Per-Dataset Component Breakdown")
    lines.append("")
    for ds in sorted_datasets:
        ds_data = dataset_results[ds]
        lines.append("### {}".format(ds))
        lines.append("")
        lines.append("| config | seeds | n_samples | acc (%) |")
        lines.append("|---|---|---|---|")
        for cn in config_names:
            info = ds_data.get(cn)
            if info and info["accs"]:
                n_seeds = len(info["accs"])
                avg_acc = sum(info["accs"]) / n_seeds
                lines.append("| {} | {} | {} | {} |".format(
                    cn, n_seeds, info["total"], _fmt(avg_acc)))
            else:
                lines.append("| {} | 0 | 0 | — |".format(cn))
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Component ablation analysis (Exp 4.1): "
                    "compare PTA vs PatchModPTA+QG vs PatchModPTA+Vote.")
    parser.add_argument(
        "--records", required=True,
        help="Record dir containing DIR/<LABEL>/records.jsonl")
    parser.add_argument(
        "--out", required=True,
        help="Output markdown file path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records_dir = Path(args.records)
    out_path = Path(args.out)

    if not records_dir.is_dir():
        print("[ERROR] records dir not found: {}".format(records_dir),
              file=sys.stderr)
        return 1

    dataset_results, all_labels_by_config = analyze(str(records_dir))
    if not dataset_results:
        print("[ERROR] no records found for any component configuration",
              file=sys.stderr)
        return 1

    lines = build_report(dataset_results, all_labels_by_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] component ablation written to {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
