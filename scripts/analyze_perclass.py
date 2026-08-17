#!/usr/bin/env python3
"""
analyze_perclass.py — Per-class accuracy analysis comparing PTA vs PatchModPTA.

Loads records for PTA and PatchModPTA (main + voting variants) across all
datasets, computes per-class accuracy, identifies hard classes, and analyzes
patch-level contribution for those classes.

CLI:
    python scripts/analyze_perclass.py --records DIR --out FILE

Outputs:
    FILE          — Markdown report with per-class accuracy tables
    FILE.csv      — CSV with columns: class, pta_acc, patch_acc, delta
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Reuse patterns from analyze_records.py
# ---------------------------------------------------------------------------

PATCH_PREFIX = "PatchModPTA-CS"
PTA_PREFIX = "PTA-CS"
ZERO_PREFIX = "ZeroShot-CS"
ABLATION_MARKERS = ("-alpha10", "-multigate")

METHOD_PATCH = "PatchModPTA"
METHOD_PTA = "PTA"
METHOD_ZERO = "ZeroShot"

HARD_K_DEFAULT = 5


# ---------------------------------------------------------------------------
# Record loading (adapted from analyze_records.py)
# ---------------------------------------------------------------------------

def list_labels(records_dir):
    """Sorted labels = subdirs of records_dir containing records.jsonl."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    labels = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "records.jsonl").is_file():
            labels.append(sub.name)
    return labels


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


def classify_label(label):
    """Map a record-dir label to a method family; None for ablations/unknown."""
    if not label:
        return None
    if any(m in label for m in ABLATION_MARKERS):
        return None
    if label.startswith(PATCH_PREFIX):
        return METHOD_PATCH
    if label.startswith(PTA_PREFIX):
        return METHOD_PTA
    if label.startswith(ZERO_PREFIX):
        return METHOD_ZERO
    return None


def label_seed(label, header):
    """Run seed from header or label suffix."""
    h = header or {}
    if h.get("seed") is not None:
        try:
            return int(h["seed"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"-s(\d+)(?:-|$)", label)
    if m:
        return int(m.group(1))
    return None


def header_classnames(header):
    cn = (header or {}).get("classnames") or []
    return [str(c) for c in cn]


def dataset_classnames_fallback(header, data_root=None):
    """Best-effort classnames from the datasets module."""
    try:
        dataset = (header or {}).get("dataset") or "dtd"
        from datasets import build_dataset
        if data_root is None:
            data_root = Path.cwd() / "data"
        d = build_dataset(dataset, str(data_root))
        cn = list(getattr(d, "classnames", []))
        return [str(c) for c in cn] if cn else []
    except Exception:
        return []


def resolve_classnames(records_dir, labels, headers):
    """Dataset-ordered classnames: first non-empty header list, else lookup.

    Returns [] when no classnames can be resolved (instead of None) so
    callers never need to guard against None.
    """
    del labels
    for h in headers:
        cn = header_classnames(h)
        if cn:
            return cn
    for h in headers:
        cn = dataset_classnames_fallback(h, records_dir)
        if cn:
            return cn
    return []


def _classname_of(classnames, idx):
    if classnames and idx < len(classnames):
        return classnames[idx]
    return "class_{}".format(idx)


def _name_sort_key(name):
    m = re.match(r"^class_(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


# ---------------------------------------------------------------------------
# Per-class statistics
# ---------------------------------------------------------------------------

def per_class_stats(samples):
    """{target_idx: {"total": int, "correct": int}}"""
    stats = {}
    for s in samples:
        t = int(s.get("target", -1))
        st = stats.setdefault(t, {"total": 0, "correct": 0})
        st["total"] += 1
        if s.get("correct"):
            st["correct"] += 1
    return stats


def per_class_named(samples, classnames):
    """{classname: {"total", "correct", "acc"}} in dataset-name order."""
    stats = per_class_stats(samples)
    out = {}
    for t, st in stats.items():
        name = _classname_of(classnames, t)
        out[name] = {
            "total": st["total"],
            "correct": st["correct"],
            "acc": 100.0 * st["correct"] / st["total"] if st["total"] else 0.0,
        }
    return out


def overall_acc(samples):
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


def _class_means(by_label):
    """{classname: mean acc across labels} from {label: {name: st}}."""
    accs = defaultdict(list)
    for named in by_label.values():
        for name, st in named.items():
            accs[name].append(st["acc"])
    return {name: sum(v) / len(v) for name, v in accs.items()}


def _fmt(v):
    return "{:.2f}".format(v) if v is not None else "---"


def _fmt_delta(v):
    if v is None:
        return "---"
    return "{:+.2f}".format(v)


# ---------------------------------------------------------------------------
# Patch contribution analysis
# ---------------------------------------------------------------------------

def argmax_idx(vec):
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def patch_contribution_for_class(
    patch_samples: List[Dict[str, Any]], target_idx: int
) -> Dict[str, Any]:
    """Analyze patch-level contribution for a specific class.

    For each sample of this class, check:
    - Does clip.argmax match ground truth?
    - Does patch_proto.argmax match ground truth when clip doesn't?
    - Final prediction correct?

    Returns dict with patch_helps, clip_wrong_count, etc.
    """
    n = 0
    clip_correct = 0
    clip_wrong = 0
    patch_helps = 0
    final_correct = 0
    n_clusters_samples = 0
    clusters_list: List[int] = []  # n_clusters values for correlation

    for s in patch_samples:
        t = int(s.get("target", -1))
        if t != target_idx:
            continue
        n += 1

        lg = s.get("logits") or {}
        clip = lg.get("clip")
        patch = lg.get("patch_proto")

        if not clip:
            continue

        clip_pred = argmax_idx(clip)
        if clip_pred is None:
            continue

        is_clip_correct = (clip_pred == t)
        if is_clip_correct:
            clip_correct += 1
        else:
            clip_wrong += 1

        # Patch contribution: only when clip is wrong AND patch exists
        if patch is not None and not is_clip_correct:
            patch_pred = argmax_idx(patch)
            if patch_pred is not None and patch_pred == t:
                patch_helps += 1

        # Final prediction
        final = lg.get("final")
        if final:
            final_pred = argmax_idx(final)
            if final_pred is not None and final_pred == t:
                final_correct += 1

        # Cluster stats
        ps = (s.get("proto_stats") or {}).get("true") or {}
        ncl = ps.get("n_clusters")
        if ncl is not None:
            n_clusters_samples += 1
            clusters_list.append(ncl)

    return {
        "n": n,
        "clip_correct": clip_correct,
        "clip_wrong": clip_wrong,
        "patch_helps": patch_helps,
        "final_correct": final_correct,
        "n_clusters_samples": n_clusters_samples,
        "clusters_list": clusters_list,
    }


def _pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Collect method records
# ---------------------------------------------------------------------------

def collect_method_samples(records_dir, labels, method):
    """Aggregate all samples for a given method across seeds/variants.

    Returns list of (label, header, samples) tuples.
    """
    results = []
    for label in labels:
        if classify_label(label) != method:
            continue
        h, samples = load_label_records(records_dir, label)
        if samples:
            results.append((label, h, samples))
    return results


def is_voting_variant(label):
    """Check if a PatchModPTA label is a voting variant."""
    voting_markers = ("-MajorityVote", "-QualityGated", "-AgreementGate")
    return any(m in label for m in voting_markers)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records_dir, out_path, hard_k):
    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return 1

    headers = [load_label_records(records_dir, label)[0] for label in labels]
    classnames = resolve_classnames(records_dir, labels, headers)

    # --- Collect PTA and PatchModPTA records ---
    pta_runs = collect_method_samples(records_dir, labels, METHOD_PTA)
    patch_runs = collect_method_samples(records_dir, labels, METHOD_PATCH)

    if not pta_runs:
        print("[ERROR] no PTA records found", file=sys.stderr)
        return 1
    if not patch_runs:
        print("[ERROR] no PatchModPTA records found", file=sys.stderr)
        return 1

    # Separate main PatchModPTA from voting variants
    patch_main_runs = [(l, h, s) for l, h, s in patch_runs if not is_voting_variant(l)]
    patch_vote_runs = [(l, h, s) for l, h, s in patch_runs if is_voting_variant(l)]

    if not patch_main_runs:
        # Fall back to all patch runs if no main found
        patch_main_runs = patch_runs

    # --- Compute per-class accuracy for each method ---
    pta_by_label = {}
    for label, header, samples in pta_runs:
        pta_by_label[label] = per_class_named(samples, classnames)

    patch_by_label = {}
    for label, header, samples in patch_main_runs:
        patch_by_label[label] = per_class_named(samples, classnames)

    # Mean per-class accuracy across seeds
    pta_means = _class_means(pta_by_label)
    patch_means = _class_means(patch_by_label)

    # All class names
    all_classes = sorted(
        set(pta_means.keys()) | set(patch_means.keys()),
        key=_name_sort_key,
    )

    # --- Compute delta ---
    perclass_rows = []
    for name in all_classes:
        p_acc = pta_means.get(name)
        pa_acc = patch_means.get(name)
        delta = (pa_acc - p_acc) if (p_acc is not None and pa_acc is not None) else None
        perclass_rows.append({
            "class": name,
            "pta_acc": p_acc,
            "patch_acc": pa_acc,
            "delta": delta,
        })

    # Sort by delta ascending (hard classes = most negative delta or lowest PTA acc)
    # "Hard" = bottom-K by PTA accuracy
    perclass_by_pta = sorted(
        perclass_rows,
        key=lambda r: (r["pta_acc"] if r["pta_acc"] is not None else -1, r["class"]),
    )
    hard_classes = [r["class"] for r in perclass_by_pta[:hard_k]]

    # --- Build PatchModPTA main samples for patch analysis ---
    patch_main_samples = []
    for _, _, samples in patch_main_runs:
        patch_main_samples.extend(samples)

    # --- Build output ---
    lines = ["# Per-Class Accuracy Analysis", ""]
    lines.append("Generated from `{}`".format(records_dir))
    lines.append("")

    # Summary stats
    pta_accs = [overall_acc(s) for _, _, s in pta_runs]
    patch_accs = [overall_acc(s) for _, _, s in patch_main_runs]
    lines.append("**Methods:**")
    lines.append("- PTA: {} run(s), overall acc = {}".format(
        len(pta_runs),
        ", ".join("{:.2f}%".format(a) if a is not None else "n/a" for a in pta_accs),
    ))
    lines.append("- PatchModPTA (main): {} run(s), overall acc = {}".format(
        len(patch_main_runs),
        ", ".join("{:.2f}%".format(a) if a is not None else "n/a" for a in patch_accs),
    ))
    if patch_vote_runs:
        vote_accs = [overall_acc(s) for _, _, s in patch_vote_runs]
        lines.append("- PatchModPTA (voting variants): {} run(s), overall acc = {}".format(
            len(patch_vote_runs),
            ", ".join("{:.2f}%".format(a) if a is not None else "n/a" for a in vote_accs),
        ))
    lines.append("")

    # --- Per-class accuracy table ---
    lines.append("## Per-Class Accuracy")
    lines.append("")
    lines.append("| class | PTA acc (%) | PatchModPTA acc (%) | delta (pp) |")
    lines.append("|---|---|---|---|")
    for r in perclass_rows:
        lines.append("| {} | {} | {} | {} |".format(
            r["class"], _fmt(r["pta_acc"]), _fmt(r["patch_acc"]),
            _fmt_delta(r["delta"]),
        ))
    lines.append("")

    # --- Hard classes ---
    lines.append("## Hard Classes (bottom-{} by PTA accuracy)".format(hard_k))
    lines.append("")
    if hard_classes:
        lines.append("| class | PTA acc (%) | PatchModPTA acc (%) | delta (pp) |")
        lines.append("|---|---|---|---|")
        for name in hard_classes:
            r = next(rr for rr in perclass_rows if rr["class"] == name)
            lines.append("| {} | {} | {} | {} |".format(
                r["class"], _fmt(r["pta_acc"]), _fmt(r["patch_acc"]),
                _fmt_delta(r["delta"]),
            ))
        lines.append("")

        # --- Patch contribution for hard classes ---
        lines.append("### Patch-Level Contribution for Hard Classes")
        lines.append("")
        lines.append("For each hard class, showing how often `patch_proto.argmax` "
                     "matches ground truth when `clip.argmax` does not.")
        lines.append("")
        lines.append("| class | n | clip_correct | clip_wrong | patch_helps | "
                     "patch_help_rate (%) | final_correct |")
        lines.append("|---|---|---|---|---|---|---|")

        for name in hard_classes:
            # Find the target index for this class name
            target_idx = None
            for idx, cn in enumerate(classnames):
                if cn == name:
                    target_idx = idx
                    break
            if target_idx is None:
                # Try class_N parsing
                m = re.match(r"^class_(\d+)$", name)
                if m:
                    target_idx = int(m.group(1))
            if target_idx is None:
                lines.append("| {} | --- | --- | --- | --- | --- | --- |".format(name))
                continue

            contrib = patch_contribution_for_class(patch_main_samples, target_idx)
            help_rate = (100.0 * contrib["patch_helps"] / contrib["clip_wrong"]
                         if contrib["clip_wrong"] else None)
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                name, contrib["n"], contrib["clip_correct"], contrib["clip_wrong"],
                contrib["patch_helps"], _fmt(help_rate), contrib["final_correct"],
            ))
        lines.append("")

        # --- Cluster growth vs accuracy for hard classes ---
        lines.append("### Cluster Growth vs Accuracy (Hard Classes)")
        lines.append("")
        lines.append("| class | n_cluster_samples | mean_n_clusters | final_n_clusters |")
        lines.append("|---|---|---|---|")
        for name in hard_classes:
            target_idx = None
            for idx, cn in enumerate(classnames):
                if cn == name:
                    target_idx = idx
                    break
            if target_idx is None:
                m = re.match(r"^class_(\d+)$", name)
                if m:
                    target_idx = int(m.group(1))
            if target_idx is None:
                lines.append("| {} | --- | --- | --- |".format(name))
                continue

            contrib = patch_contribution_for_class(patch_main_samples, target_idx)
            clusters = contrib["clusters_list"]
            if clusters:
                mean_cl = sum(clusters) / len(clusters)
                final_cl = clusters[-1]
                lines.append("| {} | {} | {:.1f} | {} |".format(
                    name, contrib["n_clusters_samples"], mean_cl, final_cl,
                ))
            else:
                lines.append("| {} | 0 | --- | --- |".format(name))
        lines.append("")
    else:
        lines.append("*No hard classes identified.*")
        lines.append("")

    # --- Write markdown ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] per-class analysis written to {}".format(out_path))

    # --- Write CSV ---
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "pta_acc", "patch_acc", "delta"])
        writer.writeheader()
        for r in perclass_rows:
            writer.writerow({
                "class": r["class"],
                "pta_acc": round(r["pta_acc"], 2) if r["pta_acc"] is not None else "",
                "patch_acc": round(r["patch_acc"], 2) if r["patch_acc"] is not None else "",
                "delta": round(r["delta"], 2) if r["delta"] is not None else "",
            })
    print("[OK] CSV written to {}".format(csv_path))

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Per-class accuracy analysis: PTA vs PatchModPTA.")
    parser.add_argument("--records", required=True,
                        help="record dir DIR/<LABEL>/records.jsonl")
    parser.add_argument("--out", required=True,
                        help="markdown output path (CSV written alongside)")
    parser.add_argument("--hard-k", type=int, default=HARD_K_DEFAULT,
                        help="number of hard classes to identify (default: 5)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return analyze(Path(args.records), Path(args.out), args.hard_k)


if __name__ == "__main__":
    sys.exit(main())
