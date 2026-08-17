#!/usr/bin/env python3
"""Tie-breaking analysis (Exp 1.2): accuracy on samples where clip and
image_proto disagree.

A "tie" is defined as ``clip.argmax != image_proto.argmax`` (component
disagreement).  For each dataset reports:

  - Total samples, tie count, tie rate
  - PTA accuracy on tie samples
  - PatchModPTA accuracy on tie samples
  - Accuracy delta on ties
  - Breakdown: patch helps (patch correct, global wrong) vs hurts
    (patch wrong, global correct)

Null-guard: ``logits.patch_proto`` is ``null`` for ZeroShot and PTA records;
the script never dereferences it without checking.

Usage::

    python scripts/analyze_ties.py --records DIR --out FILE
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (shared with analyze_records.py)
# ---------------------------------------------------------------------------

PATCH_PREFIX = "PatchModPTA-CS"
PTA_PREFIX = "PTA-CS"
ZERO_PREFIX = "ZeroShot-CS"
ABLATION_MARKERS = ("-alpha10", "-multigate")

METHOD_PATCH = "PatchModPTA"
METHOD_PTA = "PTA"
METHOD_ZERO = "ZeroShot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def argmax(vec):
    """Index of the max element; ``None`` for empty lists."""
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def classify_label(label):
    """Map a record-dir label to a method family; ``None`` for ablations."""
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


def list_labels(records_dir):
    """Sorted labels = subdirs of *records_dir* containing ``records.jsonl``."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "records.jsonl").is_file()
    )


def load_label_records(records_dir, label):
    """Return ``(header, samples)`` for ``DIR/label/records.jsonl``."""
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


def is_tie(sample):
    """True when ``clip.argmax != image_proto.argmax`` (component disagreement).

    Returns ``False`` when either logit vector is missing or empty.
    """
    lg = sample.get("logits") or {}
    clip = lg.get("clip")
    image_proto = lg.get("image_proto")
    if not clip or not image_proto:
        return False
    ca = argmax(clip)
    ia = argmax(image_proto)
    if ca is None or ia is None:
        return False
    return ca != ia


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def _samples_by_dataset(records_dir, labels, method):
    """Group samples by dataset for a given method.

    Returns ``{dataset_name: {batch_idx: sample_dict}}``.
    """
    by_ds = defaultdict(dict)
    for label in labels:
        if classify_label(label) != method:
            continue
        h, samples = load_label_records(records_dir, label)
        ds = (h or {}).get("dataset") or ""
        for s in samples:
            by_ds[ds][s["batch_idx"]] = s
    return dict(by_ds)


def analyze_dataset(tie_samples, pta_by_idx, patch_by_idx):
    """Compute tie stats for a single dataset.

    *tie_samples* — list of ``batch_idx`` where clip != image_proto argmax.
    *pta_by_idx* / *patch_by_idx* — ``{batch_idx: sample}`` dicts.
    """
    n_ties = len(tie_samples)

    if n_ties == 0:
        return {
            "total": 0,
            "pta_acc": None,
            "patch_acc": None,
            "delta": None,
            "patch_helps": 0,
            "patch_hurts": 0,
            "patch_both_right": 0,
            "patch_both_wrong": 0,
            "patch_help_rate": None,
        }

    pta_correct = 0
    patch_correct = 0
    patch_helps = 0    # patch correct AND global wrong
    patch_hurts = 0    # patch wrong AND global correct
    both_right = 0
    both_wrong = 0

    for idx in tie_samples:
        pta_s = pta_by_idx.get(idx)
        patch_s = patch_by_idx.get(idx)

        pta_ok = bool(pta_s and pta_s.get("correct"))
        patch_ok = bool(patch_s and patch_s.get("correct"))

        if pta_ok:
            pta_correct += 1
        if patch_ok:
            patch_correct += 1

        if patch_ok and not pta_ok:
            patch_helps += 1
        elif not patch_ok and pta_ok:
            patch_hurts += 1
        elif patch_ok and pta_ok:
            both_right += 1
        else:
            both_wrong += 1

    pta_acc = 100.0 * pta_correct / n_ties
    patch_acc = 100.0 * patch_correct / n_ties
    delta = patch_acc - pta_acc

    help_plus_hurt = patch_helps + patch_hurts
    help_rate = (100.0 * patch_helps / help_plus_hurt) if help_plus_hurt > 0 else None

    return {
        "total": n_ties,
        "pta_acc": pta_acc,
        "patch_acc": patch_acc,
        "delta": delta,
        "patch_helps": patch_helps,
        "patch_hurts": patch_hurts,
        "patch_both_right": both_right,
        "patch_both_wrong": both_wrong,
        "patch_help_rate": help_rate,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_pct(v):
    """Format a percentage or return 'N/A'."""
    if v is None:
        return "N/A"
    return f"{v:.1f}%"


def _fmt_delta(v):
    if v is None:
        return "N/A"
    return f"{v:+.1f}pp"


def render_markdown(dataset_results, records_dir):
    """Build markdown lines from per-dataset analysis dicts."""
    lines = ["# Tie-Breaking Analysis (Exp 1.2)", ""]
    lines.append(f"Generated from `{records_dir}`")
    lines.append("")
    lines.append("A **tie** is defined as `clip.argmax != image_proto.argmax` "
                 "(component disagreement).")
    lines.append("")
    lines.append("## Tie Rate")
    lines.append("")
    lines.append("| Dataset | Total Samples | Ties | Tie Rate |")
    lines.append("|---|---|---|---|")
    for ds_name, r in dataset_results.items():
        total_all = r.get("total_all", r["total"])
        n_ties = r["total"]
        rate = 100.0 * n_ties / total_all if total_all else 0.0
        lines.append(f"| {ds_name} | {total_all} | {n_ties} | {rate:.1f}% |")
    lines.append("")

    lines.append("## Accuracy on Ties")
    lines.append("")
    lines.append("| Dataset | PTA Acc | PatchModPTA Acc | Delta |")
    lines.append("|---|---|---|---|")
    for ds_name, r in dataset_results.items():
        lines.append(
            f"| {ds_name} | {_fmt_pct(r['pta_acc'])} "
            f"| {_fmt_pct(r['patch_acc'])} | {_fmt_delta(r['delta'])} |"
        )
    lines.append("")

    lines.append("## Patch Breakdown on Ties")
    lines.append("")
    lines.append("| Dataset | Helps | Hurts | Both Right | Both Wrong | Help Rate |")
    lines.append("|---|---|---|---|---|---|")
    for ds_name, r in dataset_results.items():
        lines.append(
            f"| {ds_name} | {r['patch_helps']} | {r['patch_hurts']} "
            f"| {r['patch_both_right']} | {r['patch_both_wrong']} "
            f"| {_fmt_pct(r['patch_help_rate'])} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tie-breaking analysis — accuracy on clip/image disagreement"
    )
    parser.add_argument(
        "--records", required=True,
        help="Directory containing per-label record subdirs",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output markdown file path",
    )
    args = parser.parse_args()

    records_dir = Path(args.records)
    if not records_dir.is_dir():
        print(f"[ERROR] records dir not found: {records_dir}", file=sys.stderr)
        return 1

    labels = list_labels(records_dir)
    if not labels:
        print(f"[ERROR] no records found in {records_dir}", file=sys.stderr)
        return 1

    # Group by dataset per method
    pta_by_ds = _samples_by_dataset(records_dir, labels, METHOD_PTA)
    patch_by_ds = _samples_by_dataset(records_dir, labels, METHOD_PATCH)

    # Union of datasets from both methods (plus any other method present)
    all_datasets = sorted(set(list(pta_by_ds.keys()) + list(patch_by_ds.keys())))
    if not all_datasets:
        print("[ERROR] no datasets found across records", file=sys.stderr)
        return 1

    dataset_results = {}

    for ds in all_datasets:
        pta_by_idx = pta_by_ds.get(ds, {})
        patch_by_idx = patch_by_ds.get(ds, {})

        # Determine the reference sample set (PTA preferred, fallback PatchModPTA)
        ref_by_idx = pta_by_idx or patch_by_idx
        if not ref_by_idx:
            continue

        total_all = len(ref_by_idx)

        # Identify ties from the reference set
        tie_idxs = sorted(
            idx for idx, s in ref_by_idx.items() if is_tie(s)
        )

        r = analyze_dataset(tie_idxs, pta_by_idx, patch_by_idx)
        r["total_all"] = total_all
        dataset_results[ds] = r

    if not dataset_results:
        print("[ERROR] no dataset results computed", file=sys.stderr)
        return 1

    lines = render_markdown(dataset_results, records_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] tie analysis written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
