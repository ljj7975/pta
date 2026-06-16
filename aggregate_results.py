#!/usr/bin/env python3
"""
aggregate_results.py — Parse outputs/result.txt into a structured comparison table.

Compares four method columns:
  PTA          — original paper method (last result per dataset)
  MPTA         — MultiProtoPTA best complete run (lines 187-194 block)
  MPTAv2       — MultiProtoPTAv2 best complete run
  MPTA-new     — latest MultiProtoPTA run (appended after the new eval job)

Usage:
    python aggregate_results.py [--output comparison_table.txt]
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paper reference values (ViT-B/16)
# ---------------------------------------------------------------------------

PAPER_TABLE1 = {
    "caltech101":     94.81,
    "dtd":            47.70,
    "eurosat":        61.57,
    "fgvc":           26.10,
    "food101":        86.44,
    "oxford_flowers": 75.23,
    "oxford_pets":    91.06,
    "stanford_cars":  68.55,
    "sun397":         69.21,
    "ucf101":         73.12,
}

PAPER_TABLE2 = {
    "I":  70.28,
    "V":  64.85,
    "R":  80.79,
    "S":  51.00,
    "A":  61.15,
}

CD_DATASETS  = list(PAPER_TABLE1.keys())
OOD_DATASETS = list(PAPER_TABLE2.keys())

DATASET_LABELS = {
    "caltech101":     "Caltech101",
    "dtd":            "DTD",
    "eurosat":        "EuroSAT",
    "fgvc":           "FGVC Aircraft",
    "food101":        "Food101",
    "oxford_flowers": "Oxford Flowers",
    "oxford_pets":    "Oxford Pets",
    "stanford_cars":  "Stanford Cars",
    "sun397":         "SUN397",
    "ucf101":         "UCF101",
    "I":              "ImageNet",
    "V":              "ImageNetV2",
    "R":              "ImageNet-R",
    "S":              "ImageNet-Sketch",
    "A":              "ImageNet-A",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Prefixes we care about — order matters for "latest run" detection
METHOD_PATTERNS = {
    "pta":       re.compile(r"^PTA's performance on (.+?): Top1- ([\d.]+)\."),
    "mpta":      re.compile(r"^MultiProtoPTA's performance on (.+?): Top1- ([\d.]+)\."),
    "mptav2":    re.compile(r"^MultiProtoPTAv2's performance on (.+?): Top1- ([\d.]+)\."),
}

# Datasets that are internal smoke-test artefacts — skip them
_SMOKE_SKIP = re.compile(r"smoke|_smoke_|caltech101_smoke")


def parse_all_runs(result_file: Path) -> dict:
    """
    Returns a dict:
      {
        "pta":    { dataset: [acc, acc, ...], ... },  # all PTA runs
        "mpta":   { dataset: [acc, acc, ...], ... },
        "mptav2": { dataset: [acc, acc, ...], ... },
      }
    All runs chronologically ordered (last = most recent).
    """
    runs: dict = {k: defaultdict(list) for k in METHOD_PATTERNS}

    if not result_file.exists():
        return runs

    for raw_line in result_file.read_text().splitlines():
        line = raw_line.strip()
        for method, pat in METHOD_PATTERNS.items():
            m = pat.match(line)
            if m:
                dataset = m.group(1).strip()
                if _SMOKE_SKIP.search(dataset):
                    break
                acc = float(m.group(2))
                runs[method][dataset].append(acc)
                break

    return runs


def _last(runs: dict, method: str, dataset: str):
    """Return the most recent recorded accuracy or None."""
    vals = runs.get(method, {}).get(dataset)
    return vals[-1] if vals else None


def _best_complete_run(runs: dict, method: str, datasets: list):
    """
    Find the most recent *complete* run (all datasets present).
    Returns dict{dataset: acc} or {} if no complete run exists.
    """
    per_ds = runs.get(method, {})
    # Build parallel lists of runs per dataset
    lengths = [len(per_ds.get(d, [])) for d in datasets if d in per_ds]
    if not lengths or min(lengths) == 0:
        return {}

    # Walk backwards through the shortest dataset's run list to find a
    # complete matching run. Since runs are appended sequentially, we match
    # the i-th occurrence across all datasets.
    min_len = min(len(per_ds.get(d, [])) for d in datasets)
    # Try from newest to oldest complete set
    for idx in range(min_len - 1, -1, -1):
        candidate = {}
        ok = True
        for d in datasets:
            vals = per_ds.get(d, [])
            if idx >= len(vals):
                ok = False
                break
            candidate[d] = vals[idx]
        if ok:
            return candidate
    return {}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val) -> str:
    return f"{val:.2f}" if val is not None else "—"


def _delta(new_val, old_val) -> str:
    if new_val is None or old_val is None:
        return "—"
    d = new_val - old_val
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_comparison_table(
    datasets: list,
    paper: dict,
    pta_vals: dict,
    mpta_vals: dict,
    mptav2_vals: dict,
    new_vals: dict,
    title: str,
    file=sys.stdout,
):
    """
    Columns: Dataset | Paper(PTA) | MPTA | MPTAv2 | MPTA-new | Δ vs PTA | Δ vs MPTA | Δ vs MPTAv2
    """
    C = [22, 9, 9, 9, 9, 9, 9, 9]
    total_w = sum(C)
    sep = "-" * total_w

    def row(*cells):
        parts = []
        for i, c in enumerate(cells):
            if i == 0:
                parts.append(f"{c:<{C[i]}}")
            else:
                parts.append(f"{c:>{C[i]}}")
        return " ".join(parts)

    print(f"\n{'='*total_w}", file=file)
    print(f"  {title}", file=file)
    print(f"{'='*total_w}", file=file)
    print(row("Dataset", "PTA(paper)", "MPTA", "MPTAv2", "MPTA-new",
              "Δ vs PTA", "Δ vs MPTA", "Δ vs v2"), file=file)
    print(sep, file=file)

    run_new, run_paper, run_mpta, run_mptav2 = [], [], [], []

    for key in sorted(datasets, key=lambda k: DATASET_LABELS.get(k, k)):
        label    = DATASET_LABELS.get(key, key)
        paper_v  = paper.get(key)
        mpta_v   = mpta_vals.get(key)
        mptav2_v = mptav2_vals.get(key)
        new_v    = new_vals.get(key)
        pta_v    = pta_vals.get(key, paper_v)   # fall back to paper if no run

        print(row(
            label,
            _fmt(pta_v),
            _fmt(mpta_v),
            _fmt(mptav2_v),
            _fmt(new_v),
            _delta(new_v, pta_v),
            _delta(new_v, mpta_v),
            _delta(new_v, mptav2_v),
        ), file=file)

        if new_v is not None:
            run_new.append(new_v)
            if pta_v   is not None: run_paper.append(pta_v)
            if mpta_v  is not None: run_mpta.append(mpta_v)
            if mptav2_v is not None: run_mptav2.append(mptav2_v)

    print(sep, file=file)

    if run_new:
        avg_new = sum(run_new) / len(run_new)
        avg_pta  = sum(run_paper)  / len(run_paper)  if run_paper  else None
        avg_mpta = sum(run_mpta)   / len(run_mpta)   if run_mpta   else None
        avg_v2   = sum(run_mptav2) / len(run_mptav2) if run_mptav2 else None
        print(row(
            f"Average ({len(run_new)} dsets)",
            _fmt(avg_pta),
            _fmt(avg_mpta),
            _fmt(avg_v2),
            _fmt(avg_new),
            _delta(avg_new, avg_pta),
            _delta(avg_new, avg_mpta),
            _delta(avg_new, avg_v2),
        ), file=file)
    else:
        paper_vals_cd = [paper.get(k) for k in datasets if paper.get(k) is not None]
        avg_paper = sum(paper_vals_cd) / len(paper_vals_cd) if paper_vals_cd else None
        print(row("Average (paper)", _fmt(avg_paper), "—", "—", "—", "—", "—", "—"), file=file)

    print(f"{'='*total_w}", file=file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate PTA benchmark results — multi-method comparison")
    parser.add_argument("--results", default="outputs/result.txt",
                        help="Path to results file (default: outputs/result.txt)")
    parser.add_argument("--output", default=None,
                        help="Save table to this file in addition to stdout")
    args = parser.parse_args()

    result_file = Path(args.results)
    runs = parse_all_runs(result_file)

    # PTA: last result per dataset (always a single complete pass)
    pta_vals = {d: _last(runs, "pta", d) for d in CD_DATASETS + OOD_DATASETS}
    pta_vals = {k: v for k, v in pta_vals.items() if v is not None}

    # MPTA: best complete CD run
    mpta_cd   = _best_complete_run(runs, "mpta", [d for d in CD_DATASETS if d in runs["mpta"]])
    mpta_ood  = {d: _last(runs, "mpta", d) for d in OOD_DATASETS}
    mpta_ood  = {k: v for k, v in mpta_ood.items() if v is not None}
    mpta_vals = {**mpta_cd, **mpta_ood}

    # MPTAv2: best complete CD run
    mptav2_cd   = _best_complete_run(runs, "mptav2", [d for d in CD_DATASETS if d in runs["mptav2"]])
    mptav2_ood  = {d: _last(runs, "mptav2", d) for d in OOD_DATASETS}
    mptav2_ood  = {k: v for k, v in mptav2_ood.items() if v is not None}
    mptav2_vals = {**mptav2_cd, **mptav2_ood}

    # MPTA-new: the very last recorded result per dataset for MPTA
    # (appended by the new job — will be the freshest entry after the eval run)
    new_vals = {d: _last(runs, "mpta", d) for d in CD_DATASETS + OOD_DATASETS}
    new_vals = {k: v for k, v in new_vals.items() if v is not None}

    outputs = [sys.stdout]
    out_file = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w")
        outputs.append(out_file)

    for f in outputs:
        print(f"\nPTA Method Comparison — ViT-B/16 backbone", file=f)
        print(f"Generated from: {result_file}", file=f)
        print(f"Columns: PTA (paper baseline) | MPTA (best prev run) | MPTAv2 (best prev run) | MPTA-new (latest run)", file=f)

        # CD table — only datasets that have at least one result
        cd_available = [d for d in CD_DATASETS if
                        d in pta_vals or d in mpta_vals or d in mptav2_vals or d in new_vals]
        print_comparison_table(
            cd_available, PAPER_TABLE1,
            pta_vals, mpta_vals, mptav2_vals, new_vals,
            "Cross-Domain Generalization (CD) — 7 datasets", file=f,
        )

        # OOD table
        ood_available = [d for d in OOD_DATASETS if
                         d in pta_vals or d in mpta_vals or d in mptav2_vals or d in new_vals]
        if ood_available:
            print_comparison_table(
                ood_available, PAPER_TABLE2,
                pta_vals, mpta_vals, mptav2_vals, new_vals,
                "Out-of-Distribution (OOD)", file=f,
            )

        missing_cd = [d for d in CD_DATASETS if d not in new_vals]
        if missing_cd:
            print(f"\n[INFO] MPTA-new not yet run for: {', '.join(DATASET_LABELS.get(k,k) for k in missing_cd)}", file=f)

    if out_file:
        out_file.close()
        print(f"\n[OK] Table saved to {args.output}")


if __name__ == "__main__":
    main()
