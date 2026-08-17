#!/usr/bin/env python3
"""
make_flower_perclass_tables.py — Task 5: oxford_flowers per-class tables.

Generates ``outputs/flower_perclass_tables.md`` from the 4 flowers record
dirs (PTA-CS-flowers-s1/s2, Combo1-flowers-s1/s2) in the same style as
``outputs/perclass_tables.md`` (method/seed/class/n/correct/acc), plus the
cross-method delta table (Combo1 − PTA-CS, mean across seeds) at the end.

Every table is labeled "Zhou split (non-canonical) — random ~50/20/30 split,
not the canonical 10/10/rest" (flowers test counts are IMBALANCED, n in
12-77, NOT the uniform ~24/class the canonical 10/10/rest split would give).

Pure stdlib + repo-local modules; no GPU, no new runs.

Record dir layout (schema fixed in .omo/plans/dtd-subset-proxy.md, T1):

    DIR/<LABEL>/records.jsonl     per-sample stream (header line + sample lines)

Per-class stats are RECOMPUTED from records.jsonl (source of truth) via
``scripts.analyze_records.per_class_named``, not read from summary.json.

GOTCHA: the flowers record headers carry ``classnames=[]`` (flowers has no
subset remap but the header was written without classnames), so this script
passes explicit classnames from ``datasets.build_dataset("oxford_flowers",
"./data").classnames`` — the exact 102-name order used at run time (labels
sorted ascending, mapped via cat_to_name.json).  The repo root is prepended
to sys.path so ``import datasets`` resolves to the repo-local package (the
pta env's site-packages ``datasets`` is the HuggingFace one and fails
silently otherwise — see learnings, Task 2).

Layout note (matching the pre-registered QA in dtd-subset-proxy.md T5):
one table per method/seed run, introduced by a header row whose first cell
carries the method and second cell the seed, followed by 102 class rows
(class/n/correct/acc).  This keeps the method/seed/class/n/correct/acc
columns and gives exactly one ``| PTA-CS |`` / ``| Combo1 |`` line per seed
table (grep -c "^| PTA-CS" == 2, "^| Combo1" == 2).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_records import (  # noqa: E402  (needs REPO_ROOT on path)
    _fmt,
    _fmt_delta,
    load_label_records,
    per_class_named,
)

SPLIT_LABEL = ("Zhou split (non-canonical) — random ~50/20/30 split, "
               "not the canonical 10/10/rest")

# Fixed label -> (method, seed) for the 4 flowers record dirs (Task 4 array).
FLOWER_RUNS = [
    ("PTA-CS-flowers-s1", "PTA-CS", 1),
    ("PTA-CS-flowers-s2", "PTA-CS", 2),
    ("Combo1-flowers-s1", "Combo1", 1),
    ("Combo1-flowers-s2", "Combo1", 2),
]

N_FLOWERS_CLASSES = 102


def load_flower_classnames(data_root):
    """102 classnames in runtime (dataset) order via the repo-local datasets module."""
    from datasets import build_dataset
    d = build_dataset("oxford_flowers", str(data_root))
    cn = list(getattr(d, "classnames", []))
    if len(cn) != N_FLOWERS_CLASSES:
        raise RuntimeError(
            "expected {} flowers classnames, got {}".format(N_FLOWERS_CLASSES, len(cn)))
    return [str(c) for c in cn]


def label_suffix_seed(label):
    """Run seed from the ``-s<N>`` label suffix.

    The Task 4 array wrote ``seed: 1`` into EVERY flowers header (even the
    ``-s2`` runs), so the header seed field is not trustworthy for these
    record dirs — the label suffix is the distinguishing seed.
    """
    m = re.search(r"-s(\d+)$", label)
    if not m:
        raise RuntimeError("no -s<N> suffix in label {}".format(label))
    return int(m.group(1))


def mean_across_seeds(named_list):
    """``{classname: mean acc across seeds}`` from a list of named per-class stats."""
    accs = defaultdict(list)
    for named in named_list:
        for name, st in named.items():
            accs[name].append(st["acc"])
    return {name: sum(v) / len(v) for name, v in accs.items()}


def cmd_generate(records_dir, out_path, data_root):
    records_dir = Path(records_dir)
    classnames = load_flower_classnames(data_root)

    named_by_label = {}
    for label, method, seed in FLOWER_RUNS:
        header, samples = load_label_records(records_dir, label)
        if not samples:
            raise RuntimeError("no samples for {}".format(label))
        got_seed = label_suffix_seed(label)
        if got_seed != seed:
            raise RuntimeError(
                "seed mismatch for {}: label/{} vs expected {}".format(
                    label, got_seed, seed))
        _ = header  # header classnames are [] (flowers has no subset remap);
        # classnames come from the dataset class list (see module docstring).
        named = per_class_named(samples, classnames)
        named_by_label[label] = named
        missing = [c for c in classnames if c not in named]
        if missing:
            raise RuntimeError(
                "{} classes missing from {}: {}".format(
                    len(missing), label, missing[:5]))

    lines = []
    lines.append("# Flowers Per-Class Accuracy Tables")
    lines.append("")
    lines.append("Generated from `{}` (flowers record dirs: {})".format(
        records_dir, ", ".join(l for l, _, _ in FLOWER_RUNS)))
    lines.append("")
    lines.append("Split: {}.".format(SPLIT_LABEL))
    lines.append("")

    lines.append("## Per-method / per-seed per-class accuracy")
    lines.append("")
    lines.append("Split: {}.".format(SPLIT_LABEL))
    lines.append("")
    for label, method, seed in FLOWER_RUNS:
        named = named_by_label[label]
        lines.append("### {} s{} — {}".format(method, seed, SPLIT_LABEL))
        lines.append("")
        lines.append("| {} | s{} | class | n | correct | acc (%) |".format(method, seed))
        lines.append("|---|---|---|---|---|---|")
        for name in classnames:
            st = named[name]
            lines.append("| {} | {} | {} | {:.2f} |".format(
                name, st["total"], st["correct"], st["acc"]))
        lines.append("")

    lines.append("## Cross-method per-class diff (mean across seeds; Combo1 − PTA-CS)")
    lines.append("")
    lines.append("Split: {}.".format(SPLIT_LABEL))
    lines.append("")
    lines.append("| class | PTA-CS | Combo1 | Combo1-PTA-CS |")
    lines.append("|---|---|---|---|")
    pta = mean_across_seeds(
        [named_by_label[l] for l, m, _ in FLOWER_RUNS if m == "PTA-CS"])
    cmb = mean_across_seeds(
        [named_by_label[l] for l, m, _ in FLOWER_RUNS if m == "Combo1"])
    for name in classnames:
        p, c = pta[name], cmb[name]
        lines.append("| {} | {} | {} | {} |".format(
            name, _fmt(p), _fmt(c), _fmt_delta(c - p)))
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] flowers per-class tables written to {}".format(out_path))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default="outputs/records_proxy_validation",
        help="record dir DIR/<LABEL>/records.jsonl (default: %(default)s)")
    parser.add_argument(
        "--out", default="outputs/flower_perclass_tables.md",
        help="markdown output path (default: %(default)s)")
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "data"),
        help="dataset root for classname lookup (default: %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return cmd_generate(args.records, Path(args.out), Path(args.data_root))


if __name__ == "__main__":
    sys.exit(main())
