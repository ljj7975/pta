#!/usr/bin/env python3
"""
proxy_validation.py — pre-registered transfer criterion for the DTD proxy subset.

Applies the PRE-REGISTERED criterion (dtd-subset-proxy.md Task 6):
    PASS iff delta_subset has the SAME SIGN as delta_full
           AND |delta_subset - delta_full| <= 2pp
(Fixed thresholds — no post-hoc tuning.)

Deltas are Combo1 - PTA-CS mean per-class accuracy:
    delta_subset  — closed-set 12-way, subset-PTA-CS-s1 vs subset-Combo1-s1
                    (per_class_named with EXPLICIT classnames from
                    outputs/subset_classes_proxy.txt — never dataset fallback)
    delta_full    — open-set 47-way, recomputed from records:
                    Combo1-full-dtd-s1 (per_class populated) and
                    outputs/records/PTA-CS-s1 (per-class from JSONL), using the
                    47 DTD classnames in dataset order.

Secondary diagnostic: per-class sign-agreement % (sign of Combo1-PTA-CS per
class on subset vs full) is printed to stdout for the evidence capture.

Outputs:
    outputs/proxy_validation_result.txt  — EXACTLY 1 line: PASS|FAIL ...
"""

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root MUST be on sys.path so `import datasets` resolves to the repo-local
# package, NOT the HuggingFace site-packages one (see learnings.md Task 2).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_records import load_label_records, per_class_named  # noqa: E402

SUBSET_DIR = REPO_ROOT / "outputs/records_proxy_validation"
FULL_COMBO_DIR = SUBSET_DIR / "Combo1-full-dtd-s1"
FULL_PTA_DIR = REPO_ROOT / "outputs/records" / "PTA-CS-s1"
SUBSET_CLASSES_FILE = REPO_ROOT / "outputs/subset_classes_proxy.txt"
RESULT_FILE = REPO_ROOT / "outputs/proxy_validation_result.txt"

# Fixed pre-registered thresholds — do not change.
MAX_DIFF_PP = 2.0


def mean_acc(named):
    """Mean of per-class accuracies (percent)."""
    if not named:
        raise RuntimeError("empty per-class stats — cannot compute mean")
    return statistics.mean(st["acc"] for st in named.values())


def dtd_classnames():
    """47 DTD classnames in dataset order (matches full-record target idx)."""
    from datasets import build_dataset
    d = build_dataset("dtd", str(REPO_ROOT / "data"))
    return [str(c) for c in d.classnames]


def load_named(label_dir, classnames):
    """per_class_named over ``label_dir``/records.jsonl with explicit classnames."""
    header, samples = load_label_records(label_dir.parent, label_dir.name)
    if not samples:
        raise RuntimeError("no samples in {}".format(label_dir))
    named = per_class_named(samples, classnames)
    return header, samples, named


def per_class_delta(combo_named, pta_named):
    """{classname: Combo1 acc - PTA-CS acc} over the intersection of classes."""
    out = {}
    for name in combo_named:
        if name in pta_named:
            out[name] = combo_named[name]["acc"] - pta_named[name]["acc"]
    return out


def main():
    # --- subset classnames: explicit, source of truth ----------------------
    subset_classes = [
        line.strip() for line in SUBSET_CLASSES_FILE.read_text().splitlines()
        if line.strip()
    ]
    assert len(subset_classes) == 12, "subset class file must hold 12 names"

    # --- subset deltas (closed-set 12-way) ---------------------------------
    _, _, sub_pta = load_named(SUBSET_DIR / "subset-PTA-CS-s1", subset_classes)
    _, _, sub_combo = load_named(SUBSET_DIR / "subset-Combo1-s1", subset_classes)
    delta_subset = mean_acc(sub_combo) - mean_acc(sub_pta)

    # --- full-DTD deltas (open-set 47-way, recomputed from records) --------
    dtd_cn = dtd_classnames()
    assert len(dtd_cn) == 47, "expected 47 DTD classnames, got {}".format(len(dtd_cn))

    _, _, full_combo = load_named(FULL_COMBO_DIR, dtd_cn)
    # PTA-CS full-DTD per-class is recomputed from outputs/records/PTA-CS-s1
    # JSONL (summary.json has empty per_class from the pre-fix run).
    _, _, full_pta = load_named(FULL_PTA_DIR, dtd_cn)
    delta_full = mean_acc(full_combo) - mean_acc(full_pta)

    # --- pre-registered criterion ------------------------------------------
    diff = delta_subset - delta_full
    same_sign = (delta_subset > 0) == (delta_full > 0)
    within_pp = abs(diff) <= MAX_DIFF_PP
    verdict = "PASS" if (same_sign and within_pp) else "FAIL"

    result_line = "{} delta_subset={:.3f} delta_full={:.3f} diff={:.3f}".format(
        verdict, delta_subset, delta_full, diff)
    RESULT_FILE.write_text(result_line + "\n", encoding="utf-8")

    # --- secondary diagnostic: per-class sign-agreement % ------------------
    sub_delta = per_class_delta(sub_combo, sub_pta)
    full_delta = per_class_delta(full_combo, full_pta)
    pairs = []
    agree = 0
    for name in subset_classes:
        s, f = sub_delta.get(name), full_delta.get(name)
        if s is None or f is None:
            continue
        same = (s > 0) == (f > 0)
        if s == 0 and f == 0:
            same = True
        agree += 1 if same else 0
        pairs.append((name, s, f, same))
    agree_pct = 100.0 * agree / len(pairs) if pairs else None

    # --- stdout (captured to evidence) -------------------------------------
    print("=== DTD proxy-subset transfer validation (pre-registered) ===")
    print("subset classes (explicit): {}".format(", ".join(subset_classes)))
    print("subset-PTA-CS mean per-class acc: {:.3f}".format(mean_acc(sub_pta)))
    print("subset-Combo1 mean per-class acc: {:.3f}".format(mean_acc(sub_combo)))
    print("delta_subset (Combo1-PTA, 12-way closed-set): {:+.3f} pp".format(delta_subset))
    print("full-PTA-CS mean per-class acc: {:.3f}".format(mean_acc(full_pta)))
    print("full-Combo1 mean per-class acc: {:.3f}".format(mean_acc(full_combo)))
    print("delta_full (Combo1-PTA, 47-way open-set): {:+.3f} pp".format(delta_full))
    print("|diff| = {:.3f} pp (threshold {:.1f} pp)".format(abs(diff), MAX_DIFF_PP))
    print("same sign: {}".format(same_sign))
    print("criterion: {} (sign-agreement={} AND |diff|<=2pp={})".format(
        verdict, same_sign, within_pp))
    print("")
    print("Secondary diagnostic — per-class Combo1-PTA delta sign agreement:")
    print("| class | subset delta | full delta | same sign |")
    for name, s, f, same in pairs:
        print("| {} | {:+.3f} | {:+.3f} | {} |".format(
            name, s, f, "yes" if same else "NO"))
    print("per-class sign-agreement: {:.1f}% ({}/{})".format(
        agree_pct, agree, len(pairs)))
    print("")
    print("RESULT: {}".format(result_line))


if __name__ == "__main__":
    main()
