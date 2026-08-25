#!/usr/bin/env python3
"""Patch-level prototype bank write-purity diagnostic.

Motivation
----------
`outputs/patch_benefit_report.md` found (pre-registered study, REFUTE
verdict) that no patch-fusion arm beats PTA baseline by more than noise on
dtd/oxford_flowers/oxford_pets. This script tests one candidate mechanism:
is the patch-level Gaussian prototype bank (``models/patch_level/gaussian_patch.py``)
being **contaminated at write time** because it is built from the model's own
(unlabeled, test-time) predictions rather than ground truth?

Grounded in code (``models/patch_modulated_pta.py:203-260``,
``configs/base.yaml``, ``configs/patch_modulated_pta/patch_modulated_pta.yaml``):

- The patch bank write gate is **independent of** the ``write_source``/
  ``write_rule`` knobs swept in ``results_dev_writerule.md`` — those only
  control the separate *image-level* EMA prototype (``_compute_write_mask``
  called at ``patch_modulated_pta.py:359``). The "fair write rule" evidence
  the prior sweep produced says nothing about the patch-level bank.
- The patch bank uses ``multi_gate=True`` (default in both configs), which
  writes an image's patches into the bank of **every** class whose zero-shot
  CLIP softmax probability exceeds ``conf_threshold`` (0.3, from
  ``configs/patch_modulated_pta/patch_modulated_pta.yaml``) — not just the
  predicted top-1 class, and never checked against the (unavailable at test
  time) ground-truth label.
- ``conf_source`` defaults to ``"text"`` (plain zero-shot ``clip_logits``,
  computed once per image, independent of any adaptation state) and none of
  this study's 6 patch arms override it — so ``logits.clip`` in every arm's
  stored records is the exact tensor this gate thresholds, and the gate
  behavior is identical across all 6 arms for a given (dataset, seed).

Hypothesis this script tests: on datasets with many visually-similar
fine-grained classes (oxford_flowers: 102 classes, oxford_pets: 37 classes),
CLIP's softmax mass is more often spread over 2+ semantically-close classes,
so multi_gate fires "collateral" writes into a wrong class's bank far more
often than on a coarser dataset like dtd (47 broad texture classes) — which
would explain the observed pattern (dtd mildly positive, flowers/pets
consistently negative) as bank contamination rather than a fusion-mechanism
problem.

This is a **replay** — it re-derives the write mask from already-stored
per-sample ``logits.clip`` and ``target`` fields in
``outputs/records_patch_benefit/``. No new GPU runs.

Usage::

    python scripts/analyze_prototype_purity.py \\
        --records outputs/records_patch_benefit \\
        --out outputs/prototype_purity_report.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_patch_benefit import DATASETS, SEEDS, load_all  # noqa: E402

# ---------------------------------------------------------------------------
# Write-gate config — replayed exactly from models/patch_modulated_pta.py's
# multi_gate branch (run(), lines ~434-443), using the values actually in
# effect for this study's records (configs/patch_modulated_pta/
# patch_modulated_pta.yaml overrides configs/base.yaml's patch_level block).
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.3       # patch_modulated_pta.yaml: patch_level.conf_threshold
CANONICAL_ARM = "PatchModPTA-CS"  # any arm works; clip_logits is state-independent


def softmax(vec):
    import math
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def replay_write_mask(clip_logits):
    """Classes an image's patches get written into, per the multi_gate rule.

    ``pred_conf > conf_thresh`` (no margin check — multi_gate branch never
    consults conf_margin_threshold, only the else/single-class branch does).
    """
    probs = softmax(clip_logits)
    return [c for c, p in enumerate(probs) if p > CONF_THRESHOLD]


def analyze_dataset(records_dir, dataset, seeds, canonical_arm=CANONICAL_ARM):
    """Pool write-purity stats over all seeds for one dataset."""
    total_images = 0
    total_writes = 0          # sum of |written_classes| across images
    multi_write_images = 0    # images that wrote to >1 class
    zero_write_images = 0     # images that wrote to 0 classes (below threshold everywhere)
    correct_writes = 0        # write events where written_class == target
    collateral_writes = 0     # write events where written_class != argmax(clip) (i.e. not the "primary" pick)
    collateral_correct = 0    # of those, how many still happen to equal target
    primary_writes = 0
    primary_correct = 0

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    index = load_all(records_dir)
    for seed in seeds:
        key = (canonical_arm, dataset, seed)
        if key not in index:
            print(f"[WARN] missing {key}, skipping", file=sys.stderr)
            continue
        _, samples = index[key]
        for s in samples:
            clip_logits = s["logits"]["clip"]
            target = s["target"]
            written = replay_write_mask(clip_logits)
            primary_cls = max(range(len(clip_logits)), key=lambda i: clip_logits[i])

            total_images += 1
            if len(written) > 1:
                multi_write_images += 1
            if len(written) == 0:
                zero_write_images += 1

            for c in written:
                total_writes += 1
                is_correct = (c == target)
                per_class_total[c] += 1
                if is_correct:
                    correct_writes += 1
                    per_class_correct[c] += 1
                if c == primary_cls:
                    primary_writes += 1
                    if is_correct:
                        primary_correct += 1
                else:
                    collateral_writes += 1
                    if is_correct:
                        collateral_correct += 1

    per_class_purity = []
    for c, tot in per_class_total.items():
        if tot > 0:
            per_class_purity.append(100.0 * per_class_correct[c] / tot)
    per_class_purity.sort()
    n_classes_written = len(per_class_purity)
    median_class_purity = (
        per_class_purity[n_classes_written // 2] if n_classes_written else None
    )

    return {
        "total_images": total_images,
        "total_writes": total_writes,
        "writes_per_image": (total_writes / total_images) if total_images else None,
        "multi_write_rate": (100.0 * multi_write_images / total_images) if total_images else None,
        "zero_write_rate": (100.0 * zero_write_images / total_images) if total_images else None,
        "overall_purity": (100.0 * correct_writes / total_writes) if total_writes else None,
        "primary_writes": primary_writes,
        "primary_purity": (100.0 * primary_correct / primary_writes) if primary_writes else None,
        "collateral_writes": collateral_writes,
        "collateral_purity": (100.0 * collateral_correct / collateral_writes) if collateral_writes else None,
        "n_classes_written": n_classes_written,
        "median_class_purity": median_class_purity,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v, suffix="%"):
    return "N/A" if v is None else f"{v:.1f}{suffix}"


def render_report(results, known_deltas):
    lines = ["# Patch-Level Prototype Bank — Write-Time Purity Diagnostic", ""]
    lines.append(
        "Replays the patch bank's `multi_gate` write mask "
        "(`models/patch_modulated_pta.py` run(), `conf_source=\"text\"`, "
        f"`conf_threshold={CONF_THRESHOLD}`) from stored `logits.clip` + `target` "
        f"in `outputs/records_patch_benefit/` (canonical arm `{CANONICAL_ARM}` — "
        "clip_logits is state-independent zero-shot CLIP, identical across all "
        "6 patch arms for a given dataset/seed). No new GPU runs."
    )
    lines.append("")
    lines.append(
        "**Scope note**: this gate is *not* touched by the `write_source`/"
        "`write_rule` sweep in `results_dev_writerule.md` — that sweep only "
        "governs the separate image-level EMA prototype. The patch bank's "
        "write purity has not been previously measured."
    )
    lines.append("")

    lines.append("## Per-dataset write-purity summary (pooled over 4 seeds)")
    lines.append("")
    lines.append(
        "| Dataset | Images | Writes/image | Multi-write rate | Zero-write rate "
        "| Overall purity | Primary-write purity | Collateral-write purity | Collateral share |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        r = results[ds]
        collateral_share = (
            100.0 * r["collateral_writes"] / r["total_writes"]
            if r["total_writes"] else None
        )
        lines.append(
            f"| {ds} | {r['total_images']} | {r['writes_per_image']:.2f} "
            f"| {_fmt(r['multi_write_rate'])} | {_fmt(r['zero_write_rate'])} "
            f"| {_fmt(r['overall_purity'])} | {_fmt(r['primary_purity'])} "
            f"| {_fmt(r['collateral_purity'])} | {_fmt(collateral_share)} |"
        )
    lines.append("")

    lines.append("## Per-class purity distribution")
    lines.append("")
    lines.append("| Dataset | Classes ever written | Median per-class purity |")
    lines.append("|---|---|---|")
    for ds in DATASETS:
        r = results[ds]
        lines.append(
            f"| {ds} | {r['n_classes_written']} | {_fmt(r['median_class_purity'])} |"
        )
    lines.append("")

    lines.append("## Cross-reference: does purity track the observed accuracy pattern?")
    lines.append("")
    lines.append(
        "`outputs/patch_benefit_report.md` / `outputs/experiment_summary.md` found: "
        "dtd mildly **positive** for MVote/AGate (+0.6-0.99pp), oxford_flowers/"
        "oxford_pets consistently **negative** across every arm (-0.1 to -2.3pp)."
    )
    lines.append("")
    lines.append("| Dataset | Overall write purity | Collateral-write purity | Known MVote/AGate delta |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        r = results[ds]
        lines.append(
            f"| {ds} | {_fmt(r['overall_purity'])} | {_fmt(r['collateral_purity'])} "
            f"| {known_deltas.get(ds, 'N/A')} |"
        )
    lines.append("")

    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="outputs/records_patch_benefit")
    parser.add_argument("--out", default="outputs/prototype_purity_report.md")
    args = parser.parse_args(argv)

    records_dir = Path(args.records)
    results = {}
    for ds in DATASETS:
        print(f"[analyze_prototype_purity] replaying {ds} ...", file=sys.stderr)
        results[ds] = analyze_dataset(records_dir, ds, SEEDS)

    # Known deltas from outputs/patch_benefit_report.md / experiment_summary.md,
    # cited here only for narrative cross-reference — not recomputed by this script.
    known_deltas = {
        "dtd": "MVote +0.65pp / AGate +0.60pp",
        "oxford_flowers": "MVote -0.13pp / AGate -0.24pp",
        "oxford_pets": "MVote -0.39pp / AGate -0.31pp",
    }

    lines = render_report(results, known_deltas)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] purity report written to {out_path}")

    # Also dump raw numbers for programmatic follow-up.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] raw numbers written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
