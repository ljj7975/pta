#!/usr/bin/env python3
"""Experiment G — candidate-set-restricted patch fusion.

See outputs/patch_fix_experiments_plan.md ("G"). Patches currently can move
any class's final logit, including classes CLIP considered implausible for
a given image. This restricts patch influence to a per-image candidate set
of classes the *baseline* (CLIP + image-level prototype, same "baseline"
concept as Experiment D) already considers plausible -- patches can re-rank
a shortlist, never promote a class ruled out.

Two interchangeable ways to build the candidate set, both swept here:
  - top-K by baseline rank (K = 1 means patches can never change the
    prediction; K = C is the unrestricted original)
  - margin-based: every class within `margin` of the baseline's top
    softmax probability (adapts per image -- shrinks to 1 class on easy
    images, grows automatically on ambiguous ones)

Computed entirely from already-stored records (reuses
analyze_uncertainty_gated_fusion.py's per-sample reconstruction) -- no new
GPU work.

Usage::

    python scripts/analyze_candidate_restricted_fusion.py \\
        --records outputs/records_patch_benefit \\
        --prereg outputs/patch_benefit_prereg.md \\
        --out outputs/candidate_restricted_fusion_report.md
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_uncertainty_gated_fusion import process_cell, softmax  # noqa: E402
from analyze_patch_benefit import (  # noqa: E402
    DATASETS,
    SEEDS,
    read_locked_thresholds,
    sign_permutation_pvalue,
    seed_level_bootstrap_ci,
)

TOPK_GRID = [1, 2, 3, 5, 10, 20, None]      # None = unrestricted (full C)
MARGIN_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]  # 1.0 = unrestricted


def candidate_mask_topk(baseline_logits, k):
    C = len(baseline_logits)
    if k is None or k >= C:
        return [True] * C
    ranked = sorted(range(C), key=lambda i: baseline_logits[i], reverse=True)
    keep = set(ranked[:k])
    return [i in keep for i in range(C)]


def candidate_mask_margin(baseline_logits, margin):
    probs = softmax(baseline_logits)
    top1 = max(probs)
    return [p >= top1 - margin for p in probs]


def score_cell(cell_samples, mode, param):
    correct = 0
    for s in cell_samples:
        if mode == "topk":
            mask = candidate_mask_topk(s["baseline_logits"], param)
        else:
            mask = candidate_mask_margin(s["baseline_logits"], param)
        gated = [
            b + (p if m else 0.0)
            for b, p, m in zip(s["baseline_logits"], s["patch_term"], mask)
        ]
        pred = max(range(len(gated)), key=lambda i: gated[i])
        if pred == s["target"]:
            correct += 1
    return 100.0 * correct / len(cell_samples) if cell_samples else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="outputs/records_patch_benefit")
    parser.add_argument("--prereg", default="outputs/patch_benefit_prereg.md")
    parser.add_argument("--out", default="outputs/candidate_restricted_fusion_report.md")
    args = parser.parse_args(argv)

    thresholds = read_locked_thresholds(args.prereg)

    cell_samples = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[load] {dataset}-s{seed}", file=sys.stderr)
            cell_samples[(dataset, seed)] = process_cell(args.records, dataset, seed)

    settings = [("topk", k) for k in TOPK_GRID] + [("margin", m) for m in MARGIN_GRID]

    stats = {}
    for dataset in DATASETS:
        for mode, param in settings:
            # topk=1 forces patch_term to 0 for every class except the single
            # top baseline pick -- exactly the baseline-only (tau=0.0)
            # reconstruction from Experiment D, used below as the reference.
            accs = [score_cell(cell_samples[(dataset, seed)], mode, param) for seed in SEEDS]
            stats.setdefault(dataset, {})[(mode, param)] = accs

    # Reference: unrestricted (topk=None) == original PatchModPTA-CS reconstruction.
    # Baseline-only reference: topk=1 == Experiment D's tau=0.0 reconstruction.
    report_stats = {}
    for dataset in DATASETS:
        baseline_accs = stats[dataset][("topk", 1)]
        for mode, param in settings:
            accs = stats[dataset][(mode, param)]
            deltas = [a - b for a, b in zip(accs, baseline_accs)]
            delta_mean = statistics.mean(deltas)
            p_sign = sign_permutation_pvalue(deltas)
            ci_lo, ci_hi = seed_level_bootstrap_ci(deltas)
            threshold = thresholds[dataset]
            if delta_mean > threshold and p_sign <= 0.0625 and (ci_lo is None or ci_lo > 0):
                verdict = "SUPPORT"
            elif delta_mean <= 0 or p_sign > 0.5:
                verdict = "REFUTE"
            else:
                verdict = "INCONCLUSIVE"
            report_stats[(dataset, mode, param)] = {
                "accs": accs, "mean": statistics.mean(accs),
                "delta_mean": delta_mean, "p_sign": p_sign, "ci": (ci_lo, ci_hi),
                "verdict": verdict,
            }

    render_report(report_stats, thresholds, args.out)
    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(
        json.dumps({f"{d}|{m}|{p}": v for (d, m, p), v in report_stats.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {json_path}")


def render_report(stats, thresholds, out_path):
    lines = ["# Experiment G — Candidate-Set-Restricted Patch Fusion — Raw Results", ""]
    lines.append(
        "Patches can only influence classes in a per-image candidate set "
        "derived from the baseline (CLIP + image-proto) ranking. `topk=1` "
        "means patches can never change the prediction (baseline-only, same "
        "as Experiment D's tau=0.0); `topk=None`/`margin=1.0` is unrestricted "
        "(original always-fused behavior, same as Experiment D's tau=1.0). "
        "Deltas below are vs. the topk=1 (baseline-only) reconstruction."
    )
    lines.append("")
    for dataset in DATASETS:
        lines.append(f"## {dataset} (noise-floor threshold: {thresholds[dataset]:.3f}pp)")
        lines.append("")
        lines.append("| Mode | Param | Accs (s1..s4) | Mean | Delta vs baseline-only | p_sign | 95% CI | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for mode, param in [("topk", k) for k in TOPK_GRID] + [("margin", m) for m in MARGIN_GRID]:
            s = stats[(dataset, mode, param)]
            acc_str = " / ".join(f"{a:.2f}" for a in s["accs"])
            ci = s["ci"]
            ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "N/A"
            param_str = str(param) if param is not None else "None (full)"
            lines.append(
                f"| {mode} | {param_str} | {acc_str} | {s['mean']:.2f} "
                f"| {s['delta_mean']:+.2f}pp | {s['p_sign']:.4f} | {ci_str} | {s['verdict']} |"
            )
        lines.append("")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
