#!/usr/bin/env python3
"""Experiment K -- statistical validation at disagree_discount=0.3.

See outputs/patch_fix_experiments_plan.md ("Experiment K"). The narrower
pilot sweep (single seed) found disagree_discount=0.3 was the best point in
[0.3, 0.9] on oxford_flowers/oxford_pets for both topk20 and otsu_mean,
while dtd stayed flat/noisy. This applies the exact same statistical bar
used throughout this investigation (outputs/patch_benefit_prereg.md's
locked noise-floor thresholds, paired sign-permutation p-value, seed-level
bootstrap CI, and the flip_metrics_v2 mechanistic check) to the full
3-dataset x 4-seed runs at that setting, for both aggregation candidates.

Usage::

    python scripts/analyze_K_discount03.py
"""

import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_patch_benefit import (  # noqa: E402
    DATASETS,
    SEEDS,
    read_locked_thresholds,
    sign_permutation_pvalue,
    seed_level_bootstrap_ci,
    cell_accuracy,
)
from analyze_records import (  # noqa: E402
    load_label_records,
    flip_metrics_v2,
    resolve_tau_img,
    resolve_tau_patch_proto,
)

AGGREGATIONS = ["topk20", "otsu_mean"]
BASELINE_ARM_DIR = "outputs/records_patch_benefit"
K_ARM_DIR = "outputs/records_K"

# Known PatchModPTA-CS (vanilla patch fusion, no corroboration gate)
# accuracies -- outputs/result_patch_benefit.txt / patch_benefit_report.md.
# This is the scientifically relevant comparison for "did the gate help":
# vs. PTA-CS answers a much harder, already-answered question (does patch
# fusion beat plain PTA at all -- known REFUTE on flowers/pets before any
# of this round's fixes).
VANILLA_PATCH_ACC = {
    "dtd": [46.75, 48.23, 47.22, 46.63],
    "oxford_flowers": [71.66, 72.39, 72.39, 72.43],
    "oxford_pets": [89.48, 89.34, 88.83, 88.66],
}


def main():
    thresholds = read_locked_thresholds("outputs/patch_benefit_prereg.md")

    report = {}
    for agg in AGGREGATIONS:
        for dataset in DATASETS:
            pta_accs, k_accs = [], []
            k_samples_all_seeds = []
            k_header = None
            for seed in SEEDS:
                _, pta_samples = load_label_records(
                    BASELINE_ARM_DIR, f"PTA-CS-{dataset}-s{seed}"
                )
                pta_accs.append(cell_accuracy(pta_samples))

                k_label = f"PatchModPTA-CS-K-d03-{agg}-{dataset}-s{seed}"
                k_header, k_samples = load_label_records(K_ARM_DIR, k_label)
                k_accs.append(cell_accuracy(k_samples))
                k_samples_all_seeds.extend(k_samples)

            deltas = [k - p for k, p in zip(k_accs, pta_accs)]
            delta_mean = statistics.mean(deltas)
            p_sign = sign_permutation_pvalue(deltas)
            ci_lo, ci_hi = seed_level_bootstrap_ci(deltas)

            # Secondary, more direct comparison: does the gate improve on
            # vanilla PatchModPTA-CS (same patch method, no gate)?
            vanilla_accs = VANILLA_PATCH_ACC[dataset]
            deltas_vs_vanilla = [k - v for k, v in zip(k_accs, vanilla_accs)]
            delta_vs_vanilla_mean = statistics.mean(deltas_vs_vanilla)
            p_sign_vanilla = sign_permutation_pvalue(deltas_vs_vanilla)
            ci_vanilla_lo, ci_vanilla_hi = seed_level_bootstrap_ci(deltas_vs_vanilla)

            cfg = k_header.get("resolved_config") if k_header else None
            tau_img = resolve_tau_img(cfg)
            tau_patch = resolve_tau_patch_proto(cfg)
            flip = flip_metrics_v2(k_samples_all_seeds, tau_img=tau_img, tau_patch_proto=tau_patch)
            pi = flip["aggregate"]["patch_image"]
            patch_image_net = pi["corr"] - pi["reg"]

            threshold = thresholds[dataset]
            support = (
                delta_mean > threshold
                and p_sign <= 0.0625
                and (ci_lo is None or ci_lo > 0)
                and patch_image_net > 0
            )
            if support:
                verdict = "SUPPORT"
            elif delta_mean <= 0 or p_sign > 0.5:
                verdict = "REFUTE"
            else:
                verdict = "INCONCLUSIVE"

            report[(agg, dataset)] = {
                "pta_accs": pta_accs,
                "k_accs": k_accs,
                "vanilla_accs": vanilla_accs,
                "deltas": deltas,
                "delta_mean": delta_mean,
                "p_sign": p_sign,
                "ci": (ci_lo, ci_hi),
                "delta_vs_vanilla_mean": delta_vs_vanilla_mean,
                "p_sign_vanilla": p_sign_vanilla,
                "ci_vanilla": (ci_vanilla_lo, ci_vanilla_hi),
                "threshold": threshold,
                "patch_image_net": patch_image_net,
                "verdict": verdict,
            }

    render(report)


def render(report):
    lines = ["# Experiment K — disagree_discount=0.3 — Statistical Results", ""]
    lines.append(
        "Comparison baseline is plain PTA-CS (same convention as every prior "
        "experiment in this series). `patch_image.net` is the "
        "`flip_metrics_v2` mechanistic check (corrections minus regressions "
        "from adding the patch term on top of image-level PTA), pooled "
        "across the arm's 4-seed record sets per dataset -- must be > 0 for "
        "SUPPORT, same as Stage B's decision rule."
    )
    lines.append("")
    for agg in AGGREGATIONS:
        lines.append(f"## {agg}")
        lines.append("")
        lines.append(
            "| Dataset | PTA-CS (s1-4) | K-arm (s1-4) | Delta mean | p_sign | "
            "95% CI | patch_image.net | Threshold | Verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for dataset in DATASETS:
            r = report[(agg, dataset)]
            pta_s = " / ".join(f"{a:.2f}" for a in r["pta_accs"])
            k_s = " / ".join(f"{a:.2f}" for a in r["k_accs"])
            ci = r["ci"]
            ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "N/A"
            lines.append(
                f"| {dataset} | {pta_s} | {k_s} | {r['delta_mean']:+.2f}pp | "
                f"{r['p_sign']:.4f} | {ci_s} | {r['patch_image_net']:+d} | "
                f"{r['threshold']:.3f}pp | {r['verdict']} |"
            )
        lines.append("")

        lines.append(
            f"### {agg} — secondary comparison: vs. vanilla PatchModPTA-CS (no gate)"
        )
        lines.append("")
        lines.append(
            "The more direct \"did the gate help\" question — same patch "
            "method, gate on vs. off — not gated by the (much harder, "
            "already-known) \"beats plain PTA\" bar above."
        )
        lines.append("")
        lines.append(
            "| Dataset | Vanilla PatchModPTA-CS (s1-4) | K-arm (s1-4) | "
            "Delta mean | p_sign | 95% CI |"
        )
        lines.append("|---|---|---|---|---|---|")
        for dataset in DATASETS:
            r = report[(agg, dataset)]
            van_s = " / ".join(f"{a:.2f}" for a in r["vanilla_accs"])
            k_s = " / ".join(f"{a:.2f}" for a in r["k_accs"])
            ci = r["ci_vanilla"]
            ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "N/A"
            lines.append(
                f"| {dataset} | {van_s} | {k_s} | "
                f"{r['delta_vs_vanilla_mean']:+.2f}pp | {r['p_sign_vanilla']:.4f} | {ci_s} |"
            )
        lines.append("")

    out_path = Path("outputs/K_discount03_report.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
